//! Per-pseudobulk statistics + Gamma posterior optimization.
//!
//! Three layers:
//! - **Visitors** (`collect_basic_stat_visitor`,
//!   `collect_batch_stat_visitor`, `collect_matched_stat_visitor`):
//!   accumulate per-group sufficient statistics across the sparse data
//!   in parallel.
//! - **`CollapsedStat` / `CollapsedOut`**: the sufficient-statistic
//!   buffer and the Gamma posteriors derived from it.
//! - **`optimize`** and **`resample_and_optimize`**: fit the per-PB
//!   Gamma posteriors from a populated `CollapsedStat`.
//!
//! Also houses the cross-batch matched-stat helper used by the
//! multi-level refinement path
//! (`collect_matched_stat_coarse`) and the coarse-level descent
//! (`merge_stat`).

use super::*;
use nalgebra::DVector;

pub(super) struct KnnParams<'a> {
    pub(super) knn_batches: usize,
    pub(super) knn_cells: usize,
    pub(super) reference_indices: Option<&'a [usize]>,
}

pub(super) fn collect_matched_stat_visitor(
    sample: usize,
    cells: &[usize],
    data_vec: &SparseIoVec,
    knn_params: &KnnParams,
    arc_stat: Arc<Mutex<&mut CollapsedStat>>,
) -> anyhow::Result<()> {
    let knn_batches = knn_params.knn_batches;
    let knn_cells = knn_params.knn_cells;

    let (y0_matched, source_columns, euclidean_distances) = match knn_params.reference_indices {
        Some(target_indices) => data_vec.read_matched_columns_csc(
            cells.iter().cloned(),
            target_indices,
            knn_cells,
            true,
        )?,
        None => {
            let (mat, src, _matched, dist) = data_vec.read_neighbouring_columns_csc(
                cells.iter().cloned(),
                knn_batches,
                knn_cells,
                true,
                None,
            )?;
            (mat, src, dist)
        }
    };

    let y1_pos: HashMap<_, _> = cells
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();

    let neg_distance_triplets = source_columns
        .iter()
        .zip(euclidean_distances.iter())
        .enumerate()
        .map(|(t, (&s, &d))| (t, y1_pos[&s], -d))
        .collect::<Vec<_>>();

    ////////////////////////////////////////////////////////
    // zhat[g,j]  =  sum_k w[j,k] * z[g,k] / sum_k w[j,k] //
    // zsum[g,s]  =  sum_j zhat[g,j]                      //
    ////////////////////////////////////////////////////////

    // Normalize distance for each source cell and take a
    // weighted average of the matched vectors using this
    // weight vector
    let ww = CscMat::from_nonzero_triplets(
        y0_matched.ncols(),
        cells.len(),
        neg_distance_triplets.as_ref(),
    )?
    .normalize_exp_logits_columns();

    let y1_hat = &y0_matched * &ww;
    let source_batches = data_vec.get_batch_membership(source_columns.iter().cloned());

    let mut stat = arc_stat.lock().expect("lock stat");

    // Source mass per batch: each cell's weights sum to 1 over its matches.
    for w_j in ww.col_iter() {
        for (&k, &w) in w_j.row_indices().iter().zip(w_j.values().iter()) {
            stat.matched_bs[(source_batches[k], sample)] += w;
        }
    }

    for y_j in y1_hat.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.imputed_sum_ds[(gene, sample)] += y;
        }
    }

    Ok(())
}

pub(super) fn collect_basic_stat_visitor(
    sample: usize,
    cells: &[usize],
    data_vec: &SparseIoVec,
    _: &EmptyArg,
    arc_stat: Arc<Mutex<&mut CollapsedStat>>,
) -> anyhow::Result<()> {
    let yy = data_vec.read_columns_csc(cells.iter().cloned())?;

    let mut stat = arc_stat.lock().expect("lock stat");

    // `w` is how many observations the column stands for — 1 for a cell, `m`
    // for a column carrying the mean profile of `m` cells. Scaling both the
    // counts and the size keeps `μ = Σy / n` the per-cell rate either way.
    for (y_j, &col) in yy.col_iter().zip(cells.iter()) {
        let w = data_vec.column_multiplicity(col);
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.observed_sum_ds[(gene, sample)] += y * w;
        }
        stat.size_s[sample] += w;
    }
    Ok(())
}

pub(super) fn collect_batch_stat_visitor(
    sample: usize,
    cells_in_sample: &[usize],
    data_vec: &SparseIoVec,
    _: &EmptyArg,
    arc_stat: Arc<Mutex<&mut CollapsedStat>>,
) -> anyhow::Result<()> {
    let yy = data_vec.read_columns_csc(cells_in_sample.iter().cloned())?;

    // cells_in_sample: sample s -> cell j
    // batches: cell j -> batch b
    let batches = data_vec.get_batch_membership(cells_in_sample.iter().cloned());

    let mut stat = arc_stat.lock().expect("lock stat");

    yy.col_iter()
        .zip(batches.iter())
        .zip(cells_in_sample.iter())
        .for_each(|((y_j, &b), &col)| {
            let w = data_vec.column_multiplicity(col);
            let rows = y_j.row_indices();
            let vals = y_j.values();
            for (&gene, &y) in rows.iter().zip(vals.iter()) {
                stat.observed_sum_db[(gene, b)] += y * w;
            }
            stat.n_bs[(b, sample)] += w;
        });
    Ok(())
}

/// Per-feature-block kernel for [`optimize`]. Runs the DC-Poisson
/// coordinate descent on a (sub)stat and returns a `CollapsedOut` whose row
/// count matches `stat.num_genes()`. When `prog` is `Some`, ticks it once per
/// descent iteration (batched path only) so the driver's bar keeps moving
/// while a block is mid-fit; `inc` is atomic, so concurrent blocks may share
/// one bar.
/// `denom += effective size`: per-(gene, sample) when observability is
/// attached, the historical per-sample scalar otherwise. The `None` arm is
/// the exact old code — full observability is bitwise-identical by
/// construction, not by tolerance.
fn add_effective_size(denom_ds: &mut nalgebra::DMatrix<f32>, stat: &CollapsedStat) {
    match stat.size_ds.as_ref() {
        Some(size_ds) => {
            debug_assert_eq!(denom_ds.shape(), size_ds.shape());
            *denom_ds += size_ds;
        }
        None => {
            for s in 0..denom_ds.ncols() {
                denom_ds.column_mut(s).add_scalar_mut(stat.size_s[s]);
            }
        }
    }
}

/// Rescale δ's Gamma denominators so that, per gene, the frame batches'
/// weighted geometric mean of the posterior mean `(a0 + num) / (b0 + den)` is
/// exactly 1. A batch masked out for a gene (`obs_mask_db == 0`) sits on the
/// prior and does not vote. Returns the adjusted denominators.
fn pin_delta_scale(
    num_db: &DMatrix<f32>,
    den_db: &DMatrix<f32>,
    (a0, b0): (f32, f32),
    frame_w: &DVector<f32>,
    mask_db: Option<&DMatrix<f32>>,
) -> DMatrix<f32> {
    let (ng, nb) = num_db.shape();
    let mut out = den_db.clone();
    for g in 0..ng {
        let (mut lsum, mut wsum) = (0f64, 0f64);
        for b in 0..nb {
            let w = f64::from(frame_w[b]) * mask_db.map_or(1.0, |m| f64::from(m[(g, b)]));
            if w > 0.0 {
                let m = f64::from(a0 + num_db[(g, b)]) / f64::from(b0 + den_db[(g, b)]);
                lsum += w * m.max(f64::MIN_POSITIVE).ln();
                wsum += w;
            }
        }
        if wsum > 0.0 {
            let gm = (lsum / wsum).exp() as f32;
            for b in 0..nb {
                // A masked entry carries no evidence and stays on the prior.
                if mask_db.is_some_and(|m| m[(g, b)] == 0.0) {
                    continue;
                }
                // (a0 + num) / (b0 + den') = mean / gm  ⇔  den' = (b0 + den)·gm − b0.
                // The clamp only binds when gm < 1 on an entry with less than a
                // cell's worth of evidence, where the prior dominates anyway.
                out[(g, b)] = ((b0 + den_db[(g, b)]) * gm - b0).max(0.0);
            }
        }
    }
    out
}

fn optimize_block(
    stat: &CollapsedStat,
    hyper: (f32, f32),
    num_iter: usize,
    out_target: CalibrateTarget,
    prog: Option<&indicatif::ProgressBar>,
) -> anyhow::Result<CollapsedOut> {
    let (a0, b0) = hyper;
    let num_genes = stat.num_genes();
    let num_samples = stat.num_samples();
    let num_batches = stat.num_batches();
    let mut mu_param = GammaMatrix::new((num_genes, num_samples), a0, b0);

    if num_batches > 1 {
        //////////////////////////////////////////////////////////////////
        // One frame for every batch                                    //
        //                                                              //
        //   E[observed_gs] = μ_gs · Σ_b δ_gb · n_bs        (own cells) //
        //   E[imputed_gs]  = μ_gs · Σ_b δ_gb · w_bs   (counterfactual) //
        //                                                              //
        // μ is the batch-free rate, δ_gb the per-batch fold, n_bs the  //
        // own mass and w_bs the counterfactual's source mass. Both     //
        // sides of a pseudobulk are Poisson at the SAME μ, so μ cannot //
        // slide into another batch's frame; δ is identified by the     //
        // counterfactual side and pinned per gene to geometric mean 1  //
        // over the frame batches (all of them, or the anchors).        //
        //////////////////////////////////////////////////////////////////
        let n_bs = &stat.n_bs;
        let w_bs = &stat.matched_bs;
        let own_plus_src = n_bs + w_bs; // [b × s]
        let obs_plus_imp = &stat.observed_sum_ds + &stat.imputed_sum_ds;
        // Fraction of each (gene, sample)'s mass whose source measures the gene;
        // `None` = fully observed.
        let obs_frac: Option<DMatrix<f32>> = stat.size_ds.as_ref().map(|size_ds| {
            DMatrix::from_fn(num_genes, num_samples, |g, s| {
                let n = stat.size_s[s];
                if n > 0.0 {
                    size_ds[(g, s)] / n
                } else {
                    0.0
                }
            })
        });
        let frame_w = stat.frame_weights();
        let own_plus_src_t = own_plus_src.transpose(); // [s × b]
        let w_bs_t = w_bs.transpose();

        let mut mu_adj_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut delta_param = GammaMatrix::new((num_genes, num_batches), a0, b0);
        let mut delta_gb = DMatrix::<f32>::from_element(num_genes, num_batches, 1.0);

        // μ given δ: (obs + imp) / (frac · Σ_b δ_gb (n_bs + w_bs))
        let update_mu = |mu_adj_param: &mut GammaMatrix, delta_gb: &DMatrix<f32>| {
            let mut denom_ds = delta_gb * &own_plus_src;
            if let Some(f) = obs_frac.as_ref() {
                denom_ds.component_mul_assign(f);
            }
            mu_adj_param.update_stat(&obs_plus_imp, &denom_ds);
            mu_adj_param.calibrate_with(CalibrateTarget::MeanOnly);
        };

        // Scratch planes, allocated once: every [g × s] temporary below is
        // rewritten in place each iteration rather than reallocated.
        let mut imp_share = DMatrix::<f32>::zeros(num_genes, num_samples);
        let mut mu_frac = DMatrix::<f32>::zeros(num_genes, num_samples);
        for _opt_iter in 0..num_iter {
            #[cfg(debug_assertions)]
            {
                debug!("iteration: {}", &_opt_iter);
            }

            update_mu(&mut mu_adj_param, &delta_gb);
            let mu_ds = mu_adj_param.posterior_mean();

            // δ given μ (one EM step): the observed side is exact per batch
            // (`observed_sum_db`); the counterfactual side splits each
            // sample's imputed sum over its source batches in proportion to
            // δ_gb · w_bs.
            //   num_gb = obs_db + Σ_s imp_gs · δ_gb w_bs / Σ_b' δ_gb' w_b's
            //   den_gb = Σ_s μ_gs · frac_gs · (n_bs + w_bs)
            imp_share.gemm(1.0, &delta_gb, w_bs, 0.0); // Σ_b δ_gb w_bs
            imp_share.zip_apply(&stat.imputed_sum_ds, |z, x| {
                *z = if *z > 0.0 { x / *z } else { 0.0 };
            });
            let mut num_db =
                &stat.observed_sum_db + (&imp_share * &w_bs_t).component_mul(&delta_gb);
            let mu_frac_ref: &DMatrix<f32> = match obs_frac.as_ref() {
                Some(f) => {
                    mu_frac.copy_from(mu_ds);
                    mu_frac.component_mul_assign(f);
                    &mu_frac
                }
                None => mu_ds,
            };
            let mut den_db = mu_frac_ref * &own_plus_src_t; // [g × b]
            if let Some(mask) = stat.obs_mask_db.as_ref() {
                // Zeroing a (gene, batch) entry means "this batch carries no δ
                // evidence for this gene"; masking BOTH sides lands the
                // posterior on the prior exactly (≈ 1, "no adjustment").
                num_db.component_mul_assign(mask);
                den_db.component_mul_assign(mask);
            }
            // Pin the per-gene scale: the frame batches' weighted geometric
            // mean of δ is 1. Folded into the denominators so the Gamma
            // posterior itself carries the normalised value.
            let den_db = pin_delta_scale(
                &num_db,
                &den_db,
                (a0, b0),
                &frame_w,
                stat.obs_mask_db.as_ref(),
            );
            delta_param.update_stat(&num_db, &den_db);
            delta_param.calibrate_with(CalibrateTarget::MeanOnly);
            delta_gb.copy_from(delta_param.posterior_mean());

            if let Some(p) = prog {
                p.inc(1);
            }
        }
        // μ consistent with the final δ.
        update_mu(&mut mu_adj_param, &delta_gb);
        mu_adj_param.calibrate_with(out_target);
        delta_param.calibrate_with(out_target);

        // Per-pseudobulk readouts against the batch-free μ, on the documented
        // scale: E[y] = μ_resid · μ (own fold) and E[ŷ] = γ · μ (source fold).
        // Each denominator is μ times a per-sample mass (own cells, or the
        // counterfactual's matched cells), scaled by the observed fraction when
        // a panel gap makes it less than full.
        let mu_ds = mu_adj_param.posterior_mean();
        let mass_times_mu = |per_sample: &dyn Fn(usize) -> f32| {
            let mut m = mu_ds.clone();
            for s in 0..num_samples {
                m.column_mut(s).scale_mut(per_sample(s));
            }
            if let Some(f) = obs_frac.as_ref() {
                m.component_mul_assign(f);
            }
            m
        };
        let resid_denom = mass_times_mu(&|s| stat.size_s[s]);
        let mut mu_resid_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        mu_resid_param.update_stat(&stat.observed_sum_ds, &resid_denom);
        mu_resid_param.calibrate_with(out_target);

        let src_mass_s: Vec<f32> = (0..num_samples).map(|s| w_bs.column(s).sum()).collect();
        let gamma_denom = mass_times_mu(&|s| src_mass_s[s]);
        let mut gamma_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        gamma_param.update_stat(&stat.imputed_sum_ds, &gamma_denom);
        gamma_param.calibrate_with(out_target);

        // Take the observed mean over the own mass.
        {
            let mut own_mass = DMatrix::<f32>::zeros(num_genes, num_samples);
            add_effective_size(&mut own_mass, stat);
            mu_param.update_stat(&stat.observed_sum_ds, &own_mass);
            mu_param.calibrate_with(out_target);
        };

        // Sparse output (bge / MeanOnly): drop each mean's per-column prior
        // baseline so its support is exactly the observed∪imputed cells.
        // Downstream `from_pseudobulks` then yields sparse pb_blobs (held
        // across all training epochs). `All` consumers keep the dense mean.
        if matches!(out_target, CalibrateTarget::MeanOnly) {
            mu_param.sparsify_mean_to_support(&stat.observed_sum_ds);
            mu_adj_param.sparsify_mean_to_support(&obs_plus_imp);
            gamma_param.sparsify_mean_to_support(&stat.imputed_sum_ds);
            mu_resid_param.sparsify_mean_to_support(&stat.observed_sum_ds);
        }

        Ok(CollapsedOut {
            mu_observed: mu_param,
            mu_adjusted: Some(mu_adj_param),
            mu_residual: Some(mu_resid_param),
            gamma: Some(gamma_param),
            delta: Some(delta_param),
        })
    } else {
        let mut denom_ds = DMatrix::<f32>::zeros(num_genes, num_samples);
        add_effective_size(&mut denom_ds, stat);
        mu_param.update_stat(&stat.observed_sum_ds, &denom_ds);
        mu_param.calibrate_with(out_target);
        if matches!(out_target, CalibrateTarget::MeanOnly) {
            mu_param.sparsify_mean_to_support(&stat.observed_sum_ds);
        }
        Ok(CollapsedOut {
            mu_observed: mu_param,
            mu_adjusted: None,
            mu_residual: None,
            gamma: None,
            delta: None,
        })
    }
}

/// Optimize the mean parameters for the DC-Poisson collapse, **blocked over
/// feature rows** so peak working memory scales with a block, not the full
/// feature axis. Every update is elementwise per `(gene, sample)` and δ is
/// per-gene given the shared per-sample / per-batch sizes, so the fit is
/// separable across features — block-independent descent is numerically
/// identical to the joint fit. For `MeanOnly` output the heavy
/// `a_stat`/`b_stat` planes are dropped per block, so the assembled result
/// carries only posterior estimates (bge never calls `posterior_sample`).
pub(super) fn optimize(
    stat: &CollapsedStat,
    hyper: (f32, f32),
    num_iter: usize,
    label: &str,
    out_target: CalibrateTarget,
    // Retain a_stat/b_stat even under MeanOnly — the finest level of an
    // `--emit-pb-reference` run serializes `evidence_mean`, which reads them.
    keep_stats: bool,
) -> anyhow::Result<CollapsedOut> {
    let num_genes = stat.num_genes();
    let num_samples = stat.num_samples();
    let num_batches = stat.num_batches();

    // Block width bounds a single working plane, not the whole fit. Each block
    // holds ~19 `block_rows × num_samples` planes live (4 Gamma params × 3
    // planes + sufficient stats + descent scratch), so a 2M-element plane
    // (~8 MB in f32) caps a block at ~150 MB regardless of how many features
    // the panel carries. `block_rows ≈ 2M/num_samples` keeps the plane fixed
    // at every refinement level (few samples → wider blocks, same plane).
    //
    // The old 32M target barely blocked at all: at `k=1024` it gave
    // `block_rows ≈ 31k`, so a ~41k-gene panel split into just 2 uneven blocks
    // and peak sat at ~76% of the un-blocked cost. 2M gives ~21 even blocks.
    const BLOCK_ELEMS: usize = 2_000_000;
    let block_rows = (BLOCK_ELEMS / num_samples.max(1)).clamp(1, num_genes.max(1));
    // Same crate helper the sparse block-visitors use; returns half-open
    // `(lb, ub)` gene ranges of `block_rows` each (last one short).
    let jobs = create_jobs(num_genes, num_samples, Some(block_rows));
    let n_blocks = jobs.len();

    let dims = format!("{num_genes} genes × {num_samples} samples");

    // `posterior_sample` (topic path) reads a_stat/b_stat; bge (MeanOnly)
    // does not, so those planes can be discarded per block — that's what
    // keeps the assembled output from holding the full sufficient stats.
    let keep_stats = keep_stats || matches!(out_target, CalibrateTarget::All);

    // The moving unit is one descent iteration whenever the batch-correction
    // loop runs: each block ticks `num_iter` times (`optimize_block` handed the
    // bar), so the bar is `n_blocks × num_iter` ticks and keeps advancing even
    // while a single block is mid-fit. Without batches a block is one
    // closed-form pass with no inner loop, so there the unit is the block and
    // the loop below ticks once per finished block instead.
    let batched = num_batches > 1;
    let total = if batched {
        n_blocks * num_iter
    } else {
        n_blocks
    };
    let msg = if batched {
        format!("{label} opt-iters · {dims} · {n_blocks} blocks")
    } else {
        format!("{label} gene-blocks · {dims}")
    };

    // One block covers the whole panel: `select_rows` would clone the full
    // stat for nothing, and there is no job to parallelize. A moving
    // iteration bar (batched) still beats a spinner; without batches there is
    // nothing to count, so a spinner avoids a `0/1` bar frozen at zero.
    if n_blocks <= 1 {
        if batched {
            let prog = styled_progress_bar(total as u64, &msg);
            let out = optimize_block(stat, hyper, num_iter, out_target, Some(&prog));
            prog.finish_and_clear();
            return out;
        }
        let spin = matrix_util::progress::new_spinner("{spinner} [{elapsed_precise}] {msg}")
            .with_message(format!("{label} single gene-block · {dims}"));
        let out = optimize_block(stat, hyper, num_iter, out_target, None);
        spin.finish_and_clear();
        return out;
    }

    // The gene axis is separable — a block's fit depends only on its own rows
    // and the shared per-sample sizes, never on other genes — so the blocks
    // are independent jobs run concurrently. Peak memory is now
    // (rayon width × ~150 MB), bounded by the plane cap above rather than by
    // the feature count.
    let prog = styled_progress_bar(total as u64, &msg);

    // `.map(..).collect::<Result<Vec>>()` preserves input order, so blocks
    // reassemble in gene order. `prog.inc` is atomic: batched blocks tick per
    // iteration inside `optimize_block`; otherwise tick once per finished block.
    let outs = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<CollapsedOut> {
            let sub = stat.select_rows(lb, ub - lb);
            let mut out_b =
                optimize_block(&sub, hyper, num_iter, out_target, batched.then_some(&prog))?;
            if !keep_stats {
                // Free a_stat/b_stat now so the accumulated blocks never add
                // up to the full sufficient-stat planes.
                out_b.release_stats();
            }
            if !batched {
                prog.inc(1);
            }
            Ok(out_b)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    prog.finish_and_clear();

    let mut mu_obs: Vec<GammaMatrix> = Vec::with_capacity(n_blocks);
    let mut mu_adj: Vec<GammaMatrix> = Vec::new();
    let mut mu_res: Vec<GammaMatrix> = Vec::new();
    let mut gam: Vec<GammaMatrix> = Vec::new();
    let mut del: Vec<GammaMatrix> = Vec::new();
    for out_b in outs {
        mu_obs.push(out_b.mu_observed);
        if let Some(x) = out_b.mu_adjusted {
            mu_adj.push(x);
        }
        if let Some(x) = out_b.mu_residual {
            mu_res.push(x);
        }
        if let Some(x) = out_b.gamma {
            gam.push(x);
        }
        if let Some(x) = out_b.delta {
            del.push(x);
        }
    }

    let join = |v: Vec<GammaMatrix>| -> Option<GammaMatrix> {
        (!v.is_empty()).then(|| GammaMatrix::vconcat(v, keep_stats))
    };
    Ok(CollapsedOut {
        mu_observed: GammaMatrix::vconcat(mu_obs, keep_stats),
        mu_adjusted: join(mu_adj),
        mu_residual: join(mu_res),
        gamma: join(gam),
        delta: join(del),
    })
}

/// output struct to make the model parameters more accessible
#[derive(Debug, Clone)]
pub struct CollapsedOut {
    pub mu_observed: GammaMatrix,
    pub mu_adjusted: Option<GammaMatrix>,
    pub mu_residual: Option<GammaMatrix>,
    pub gamma: Option<GammaMatrix>,
    pub delta: Option<GammaMatrix>,
}

impl CollapsedOut {
    /// Drop `a_stat`/`b_stat` on every contained parameter. Safe when the
    /// consumer reads posterior means/log-means but never `posterior_sample`
    /// (bge). Used by the gene-blocked `optimize` to keep accumulated blocks
    /// from summing to the full sufficient-stat planes.
    fn release_stats(&mut self) {
        self.mu_observed.release_stats();
        for p in [
            &mut self.mu_adjusted,
            &mut self.mu_residual,
            &mut self.gamma,
            &mut self.delta,
        ] {
            if let Some(g) = p.as_mut() {
                g.release_stats();
            }
        }
    }
}

/// a struct to hold the sufficient statistics for the model
#[derive(Debug, Clone)]
pub struct CollapsedStat {
    pub observed_sum_ds: nalgebra::DMatrix<f32>, // observed sum within each sample
    pub imputed_sum_ds: nalgebra::DMatrix<f32>,  // counterfactual sum within each sample
    pub size_s: nalgebra::DVector<f32>,          // sample s size
    pub observed_sum_db: nalgebra::DMatrix<f32>, // divergence numerator
    pub n_bs: nalgebra::DMatrix<f32>,            // batch-specific sample size
    /// Counterfactual SOURCE mass per (batch, sample): how much of sample `s`'s
    /// imputed profile was drawn from batch `b`'s cells (each matched cell's
    /// weights sum to 1, so a column sums to the matched cell count). With
    /// `n_bs` this is what identifies `δ`: `E[imputed_gs] = μ_gs Σ_b δ_gb w_bs`
    /// against `E[observed_gs] = μ_gs Σ_b δ_gb n_bs`.
    pub matched_bs: nalgebra::DMatrix<f32>,
    /// Batches whose frame IS the frame: `δ` is normalised so their per-gene
    /// geometric mean is 1. Empty = pooled, where every batch's mass weighs in.
    pub anchor_batches: Vec<usize>,
    /// Per-(gene, sample) effective size: the mass of sample `s` whose SOURCE
    /// actually measures gene `g`. `None` means fully observed — every gene
    /// sees `size_s[s]`, and `optimize` takes the exact historical code path.
    ///
    /// This is what distinguishes "unmeasured" from "measured as zero": a
    /// gene absent from one cohort's panel reads 0 in all of its cells, and
    /// without this denominator the estimate is silently dragged toward zero
    /// by cells that never looked. Set by [`attach_observability`]; see the
    /// plan note there for the factorized construction.
    pub size_ds: Option<nalgebra::DMatrix<f32>>,
    /// Per-(gene, batch) observability of δ's evidence: 1.0 when any source
    /// in batch `b` measures gene `g`, else 0.0. Zeroing δ's denominator
    /// there sends δ to its prior (≈ 1, "no adjustment") instead of to an
    /// extreme driven by structurally-absent counts. `None` = all observed.
    pub obs_mask_db: Option<nalgebra::DMatrix<f32>>,
}

impl CollapsedStat {
    pub fn new(ngene: usize, nsample: usize, nbatch: usize) -> Self {
        Self {
            observed_sum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            imputed_sum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            size_s: nalgebra::DVector::<f32>::zeros(nsample),
            observed_sum_db: nalgebra::DMatrix::<f32>::zeros(ngene, nbatch),
            n_bs: nalgebra::DMatrix::<f32>::zeros(nbatch, nsample),
            matched_bs: nalgebra::DMatrix::<f32>::zeros(nbatch, nsample),
            anchor_batches: Vec::new(),
            size_ds: None,
            obs_mask_db: None,
        }
    }

    pub fn num_genes(&self) -> usize {
        self.observed_sum_ds.nrows()
    }

    pub fn num_samples(&self) -> usize {
        self.observed_sum_ds.ncols()
    }

    pub fn num_batches(&self) -> usize {
        self.observed_sum_db.ncols()
    }

    /// Per-batch weights that define the frame `δ` is pinned to: the anchors
    /// when there are any (each weighted 1), else every batch by its own cell
    /// mass.
    pub fn frame_weights(&self) -> DVector<f32> {
        let nb = self.num_batches();
        if self.anchor_batches.is_empty() {
            DVector::from_fn(nb, |b, _| self.n_bs.row(b).sum())
        } else {
            let mut w = DVector::<f32>::zeros(nb);
            for &b in &self.anchor_batches {
                w[b] = 1.0;
            }
            w
        }
    }

    pub fn clear(&mut self) {
        self.observed_sum_ds.fill(0_f32);
        self.imputed_sum_ds.fill(0_f32);
        self.observed_sum_db.fill(0_f32);
        self.size_s.fill(0_f32);
        self.n_bs.fill(0_f32);
        self.matched_bs.fill(0_f32);
        self.anchor_batches.clear();
        self.size_ds = None;
        self.obs_mask_db = None;
    }

    /// Select a subset of sample columns (groups) by index.
    pub fn select_columns(&self, indices: &[usize]) -> Self {
        let n_new = indices.len();
        let ng = self.num_genes();
        let nb = self.num_batches();
        let mut out = Self::new(ng, n_new, nb);
        for (new_col, &old_col) in indices.iter().enumerate() {
            out.observed_sum_ds
                .column_mut(new_col)
                .copy_from(&self.observed_sum_ds.column(old_col));
            out.imputed_sum_ds
                .column_mut(new_col)
                .copy_from(&self.imputed_sum_ds.column(old_col));
            out.size_s[new_col] = self.size_s[old_col];
            for b in 0..nb {
                out.n_bs[(b, new_col)] = self.n_bs[(b, old_col)];
                out.matched_bs[(b, new_col)] = self.matched_bs[(b, old_col)];
            }
        }
        if let Some(size_ds) = self.size_ds.as_ref() {
            out.size_ds = Some(nalgebra::DMatrix::from_fn(ng, n_new, |g, new_col| {
                size_ds[(g, indices[new_col])]
            }));
        }
        out.obs_mask_db = self.obs_mask_db.clone();
        out.observed_sum_db.copy_from(&self.observed_sum_db);
        out.anchor_batches = self.anchor_batches.clone();
        out
    }

    /// Select a contiguous block of feature rows (`r0..r0+nrows`). Per-gene
    /// stats (`observed`/`imputed`/`observed_db`) are sliced; the per-sample
    /// `size_s` and per-batch `n_bs`/`matched_bs` are shared, so they're copied
    /// whole. Used by the gene-blocked `optimize`.
    pub fn select_rows(&self, r0: usize, nrows: usize) -> Self {
        Self {
            observed_sum_ds: self.observed_sum_ds.rows(r0, nrows).into_owned(),
            imputed_sum_ds: self.imputed_sum_ds.rows(r0, nrows).into_owned(),
            size_s: self.size_s.clone(),
            observed_sum_db: self.observed_sum_db.rows(r0, nrows).into_owned(),
            n_bs: self.n_bs.clone(),
            matched_bs: self.matched_bs.clone(),
            anchor_batches: self.anchor_batches.clone(),
            size_ds: self
                .size_ds
                .as_ref()
                .map(|m| m.rows(r0, nrows).into_owned()),
            obs_mask_db: self
                .obs_mask_db
                .as_ref()
                .map(|m| m.rows(r0, nrows).into_owned()),
        }
    }
}

/// Resample from over-resolved sufficient statistics: randomly select
/// ~half the groups, then optimise to produce a fresh `CollapsedOut`.
pub fn resample_and_optimize(
    stat: &CollapsedStat,
    rng: &mut impl rand::Rng,
    opt_iter: usize,
) -> anyhow::Result<CollapsedOut> {
    use rand::seq::SliceRandom;
    let n = stat.num_samples();
    let target = n / 2;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    indices.truncate(target);
    indices.sort_unstable();
    let sub_stat = stat.select_columns(&indices);
    optimize(
        &sub_stat,
        (1.0, 1.0),
        opt_iter,
        "Optimizing",
        CalibrateTarget::All,
        false,
    )
}

///////////////////////////////////////////////////////////////
// Multi-level (METIS-inspired) collapsing for batch effects //
///////////////////////////////////////////////////////////////

pub(super) const DEFAULT_NUM_LEVELS: usize = 2;
pub(super) const DEFAULT_COARSEST_SORT_DIM: usize = 7;

/// Cross-batch matched-stat accumulation on top of the pb-sample
/// layout. For each fine-group, fetches `knn` matched cells from each
/// non-own batch via `batch_knn_lookup`, dedupes them through their
/// pb-samples, and emits the counterfactual gene sums into
/// `stat.imputed_sum_ds` and the per-batch source mass into `stat.matched_bs`.
pub(super) fn collect_matched_stat_coarse(
    layout: &PbSampleLayout,
    gene_sums: &[Vec<(usize, f32)>],
    pbsamp_to_group: &[usize],
    batch_knn_lookup: &[ColumnDict<usize>],
    knn: usize,
    anchor_batches: Option<&[usize]>,
    stat: &mut CollapsedStat,
) -> anyhow::Result<()> {
    let num_pb = layout.cell_counts.len();
    debug_assert_eq!(pbsamp_to_group.len(), num_pb);

    // The batches that source the counterfactual are the batches that pin δ:
    // recorded here, where that mass is written, so no caller can forget.
    stat.anchor_batches = anchor_batches.map(<[usize]>::to_vec).unwrap_or_default();
    let neighbors_per_sc = per_batch_sc_neighbors(layout, batch_knn_lookup, knn, anchor_batches)?;

    use indicatif::ParallelProgressIterator;
    let prog_bar = styled_progress_bar(num_pb as u64, "pb-samples (matched stats)");
    let arc_stat = Arc::new(Mutex::new(stat));

    (0..num_pb)
        .into_par_iter()
        .progress_with(prog_bar.clone())
        .for_each(|pbsamp_idx| {
            let pbsamp_group = pbsamp_to_group[pbsamp_idx];
            let sc_count = layout.cell_counts[pbsamp_idx];

            if sc_count < 1.0 {
                return;
            }

            let filtered = &neighbors_per_sc[pbsamp_idx];
            if filtered.is_empty() {
                return;
            }

            // Softmax weights from negative distances
            let max_neg_d = filtered
                .iter()
                .map(|(_, d)| -d)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut weights: Vec<f32> = filtered
                .iter()
                .map(|(_, d)| (-d - max_neg_d).exp())
                .collect();
            let w_sum: f32 = weights.iter().sum();
            if w_sum > 0.0 {
                weights.iter_mut().for_each(|w| *w /= w_sum);
            }

            // Counterfactual: weighted average of matched pb-samples'
            // per-cell gene expression
            // y_hat[g] = sum_k w[k] * gene_sums[k][g] / cell_counts[k]
            let mut y_hat: HashMap<usize, f32> = HashMap::default();
            for ((matched_sc, _), &w) in filtered.iter().zip(weights.iter()) {
                let matched_count = layout.cell_counts[*matched_sc];
                if matched_count < 1.0 {
                    continue;
                }
                let inv_count = 1.0 / matched_count;
                for &(gene, val) in &gene_sums[*matched_sc] {
                    *y_hat.entry(gene).or_default() += w * val * inv_count;
                }
            }

            let mut stat = arc_stat.lock().expect("lock stat");

            // Accumulate imputed_sum_ds[g, s] += cell_counts[pbsamp] * y_hat[g]
            for (&gene, &y) in &y_hat {
                stat.imputed_sum_ds[(gene, pbsamp_group)] += sc_count * y;
            }

            // Source mass per batch: this pb-sample's `sc_count` cells split their
            // unit weight over the matched pb-samples' batches.
            for ((matched_sc, _), &w) in filtered.iter().zip(weights.iter()) {
                if layout.cell_counts[*matched_sc] < 1.0 {
                    continue;
                }
                stat.matched_bs[(layout.pb_sample_to_batch[*matched_sc], pbsamp_group)] +=
                    sc_count * w;
            }
        });

    prog_bar.finish_and_clear();
    Ok(())
}

/// Format a per-cell group-index vector as fixed-width zero-padded strings
/// so that `SparseIoVec::assign_groups`' lexicographic key sort agrees with
/// numeric order. `k` is the number of distinct groups (`group ∈ 0..k`).
/// (batch-marginal).
pub(super) fn merge_stat(
    fine_stat: &CollapsedStat,
    fine_to_coarse: &[usize],
    num_coarse_groups: usize,
) -> CollapsedStat {
    let num_genes = fine_stat.num_genes();
    let num_batches = fine_stat.num_batches();
    let mut coarse = CollapsedStat::new(num_genes, num_coarse_groups, num_batches);

    for (fine_g, &coarse_g) in fine_to_coarse.iter().enumerate() {
        coarse
            .observed_sum_ds
            .column_mut(coarse_g)
            .add_assign(&fine_stat.observed_sum_ds.column(fine_g));
        coarse
            .imputed_sum_ds
            .column_mut(coarse_g)
            .add_assign(&fine_stat.imputed_sum_ds.column(fine_g));
        coarse.size_s[coarse_g] += fine_stat.size_s[fine_g];
        for b in 0..num_batches {
            coarse.n_bs[(b, coarse_g)] += fine_stat.n_bs[(b, fine_g)];
            coarse.matched_bs[(b, coarse_g)] += fine_stat.matched_bs[(b, fine_g)];
        }
    }

    // Observability descends with the merge: effective sizes add like any
    // other per-sample mass, and the per-batch mask is sample-free.
    if let Some(fine_size_ds) = fine_stat.size_ds.as_ref() {
        let mut size_ds = nalgebra::DMatrix::<f32>::zeros(num_genes, num_coarse_groups);
        for (fine_g, &coarse_g) in fine_to_coarse.iter().enumerate() {
            size_ds
                .column_mut(coarse_g)
                .add_assign(&fine_size_ds.column(fine_g));
        }
        coarse.size_ds = Some(size_ds);
    }
    coarse.obs_mask_db = fine_stat.obs_mask_db.clone();
    coarse.anchor_batches = fine_stat.anchor_batches.clone();

    coarse.observed_sum_db.copy_from(&fine_stat.observed_sum_db);
    coarse
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod gene_block_tests;
