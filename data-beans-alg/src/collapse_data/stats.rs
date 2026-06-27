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

    let mut y1 = data_vec.read_columns_csc(cells.iter().cloned())?;

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
        y1.ncols(),
        neg_distance_triplets.as_ref(),
    )?
    .normalize_exp_logits_columns();

    let y1_hat = &y0_matched * ww;
    y1.adjust_by_division_inplace(&y1_hat);

    let mut stat = arc_stat.lock().expect("lock stat");

    for y_j in y1_hat.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.imputed_sum_ds[(gene, sample)] += y;
        }
    }

    for y_j in y1.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.residual_sum_ds[(gene, sample)] += y;
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

    for y_j in yy.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.observed_sum_ds[(gene, sample)] += y;
        }
        stat.size_s[sample] += 1_f32; // each column is a sample
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

    yy.col_iter().zip(batches.iter()).for_each(|(y_j, &b)| {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.observed_sum_db[(gene, b)] += y;
        }
        stat.n_bs[(b, sample)] += 1_f32;
    });
    Ok(())
}

/// Per-feature-block kernel for [`optimize`]. Runs the DC-Poisson
/// coordinate descent on a (sub)stat and returns a `CollapsedOut` whose row
/// count matches `stat.num_genes()`. `progress_label` shows the
/// per-iteration bar only when this is run as a single block; gene-blocked
/// runs pass `None` and let the driver show a block-level bar.
fn optimize_block(
    stat: &CollapsedStat,
    hyper: (f32, f32),
    num_iter: usize,
    out_target: CalibrateTarget,
    progress_label: Option<&str>,
) -> anyhow::Result<CollapsedOut> {
    let (a0, b0) = hyper;
    let num_genes = stat.num_genes();
    let num_samples = stat.num_samples();
    let num_batches = stat.num_batches();
    let mut mu_param = GammaMatrix::new((num_genes, num_samples), a0, b0);

    if num_batches > 1 {
        // temporary denominator
        let mut denom_ds = nalgebra::DMatrix::<f32>::zeros(num_genes, num_samples);

        // parameters
        let mut mu_adj_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut mu_resid_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut gamma_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut delta_param = GammaMatrix::new((num_genes, num_batches), a0, b0);

        ////////////////////////////////////
        // E[y_resid] = E[μ_resid]        //
        //       E[y] = E[μ_resid] * E[μ] //
        //   E[y_hat] = E[γ] * E[μ]       //
        //   E[y_bat] = E[δ] * E[μ]       //
        ////////////////////////////////////

        //            residual_sum_ds
        // μ_resid = -----------------
        //            1_d * size_s'

        {
            for s in 0..num_samples {
                denom_ds.column_mut(s).add_scalar_mut(stat.size_s[s]);
            }
            mu_resid_param.update_stat(&stat.residual_sum_ds, &denom_ds);
            // mu_resid is fixed across the loop (read via posterior_mean);
            // calibrate to the output target now so it carries sd/log only
            // when the caller actually needs them.
            mu_resid_param.calibrate_with(out_target);
        };

        let prog_bar = progress_label.map(|label| {
            matrix_util::progress::new_progress_bar(num_iter as u64)
                .with_message(format!("{label} iterations"))
        });
        for _opt_iter in 0..num_iter {
            #[cfg(debug_assertions)]
            {
                debug!("iteration: {}", &_opt_iter);
            }

            let resid_ds = mu_resid_param.posterior_mean();
            let gamma_ds = gamma_param.posterior_mean();

            //      observed_ds + imputed_sum_ds
            // μ = ---------------------------------
            //      (μ_resid + γ) .* (1_d * size_s')

            denom_ds.copy_from(&(resid_ds + gamma_ds));
            for s in 0..num_samples {
                denom_ds.column_mut(s).scale_mut(stat.size_s[s]);
            }

            mu_adj_param.update_stat(&(&stat.observed_sum_ds + &stat.imputed_sum_ds), &denom_ds);
            mu_adj_param.calibrate_with(CalibrateTarget::MeanOnly);

            let mu_ds = mu_adj_param.posterior_mean();

            //      imputed_sum_ds
            // γ = ---------------------
            //      μ .* (1_d * size_s')

            denom_ds.copy_from(mu_ds);
            for s in 0..num_samples {
                denom_ds.column_mut(s).scale_mut(stat.size_s[s]);
            }
            gamma_param.update_stat(&stat.imputed_sum_ds, &denom_ds);
            gamma_param.calibrate_with(CalibrateTarget::MeanOnly);

            if let Some(pb) = &prog_bar {
                pb.inc(1);
            }
        }
        if let Some(pb) = &prog_bar {
            pb.finish_and_clear();
        }

        // Output calibration after loop. `out_target` decides whether the
        // sd / log_mean / log_sd planes are materialized (only when the
        // caller writes them) or left empty (e.g. bge, which reads means).
        mu_adj_param.calibrate_with(out_target);
        gamma_param.calibrate_with(out_target);

        //      observed_db
        // δ = ---------------------
        //      μ * size_bs'
        {
            let mu_ds = mu_adj_param.posterior_mean();
            delta_param.update_stat(&stat.observed_sum_db, &(mu_ds * &stat.n_bs.transpose()));
            delta_param.calibrate_with(out_target);
        }

        // Take the observed mean
        {
            let mut denom_ds = DMatrix::<f32>::zeros(num_genes, num_samples);
            for s in 0..num_samples {
                denom_ds.column_mut(s).add_scalar_mut(stat.size_s[s]);
            }
            mu_param.update_stat(&stat.observed_sum_ds, &denom_ds);
            mu_param.calibrate_with(out_target);
        };

        // Sparse output (bge / MeanOnly): drop each mean's per-column prior
        // baseline so its support is exactly the observed∪imputed cells.
        // Downstream `from_pseudobulks` then yields sparse pb_blobs (held
        // across all training epochs). `All` consumers keep the dense mean.
        if matches!(out_target, CalibrateTarget::MeanOnly) {
            mu_param.sparsify_mean_to_support(&stat.observed_sum_ds);
            mu_adj_param.sparsify_mean_to_support(&(&stat.observed_sum_ds + &stat.imputed_sum_ds));
            gamma_param.sparsify_mean_to_support(&stat.imputed_sum_ds);
            mu_resid_param.sparsify_mean_to_support(&stat.residual_sum_ds);
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
        for s in 0..num_samples {
            denom_ds.column_mut(s).add_scalar_mut(stat.size_s[s]);
        }
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
) -> anyhow::Result<CollapsedOut> {
    let num_genes = stat.num_genes();
    let num_samples = stat.num_samples();

    // Block width targets a fixed per-plane footprint regardless of how
    // many features the panel carries.
    const BLOCK_ELEMS: usize = 32_000_000;
    let block_rows = (BLOCK_ELEMS / num_samples.max(1)).clamp(1, num_genes.max(1));
    let n_blocks = num_genes.div_ceil(block_rows.max(1));

    // Small problems (e.g. topic/svd with modest pb counts) run as a single
    // block, preserving the familiar per-iteration progress bar.
    if n_blocks <= 1 {
        return optimize_block(stat, hyper, num_iter, out_target, Some(label));
    }

    // `posterior_sample` (topic path) reads a_stat/b_stat; bge (MeanOnly)
    // does not, so those planes can be discarded per block — that's what
    // keeps the assembled output from holding the full sufficient stats.
    let keep_stats = matches!(out_target, CalibrateTarget::All);

    let prog = matrix_util::progress::new_progress_bar(n_blocks as u64)
        .with_message(format!("{label} gene-blocks"));

    let mut mu_obs: Vec<GammaMatrix> = Vec::with_capacity(n_blocks);
    let mut mu_adj: Vec<GammaMatrix> = Vec::new();
    let mut mu_res: Vec<GammaMatrix> = Vec::new();
    let mut gam: Vec<GammaMatrix> = Vec::new();
    let mut del: Vec<GammaMatrix> = Vec::new();

    let mut r0 = 0;
    while r0 < num_genes {
        let nr = block_rows.min(num_genes - r0);
        let sub = stat.select_rows(r0, nr);
        let mut out_b = optimize_block(&sub, hyper, num_iter, out_target, None)?;
        if !keep_stats {
            // Free a_stat/b_stat now so the accumulated blocks never add up
            // to the full sufficient-stat planes.
            out_b.release_stats();
        }
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
        prog.inc(1);
        r0 += nr;
    }
    prog.finish_and_clear();

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
#[derive(Debug)]
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
    pub residual_sum_ds: nalgebra::DMatrix<f32>, // residual sum within each sample
    pub size_s: nalgebra::DVector<f32>,          // sample s size
    pub observed_sum_db: nalgebra::DMatrix<f32>, // divergence numerator
    pub n_bs: nalgebra::DMatrix<f32>,            // batch-specific sample size
}

impl CollapsedStat {
    pub fn new(ngene: usize, nsample: usize, nbatch: usize) -> Self {
        Self {
            observed_sum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            imputed_sum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            residual_sum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            size_s: nalgebra::DVector::<f32>::zeros(nsample),
            observed_sum_db: nalgebra::DMatrix::<f32>::zeros(ngene, nbatch),
            n_bs: nalgebra::DMatrix::<f32>::zeros(nbatch, nsample),
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

    pub fn clear(&mut self) {
        self.observed_sum_ds.fill(0_f32);
        self.imputed_sum_ds.fill(0_f32);
        self.residual_sum_ds.fill(0_f32);
        self.observed_sum_db.fill(0_f32);
        self.size_s.fill(0_f32);
        self.n_bs.fill(0_f32);
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
            out.residual_sum_ds
                .column_mut(new_col)
                .copy_from(&self.residual_sum_ds.column(old_col));
            out.size_s[new_col] = self.size_s[old_col];
            for b in 0..nb {
                out.n_bs[(b, new_col)] = self.n_bs[(b, old_col)];
            }
        }
        out.observed_sum_db.copy_from(&self.observed_sum_db);
        out
    }

    /// Select a contiguous block of feature rows (`r0..r0+nrows`). Per-gene
    /// stats (`observed`/`imputed`/`residual`/`observed_db`) are sliced;
    /// the per-sample `size_s` and per-batch `n_bs` are shared, so they're
    /// copied whole. Used by the gene-blocked `optimize`.
    pub fn select_rows(&self, r0: usize, nrows: usize) -> Self {
        Self {
            observed_sum_ds: self.observed_sum_ds.rows(r0, nrows).into_owned(),
            imputed_sum_ds: self.imputed_sum_ds.rows(r0, nrows).into_owned(),
            residual_sum_ds: self.residual_sum_ds.rows(r0, nrows).into_owned(),
            size_s: self.size_s.clone(),
            observed_sum_db: self.observed_sum_db.rows(r0, nrows).into_owned(),
            n_bs: self.n_bs.clone(),
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
    )
}

/////////////////////////////////////////////////////////////
// Multi-level (METIS-inspired) collapsing for batch effects
/////////////////////////////////////////////////////////////

pub(super) const DEFAULT_NUM_LEVELS: usize = 2;
pub(super) const DEFAULT_COARSEST_SORT_DIM: usize = 7;

/// Cross-batch matched-stat accumulation on top of the pb-sample
/// layout. For each fine-group, fetches `knn` matched cells from each
/// non-own batch via `batch_knn_lookup`, dedupes them through their
/// pb-samples, and emits the matched gene sums into `stat.matched_dn`.
pub(super) fn collect_matched_stat_coarse(
    layout: &PbSampleLayout,
    gene_sums: &[Vec<(usize, f32)>],
    pbsamp_to_group: &[usize],
    batch_knn_lookup: &[ColumnDict<usize>],
    knn: usize,
    stat: &mut CollapsedStat,
) -> anyhow::Result<()> {
    let num_pb = layout.cell_counts.len();
    debug_assert_eq!(pbsamp_to_group.len(), num_pb);

    let neighbors_per_sc = per_batch_sc_neighbors(layout, batch_knn_lookup, knn)?;

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

            // Accumulate residual_sum_ds[g, s] += y_obs[g] / y_hat[g]
            // where y_obs[g] = gene_sums[pbsamp][g] / cell_counts[pbsamp]
            // -> residual_sum_ds[g, s] += gene_sums[pbsamp][g] / (cell_counts[pbsamp] * y_hat[g])
            //    then × cell_counts[pbsamp] to match original scaling
            // = gene_sums[pbsamp][g] / y_hat[g]
            for &(gene, val) in &gene_sums[pbsamp_idx] {
                if let Some(&y_h) = y_hat.get(&gene) {
                    if y_h > 0.0 {
                        stat.residual_sum_ds[(gene, pbsamp_group)] += val / y_h;
                    }
                }
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
        coarse
            .residual_sum_ds
            .column_mut(coarse_g)
            .add_assign(&fine_stat.residual_sum_ds.column(fine_g));
        coarse.size_s[coarse_g] += fine_stat.size_s[fine_g];
        for b in 0..num_batches {
            coarse.n_bs[(b, coarse_g)] += fine_stat.n_bs[(b, fine_g)];
        }
    }

    coarse.observed_sum_db.copy_from(&fine_stat.observed_sum_db);
    coarse
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod gene_block_tests;
