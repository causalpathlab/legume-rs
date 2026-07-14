//! Public API: `annotate()` wires specificity → ES → Efron–Tibshirani
//! restandardization → pooled sample-permutation p-value → FDR → Q matrix
//! → cell labels.
//!
//! **Two nulls, combined per Efron–Tibshirani 2007.**
//!
//! 1. **Row randomization** (randomization model): draw random size-|M_c|
//!    gene sets, compute ES on observed β specificity. Gives the (mean*, SD*)
//!    moments used to restandardize observed and permuted ES. Captures the
//!    marginal scale of per-gene scores correctly but destroys gene-gene
//!    correlation.
//!
//! 2. **Sample permutation** (permutation model): shuffle `pb_membership_pk`
//!    rows (optionally within batch blocks), recompute
//!    `β̃ = pb_gene · shuffled_membership`, re-specificity, re-rank, re-ES.
//!    Preserves gene-gene correlation; destroys PB-topic coupling. Naive
//!    per-(k, c) comparison suffers from topic label-switching (~1/K of
//!    permutations happen to be equivalent to a topic relabeling), so we
//!    **pool** the permuted ES across all K topics per celltype — the null
//!    distribution for (k, c) is {es_std_perm[k', c] : all k', all perms}.
//!    This is invariant to topic relabeling.
//!
//! Restandardized observed ES = (ES − mean*) / SD*. Same restandardization
//! applied to each permuted ES. p-value = fraction of pooled permuted
//! restandardized ES ≥ observed restandardized ES.

use crate::cellproj::{label_cells, LabelWithConfidence};
use crate::consensus::UNASSIGNED;
use crate::es::{rank_descending, weighted_ks_es};
use crate::fdr::bh_fdr;
use crate::gene_strata::GeneStrata;
use crate::marker_bootstrap::{run_cluster_bootstrap, ClusterBootstrap, EnrichmentBootstrapConfig};
use crate::null::permute_indices;
use crate::q_matrix::build_q_matrix;
use crate::specificity::{compute_specificity, SpecificityMode};
use crate::Mat;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

pub struct GroupInputs {
    /// G × K group-profile matrix: topic β (simplex), SVD loadings (signed),
    /// or cluster centroids (gene-space means).
    pub profile_gk: Mat,
    /// G × P pseudobulk gene aggregates. Used by the sample-permutation null
    /// to recompute `β̃_perm = pb_gene · pb_membership[π]`.
    pub pb_gene_gp: Mat,
    /// P × K PB-level membership (topic θ_PB, SVD scores, or one-hot clusters).
    /// Row-permuted under sample permutation.
    pub pb_membership_pk: Mat,
    /// N × K cell-level membership: θ_cell for topic kinds, one-hot for clusters.
    pub cell_membership_nk: Mat,
    pub gene_names: Vec<Box<str>>,
    pub cell_names: Vec<Box<str>>,
}

#[derive(Debug, Clone)]
pub struct AnnotateConfig {
    pub specificity: SpecificityMode,
    /// Number of random size-|M_c| gene draws per celltype for the
    /// Efron–Tibshirani row-randomization moments (used to restandardize
    /// both observed and sample-permutation ES values).
    pub num_row_randomization: usize,
    /// Number of PB-level sample permutations (shuffle `pb_membership_pk`
    /// rows, recompute β̃ = pb_gene · shuffled_membership). Per celltype,
    /// pooled across all K topics to form a topic-relabeling-invariant null
    /// (B_perm × K null values per celltype).
    pub num_sample_perm: usize,
    /// Optional per-PB batch labels; when provided, permutations shuffle
    /// within batch blocks.
    pub batch_labels: Option<Vec<u32>>,
    pub fdr_alpha: f32,
    pub q_softmax_temperature: f32,
    pub min_confidence: f32,
    pub seed: u64,
    /// Draw the row-randomization null **within gene-abundance strata**, so a null gene set
    /// reproduces each panel's own abundance profile (GOseq; see [`crate::gene_strata`]).
    ///
    /// `false` restores the old uniform draw over every gene — which is ~30% undetected genes that
    /// sort to the bottom of every ranking and can never be enriched, i.e. a null that is trivially
    /// easy to beat. Kept only as an escape hatch and as the control the tests measure against.
    pub stratify_null: bool,
    /// When set, the shipped per-cluster call is the consensus of a **marker bootstrap**
    /// ([`crate::marker_bootstrap`]) rather than a single pass over one panel: each celltype's
    /// panel is resampled with replacement, the ES re-walked and the FDR re-called, so every call
    /// carries the support it earned across resamples and an unreproducible one abstains.
    /// `None` ⇒ the point-estimate path, unchanged.
    ///
    /// The library default is `None`; senna's CLI turns it **on**, as `faba annotate` does.
    pub bootstrap: Option<EnrichmentBootstrapConfig>,
}

impl Default for AnnotateConfig {
    fn default() -> Self {
        Self {
            specificity: SpecificityMode::Simplex,
            num_row_randomization: 1000,
            num_sample_perm: 500,
            batch_labels: None,
            fdr_alpha: 0.10,
            q_softmax_temperature: 1.0,
            min_confidence: 0.0,
            seed: 42,
            stratify_null: true,
            bootstrap: None,
        }
    }
}

pub struct AnnotateOutputs {
    pub q_kc: Mat,
    pub es_kc: Mat,
    pub es_restandardized_kc: Mat,
    /// Sample-permutation z = (ES − perm_mean)/perm_sd against the correlation-
    /// preserving null. `None` when `num_sample_perm == 0`. Graded (unlike the
    /// pooled, FWER-style `pvalue_kc`) — the preferred ontology-annotator input.
    pub perm_z_kc: Option<Mat>,
    pub pvalue_kc: Mat,
    pub qvalue_kc: Mat,
    pub cell_annotation_nc: Mat,
    pub argmax_labels: Vec<LabelWithConfidence>,
    /// The marker bootstrap's per-**cluster** consensus, when `config.bootstrap` was set. When it
    /// is present, `cell_annotation_nc` and `argmax_labels` are *derived from it* (broadcast to
    /// cells), not from the single-pass `q_kc` — so the shipped `confidence` is a resampling
    /// frequency rather than a softmaxed test statistic.
    pub bootstrap: Option<ClusterBootstrap>,
}

/// Run the full bipartite enrichment annotation pipeline.
pub fn annotate(
    inputs: &GroupInputs,
    markers_gc: &Mat,
    celltype_names: &[Box<str>],
    config: &AnnotateConfig,
) -> anyhow::Result<AnnotateOutputs> {
    let g = inputs.profile_gk.nrows();
    let k = inputs.profile_gk.ncols();
    let p = inputs.pb_membership_pk.nrows();
    let c = markers_gc.ncols();

    anyhow::ensure!(markers_gc.nrows() == g, "markers G dim mismatch");
    anyhow::ensure!(
        inputs.cell_membership_nk.ncols() == k,
        "cell_membership K mismatch: cells {} vs profile {}",
        inputs.cell_membership_nk.ncols(),
        k
    );
    anyhow::ensure!(celltype_names.len() == c, "celltype_names length mismatch");
    anyhow::ensure!(
        inputs.pb_gene_gp.nrows() == g,
        "pb_gene G mismatch: {} vs profile {}",
        inputs.pb_gene_gp.nrows(),
        g
    );
    anyhow::ensure!(
        inputs.pb_gene_gp.ncols() == p,
        "pb_gene P {} ≠ pb_membership P {}",
        inputs.pb_gene_gp.ncols(),
        p
    );
    anyhow::ensure!(
        inputs.pb_membership_pk.ncols() == k,
        "pb_membership K {} ≠ profile K {}",
        inputs.pb_membership_pk.ncols(),
        k
    );

    // Per-celltype hit-weight vectors derived from the (already-IDF-weighted)
    // marker matrix. Each `hit_weights[c][g]` is `markers_gc[g, c]`:
    // - 0 for non-markers (treated as misses by `weighted_ks_es`)
    // - positive IDF weight `ln(C / c_g)` for markers (specific markers
    //   contribute more to ES walks than markers shared by many celltypes)
    let hit_weights: Vec<Vec<f32>> = (0..c)
        .map(|cc| (0..g).map(|gi| markers_gc[(gi, cc)].max(0.0)).collect())
        .collect();
    let marker_sizes: Vec<usize> = hit_weights
        .iter()
        .map(|hw| hw.iter().filter(|&&w| w > 0.0).count())
        .collect();

    // Observed specificity and ranked orderings per topic.
    let specificity_obs = compute_specificity(&inputs.profile_gk, config.specificity);
    let ranked_per_k: Vec<Vec<u32>> = (0..k)
        .map(|kk| {
            let scores: Vec<f32> = (0..g).map(|gi| specificity_obs[(gi, kk)]).collect();
            rank_descending(&scores)
        })
        .collect();

    // Observed ES (K × C). IDF-weighted hits via hit_weights[c].
    let mut es_obs = Mat::zeros(k, c);
    for kk in 0..k {
        for cc in 0..c {
            es_obs[(kk, cc)] = weighted_ks_es(&ranked_per_k[kk], &hit_weights[cc]);
        }
    }

    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap();

    ////////////////////////////////////////////////////////////////////////////
    // Null 1: row randomization → (mean*, sd*) restandardization moments.    //
    // For each celltype c, sample random gene sets MATCHED on abundance and  //
    // on the panel's own weights, compute ES against every topic's observed  //
    // ranking, accumulate per-(k, c) mean and sd via Welford. Does not set    //
    // the p-value — that comes from null 2.                                  //
    //                                                                        //
    // The matching is load-bearing, not a refinement. A uniform draw is 30%  //
    // undetected genes (measured on BMMNC) — genes pinned to the bottom of    //
    // every ranking, which can never be enriched — so it is trivially easy   //
    // to beat, and easy to beat by an amount that DIFFERS PER CELLTYPE, since //
    // panels differ in how well-expressed their markers are. `es_std` is the  //
    // decision variable (`argmax_c`), so that differential lands straight in  //
    // the label: measured, a celltype's mean es_std was a perfectly monotone  //
    // function of its markers' mean expression (Spearman +1.000 over 8 types).//
    // See `crate::gene_strata`.                                              //
    ////////////////////////////////////////////////////////////////////////////
    let b_rand = config.num_row_randomization;
    let strata = if config.stratify_null {
        GeneStrata::by_abundance(&inputs.profile_gk)
    } else {
        GeneStrata::unstratified(g)
    };
    // Each panel's live markers, and the abundance profile a null draw must reproduce.
    let live: Vec<Vec<(u32, f32)>> = (0..c)
        .map(|cc| {
            (0..g)
                .filter(|&gi| hit_weights[cc][gi] > 0.0)
                .map(|gi| (gi as u32, hit_weights[cc][gi]))
                .collect()
        })
        .collect();
    let strata_profile: Vec<Vec<Vec<f32>>> = live.iter().map(|p| strata.profile_of(p)).collect();

    let row_moments: Vec<(Vec<f32>, Vec<f32>)> = (0..c)
        .into_par_iter()
        .progress_with_style(style.clone())
        .with_message("row randomization")
        .map(|cc| {
            let m_size = marker_sizes[cc];
            let mut mean = vec![0.0f32; k];
            let mut m2 = vec![0.0f32; k];
            if m_size == 0 || m_size >= g {
                return (mean, m2);
            }
            let mut rng = SmallRng::seed_from_u64(
                config
                    .seed
                    .wrapping_add(1_000_003u64.wrapping_mul(cc as u64 + 1)),
            );
            let mut scratch = strata.scratch();
            let mut drawn: Vec<(u32, f32)> = Vec::with_capacity(m_size);
            let mut hit_buf = vec![0.0f32; g];
            for draw in 0..b_rand {
                strata.draw_matched(&strata_profile[cc], &mut scratch, &mut drawn, &mut rng);
                // The null carries the panel's OWN weights, not a binary 1.0: the observed ES is
                // IDF/specificity-weighted, and standardizing a weighted statistic against an
                // unweighted null compares two different quantities.
                for &(gi, w) in &drawn {
                    hit_buf[gi as usize] = w;
                }
                for kk in 0..k {
                    let es = weighted_ks_es(&ranked_per_k[kk], &hit_buf);
                    let prev_mean = mean[kk];
                    let new_mean = prev_mean + (es - prev_mean) / (draw as f32 + 1.0);
                    m2[kk] += (es - prev_mean) * (es - new_mean);
                    mean[kk] = new_mean;
                }
                // Reset only what was touched; `fill(0)` over G would dominate the loop.
                for &(gi, _) in &drawn {
                    hit_buf[gi as usize] = 0.0;
                }
            }
            (mean, m2)
        })
        .collect();

    let mut row_mean = Mat::zeros(k, c);
    let mut row_sd = Mat::zeros(k, c);
    let bf = b_rand as f32;
    for cc in 0..c {
        let (mean_vec, m2_vec) = &row_moments[cc];
        for kk in 0..k {
            row_mean[(kk, cc)] = mean_vec[kk];
            let var = if bf > 1.0 {
                m2_vec[kk] / (bf - 1.0)
            } else {
                0.0
            };
            row_sd[(kk, cc)] = var.sqrt().max(1e-8);
        }
    }

    // Restandardize observed ES.
    let mut es_obs_std = Mat::zeros(k, c);
    for kk in 0..k {
        for cc in 0..c {
            es_obs_std[(kk, cc)] = (es_obs[(kk, cc)] - row_mean[(kk, cc)]) / row_sd[(kk, cc)];
        }
    }

    /////////////////////////////////////////////////////////////////
    // Null 2: sample (PB) permutation → pooled null for p-values. //
    /////////////////////////////////////////////////////////////////
    // Each perm: shuffle pb_membership rows (within batch if provided),
    // recompute β̃, specificity, rank, ES. Pool permuted restandardized
    // ES across all K topics per celltype to produce a topic-labeling-
    // invariant null distribution.
    //
    // When `num_sample_perm == 0`, fall back to row-randomization-based
    // p-values (count of row-rand ES ≥ observed). Useful for small-K
    // settings where the pooled null floor at ~1/K is too conservative.
    let mut pvalue = Mat::zeros(k, c);
    let b_perm = config.num_sample_perm;
    let mut perm_z_kc: Option<Mat> = None;

    // The pooled permuted z's, per celltype, **kept** (sorted ascending) rather than dropped:
    // the marker bootstrap scores each of its draws against this same frozen pool. Rebuilding it
    // per draw would be `b_perm × n_boot × K × C` ES walks — see `marker_bootstrap`'s module doc
    // on why it is frozen, and why `num_sample_perm == 0` is the exact alternative.
    // Empty ⇒ the bootstrap falls back to exact row-randomization counts.
    let mut pooled_null: Vec<Vec<f32>> = Vec::new();

    if b_perm == 0 {
        // Fallback: row-randomization p-value. Recount with the second pass — same abundance- and
        // weight-matched null as the moments above, or the p-value would be calibrated against a
        // different gene population than the one that standardized the statistic.
        let counts: Vec<Vec<u32>> = (0..c)
            .into_par_iter()
            .map(|cc| {
                let m_size = marker_sizes[cc];
                let mut count = vec![0u32; k];
                if m_size == 0 || m_size >= g {
                    return count;
                }
                let mut rng = SmallRng::seed_from_u64(
                    config
                        .seed
                        .wrapping_add(3_000_003u64.wrapping_mul(cc as u64 + 1)),
                );
                let mut scratch = strata.scratch();
                let mut drawn: Vec<(u32, f32)> = Vec::with_capacity(m_size);
                let mut hit_buf = vec![0.0f32; g];
                for _ in 0..b_rand {
                    strata.draw_matched(&strata_profile[cc], &mut scratch, &mut drawn, &mut rng);
                    for &(gi, w) in &drawn {
                        hit_buf[gi as usize] = w;
                    }
                    for kk in 0..k {
                        let es = weighted_ks_es(&ranked_per_k[kk], &hit_buf);
                        if es >= es_obs[(kk, cc)] {
                            count[kk] += 1;
                        }
                    }
                    for &(gi, _) in &drawn {
                        hit_buf[gi as usize] = 0.0;
                    }
                }
                count
            })
            .collect();
        for cc in 0..c {
            for kk in 0..k {
                pvalue[(kk, cc)] = (counts[cc][kk] as f32 + 1.0) / (bf + 1.0);
            }
        }
    } else {
        // Sample permutations: first collect raw ES values per (k, c, perm).
        let perm_es_raw: Vec<Mat> = (0..b_perm)
            .into_par_iter()
            .progress_with_style(style.clone())
            .with_message("sample permutations")
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    config
                        .seed
                        .wrapping_add(2_000_003u64.wrapping_mul(i as u64 + 1)),
                );
                let pi = permute_indices(p, config.batch_labels.as_deref(), &mut rng);
                let mut permuted = Mat::zeros(p, k);
                for (new_row, &src) in pi.iter().enumerate() {
                    permuted
                        .row_mut(new_row)
                        .copy_from(&inputs.pb_membership_pk.row(src));
                }
                let beta_tilde = &inputs.pb_gene_gp * &permuted;
                let spec_perm = compute_specificity(&beta_tilde, config.specificity);
                let mut es_raw = Mat::zeros(k, c);
                for kk in 0..k {
                    let scores: Vec<f32> = (0..g).map(|gi| spec_perm[(gi, kk)]).collect();
                    let ranked = rank_descending(&scores);
                    for cc in 0..c {
                        es_raw[(kk, cc)] = weighted_ks_es(&ranked, &hit_weights[cc]);
                    }
                }
                es_raw
            })
            .collect();

        // Per-(k, c) sample-permutation moments via Welford across B perms.
        let mut perm_mean = Mat::zeros(k, c);
        let mut perm_m2 = Mat::zeros(k, c);
        for (idx, es_raw) in perm_es_raw.iter().enumerate() {
            let n = (idx as f32) + 1.0;
            for kk in 0..k {
                for cc in 0..c {
                    let x = es_raw[(kk, cc)];
                    let prev_mean = perm_mean[(kk, cc)];
                    let new_mean = prev_mean + (x - prev_mean) / n;
                    perm_m2[(kk, cc)] += (x - prev_mean) * (x - new_mean);
                    perm_mean[(kk, cc)] = new_mean;
                }
            }
        }
        let mut perm_sd = Mat::zeros(k, c);
        let mut sd_floor_hits = 0usize;
        for kk in 0..k {
            for cc in 0..c {
                let var = if b_perm > 1 {
                    perm_m2[(kk, cc)] / (b_perm as f32 - 1.0)
                } else {
                    0.0
                };
                let sd = var.sqrt();
                if sd <= 1e-8 {
                    sd_floor_hits += 1;
                }
                perm_sd[(kk, cc)] = sd.max(1e-8);
            }
        }

        // Sample-permutation z: observed raw ES standardized by the permutation
        // null's OWN moments — correlation-preserving and graded (unlike the
        // pooled count p). An entry whose null variance collapsed (perm_sd at the
        // floor) carries no usable signal → z = 0; and if the null is degenerate
        // overall (too few pseudobulk samples to permute — e.g. a single sample),
        // perm-z is suppressed so downstream falls back to the robust
        // row-randomization es_std instead of `(es − mean)/~0` garbage.
        let mut pz = Mat::zeros(k, c);
        for kk in 0..k {
            for cc in 0..c {
                pz[(kk, cc)] = if perm_sd[(kk, cc)] <= 1e-8 {
                    0.0
                } else {
                    (es_obs[(kk, cc)] - perm_mean[(kk, cc)]) / perm_sd[(kk, cc)]
                };
            }
        }
        // Floored entries are inert (z = 0 ⇒ p = 0.5, never reject), so partial
        // degeneracy ships safely. Fall back to es_std only when the null is
        // MOSTLY un-estimable — i.e. too few permutable pseudobulk samples, which
        // collapses the majority of per-(k,c) variances to the floor.
        let floored_frac = sd_floor_hits as f32 / (k * c).max(1) as f32;
        if floored_frac < 0.5 {
            perm_z_kc = Some(pz);
        } else {
            log::warn!(
                "sample-permutation null mostly degenerate ({:.0}% of {} k×c perm_sd at floor); \
                 suppressing perm-z, ontology falls back to row-randomization es_std",
                100.0 * floored_frac,
                k * c
            );
        }

        // Efron–Tibshirani-style restandardization at set level (adapted):
        //   obs_std[k, c]    = (obs[k, c]    − row_mean[k, c]) / row_sd[k, c]
        //   perm_std[k, c, i] = (perm[k, c, i] − perm_mean[k, c]) / perm_sd[k, c]
        // Different moments for each side → the ratio row_sd/perm_sd carries
        // the correlation correction. Count-based p then compares observed
        // z (gene-level scale) against permutation z-scores (perm-self scale).
        let total_pool = (b_perm * k) as f32;
        pooled_null.reserve(c);
        for cc in 0..c {
            let mut pool: Vec<f32> = Vec::with_capacity(b_perm * k);
            for es_raw in &perm_es_raw {
                for kk in 0..k {
                    let z = (es_raw[(kk, cc)] - perm_mean[(kk, cc)]) / perm_sd[(kk, cc)];
                    pool.push(z);
                }
            }
            for kk in 0..k {
                let obs_z = es_obs_std[(kk, cc)]; // uses row_mean / row_sd
                let n_ge = pool.iter().filter(|&&v| v >= obs_z).count() as f32;
                pvalue[(kk, cc)] = (n_ge + 1.0) / (total_pool + 1.0);
            }
            // Sorted once so the bootstrap can binary-search the tail instead of rescanning
            // `b_perm × K` values for every one of its `n_boot × K × C` lookups.
            pool.sort_by(f32::total_cmp);
            pooled_null.push(pool);
        }
    }

    // BH FDR per topic row (C tests per row). This is the right scale for
    // two reasons: (a) Q is built per-topic (softmax over significant edges
    // in a row), so per-row error control matches downstream use; (b) the
    // pooled sample-permutation null shares values across topics within a
    // celltype, so global K × C BH over-corrects for dependent tests.
    let mut qvalue = Mat::zeros(k, c);
    for kk in 0..k {
        let row_p: Vec<f32> = (0..c).map(|cc| pvalue[(kk, cc)]).collect();
        let row_q = bh_fdr(&row_p);
        for cc in 0..c {
            qvalue[(kk, cc)] = row_q[cc];
        }
    }

    // Q matrix (FDR-sparse, row-softmax).
    let q_mat = build_q_matrix(
        &es_obs_std,
        &qvalue,
        config.fdr_alpha,
        config.q_softmax_temperature,
    );

    // Cell-level posterior + labels — the single-pass point estimate.
    let (mut posterior, mut labels) = label_cells(
        &inputs.cell_membership_nk,
        &q_mat,
        &inputs.cell_names,
        celltype_names,
        config.min_confidence,
    );

    // The stability bootstrap. When it runs, its consensus REPLACES the point estimate above: a
    // cluster whose call cannot survive resampling of its own marker panel should not ship that
    // call with a softmaxed `confidence` of 0.98 beside it.
    let bootstrap = match config.bootstrap.as_ref() {
        None => None,
        Some(bcfg) => {
            let boot = run_cluster_bootstrap(
                &ranked_per_k,
                markers_gc,
                &strata,
                &pooled_null,
                g,
                config.fdr_alpha,
                config.q_softmax_temperature,
                config.seed,
                bcfg,
            )?;
            let (bp, bl) = broadcast_to_cells(
                &boot,
                &inputs.cell_membership_nk,
                &inputs.cell_names,
                celltype_names,
                &q_mat,
            );
            posterior = bp;
            labels = bl;
            Some(boot)
        }
    };

    Ok(AnnotateOutputs {
        q_kc: q_mat,
        es_kc: es_obs,
        es_restandardized_kc: es_obs_std,
        perm_z_kc,
        pvalue_kc: pvalue,
        qvalue_kc: qvalue,
        cell_annotation_nc: posterior,
        argmax_labels: labels,
        bootstrap,
    })
}

/// Push the per-cluster consensus down onto the cells.
///
/// `cell_membership_nk` is one-hot on this path, so a cell's cluster is the column it fires on and
/// **its label is its cluster's label**. The per-cell posterior becomes the cluster's *bootstrap*
/// distribution over celltypes, and the reported confidence becomes `cluster_label_support` — the
/// fraction of resamples that agreed — rather than a softmax over one panel's test statistics.
///
/// A cell whose cluster id is out of range (the sentinel for "not clustered") stays unassigned, as
/// it already did.
fn broadcast_to_cells(
    boot: &ClusterBootstrap,
    cell_membership_nk: &Mat,
    cell_names: &[Box<str>],
    celltype_names: &[Box<str>],
    q_mat: &Mat,
) -> (Mat, Vec<LabelWithConfidence>) {
    let n = cell_membership_nk.nrows();
    let k = cell_membership_nk.ncols();
    let c = boot.c;
    let width = c + 1; // the consensus rows carry an `unassigned` column

    // Where the bootstrap and the single-pass call disagree, say so. This is the disagreement the
    // point estimate would otherwise have hidden behind a confident-looking q.
    for kk in 0..k {
        let top_q = (0..c)
            .max_by(|&a, &b| q_mat[(kk, a)].total_cmp(&q_mat[(kk, b)]))
            .filter(|&cc| q_mat[(kk, cc)] > 0.0);
        let con = boot.consensus.label[kk];
        if let Some(tq) = top_q {
            if con != tq {
                let con_name = if con == UNASSIGNED {
                    crate::UNASSIGNED_LABEL
                } else {
                    &celltype_names[con]
                };
                log::warn!(
                    "cluster {kk}: single-pass FDR calls '{}' but only {:.0}% of bootstrap \
                     resamples agreed — shipping '{}' (support {:.2})",
                    celltype_names[tq],
                    100.0 * boot.consensus.support[kk],
                    con_name,
                    boot.consensus.support[kk],
                );
            }
        }
    }

    let mut posterior = Mat::zeros(n, c);
    let mut labels: Vec<LabelWithConfidence> = Vec::with_capacity(n);

    for (i, cell_name) in cell_names.iter().enumerate().take(n) {
        // The one-hot cluster this cell fires on, if any.
        let kk = (0..k).find(|&kk| cell_membership_nk[(i, kk)] > 0.0);
        let Some(kk) = kk else {
            labels.push(LabelWithConfidence {
                cell_name: cell_name.clone(),
                label: crate::UNASSIGNED_LABEL.into(),
                confidence: 0.0,
            });
            continue;
        };

        // The type columns only: any mass the resamples spent on `unassigned` is simply absent, so
        // the row sums to less than 1 exactly when the bootstrap declined to call. That is honest.
        let row = &boot.consensus.post[kk * width..(kk + 1) * width];
        for cc in 0..c {
            posterior[(i, cc)] = row[cc];
        }

        let t = boot.consensus.label[kk];
        labels.push(LabelWithConfidence {
            cell_name: cell_name.clone(),
            label: if t == UNASSIGNED {
                crate::UNASSIGNED_LABEL.into()
            } else {
                celltype_names[t].clone()
            },
            // `cluster_label_support`: a resampling frequency, not a softmaxed statistic.
            confidence: boot.consensus.support[kk],
        });
    }

    (posterior, labels)
}
