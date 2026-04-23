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
use crate::es::{rank_descending, weighted_ks_es};
use crate::fdr::bh_fdr;
use crate::null::permute_indices;
use crate::q_matrix::build_q_matrix;
use crate::specificity::{compute_specificity, SpecificityMode};
use crate::Mat;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
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
        }
    }
}

pub struct AnnotateOutputs {
    pub q_kc: Mat,
    pub es_kc: Mat,
    pub es_restandardized_kc: Mat,
    pub pvalue_kc: Mat,
    pub qvalue_kc: Mat,
    pub cell_annotation_nc: Mat,
    pub argmax_labels: Vec<LabelWithConfidence>,
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

    // -----------------------------------------------------------------
    // Null 1: row randomization → (mean*, sd*) restandardization moments.
    // For each celltype c, sample random size-|M_c| gene sets, compute ES
    // against every topic's observed ranking, accumulate per-(k, c) mean
    // and sd via Welford. Does not set the p-value — that comes from null 2.
    // -----------------------------------------------------------------
    let b_rand = config.num_row_randomization;
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
            let mut pool: Vec<u32> = (0..g as u32).collect();
            // Reusable hit-weight buffer (binary 0/1; same size as |M_c|).
            let mut hit_buf = vec![0.0f32; g];
            for draw in 0..b_rand {
                pool.shuffle(&mut rng);
                // Reset previous draw, then mark the new random set.
                hit_buf.fill(0.0);
                for &gi in pool.iter().take(m_size) {
                    hit_buf[gi as usize] = 1.0;
                }
                for kk in 0..k {
                    let es = weighted_ks_es(&ranked_per_k[kk], &hit_buf);
                    let prev_mean = mean[kk];
                    let new_mean = prev_mean + (es - prev_mean) / (draw as f32 + 1.0);
                    m2[kk] += (es - prev_mean) * (es - new_mean);
                    mean[kk] = new_mean;
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

    // -----------------------------------------------------------------
    // Null 2: sample (PB) permutation → pooled null for p-values.
    // Each perm: shuffle pb_membership rows (within batch if provided),
    // recompute β̃, specificity, rank, ES. Pool permuted restandardized
    // ES across all K topics per celltype to produce a topic-labeling-
    // invariant null distribution.
    //
    // When `num_sample_perm == 0`, fall back to row-randomization-based
    // p-values (count of row-rand ES ≥ observed). Useful for small-K
    // settings where the pooled null floor at ~1/K is too conservative.
    // -----------------------------------------------------------------
    let mut pvalue = Mat::zeros(k, c);
    let b_perm = config.num_sample_perm;

    if b_perm == 0 {
        // Fallback: row-randomization p-value. Recount with the second pass.
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
                let mut pool_idx: Vec<u32> = (0..g as u32).collect();
                let mut hit_buf = vec![0.0f32; g];
                for _ in 0..b_rand {
                    pool_idx.shuffle(&mut rng);
                    hit_buf.fill(0.0);
                    for &gi in pool_idx.iter().take(m_size) {
                        hit_buf[gi as usize] = 1.0;
                    }
                    for kk in 0..k {
                        let es = weighted_ks_es(&ranked_per_k[kk], &hit_buf);
                        if es >= es_obs[(kk, cc)] {
                            count[kk] += 1;
                        }
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
        for kk in 0..k {
            for cc in 0..c {
                let var = if b_perm > 1 {
                    perm_m2[(kk, cc)] / (b_perm as f32 - 1.0)
                } else {
                    0.0
                };
                perm_sd[(kk, cc)] = var.sqrt().max(1e-8);
            }
        }

        // Efron–Tibshirani-style restandardization at set level (adapted):
        //   obs_std[k, c]    = (obs[k, c]    − row_mean[k, c]) / row_sd[k, c]
        //   perm_std[k, c, i] = (perm[k, c, i] − perm_mean[k, c]) / perm_sd[k, c]
        // Different moments for each side → the ratio row_sd/perm_sd carries
        // the correlation correction. Count-based p then compares observed
        // z (gene-level scale) against permutation z-scores (perm-self scale).
        let total_pool = (b_perm * k) as f32;
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

    // Cell-level posterior + labels.
    let (posterior, labels) = label_cells(
        &inputs.cell_membership_nk,
        &q_mat,
        &inputs.cell_names,
        celltype_names,
        config.min_confidence,
    );

    Ok(AnnotateOutputs {
        q_kc: q_mat,
        es_kc: es_obs,
        es_restandardized_kc: es_obs_std,
        pvalue_kc: pvalue,
        qvalue_kc: qvalue,
        cell_annotation_nc: posterior,
        argmax_labels: labels,
    })
}
