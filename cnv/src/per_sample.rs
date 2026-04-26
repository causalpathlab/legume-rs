//! Per-sample CN HMM calling on a `[G × N]` log-ratio signal.
//!
//! The signal `[G_genome_ordered × N]` is partitioned into `n_topics` blocks
//! of `n_indv` columns each (concatenated `topic_0_indv_0..topic_0_indv_{n-1}`,
//! `topic_1_indv_0..`). For each topic we:
//!
//! 1. Initialise emission means/variances either from quantiles (fixed K) or
//!    via [`kmeans_init::select_kmeans_k`] (BIC-selected K).
//! 2. Run [`hmm::fit_em`] to refine emission means, transitions and per-sample σ².
//! 3. Compute a continuous CN score per (gene × sample) as the expected
//!    normalised emission level (`Σ_k P(state=k|y) · em_norm[k] ∈ [−1, 1]`).
//!
//! Optional iterative reference refinement: cluster samples within each topic
//! on their post-HMM cn_score (kmeans K=2) and pick the lower-burden cluster
//! as the new normalisation reference for the next pass.

use crate::genome_order::{GenePosition, GenomeOrder};
use crate::hmm::{fit_em, CnvHmmParams, EmConfig, HmmResult, SampleEmissionParams};
use crate::kmeans_init::{select_kmeans_k, sort_components};
use matrix_util::clustering::{Kmeans, KmeansArgs};
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Result of per-sample CNV calling.
///
/// `cn_score[g, s] ∈ [−1, 1]` is the expected normalised CN level (gain → +1,
/// loss → −1, neutral → 0). `viterbi_paths[s]` is the per-sample MAP state
/// path on the genome-ordered gene axis.
pub struct PerSampleCnv {
    pub genome_order: GenomeOrder,
    pub n_states: usize,
    pub cn_score: DMatrix<f32>,
    pub viterbi_paths: Vec<Vec<usize>>,
}

/// Configuration for [`call_per_sample_cnv`].
#[derive(Debug, Clone)]
pub struct PerSampleCnvConfig {
    /// HMM EM max iterations.
    pub em_iter: usize,
    /// HMM EM tolerance (relative log-likelihood change).
    pub em_tol: f32,
    /// Fixed number of CN states (used when `gmm_k_max == 0`).
    pub n_states: usize,
    /// If ≥ 3, BIC-select K ∈ [3..gmm_k_max] from the marginal signal per topic.
    /// 0 ⇒ disabled.
    pub gmm_k_max: usize,
}

impl Default for PerSampleCnvConfig {
    fn default() -> Self {
        Self {
            em_iter: 50,
            em_tol: 1e-4,
            n_states: 3,
            gmm_k_max: 0,
        }
    }
}

/// Initialise + EM-fit one HMM on a single topic's signal block `[G_ord × n]`.
///
/// If `config.gmm_k_max ≥ 3`, K is BIC-selected from kmeans over
/// `[3..=gmm_k_max]`; otherwise quantile init at fixed `K = config.n_states`.
pub fn fit_topic_hmm(
    signal_topic: &DMatrix<f32>,
    config: &PerSampleCnvConfig,
    topic_label: &str,
) -> (CnvHmmParams, Vec<HmmResult>) {
    let g = signal_topic.nrows();
    let n = signal_topic.ncols();

    let all_vals: Vec<f32> = signal_topic
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .collect();

    let (n_states, emission_means, sigma_sq_init, neutral_idx) = if config.gmm_k_max >= 3 {
        let k_range: Vec<usize> = (3..=config.gmm_k_max).collect();
        let (k_sel, means, variances, weights) = select_kmeans_k(&all_vals, &k_range);
        let (means_s, variances_s, weights_s, neutral) = sort_components(means, variances, weights);
        let avg_var = if !variances_s.is_empty() {
            variances_s.iter().sum::<f32>() / (variances_s.len() as f32)
        } else {
            1e-2
        };
        log::info!(
            "CNV kmeans init [{}]: BIC-selected K={}, means={:?}, vars={:?}, weights={:?}, neutral_idx={}",
            topic_label,
            k_sel,
            means_s.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
            variances_s.iter().map(|x| (x * 10000.0).round() / 10000.0).collect::<Vec<_>>(),
            weights_s.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
            neutral
        );
        (k_sel, means_s, avg_var.max(1e-4), neutral)
    } else {
        let nn = all_vals.len();
        let mut buf = all_vals.clone();
        let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
        let quantile = |buf: &mut [f32], p: f32| -> f32 {
            if nn == 0 {
                return 0.0;
            }
            let idx = ((p * (nn as f32 - 1.0)).round() as usize).min(nn - 1);
            *buf.select_nth_unstable_by(idx, cmp).1
        };
        let loss_mean = quantile(&mut buf, 0.05);
        let gain_mean = quantile(&mut buf, 0.95);
        let neutral_vals: Vec<f32> = all_vals
            .iter()
            .copied()
            .filter(|&v| v > loss_mean * 0.5 && v < gain_mean * 0.5)
            .collect();
        let sigma_init = if neutral_vals.len() >= 10 {
            let m = neutral_vals.iter().sum::<f32>() / (neutral_vals.len() as f32);
            let var = neutral_vals.iter().map(|&v| (v - m).powi(2)).sum::<f32>()
                / (neutral_vals.len() as f32);
            var.sqrt().max(1e-3)
        } else {
            let q25 = quantile(&mut buf, 0.25);
            let q75 = quantile(&mut buf, 0.75);
            ((q75 - q25) / 1.349).max(1e-3)
        };
        let sigma_sq_init = (sigma_init * sigma_init).max(1e-4);
        let k_fixed = config.n_states;
        let neutral_idx = k_fixed / 2;
        let em: Vec<f32> = (0..k_fixed)
            .map(|i| {
                if i == neutral_idx {
                    0.0
                } else if i < neutral_idx {
                    loss_mean * ((neutral_idx - i) as f32) / (neutral_idx as f32).max(1.0)
                } else {
                    gain_mean * ((i - neutral_idx) as f32)
                        / ((k_fixed - 1 - neutral_idx) as f32).max(1.0)
                }
            })
            .collect();
        (k_fixed, em, sigma_sq_init, neutral_idx)
    };

    let init_params = CnvHmmParams::new(
        nalgebra::DVector::from_vec(emission_means),
        1e-3,
        neutral_idx,
    );
    let init_sample_params = SampleEmissionParams {
        alpha: vec![1.0f32; n],
        sigma_sq: vec![sigma_sq_init; n],
    };
    let em_cfg = EmConfig {
        max_iter: config.em_iter,
        tol: config.em_tol,
    };

    let segment_data = signal_topic.transpose();
    log::info!(
        "CNV HMM EM [{}]: {} genes × {} samples, K={}, init σ²={:.4}",
        topic_label,
        g,
        n,
        n_states,
        sigma_sq_init
    );
    let (params, sample_params, results) = fit_em(
        &init_params,
        &segment_data,
        Some(&init_sample_params),
        &em_cfg,
    );

    let (smin, smax) = sample_params
        .sigma_sq
        .iter()
        .cloned()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    let self_trans: Vec<f32> = (0..n_states)
        .map(|i| params.log_trans[(i, i)].exp())
        .collect();
    log::info!(
        "CNV HMM done [{}]: emission_means={:?}, σ²∈[{:.4}, {:.4}], self-trans={:?}",
        topic_label,
        params.emission_means.as_slice(),
        smin,
        smax,
        self_trans
    );

    (params, results)
}

/// Run a per-topic CN HMM on a `[G × (n_topics·n_indv)]` signal matrix.
///
/// Each topic gets its own emission means, transitions, and per-sample σ².
/// Topic blocks are independent and run in parallel via rayon.
pub fn call_per_sample_cnv(
    signal: &DMatrix<f32>,
    positions: &[GenePosition],
    n_topics: usize,
    n_indv: usize,
    config: &PerSampleCnvConfig,
) -> anyhow::Result<PerSampleCnv> {
    let genome_order = GenomeOrder::from_positions(positions);
    if genome_order.ordered_indices.is_empty() {
        anyhow::bail!("CNV: no genes mapped to canonical chromosomes");
    }
    let signal_ord = genome_order.reorder_rows(signal)?;
    let g = signal_ord.nrows();
    let n_total = signal_ord.ncols();
    assert_eq!(n_total, n_topics * n_indv);

    let topic_outs: Vec<(DMatrix<f32>, Vec<Vec<usize>>, usize)> = (0..n_topics)
        .into_par_iter()
        .map(|k| {
            let col_start = k * n_indv;
            let signal_topic = signal_ord.columns(col_start, n_indv).into_owned();
            let (params, results) = fit_topic_hmm(&signal_topic, config, &format!("topic_{}", k));
            let max_abs = params
                .emission_means
                .iter()
                .map(|x| x.abs())
                .fold(0f32, f32::max)
                .max(1e-6);
            let em_norm = params.emission_means.map(|x| x / max_abs);

            let mut block = DMatrix::<f32>::zeros(g, n_indv);
            let mut paths = Vec::with_capacity(n_indv);
            for (s_local, res) in results.iter().enumerate().take(n_indv) {
                block
                    .column_mut(s_local)
                    .copy_from(&(&res.posteriors * &em_norm));
                paths.push(res.viterbi_path.clone());
            }
            (block, paths, params.emission_means.len())
        })
        .collect();

    let max_n_states = topic_outs.iter().map(|t| t.2).max().unwrap_or(0);
    let mut cn_score = DMatrix::<f32>::zeros(g, n_total);
    let mut viterbi_paths: Vec<Vec<usize>> = Vec::with_capacity(n_total);
    for (k, (block, paths, _)) in topic_outs.into_iter().enumerate() {
        cn_score.columns_mut(k * n_indv, n_indv).copy_from(&block);
        viterbi_paths.extend(paths);
    }

    Ok(PerSampleCnv {
        genome_order,
        n_states: max_n_states,
        cn_score,
        viterbi_paths,
    })
}

/// Auto-detect normal-reference samples within a single-topic signal block.
///
/// Score each sample by genome-wide deviation from the per-gene cross-sample
/// median. Lowest-scoring `frac` of samples (minimum 2) are returned in
/// ascending sample-index order — these are the most "flat" / closest to
/// neutral and serve as the normalisation reference.
pub fn detect_normal_samples(log_tau: &DMatrix<f32>, frac: f32) -> Vec<usize> {
    let g = log_tau.nrows();
    let n = log_tau.ncols();
    if n == 0 {
        return Vec::new();
    }

    let median_g: Vec<f32> = (0..g)
        .into_par_iter()
        .map(|gi| {
            let mut row: Vec<f32> = (0..n).map(|j| log_tau[(gi, j)]).collect();
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = n / 2;
            if n % 2 == 1 {
                row[m]
            } else {
                0.5 * (row[m - 1] + row[m])
            }
        })
        .collect();

    let mut score = vec![0f32; n];
    for s in 0..n {
        let mut acc = 0f32;
        for gi in 0..g {
            acc += (log_tau[(gi, s)] - median_g[gi]).abs();
        }
        score[s] = acc / (g as f32);
    }

    let k = ((n as f32) * frac).round() as usize;
    let k = k.max(2).min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        score[a]
            .partial_cmp(&score[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx.sort();
    idx
}

/// Per-topic kmeans (K=2) on samples' per-gene cn_score; the lower-burden
/// cluster (flatter genome → wildtype-like) becomes the new reference.
/// Returns reference indices in topic-local `0..n_indv` space.
pub fn cluster_reference_from_cn_score(
    cn_score: &DMatrix<f32>,
    n_topics: usize,
    n_indv: usize,
) -> Vec<Vec<usize>> {
    let g = cn_score.nrows();
    (0..n_topics)
        .into_par_iter()
        .map(|k| {
            let col_start = k * n_indv;
            let feat = cn_score.columns(col_start, n_indv).transpose();
            let labels = feat.kmeans_rows(KmeansArgs {
                num_clusters: 2,
                max_iter: 100,
            });

            let burden_per_sample: Vec<f32> = (0..n_indv)
                .map(|i| feat.row(i).iter().map(|v| v.abs()).sum::<f32>() / g as f32)
                .collect();

            let mut burden = [0f32; 2];
            let mut counts = [0usize; 2];
            for (i, &l) in labels.iter().enumerate() {
                burden[l] += burden_per_sample[i];
                counts[l] += 1;
            }
            let mean_burden = |c: usize| {
                if counts[c] > 0 {
                    burden[c] / counts[c] as f32
                } else {
                    f32::INFINITY
                }
            };
            let ref_cluster = if mean_burden(0) <= mean_burden(1) { 0 } else { 1 };
            let mut refs: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter_map(|(i, &l)| (l == ref_cluster).then_some(i))
                .collect();

            if refs.len() < 2 {
                let mut idx_burden: Vec<(usize, f32)> =
                    burden_per_sample.iter().copied().enumerate().collect();
                idx_burden
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let take = (n_indv / 3).max(2).min(n_indv);
                refs = idx_burden.into_iter().take(take).map(|(i, _)| i).collect();
                refs.sort();
            }
            log::info!(
                "CNV ref refinement [topic_{}]: cluster sizes={:?}, mean_burden=[{:.4}, {:.4}], ref_cluster={} ({} samples)",
                k,
                counts,
                mean_burden(0),
                mean_burden(1),
                ref_cluster,
                refs.len()
            );
            refs
        })
        .collect()
}

/// Compute the modal Viterbi state across samples at a given ordered-gene index.
/// Returns `-1` when no path exists. Uses a stack-allocated counter (max 32 states).
pub fn modal_state_at(viterbi_paths: &[Vec<usize>], op: usize, n_states: usize) -> i32 {
    let mut counts = vec![0u32; n_states];
    for path in viterbi_paths {
        let s = path[op];
        if s < n_states {
            counts[s] += 1;
        }
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| *c)
        .map(|(s, _)| s as i32)
        .unwrap_or(-1)
}

/// Convert topic-local sample indices `[0..n_indv)` to flat indices in the
/// concatenated `[0..(n_topics·n_indv))` namespace.
pub fn topic_local_to_flat(refs: &[Vec<usize>], n_indv: usize) -> Vec<Vec<usize>> {
    refs.iter()
        .enumerate()
        .map(|(k, ix)| ix.iter().map(|&i| k * n_indv + i).collect())
        .collect()
}
