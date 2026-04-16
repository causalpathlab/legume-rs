//! Leiden clustering-based β initialization for topic model decoders.
//!
//! Clusters pseudobulk profiles with Leiden, computes per-cluster mean
//! expression, and overwrites the decoder dictionary logits. The per-gene
//! baseline (mean across all PBs) is stored in `logit_bias`, separating
//! housekeeping signal from cell-type-specific deviations in `logits`.

use crate::embed_common::*;

use candle_core::Device;
use candle_nn::VarMap;
use data_beans_alg::collapse_data::CollapsedOut;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use matrix_util::traits::ConvertMatOps;

const DICT_LOGITS_VAR: &str = "dictionary.logits";
const DICT_BIAS_VAR: &str = "dictionary.logit_bias";
const NB_LOG_PHI_VAR: &str = "log_phi";

fn decoder_var_path(level: usize, suffix: &str) -> String {
    format!("dec_{level}.{suffix}")
}

/// Initialize decoder dictionary from Leiden clustering of pseudobulk
/// profiles.
///
/// 1. Depth-normalize PB columns, log1p, select top-weighted genes.
/// 2. Leiden cluster PB profiles targeting K clusters.
/// 3. Global mean → `logit_bias` (baseline per gene, fixed).
/// 4. Per-cluster mean centered by global mean → `logits` (topic-specific).
pub(crate) fn leiden_beta_init(
    finest: &CollapsedOut,
    n_topics: usize,
    feature_weights: &[f32],
    parameters: &VarMap,
    level_coarsenings: &[Option<FeatureCoarsening>],
    dev: &Device,
) -> anyhow::Result<()> {
    let mu_gp: &Mat = match finest.mu_adjusted.as_ref() {
        Some(adj) => {
            log::info!("β init: using mu_adjusted (batch-corrected)");
            adj.posterior_mean()
        }
        None => {
            log::info!("β init: using mu_observed");
            finest.mu_observed.posterior_mean()
        }
    };
    let n_pb = mu_gp.ncols();
    let d_full = mu_gp.nrows();
    if n_pb < 2 {
        log::warn!("β init: need ≥2 pseudobulks, got {}; skipping", n_pb);
        return Ok(());
    }

    // Depth-normalize → log1p
    let x_gp = depth_normalize_log1p(mu_gp);

    // Global mean per gene (across PBs) — becomes the baseline logit_bias.
    let global_mean: Vec<f32> = (0..d_full)
        .map(|g| {
            let sum: f32 = (0..n_pb).map(|p| x_gp[(g, p)]).sum();
            sum / n_pb as f32
        })
        .collect();

    // Select top genes by weight for Leiden clustering.
    // Genes with high inverse-mean weight are informative (moderate expression).
    // Cap at ~1/3 of total genes to focus on the most discriminative features.
    let max_genes = d_full / 3;
    let mut ranked: Vec<(usize, f32)> = feature_weights.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let kept_idx: Vec<usize> = ranked
        .iter()
        .take(max_genes)
        .filter(|(_, w)| *w > 0.01)
        .map(|(i, _)| *i)
        .collect();
    let n_kept = kept_idx.len();
    log::info!(
        "β init: {}/{} top-weighted genes for Leiden clustering of {} PBs",
        n_kept,
        d_full,
        n_pb
    );

    if n_kept < 10 || n_pb < n_topics {
        log::warn!(
            "β init: too few genes ({}) or PBs ({}) for {} topics; setting bias only",
            n_kept,
            n_pb,
            n_topics
        );
        set_bias_all_levels(parameters, &global_mean, level_coarsenings, dev)?;
        return Ok(());
    }

    // Build filtered PB matrix [n_pb, n_kept] — rows are PBs, columns are genes
    let mut pb_filtered = Mat::zeros(n_pb, n_kept);
    for (new_g, &orig_g) in kept_idx.iter().enumerate() {
        for p in 0..n_pb {
            pb_filtered[(p, new_g)] = x_gp[(orig_g, p)];
        }
    }

    // Leiden clustering of PBs targeting K clusters
    let knn = (n_pb / 5).clamp(3, 30);
    let cluster_result =
        crate::cluster::leiden_clustering(&pb_filtered, knn, 1.0, Some(n_topics), Some(42))?;

    let k_actual = cluster_result.n_clusters;
    log::info!(
        "β init: Leiden found {} clusters (target {})",
        k_actual,
        n_topics
    );

    // Compute per-cluster mean expression at full resolution [K, D_full]
    let k = k_actual.min(n_topics);
    let mut cluster_mean_kd = Mat::zeros(k, d_full);
    let mut cluster_counts = vec![0usize; k];
    for (p, &label) in cluster_result.labels.iter().enumerate() {
        if label < k {
            cluster_counts[label] += 1;
            for g in 0..d_full {
                cluster_mean_kd[(label, g)] += x_gp[(g, p)];
            }
        }
    }
    for ki in 0..k {
        if cluster_counts[ki] > 0 {
            let n = cluster_counts[ki] as f32;
            for g in 0..d_full {
                cluster_mean_kd[(ki, g)] /= n;
            }
        }
    }

    // Build full [n_topics, D_full] matrix. If Leiden found fewer clusters
    // than requested, fill remaining rows with global mean (zero deviation).
    let mut full_kd = Mat::zeros(n_topics, d_full);
    for ki in 0..k {
        for g in 0..d_full {
            full_kd[(ki, g)] = cluster_mean_kd[(ki, g)];
        }
    }
    for ki in k..n_topics {
        for g in 0..d_full {
            full_kd[(ki, g)] = global_mean[g];
        }
    }

    // logit_bias[g] = log(global_mean[g] + eps) — shared baseline
    // logits[k, g] = log(cluster_mean[k, g] + eps) - logit_bias[g]
    //              = topic-specific deviation from baseline
    // β_kg = softmax_g(logits[k, :] + logit_bias[:])
    //      ∝ exp(log cluster_mean[k, g]) = cluster_mean[k, g]
    let eps = 1e-8f32;
    let bias_d: Vec<f32> = global_mean.iter().map(|&v| (v + eps).ln()).collect();
    let mut logits_kd = Mat::zeros(n_topics, d_full);
    for ki in 0..n_topics {
        for g in 0..d_full {
            logits_kd[(ki, g)] = (full_kd[(ki, g)] + eps).ln() - bias_d[g];
        }
    }

    // NB dispersion init: log_phi per gene scaled by feature weight.
    // High-weight genes (informative): log(2) ≈ 0.69 (moderate dispersion).
    // Low-weight genes (housekeeping): log(0.01) ≈ −4.6 (high overdispersion →
    // weak NB gradient contribution). Linear interpolation in log-phi space.
    let log_phi_hi = 0.693f32; // ln(2), for w=1 genes
    let log_phi_lo = -4.6f32; // ln(0.01), for w→0 genes
    let log_phi_full: Vec<f32> = feature_weights
        .iter()
        .map(|&w| {
            let t = w.clamp(0.0, 1.0);
            log_phi_lo + t * (log_phi_hi - log_phi_lo)
        })
        .collect();

    // Overwrite decoder Vars at each level
    let data = parameters.data().lock().expect("VarMap lock");
    for (level, fc) in level_coarsenings.iter().enumerate() {
        let (level_logits, level_bias) = coarsen_logits_and_bias(
            &logits_kd,
            &bias_d,
            n_topics,
            fc.as_ref(),
        );

        let logits_name = decoder_var_path(level, DICT_LOGITS_VAR);
        if let Some(var) = data.get(&logits_name) {
            match var.set(&level_logits.to_tensor(dev)?) {
                Ok(()) => log::info!("β init: set {}", logits_name),
                Err(e) => log::warn!("β init: failed {} — {}", logits_name, e),
            }
        }

        let bias_name = decoder_var_path(level, DICT_BIAS_VAR);
        if let Some(var) = data.get(&bias_name) {
            match var.set(&level_bias.to_tensor(dev)?) {
                Ok(()) => log::info!("β init: set {}", bias_name),
                Err(e) => log::warn!("β init: failed {} — {}", bias_name, e),
            }
        }

        // NB: init log_phi with filter-based overdispersion
        let phi_name = decoder_var_path(level, NB_LOG_PHI_VAR);
        if let Some(var) = data.get(&phi_name) {
            let level_phi = coarsen_bias(&log_phi_full, fc.as_ref());
            match var.set(&level_phi.to_tensor(dev)?) {
                Ok(()) => log::info!("β init: set {}", phi_name),
                Err(e) => log::warn!("β init: failed {} — {}", phi_name, e),
            }
        }
    }

    Ok(())
}

/// Set only the logit_bias at every level (fallback when clustering fails).
fn set_bias_all_levels(
    parameters: &VarMap,
    global_mean: &[f32],
    level_coarsenings: &[Option<FeatureCoarsening>],
    dev: &Device,
) -> anyhow::Result<()> {
    let eps = 1e-8f32;
    let bias_d: Vec<f32> = global_mean.iter().map(|&v| (v + eps).ln()).collect();
    let data = parameters.data().lock().expect("VarMap lock");
    for (level, fc) in level_coarsenings.iter().enumerate() {
        let level_bias = coarsen_bias(&bias_d, fc.as_ref());
        let bias_name = decoder_var_path(level, DICT_BIAS_VAR);
        if let Some(var) = data.get(&bias_name) {
            match var.set(&level_bias.to_tensor(dev)?) {
                Ok(()) => log::info!("β init: set {}", bias_name),
                Err(e) => log::warn!("β init: failed {} — {}", bias_name, e),
            }
        }
    }
    Ok(())
}

/// Coarsen logits [K, D_full] and bias [D_full] to level resolution.
fn coarsen_logits_and_bias(
    logits_kd: &Mat,
    bias_d: &[f32],
    n_topics: usize,
    fc: Option<&FeatureCoarsening>,
) -> (Mat, Mat) {
    match fc {
        Some(fc) => {
            let d_c = fc.num_coarse;
            let mut w = Mat::zeros(n_topics, d_c);
            let mut b = Mat::zeros(1, d_c);
            for (c, fine_indices) in fc.coarse_to_fine.iter().enumerate() {
                let n = fine_indices.len() as f32;
                if n > 0.0 {
                    for &f in fine_indices {
                        for kk in 0..n_topics {
                            w[(kk, c)] += logits_kd[(kk, f)];
                        }
                        b[(0, c)] += bias_d[f];
                    }
                    for kk in 0..n_topics {
                        w[(kk, c)] /= n;
                    }
                    b[(0, c)] /= n;
                }
            }
            (w, b)
        }
        None => {
            let b = Mat::from_row_slice(1, bias_d.len(), bias_d);
            (logits_kd.clone(), b)
        }
    }
}

/// Coarsen bias only (for fallback path).
fn coarsen_bias(bias_d: &[f32], fc: Option<&FeatureCoarsening>) -> Mat {
    match fc {
        Some(fc) => {
            let d_c = fc.num_coarse;
            let mut b = Mat::zeros(1, d_c);
            for (c, fine_indices) in fc.coarse_to_fine.iter().enumerate() {
                let n = fine_indices.len() as f32;
                if n > 0.0 {
                    for &f in fine_indices {
                        b[(0, c)] += bias_d[f];
                    }
                    b[(0, c)] /= n;
                }
            }
            b
        }
        None => Mat::from_row_slice(1, bias_d.len(), bias_d),
    }
}

/// Build a `[1, D_level]` feature weight tensor from continuous weights,
/// coarsened to the level's feature resolution.
pub(crate) fn coarsen_weights_for_level(
    feature_weights: &[f32],
    fc: Option<&FeatureCoarsening>,
    dev: &Device,
) -> anyhow::Result<candle_core::Tensor> {
    let w = coarsen_bias(feature_weights, fc);
    w.to_tensor(dev)
}

/// Depth-normalize each column to median depth, clamp ≥0, log1p.
fn depth_normalize_log1p(mu_gp: &Mat) -> Mat {
    let n_pb = mu_gp.ncols();
    let totals: Vec<f32> = (0..n_pb)
        .map(|p| mu_gp.column(p).iter().map(|&v| v.max(0.0)).sum::<f32>())
        .collect();
    let mut sorted: Vec<f32> = totals.iter().copied().filter(|&t| t > 0.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        1.0
    } else {
        sorted[sorted.len() / 2]
    };

    let mut out = mu_gp.clone();
    for (p, &total) in totals.iter().enumerate() {
        let scale = if total > 1e-8 { median / total } else { 0.0 };
        let mut col = out.column_mut(p);
        for v in col.iter_mut() {
            *v = (v.max(0.0) * scale).ln_1p();
        }
    }
    out
}
