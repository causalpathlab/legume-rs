//! Hidden Markov Model for copy number state calling on genomic segments.
//!
//! Gaussian emission with per-sample scale and variance.
//! States represent relative CN: {deletion, neutral, gain} or finer.
//!
//! # References
//!
//! - Patel, A. et al. (2014). "Single-cell RNA-seq highlights intratumoral heterogeneity
//!   in primary glioblastoma." Science, 344(6190), 1396-1401. (early scRNA-seq CNV inference)
//! - Tirosh, I. et al. (2016). "Dissecting the multicellular ecosystem of metastatic melanoma
//!   by single-cell RNA-seq." Science, 352(6282), 189-196. (expression-based CNV inference)
//! - inferCNV, Broad Institute. <https://github.com/broadinstitute/infercnv>
//!   (6-state HMM with hspike calibration; i3 model for emission initialization)
//! - Gao, R. et al. (2021). "Delineating copy number and clonal substructure in human tumors
//!   from single-cell transcriptomes." Nature Biotechnology, 39, 599-608. (CopyKAT:
//!   Poisson-Gamma MCMC segmentation, DLM smoothing)
//! - Teng, H. et al. (2022). "Integrative inference of subclonal tumour evolution from
//!   single-cell and bulk sequencing data." Nature Biotechnology, 40, 1543-1553. (Numbat:
//!   joint expression + phased BAF HMM, per-clone ploidy)
//! - De Falco, A. et al. (2023). "SCEVAN: joint CNV calling from single-cell data."
//!   Nature Communications, 14, 643. (Mumford-Shah variational segmentation)

use nalgebra::{DMatrix, DVector};

/// HMM parameters for CN state calling.
#[derive(Debug, Clone)]
pub struct CnvHmmParams {
    /// State-specific emission means (length K_cn).
    /// E.g., [-0.5, 0.0, 0.3] for {del, neutral, gain}.
    pub emission_means: DVector<f32>,
    /// Log transition matrix [K_cn x K_cn].
    /// log_trans[(i,j)] = log P(state_j | state_i).
    pub log_trans: DMatrix<f32>,
    /// Log initial state probabilities (length K_cn).
    pub log_pi: DVector<f32>,
}

/// Per-sample emission parameters estimated during EM.
#[derive(Debug, Clone)]
pub struct SampleEmissionParams {
    /// Per-sample scale factor: observed = alpha * emission_mean.
    pub alpha: Vec<f32>,
    /// Per-sample emission variance.
    pub sigma_sq: Vec<f32>,
}

/// HMM output for a single sample.
#[derive(Debug, Clone)]
pub struct HmmResult {
    /// Posterior state probabilities [S x K_cn] (from forward-backward).
    pub posteriors: DMatrix<f32>,
    /// Viterbi path: most likely state for each segment (length S).
    pub viterbi_path: Vec<usize>,
    /// Log-likelihood of the observation sequence.
    pub log_likelihood: f32,
}

impl CnvHmmParams {
    /// Create a K-state CN HMM with the given transition probability.
    ///
    /// `emission_means`: state-specific means (length K), should be sorted.
    /// `transition_prob`: off-diagonal transition probability (e.g., 1e-6).
    ///   Self-transition = 1 - (K-1) * transition_prob.
    /// `neutral_idx`: index of the neutral state for initial state prior.
    pub fn new(emission_means: DVector<f32>, transition_prob: f32, neutral_idx: usize) -> Self {
        let k = emission_means.len();
        let self_prob = 1.0 - (k as f32 - 1.0) * transition_prob;

        let mut log_trans = DMatrix::from_element(k, k, transition_prob.ln());
        for i in 0..k {
            log_trans[(i, i)] = self_prob.ln();
        }

        // Initial state: strongly favor neutral
        let mut log_pi = DVector::from_element(k, transition_prob.ln());
        log_pi[neutral_idx] = self_prob.ln();

        Self {
            emission_means,
            log_trans,
            log_pi,
        }
    }

    /// Create default 3-state CN HMM: {deletion, neutral, gain}.
    ///
    /// - Emission means: [-0.5, 0.0, 0.4]
    /// - High self-transition (InferCNV-style, t=1e-6)
    /// - Initial state strongly favors neutral
    pub fn default_3state() -> Self {
        Self::new(
            DVector::from_vec(vec![-0.5, 0.0, 0.4]),
            1e-6,
            1, // neutral = index 1
        )
    }

    /// Initialize 3-state HMM from segment data (InferCNV i3 approach).
    ///
    /// `segment_data`: [samples x segments] matrix of segment-level observations.
    /// `cells_per_sample`: number of cells in each pseudobulk sample (for variance scaling).
    ///
    /// Emission means are set from data quantiles:
    ///   - neutral = 0 (mu_residual is centered by construction)
    ///   - deletion = 5th percentile of segment values
    ///   - gain = 95th percentile of segment values
    ///
    /// Per-sample variance is scaled by 1/n_cells (InferCNV hspike approach):
    ///   sigma^2_b = MAD^2 / n_cells_b
    ///
    /// Returns (HMM params, per-sample emission params).
    pub fn initialize_from_data(
        segment_data: &DMatrix<f32>,
        cells_per_sample: &[f32],
    ) -> (Self, SampleEmissionParams) {
        // Collect all segment values for quantile estimation
        let mut all_vals: Vec<f32> = Vec::with_capacity(segment_data.len());
        for &v in segment_data.iter() {
            if v.is_finite() {
                all_vals.push(v);
            }
        }
        all_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let n = all_vals.len();
        let quantile = |p: f32| -> f32 {
            if n == 0 {
                return 0.0;
            }
            let idx = (p * (n - 1) as f32).round() as usize;
            all_vals[idx.min(n - 1)]
        };

        // Emission means from quantiles
        let del_mean = quantile(0.05);
        let gain_mean = quantile(0.95);
        let emission_means = DVector::from_vec(vec![del_mean, 0.0, gain_mean]);

        let params = Self::new(emission_means, 1e-6, 1);

        // Per-sample variance: MAD^2 scaled by 1/n_cells
        let median = quantile(0.5);
        let mut abs_devs: Vec<f32> = all_vals.iter().map(|&v| (v - median).abs()).collect();
        abs_devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if abs_devs.is_empty() {
            1.0
        } else {
            abs_devs[abs_devs.len() / 2] * 1.4826 // MAD → σ conversion for Normal
        };
        let base_sigma_sq = (mad * mad).max(1e-4);

        let n_samples = segment_data.nrows();
        let sample_alpha = vec![1.0f32; n_samples];
        let sample_sigma_sq: Vec<f32> = cells_per_sample
            .iter()
            .map(|&n_cells| (base_sigma_sq / n_cells.max(1.0)).max(1e-8))
            .collect();

        let sample_params = SampleEmissionParams {
            alpha: sample_alpha,
            sigma_sq: sample_sigma_sq,
        };

        (params, sample_params)
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.emission_means.len()
    }
}

/// Precompute log emission probabilities for all (t, k) pairs.
/// Returns [n x k] matrix where entry (t, j) = log N(y_t | alpha * mu_j, sigma^2).
fn precompute_log_emissions(
    params: &CnvHmmParams,
    y: &[f32],
    alpha: f32,
    sigma_sq: f32,
) -> DMatrix<f32> {
    let n = y.len();
    let k = params.n_states();
    let inv_sigma_sq = 1.0 / sigma_sq;
    let log_norm = -0.5 * (sigma_sq.ln() + std::f32::consts::TAU.ln());

    DMatrix::from_fn(n, k, |t, j| {
        let diff = y[t] - alpha * params.emission_means[j];
        log_norm - 0.5 * diff * diff * inv_sigma_sq
    })
}

/// Log-sum-exp over a row of a matrix, using a scratch buffer.
fn logsumexp_slice(buf: &mut [f32], mat: &DMatrix<f32>, row: usize, k: usize) -> f32 {
    for j in 0..k {
        buf[j] = mat[(row, j)];
    }
    logsumexp_buf(buf, k)
}

/// Log-sum-exp over a buffer of length k.
fn logsumexp_buf(buf: &[f32], k: usize) -> f32 {
    let max = buf[..k].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        return f32::NEG_INFINITY;
    }
    let sum: f32 = buf[..k].iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Run forward-backward algorithm for a single sample.
///
/// Returns posterior probabilities [S x K] and log-likelihood.
pub fn forward_backward(
    params: &CnvHmmParams,
    y: &[f32],
    alpha: f32,
    sigma_sq: f32,
) -> (DMatrix<f32>, f32) {
    let log_emit = precompute_log_emissions(params, y, alpha, sigma_sq);
    forward_backward_with_emit(params, &log_emit)
}

/// Viterbi algorithm: find the most likely state sequence.
pub fn viterbi(params: &CnvHmmParams, y: &[f32], alpha: f32, sigma_sq: f32) -> Vec<usize> {
    let log_emit = precompute_log_emissions(params, y, alpha, sigma_sq);
    viterbi_with_emit(params, &log_emit)
}

/// Run full HMM inference for a single sample: forward-backward + Viterbi.
///
/// Precomputes emissions once and shares them between both algorithms.
pub fn infer_sample(params: &CnvHmmParams, y: &[f32], alpha: f32, sigma_sq: f32) -> HmmResult {
    let log_emit = precompute_log_emissions(params, y, alpha, sigma_sq);
    let (posteriors, log_likelihood) = forward_backward_with_emit(params, &log_emit);
    let viterbi_path = viterbi_with_emit(params, &log_emit);
    HmmResult {
        posteriors,
        viterbi_path,
        log_likelihood,
    }
}

/// Forward-backward with precomputed emissions.
pub fn forward_backward_with_emit(
    params: &CnvHmmParams,
    log_emit: &DMatrix<f32>,
) -> (DMatrix<f32>, f32) {
    let n = log_emit.nrows();
    let k = params.n_states();

    if n == 0 {
        return (DMatrix::zeros(0, k), 0.0);
    }

    let mut log_alpha = DMatrix::<f32>::zeros(n, k);
    let mut log_scale = vec![0.0f32; n];
    let mut buf = vec![0.0f32; k];

    // t = 0
    for j in 0..k {
        log_alpha[(0, j)] = params.log_pi[j] + log_emit[(0, j)];
    }
    log_scale[0] = logsumexp_slice(&mut buf, &log_alpha, 0, k);
    for j in 0..k {
        log_alpha[(0, j)] -= log_scale[0];
    }

    // Forward recursion
    for t in 1..n {
        for j in 0..k {
            for i in 0..k {
                buf[i] = log_alpha[(t - 1, i)] + params.log_trans[(i, j)];
            }
            log_alpha[(t, j)] = logsumexp_buf(&buf, k) + log_emit[(t, j)];
        }
        log_scale[t] = logsumexp_slice(&mut buf, &log_alpha, t, k);
        for j in 0..k {
            log_alpha[(t, j)] -= log_scale[t];
        }
    }

    // Backward pass
    let mut log_beta = DMatrix::<f32>::zeros(n, k);
    for t in (0..n - 1).rev() {
        for i in 0..k {
            for j in 0..k {
                buf[j] = params.log_trans[(i, j)] + log_emit[(t + 1, j)] + log_beta[(t + 1, j)];
            }
            log_beta[(t, i)] = logsumexp_buf(&buf, k) - log_scale[t + 1];
        }
    }

    // Posteriors
    let mut posteriors = &log_alpha + &log_beta;
    for t in 0..n {
        let lse = logsumexp_slice(&mut buf, &posteriors, t, k);
        for j in 0..k {
            posteriors[(t, j)] = (posteriors[(t, j)] - lse).exp();
        }
    }

    let log_lik: f32 = log_scale.iter().sum();
    (posteriors, log_lik)
}

/// Viterbi with precomputed emissions.
pub fn viterbi_with_emit(params: &CnvHmmParams, log_emit: &DMatrix<f32>) -> Vec<usize> {
    let n = log_emit.nrows();
    let k = params.n_states();

    if n == 0 {
        return vec![];
    }

    let mut delta = DMatrix::<f32>::zeros(n, k);
    let mut psi = vec![vec![0usize; k]; n];

    for j in 0..k {
        delta[(0, j)] = params.log_pi[j] + log_emit[(0, j)];
    }

    for t in 1..n {
        for j in 0..k {
            let mut best_val = f32::NEG_INFINITY;
            let mut best_i = 0;
            for i in 0..k {
                let val = delta[(t - 1, i)] + params.log_trans[(i, j)];
                if val > best_val {
                    best_val = val;
                    best_i = i;
                }
            }
            delta[(t, j)] = best_val + log_emit[(t, j)];
            psi[t][j] = best_i;
        }
    }

    let mut path = vec![0usize; n];
    path[n - 1] = (0..k)
        .max_by(|&a, &b| delta[(n - 1, a)].partial_cmp(&delta[(n - 1, b)]).unwrap())
        .unwrap();
    for t in (0..n - 1).rev() {
        path[t] = psi[t + 1][path[t + 1]];
    }
    path
}

/// Combine weighted log-emissions across multiple samples into a single matrix.
fn combine_weighted_emissions(
    params: &CnvHmmParams,
    signals: &[&[f32]],
    alphas: &[f32],
    sigma_sqs: &[f32],
    weights: Option<&[f32]>,
) -> DMatrix<f32> {
    let t_len = signals[0].len();
    let k = params.n_states();
    let mut combined = DMatrix::<f32>::zeros(t_len, k);
    for (s, signal) in signals.iter().enumerate() {
        let w = weights.map_or(1.0, |ws| ws[s]);
        let log_emit = precompute_log_emissions(params, signal, alphas[s], sigma_sqs[s]);
        for t in 0..t_len {
            for j in 0..k {
                combined[(t, j)] += w * log_emit[(t, j)];
            }
        }
    }
    combined
}

/// Shared Viterbi: sum weighted log-emissions across samples,
/// run a single Viterbi for consensus state path.
pub fn shared_viterbi(
    params: &CnvHmmParams,
    signals: &[&[f32]],
    alphas: &[f32],
    sigma_sqs: &[f32],
    weights: Option<&[f32]>,
) -> Vec<usize> {
    if signals.is_empty() || signals[0].is_empty() {
        return vec![];
    }
    let combined = combine_weighted_emissions(params, signals, alphas, sigma_sqs, weights);
    viterbi_with_emit(params, &combined)
}

/// Shared forward-backward: sum weighted log-emissions across samples.
///
/// Returns (posteriors [T × K], log-likelihood).
pub fn shared_forward_backward(
    params: &CnvHmmParams,
    signals: &[&[f32]],
    alphas: &[f32],
    sigma_sqs: &[f32],
    weights: Option<&[f32]>,
) -> (DMatrix<f32>, f32) {
    if signals.is_empty() || signals[0].is_empty() {
        return (DMatrix::zeros(0, params.n_states()), 0.0);
    }
    let combined = combine_weighted_emissions(params, signals, alphas, sigma_sqs, weights);
    forward_backward_with_emit(params, &combined)
}

/// EM configuration.
pub struct EmConfig {
    pub max_iter: usize,
    pub tol: f32,
}

impl Default for EmConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-4,
        }
    }
}

/// Estimate per-sample scale (alpha) and variance (sigma_sq) given current HMM params.
///
/// Closed-form MLE given posteriors:
///   alpha_b = sum_s gamma_sk * y_s * mu_k / sum_s gamma_sk * mu_k^2
///   sigma^2_b = sum_s sum_k gamma_sk * (y_s - alpha_b * mu_k)^2 / S
pub fn estimate_sample_params(
    params: &CnvHmmParams,
    y: &[f32],
    posteriors: &DMatrix<f32>,
) -> (f32, f32) {
    let n = y.len();
    let k = params.n_states();

    // Alpha: weighted regression of y on mu
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for t in 0..n {
        for j in 0..k {
            let w = posteriors[(t, j)];
            let mu = params.emission_means[j];
            num += w * y[t] * mu;
            den += w * mu * mu;
        }
    }
    // Avoid division by zero when all emission means are 0 (e.g., all neutral)
    let alpha = if den.abs() > 1e-10 { num / den } else { 1.0 };

    // Sigma^2: weighted residual variance
    let mut ss = 0.0f32;
    let mut total_weight = 0.0f32;
    for t in 0..n {
        for j in 0..k {
            let w = posteriors[(t, j)];
            let residual = y[t] - alpha * params.emission_means[j];
            ss += w * residual * residual;
            total_weight += w;
        }
    }
    let sigma_sq = if total_weight > 0.0 {
        (ss / total_weight).max(1e-6)
    } else {
        1.0
    };

    (alpha, sigma_sq)
}

/// Run EM to fit HMM parameters and per-sample emission params.
///
/// `segment_data`: [samples x segments] matrix of segment-level observations.
/// `initial_sample_params`: optional data-driven per-sample params from `initialize_from_data`.
///   If None, defaults to alpha=1, sigma^2=1 for all samples.
/// Returns updated HMM params, per-sample params, and per-sample HMM results.
pub fn fit_em(
    initial_params: &CnvHmmParams,
    segment_data: &DMatrix<f32>,
    initial_sample_params: Option<&SampleEmissionParams>,
    config: &EmConfig,
) -> (CnvHmmParams, SampleEmissionParams, Vec<HmmResult>) {
    let n_samples = segment_data.nrows();
    let n_segments = segment_data.ncols();
    let k = initial_params.n_states();

    let mut params = initial_params.clone();
    let mut sample_alpha = initial_sample_params
        .map(|p| p.alpha.clone())
        .unwrap_or_else(|| vec![1.0f32; n_samples]);
    let mut sample_sigma_sq = initial_sample_params
        .map(|p| p.sigma_sq.clone())
        .unwrap_or_else(|| vec![1.0f32; n_samples]);
    let mut results = Vec::with_capacity(n_samples);

    let mut prev_ll = f32::NEG_INFINITY;

    // Pre-extract rows to avoid repeated allocation
    let sample_rows: Vec<Vec<f32>> = (0..n_samples)
        .map(|s| (0..n_segments).map(|t| segment_data[(s, t)]).collect())
        .collect();

    // Reusable xi buffer (hoisted out of inner loop)
    let mut xi = DMatrix::<f32>::zeros(k, k);

    for iter in 0..config.max_iter {
        // E-step: precompute emissions + forward-backward for each sample
        results.clear();
        let mut total_ll = 0.0f32;
        let mut log_emits: Vec<DMatrix<f32>> = Vec::with_capacity(n_samples);

        for s in 0..n_samples {
            let log_emit = precompute_log_emissions(
                &params,
                &sample_rows[s],
                sample_alpha[s],
                sample_sigma_sq[s],
            );
            let (posteriors, ll) = forward_backward_with_emit(&params, &log_emit);
            let viterbi_path = viterbi_with_emit(&params, &log_emit);
            total_ll += ll;
            results.push(HmmResult {
                posteriors,
                viterbi_path,
                log_likelihood: ll,
            });
            log_emits.push(log_emit);
        }

        // Check convergence
        let ll_change = (total_ll - prev_ll).abs() / (prev_ll.abs() + 1e-10);
        log::debug!(
            "EM iter {}: ll={:.4}, change={:.6}",
            iter,
            total_ll,
            ll_change
        );
        if iter > 0 && ll_change < config.tol {
            log::info!(
                "EM converged after {} iterations (ll={:.4})",
                iter,
                total_ll
            );
            break;
        }
        prev_ll = total_ll;

        // M-step: update per-sample params
        for s in 0..n_samples {
            let (alpha, sigma_sq) =
                estimate_sample_params(&params, &sample_rows[s], &results[s].posteriors);
            sample_alpha[s] = alpha;
            sample_sigma_sq[s] = sigma_sq;
        }

        // M-step: update emission means (weighted across all samples)
        let mut new_means = DVector::<f32>::zeros(k);
        let mut mean_weights = DVector::<f32>::zeros(k);
        for s in 0..n_samples {
            let alpha = sample_alpha[s];
            let post = &results[s].posteriors;
            let y_vec = DVector::from_row_slice(&sample_rows[s][..n_segments]);
            // mean_weights += column sums of posteriors
            for j in 0..k {
                mean_weights[j] += post.column(j).sum();
            }
            // new_means += posteriors' @ (y / alpha)  per state
            if alpha.abs() > 1e-10 {
                let y_scaled = &y_vec / alpha;
                for j in 0..k {
                    new_means[j] += post.column(j).dot(&y_scaled);
                }
            }
        }
        for j in 0..k {
            if mean_weights[j] > 1e-10 {
                new_means[j] /= mean_weights[j];
            }
        }

        // Pin neutral state at 0 (identifiability)
        if k >= 2 {
            let neutral_idx = k / 2;
            let shift = new_means[neutral_idx];
            for j in 0..k {
                new_means[j] -= shift;
            }
        }
        params.emission_means = new_means;

        // M-step: update transition matrix using precomputed emissions
        let mut trans_counts = DMatrix::<f32>::zeros(k, k);
        for s in 0..n_samples {
            for t in 0..n_segments - 1 {
                xi.fill(0.0);
                // xi(i,j) ∝ gamma(t,i) * a(i,j) * b(j,y_{t+1}) / gamma_marginal
                // Using posteriors as approximate forward variable (standard EM shortcut)
                for i in 0..k {
                    for j in 0..k {
                        xi[(i, j)] = results[s].posteriors[(t, i)].max(1e-30).ln()
                            + params.log_trans[(i, j)]
                            + log_emits[s][(t + 1, j)];
                    }
                }
                let max_xi = xi.max();
                xi.add_scalar_mut(-max_xi);
                xi.apply(|x| *x = x.exp());
                let xi_sum = xi.sum();
                if xi_sum > 0.0 {
                    xi.scale_mut(1.0 / xi_sum);
                }
                trans_counts += &xi;
            }
        }

        for i in 0..k {
            let row_sum: f32 = (0..k).map(|j| trans_counts[(i, j)]).sum();
            if row_sum > 1e-10 {
                for j in 0..k {
                    params.log_trans[(i, j)] = (trans_counts[(i, j)] / row_sum).max(1e-10).ln();
                }
            }
        }
    }

    // Final E-step
    results.clear();
    for s in 0..n_samples {
        results.push(infer_sample(
            &params,
            &sample_rows[s],
            sample_alpha[s],
            sample_sigma_sq[s],
        ));
    }

    let sample_params = SampleEmissionParams {
        alpha: sample_alpha,
        sigma_sq: sample_sigma_sq,
    };

    (params, sample_params, results)
}

/// Initialize from data and run EM in one call.
///
/// `segment_data`: [samples x segments] matrix.
/// `cells_per_sample`: number of cells per pseudobulk sample (for variance scaling).
/// `config`: EM configuration.
pub fn fit_cnv_hmm(
    segment_data: &DMatrix<f32>,
    cells_per_sample: &[f32],
    config: &EmConfig,
) -> (CnvHmmParams, SampleEmissionParams, Vec<HmmResult>) {
    let (init_params, init_sample_params) =
        CnvHmmParams::initialize_from_data(segment_data, cells_per_sample);

    log::info!(
        "HMM init: emission_means={:?}, base_sigma_sq range=[{:.4}, {:.4}]",
        init_params.emission_means.as_slice(),
        init_sample_params
            .sigma_sq
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        init_sample_params
            .sigma_sq
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
    );

    fit_em(
        &init_params,
        segment_data,
        Some(&init_sample_params),
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_backward_uniform() {
        let params = CnvHmmParams::default_3state();
        let y = vec![0.0; 10]; // all neutral
        let (posteriors, ll) = forward_backward(&params, &y, 1.0, 0.1);

        assert_eq!(posteriors.nrows(), 10);
        assert_eq!(posteriors.ncols(), 3);
        assert!(ll.is_finite());

        // Neutral state (index 1) should have highest posterior
        for t in 0..10 {
            let neutral_post = posteriors[(t, 1)];
            assert!(
                neutral_post > posteriors[(t, 0)] && neutral_post > posteriors[(t, 2)],
                "neutral posterior {:.3} should dominate at t={}",
                neutral_post,
                t
            );
        }
    }

    #[test]
    fn test_viterbi_step() {
        let params = CnvHmmParams::default_3state();
        // Neutral, then gain
        let mut y = vec![0.0f32; 20];
        #[allow(clippy::needless_range_loop)]
        for i in 10..20 {
            y[i] = 0.5;
        }
        let path = viterbi(&params, &y, 1.0, 0.05);

        // First half should be neutral (1), second half should be gain (2)
        for &s in &path[..5] {
            assert_eq!(s, 1, "expected neutral in first half");
        }
        for &s in &path[15..] {
            assert_eq!(s, 2, "expected gain in second half");
        }
    }
}
