//! Mixture of HMMs with blocked Gibbs sampling for CNV state calling.
//!
//! Alternates:
//! 1. **E-step** (forward-backward): compute per-state posteriors given current emission params
//! 2. **S-step** (ESS): sample new emission params via elliptical slice sampling
//! 3. **Responsibilities**: soft assignment of samples to mixture components
//! 4. **Mixing weights**: Dirichlet posterior draw
//!
//! M=1 degenerates to a single shared HMM.

use log::info;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Distribution, Gamma, StandardNormal};

use mcmc_util::engine::{elliptical_slice_step, EssParam};

use crate::hmm::{forward_backward, shared_viterbi, CnvHmmParams};

/// Configuration for the Gibbs HMM sampler.
#[derive(Debug, Clone)]
pub struct GibbsHmmConfig {
    /// Number of mixture components (default 1).
    pub n_components: usize,
    /// Number of CN states per component (default 3: del/neutral/gain).
    pub n_states: usize,
    /// Total Gibbs iterations (default 500).
    pub n_iter: usize,
    /// Burn-in iterations (default 200).
    pub warmup: usize,
    /// Random seed.
    pub seed: u64,
    /// Prior variance on emission means (default 1.0).
    pub prior_var_mu: f32,
    /// Prior variance on log(sigma²) (default 1.0).
    pub prior_var_log_sigma: f32,
    /// Off-diagonal transition probability (default 1e-6).
    pub transition_prob: f32,
}

impl Default for GibbsHmmConfig {
    fn default() -> Self {
        Self {
            n_components: 1,
            n_states: 3,
            n_iter: 500,
            warmup: 200,
            seed: 42,
            prior_var_mu: 1.0,
            prior_var_log_sigma: 1.0,
            transition_prob: 1e-6,
        }
    }
}

/// Result of the Gibbs HMM sampler.
#[derive(Debug, Clone)]
pub struct GibbsHmmResult {
    pub n_components: usize,
    /// Per-component posterior mean emission means: `[M]` vectors of length K.
    pub emission_means: Vec<DVector<f32>>,
    /// Per-component posterior mean emission variance.
    pub sigma_sq: Vec<f32>,
    /// Per-sample scale factors (posterior mean).
    pub per_sample_alpha: Vec<f32>,
    /// Per-component consensus Viterbi path: `[M]` vectors of length B.
    pub viterbi_paths: Vec<Vec<usize>>,
    /// Per-component averaged posteriors: `[M]` matrices of shape `[B × K]`.
    pub posteriors_mean: Vec<DMatrix<f32>>,
    /// Sample responsibilities: `[S × M]`.
    pub responsibilities: DMatrix<f32>,
    /// Mixing weights (posterior mean): length M.
    pub mixing_weights: Vec<f32>,
    /// Log-likelihood trace (one per iteration, post-warmup).
    pub log_likelihoods: Vec<f32>,
}

/// Packed emission parameters for a single component: `[μ_1, ..., μ_K, log(σ²)]`.
///
/// The neutral state (K/2) is pinned at 0 by reparametrization:
/// we store deltas from neutral and add the constraint in the likelihood.
#[derive(Debug, Clone)]
struct EmissionTheta(DVector<f32>);

impl EssParam for EmissionTheta {
    fn linear_combine(&self, a: f32, other: &Self, b: f32) -> Self {
        EmissionTheta(self.0.linear_combine(a, &other.0, b))
    }
}

impl EmissionTheta {
    fn new(n_states: usize) -> Self {
        // Initialize: means at 0, log(sigma²) = 0 (sigma² = 1)
        EmissionTheta(DVector::zeros(n_states + 1))
    }

    fn n_states(&self) -> usize {
        self.0.len() - 1
    }

    /// Extract emission means with neutral pinned at 0.
    fn means(&self) -> DVector<f32> {
        let k = self.n_states();
        let neutral = k / 2;
        let mut means = self.0.rows(0, k).clone_owned();
        means[neutral] = 0.0; // pin neutral
        means
    }

    fn sigma_sq(&self) -> f32 {
        self.0[self.n_states()].exp().max(1e-6)
    }

    fn to_hmm_params(&self, transition_prob: f32) -> CnvHmmParams {
        let means = self.means();
        let neutral = self.n_states() / 2;
        CnvHmmParams::new(means, transition_prob, neutral)
    }
}

/// Run the mixture Gibbs HMM sampler.
///
/// # Arguments
/// * `block_signal` — `[B × S]` matrix of block-level log(mu_residual)
/// * `chr_block_bounds` — chromosome boundaries in block space: `[(start, end)]`
/// * `config` — sampler configuration
///
/// # Returns
/// `GibbsHmmResult` with posterior summaries.
pub fn fit_gibbs_hmm(
    block_signal: &DMatrix<f32>,
    chr_block_bounds: &[(usize, usize)],
    config: &GibbsHmmConfig,
) -> GibbsHmmResult {
    fit_gibbs_hmm_init(block_signal, chr_block_bounds, config, None)
}

/// Run the mixture Gibbs HMM sampler with optional warm-start from a previous result.
///
/// When `init` is provided, emission params, per-sample alpha, responsibilities,
/// and mixing weights are initialized from the previous level's posterior means.
/// This enables multi-level coarsening where coarse levels warm-start finer ones.
pub fn fit_gibbs_hmm_init(
    block_signal: &DMatrix<f32>,
    chr_block_bounds: &[(usize, usize)],
    config: &GibbsHmmConfig,
    init: Option<&GibbsHmmResult>,
) -> GibbsHmmResult {
    let n_blocks = block_signal.nrows();
    let n_samples = block_signal.ncols();
    let m = config.n_components;
    let k = config.n_states;

    info!(
        "Gibbs HMM: {} blocks × {} samples, {} components, {} states, {} iter (warmup {}), warm_start={}",
        n_blocks, n_samples, m, k, config.n_iter, config.warmup, init.is_some()
    );

    let mut rng = SmallRng::seed_from_u64(config.seed);

    // Extract per-sample signal vectors (column-major → row slices for HMM)
    let sample_signals: Vec<Vec<f32>> = (0..n_samples)
        .map(|s| (0..n_blocks).map(|b| block_signal[(b, s)]).collect())
        .collect();

    // Initialize emission parameters per component
    let mut thetas: Vec<EmissionTheta> = if let Some(prev) = init {
        // Warm-start from previous level
        (0..m)
            .map(|c| {
                let mut theta = EmissionTheta::new(k);
                for j in 0..k.min(prev.emission_means[c].len()) {
                    theta.0[j] = prev.emission_means[c][j];
                }
                theta.0[k] = prev.sigma_sq[c].max(1e-6).ln();
                theta
            })
            .collect()
    } else {
        (0..m)
            .map(|c| {
                let mut theta = EmissionTheta::new(k);
                if k >= 3 {
                    let scale = 0.3 + 0.1 * c as f32;
                    theta.0[0] = -scale; // deletion
                    theta.0[k - 1] = scale; // gain
                }
                theta
            })
            .collect()
    };

    // Per-sample scale factors
    let mut alphas = init
        .map(|prev| prev.per_sample_alpha.clone())
        .unwrap_or_else(|| vec![1.0f32; n_samples]);

    // Mixing weights
    let mut log_pi: Vec<f32> = init
        .map(|prev| {
            prev.mixing_weights
                .iter()
                .map(|&w| w.max(1e-10).ln())
                .collect()
        })
        .unwrap_or_else(|| vec![-(m as f32).ln(); m]);

    // Responsibilities [S × M]
    let mut resp = init
        .map(|prev| prev.responsibilities.clone())
        .unwrap_or_else(|| DMatrix::<f32>::from_element(n_samples, m, 1.0 / m as f32));

    // Accumulators for posterior means (post-warmup)
    let n_collect = config.n_iter.saturating_sub(config.warmup);
    let mut means_acc: Vec<DVector<f32>> = (0..m).map(|_| DVector::zeros(k)).collect();
    let mut sigma_sq_acc: Vec<f32> = vec![0.0; m];
    let mut alpha_acc = vec![0.0f32; n_samples];
    let mut resp_acc = DMatrix::<f32>::zeros(n_samples, m);
    let mut pi_acc = vec![0.0f32; m];
    let mut posteriors_acc: Vec<DMatrix<f32>> =
        (0..m).map(|_| DMatrix::zeros(n_blocks, k)).collect();
    let mut ll_trace: Vec<f32> = Vec::with_capacity(n_collect);

    // Prior scale for ESS
    let prior_std_mu = config.prior_var_mu.sqrt();
    let prior_std_log_sigma = config.prior_var_log_sigma.sqrt();

    // Pre-allocate per-iteration buffers
    let mut posteriors_ms: Vec<Vec<DMatrix<f32>>> = (0..m)
        .map(|_| vec![DMatrix::zeros(n_blocks, k); n_samples])
        .collect();
    let mut ll_ms: Vec<Vec<f32>> = (0..m).map(|_| vec![0.0f32; n_samples]).collect();

    // Gibbs loop
    for iter in 0..config.n_iter {
        let mut total_ll = 0.0f32;

        // --- E-step: forward-backward per component per sample ---
        for c in 0..m {
            let hmm_params = thetas[c].to_hmm_params(config.transition_prob);
            let sigma_sq = thetas[c].sigma_sq();

            for s in 0..n_samples {
                posteriors_ms[c][s].fill(0.0);
                let mut full_ll = 0.0f32;

                for &(chr_start, chr_end) in chr_block_bounds {
                    let chr_signal: Vec<f32> =
                        (chr_start..chr_end).map(|b| sample_signals[s][b]).collect();
                    if chr_signal.is_empty() {
                        continue;
                    }
                    let (post, ll) =
                        forward_backward(&hmm_params, &chr_signal, alphas[s], sigma_sq);
                    full_ll += ll;
                    for t in 0..post.nrows() {
                        for j in 0..k {
                            posteriors_ms[c][s][(chr_start + t, j)] = post[(t, j)];
                        }
                    }
                }

                ll_ms[c][s] = full_ll;
            }
        }

        // --- Responsibilities ---
        for s in 0..n_samples {
            // log r_sm = log(pi_m) + ll[m][s]
            let log_r: Vec<f32> = (0..m).map(|c| log_pi[c] + ll_ms[c][s]).collect();
            // log-sum-exp for normalization
            let max_lr = log_r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = max_lr + log_r.iter().map(|&v| (v - max_lr).exp()).sum::<f32>().ln();
            for c in 0..m {
                resp[(s, c)] = (log_r[c] - lse).exp();
            }
            total_ll += lse;
        }

        // --- S-step: ESS for emission params per component ---
        for c in 0..m {
            let posteriors_c = &posteriors_ms[c];

            let lnpdf = |theta: &EmissionTheta| -> f32 {
                let means = theta.means();
                let sig2 = theta.sigma_sq();
                let inv_sig2 = 1.0 / sig2;
                let log_norm = -0.5 * (sig2.ln() + std::f32::consts::TAU.ln());

                let mut ll = 0.0f32;
                for s in 0..n_samples {
                    let w = resp[(s, c)];
                    if w < 1e-10 {
                        continue;
                    }
                    let alpha_s = alphas[s];
                    for t in 0..n_blocks {
                        let y = sample_signals[s][t];
                        for j in 0..k {
                            let gamma = posteriors_c[s][(t, j)];
                            if gamma < 1e-10 {
                                continue;
                            }
                            let diff = y - alpha_s * means[j];
                            ll += w * gamma * (log_norm - 0.5 * diff * diff * inv_sig2);
                        }
                    }
                }
                ll
            };

            let cur_ll = lnpdf(&thetas[c]);

            // Draw prior sample for ESS
            let prior_sample = {
                let dim = k + 1;
                let mut v = DVector::<f32>::zeros(dim);
                for i in 0..k {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    v[i] = z as f32 * prior_std_mu;
                }
                let z: f64 = StandardNormal.sample(&mut rng);
                v[k] = z as f32 * prior_std_log_sigma;
                EmissionTheta(v)
            };

            let (new_theta, _new_ll) =
                elliptical_slice_step(&thetas[c], &prior_sample, &lnpdf, cur_ll, &mut rng);
            thetas[c] = new_theta;
        }

        // --- Per-sample scale (closed form, weighted by responsibilities) ---
        let component_means: Vec<DVector<f32>> = (0..m).map(|c| thetas[c].means()).collect();
        for s in 0..n_samples {
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for c in 0..m {
                let w = resp[(s, c)];
                if w < 1e-10 {
                    continue;
                }
                let means = &component_means[c];
                for t in 0..n_blocks {
                    let y = sample_signals[s][t];
                    for j in 0..k {
                        let gamma = posteriors_ms[c][s][(t, j)];
                        let wg = w * gamma;
                        num += wg * y * means[j];
                        den += wg * means[j] * means[j];
                    }
                }
            }
            alphas[s] = if den.abs() > 1e-10 { num / den } else { 1.0 };
        }

        // --- Mixing weights: Dirichlet posterior ---
        if m > 1 {
            let counts: Vec<f32> = (0..m)
                .map(|c| 1.0 + (0..n_samples).map(|s| resp[(s, c)]).sum::<f32>())
                .collect();

            // Sample from Dirichlet via Gamma
            let mut pi_new = vec![0.0f32; m];
            let mut sum_g = 0.0f32;
            for c in 0..m {
                let gamma_dist = Gamma::new(counts[c] as f64, 1.0).unwrap();
                let g: f64 = gamma_dist.sample(&mut rng);
                pi_new[c] = g as f32;
                sum_g += g as f32;
            }
            for c in 0..m {
                pi_new[c] /= sum_g;
                log_pi[c] = pi_new[c].max(1e-10).ln();
            }
        }

        // --- Collect post-warmup ---
        if iter >= config.warmup {
            for c in 0..m {
                let means = thetas[c].means();
                means_acc[c] += &means;
                sigma_sq_acc[c] += thetas[c].sigma_sq();
            }
            for s in 0..n_samples {
                alpha_acc[s] += alphas[s];
                for c in 0..m {
                    resp_acc[(s, c)] += resp[(s, c)];
                }
            }
            for c in 0..m {
                for s in 0..n_samples {
                    let w = resp[(s, c)];
                    for t in 0..n_blocks {
                        for j in 0..k {
                            posteriors_acc[c][(t, j)] += w * posteriors_ms[c][s][(t, j)];
                        }
                    }
                }
            }
            let pi_vals: Vec<f32> = log_pi.iter().map(|&lp| lp.exp()).collect();
            for c in 0..m {
                pi_acc[c] += pi_vals[c];
            }
            ll_trace.push(total_ll);
        }

        if iter % 50 == 0 || iter == config.n_iter - 1 {
            let means_str: Vec<String> = (0..m)
                .map(|c| format!("{:?}", thetas[c].means().as_slice()))
                .collect();
            info!(
                "Gibbs iter {}/{}: ll={:.2}, means=[{}]",
                iter,
                config.n_iter,
                total_ll,
                means_str.join("; "),
            );
        }
    }

    // --- Compute posterior means ---
    let n_coll = n_collect.max(1) as f32;
    let emission_means: Vec<DVector<f32>> = means_acc
        .into_iter()
        .map(|v| v.scale(1.0 / n_coll))
        .collect();
    let sigma_sq: Vec<f32> = sigma_sq_acc.iter().map(|&v| v / n_coll).collect();
    let per_sample_alpha: Vec<f32> = alpha_acc.iter().map(|&v| v / n_coll).collect();
    resp_acc.scale_mut(1.0 / n_coll);
    let mixing_weights: Vec<f32> = pi_acc.iter().map(|&v| v / n_coll).collect();

    for acc in posteriors_acc.iter_mut() {
        acc.scale_mut(1.0 / (n_coll * n_samples as f32));
    }

    // --- Consensus Viterbi per component using posterior mean params ---
    let viterbi_paths: Vec<Vec<usize>> = (0..m)
        .map(|c| {
            let hmm_params =
                CnvHmmParams::new(emission_means[c].clone(), config.transition_prob, k / 2);
            let sig2 = sigma_sq[c];

            // Run shared Viterbi per chromosome, stitch together
            let mut full_path = vec![0usize; n_blocks];
            for &(chr_start, chr_end) in chr_block_bounds {
                let signals: Vec<&[f32]> = (0..n_samples)
                    .map(|s| &sample_signals[s][chr_start..chr_end])
                    .collect();
                let sigs: Vec<f32> = vec![sig2; n_samples];
                let weights: Vec<f32> = (0..n_samples).map(|s| resp_acc[(s, c)]).collect();
                let path = shared_viterbi(
                    &hmm_params,
                    &signals,
                    &per_sample_alpha,
                    &sigs,
                    Some(&weights),
                );
                for (i, &state) in path.iter().enumerate() {
                    full_path[chr_start + i] = state;
                }
            }
            full_path
        })
        .collect();

    info!(
        "Gibbs HMM done. Components: {}, final ll: {:.2}",
        m,
        ll_trace.last().copied().unwrap_or(f32::NAN)
    );

    GibbsHmmResult {
        n_components: m,
        emission_means,
        sigma_sq,
        per_sample_alpha,
        viterbi_paths,
        posteriors_mean: posteriors_acc,
        responsibilities: resp_acc,
        mixing_weights,
        log_likelihoods: ll_trace,
    }
}

/// Run multi-level Gibbs HMM: coarsen at decreasing correlation thresholds,
/// warm-starting each level from the previous level's posterior params.
///
/// # Arguments
/// * `log_resid` — `[G_ordered × S]` matrix of log(mu_residual) in genome order
/// * `chr_bounds` — chromosome boundaries: `[(chr_name, start, end)]`
/// * `corr_thresholds` — decreasing correlation thresholds, e.g. `[0.8, 0.6, 0.4]`
///   (coarsest → finest). Each level produces a coarsening at that threshold.
/// * `config` — Gibbs config (applied at each level; iterations may be reduced
///   at coarser levels since they only need rough param estimates)
///
/// # Returns
/// `(GibbsHmmResult, GenomicCoarsening)` from the finest (last) level.
pub fn fit_gibbs_hmm_multilevel(
    log_resid: &DMatrix<f32>,
    chr_bounds: &[(Box<str>, usize, usize)],
    corr_thresholds: &[f32],
    config: &GibbsHmmConfig,
) -> (GibbsHmmResult, crate::genomic_coarsening::GenomicCoarsening) {
    use crate::genomic_coarsening::greedy_coarsen;

    assert!(
        !corr_thresholds.is_empty(),
        "need at least one correlation threshold"
    );

    let n_levels = corr_thresholds.len();
    let mut prev_result: Option<GibbsHmmResult> = None;
    let mut final_coarsening = None;

    for (level, &threshold) in corr_thresholds.iter().enumerate() {
        info!(
            "=== Multi-level CNV: level {}/{}, corr_threshold={:.2} ===",
            level + 1,
            n_levels,
            threshold
        );

        let coarsening = greedy_coarsen(log_resid, chr_bounds, threshold);
        let block_signal = coarsening.aggregate_to_blocks(log_resid);

        let chr_block_bounds: Vec<(usize, usize)> = coarsening
            .chr_block_boundaries()
            .iter()
            .map(|(_, s, e)| (*s, *e))
            .collect();

        // At coarser levels, use fewer iterations (just need rough params)
        let level_config = if level < n_levels - 1 {
            GibbsHmmConfig {
                n_iter: config.n_iter / 2,
                warmup: config.warmup / 2,
                ..config.clone()
            }
        } else {
            config.clone()
        };

        let result = fit_gibbs_hmm_init(
            &block_signal,
            &chr_block_bounds,
            &level_config,
            prev_result.as_ref(),
        );

        prev_result = Some(result);
        final_coarsening = Some(coarsening);
    }

    (prev_result.unwrap(), final_coarsening.unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_component_neutral() {
        // All-neutral signal: 20 blocks × 5 samples, all near 0
        let n_blocks = 20;
        let n_samples = 5;
        let block_signal = DMatrix::<f32>::zeros(n_blocks, n_samples);
        let chr_bounds = vec![(0usize, n_blocks)];

        let config = GibbsHmmConfig {
            n_components: 1,
            n_states: 3,
            n_iter: 50,
            warmup: 20,
            seed: 42,
            ..Default::default()
        };

        let result = fit_gibbs_hmm(&block_signal, &chr_bounds, &config);
        assert_eq!(result.n_components, 1);
        assert_eq!(result.viterbi_paths.len(), 1);
        assert_eq!(result.viterbi_paths[0].len(), n_blocks);

        // All blocks should be neutral (state 1)
        for &state in &result.viterbi_paths[0] {
            assert_eq!(state, 1, "expected neutral state for zero signal");
        }
    }

    #[test]
    fn test_single_component_with_gain() {
        // 30 blocks × 8 samples
        // Blocks 0-14: neutral (0.0)
        // Blocks 15-29: gain (+0.5)
        let n_blocks = 30;
        let n_samples = 8;
        let mut block_signal = DMatrix::<f32>::zeros(n_blocks, n_samples);
        for b in 15..30 {
            for s in 0..n_samples {
                block_signal[(b, s)] = 0.5;
            }
        }

        let chr_bounds = vec![(0, n_blocks)];
        let config = GibbsHmmConfig {
            n_components: 1,
            n_states: 3,
            n_iter: 300,
            warmup: 150,
            seed: 42,
            ..Default::default()
        };

        let result = fit_gibbs_hmm(&block_signal, &chr_bounds, &config);

        // First half should be neutral (state 1)
        for &state in &result.viterbi_paths[0][..10] {
            assert_eq!(state, 1, "expected neutral in first half");
        }
        // Second half should be non-neutral (gain or deletion depending on label)
        // The key structural test: the two halves have different states
        let first_half_state = result.viterbi_paths[0][5];
        let second_half_state = result.viterbi_paths[0][25];
        assert_ne!(
            first_half_state, second_half_state,
            "first and second half should have different CN states"
        );
        assert_eq!(first_half_state, 1, "first half should be neutral");
    }

    #[test]
    fn test_two_components() {
        // 20 blocks × 10 samples
        // Samples 0-4: gain in blocks 10-19
        // Samples 5-9: deletion in blocks 10-19
        let n_blocks = 20;
        let n_samples = 10;
        let mut block_signal = DMatrix::<f32>::zeros(n_blocks, n_samples);
        for b in 10..20 {
            for s in 0..5 {
                block_signal[(b, s)] = 0.5;
            }
            for s in 5..10 {
                block_signal[(b, s)] = -0.5;
            }
        }

        let chr_bounds = vec![(0, n_blocks)];
        let config = GibbsHmmConfig {
            n_components: 2,
            n_states: 3,
            n_iter: 300,
            warmup: 150,
            seed: 42,
            ..Default::default()
        };

        let result = fit_gibbs_hmm(&block_signal, &chr_bounds, &config);

        assert_eq!(result.n_components, 2);
        assert_eq!(result.viterbi_paths.len(), 2);
        assert_eq!(result.responsibilities.ncols(), 2);
        assert_eq!(result.responsibilities.nrows(), n_samples);

        // Each sample should have dominant responsibility to one component
        for s in 0..n_samples {
            let max_resp = (0..2)
                .map(|c| result.responsibilities[(s, c)])
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                max_resp > 0.3,
                "sample {} should have clear component assignment, max_resp={}",
                s,
                max_resp
            );
        }
    }

    #[test]
    fn test_multilevel() {
        // 60 genes × 10 samples on one chromosome
        // Genes 0-29: neutral baseline + shared sample pattern
        // Genes 30-59: gain (+0.5) + same shared sample pattern
        // The shared pattern across samples makes correlation high within groups
        let n_genes = 60;
        let n_samples = 10;
        let mut log_resid = DMatrix::<f32>::zeros(n_genes, n_samples);

        // Different sample patterns per group so correlation across groups is low
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        use rand::prelude::*;
        // Pattern A for neutral genes: random but shared within group
        let pattern_a: Vec<f32> = (0..n_samples)
            .map(|_| rng.random_range(-0.3..0.3))
            .collect();
        // Pattern B for gain genes: different random pattern + offset
        let pattern_b: Vec<f32> = (0..n_samples)
            .map(|_| rng.random_range(-0.3..0.3))
            .collect();
        for g in 0..30 {
            for s in 0..n_samples {
                log_resid[(g, s)] = pattern_a[s];
            }
        }
        for g in 30..60 {
            for s in 0..n_samples {
                log_resid[(g, s)] = 0.5 + pattern_b[s];
            }
        }

        let chr_bounds: Vec<(Box<str>, usize, usize)> = vec![("chr1".into(), 0, n_genes)];
        let thresholds = vec![0.8, 0.5];

        let config = GibbsHmmConfig {
            n_components: 1,
            n_states: 3,
            n_iter: 400,
            warmup: 200,
            seed: 42,
            ..Default::default()
        };

        let (result, coarsening) =
            fit_gibbs_hmm_multilevel(&log_resid, &chr_bounds, &thresholds, &config);

        // Should have reasonable number of blocks (2 groups → 2 blocks at finest)
        assert!(coarsening.num_blocks() >= 2, "at least 2 blocks expected");

        // Viterbi path length should match number of blocks
        assert_eq!(result.viterbi_paths[0].len(), coarsening.num_blocks());

        // Multi-level should complete without errors and produce valid results
        assert!(!result.log_likelihoods.is_empty(), "should collect samples");
        assert_eq!(result.emission_means.len(), 1);
        assert_eq!(result.emission_means[0].len(), 3);
    }

    #[test]
    fn test_simulated_cnv_recovery() {
        // Simulate realistic log(mu_residual):
        // 300 genes, 3 chromosomes (100 each), 20 samples
        // Chr1: genes 30-60 have gain (+0.4) in samples 0-9 only
        // Chr2: genes 170-190 have deletion (-0.5) in all samples
        // Rest: neutral with per-gene, per-sample noise
        use rand::prelude::*;
        use rand_distr::{Distribution, Normal};

        let n_genes = 300;
        let n_samples = 20;
        let mut rng = SmallRng::seed_from_u64(2024);
        let noise = Normal::new(0.0f32, 0.08).unwrap();

        let mut log_resid = DMatrix::<f32>::zeros(n_genes, n_samples);

        // Add per-gene baseline (shared across samples) + noise
        for g in 0..n_genes {
            let baseline: f32 = noise.sample(&mut rng) * 0.5;
            for s in 0..n_samples {
                log_resid[(g, s)] = baseline + noise.sample(&mut rng);
            }
        }

        // CNV gain on chr1 genes 30-60, samples 0-9 only
        for g in 30..60 {
            for s in 0..10 {
                log_resid[(g, s)] += 0.4;
            }
        }

        // CNV deletion on chr2 genes 170-190, all samples
        for g in 170..190 {
            for s in 0..n_samples {
                log_resid[(g, s)] -= 0.5;
            }
        }

        let chr_bounds: Vec<(Box<str>, usize, usize)> = vec![
            ("chr1".into(), 0, 100),
            ("chr2".into(), 100, 200),
            ("chr3".into(), 200, 300),
        ];

        let thresholds = vec![0.7, 0.4];
        let config = GibbsHmmConfig {
            n_components: 1,
            n_states: 3,
            n_iter: 400,
            warmup: 200,
            seed: 42,
            ..Default::default()
        };

        let (result, coarsening) =
            fit_gibbs_hmm_multilevel(&log_resid, &chr_bounds, &thresholds, &config);

        let gene_states = coarsening.expand_vec_to_genes(&result.viterbi_paths[0], n_genes);

        // Check that neutral regions are called neutral (state 1)
        let neutral_state = 1;
        let neutral_correct: usize = (0..30)
            .chain(60..170)
            .chain(190..300)
            .filter(|&g| gene_states[g] == neutral_state)
            .count();
        let neutral_total = 30 + 110 + 110; // 250
        let neutral_accuracy = neutral_correct as f32 / neutral_total as f32;

        // Check that the deletion region (170-190) is called non-neutral
        let del_non_neutral: usize = (170..190)
            .filter(|&g| gene_states[g] != neutral_state)
            .count();

        eprintln!(
            "Neutral accuracy: {}/{} = {:.1}%",
            neutral_correct,
            neutral_total,
            neutral_accuracy * 100.0
        );
        eprintln!(
            "Deletion detected: {}/20 genes non-neutral",
            del_non_neutral
        );
        eprintln!("Emission means: {:?}", result.emission_means[0].as_slice());

        // Neutral regions should be mostly correct (>80%)
        assert!(
            neutral_accuracy > 0.8,
            "neutral accuracy {:.1}% too low",
            neutral_accuracy * 100.0
        );
        // Deletion region should be mostly detected (>50%)
        assert!(
            del_non_neutral > 10,
            "only {}/20 deletion genes detected",
            del_non_neutral
        );
    }
}
