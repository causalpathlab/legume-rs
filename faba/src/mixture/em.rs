/// EM algorithm parameters.
#[derive(Clone, Copy)]
pub struct EmParams {
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Convergence tolerance for log-likelihood change
    pub tol: f32,
    /// Minimum component weight before pruning
    pub min_weight: f32,
}

impl Default for EmParams {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-4,
            min_weight: 0.01,
        }
    }
}

/// Result of fixed-parameter EM (weights only).
///
/// `gamma` is stored in **row-major flat layout**: posterior γ_{n,k} lives
/// at `gamma[n * n_components + k]`. The old nested `Vec<Vec<f32>>` did
/// `n_obs` separate heap allocations and scattered the rows across the
/// allocator; the flat layout keeps the E-step's sequential read of
/// (n, *) on the same cache line and lets `par_chunks_mut(n_components)`
/// hand out non-overlapping row slices for the parallel path.
pub struct FixedEmResult {
    /// Estimated mixing weights pi_0..pi_K (pi_0 = noise)
    pub weights: Vec<f32>,
    /// Flat posterior gamma; use `gamma_row(n)` for slice access.
    pub gamma: Vec<f32>,
    /// Number of mixture components (= row stride of `gamma`).
    pub n_components: usize,
    /// Final log-likelihood
    pub log_lik: f32,
    /// BIC
    pub bic: f32,
    /// Number of iterations
    pub n_iter: usize,
}

impl FixedEmResult {
    /// Posterior γ_{n,·} for observation `n`.
    #[allow(dead_code)] // public helper; the in-crate caller unpacks `gamma` directly.
    #[inline]
    pub fn gamma_row(&self, n: usize) -> &[f32] {
        let start = n * self.n_components;
        &self.gamma[start..start + self.n_components]
    }
}

/// Result of full Gaussian mixture EM.
pub struct GmmResult {
    /// Mixing weights pi_0..pi_K (pi_0 = noise/uniform)
    pub weights: Vec<f32>,
    /// Learned component means
    pub mus: Vec<f32>,
    /// Learned component standard deviations
    pub sigmas: Vec<f32>,
    /// Posterior assignment probabilities gamma[n][k]
    #[allow(dead_code)] // set by the GMM path; bandwidth-first caller leaves it empty
    pub gamma: Vec<Vec<f32>>,
    /// BIC
    #[allow(dead_code)] // no BIC model selection on the editing path
    pub bic: f32,
}

/// Numerically stable log-sum-exp: log(exp(a) + exp(b))
pub fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// E-step parallelism gate. Below this fragment count, rayon spawn overhead
/// dominates the per-fragment work (~10 f32 ops × `n_total`); above it, the
/// heavy-UTR tail dominates and parallelising the per-fragment log-norm +
/// γ-update lets idle rayon workers steal it. 4096 was picked empirically
/// as the smallest `n_obs` where chr1's housekeeping-gene UTRs (HSPA8,
/// ACTB, …) start dwarfing all the rest combined.
const E_STEP_PARALLEL_THRESHOLD: usize = 4096;

/// Fixed-parameter EM: given precomputed per-observation component log-likelihoods,
/// estimate only the mixing weights.
///
/// **Contract:** column 0 of `component_log_liks` is treated as a catch-all
/// (uniform-noise) component that is preserved by the pruning logic and is the
/// sole survivor if every signal component is pruned below `min_weight` in the
/// same iteration. Callers without a noise column should add a flat one (e.g.
/// `-ln(domain_length)`) at index 0.
///
/// * `component_log_liks` - flat row-major (`n_obs × n_components`) log-lik
///   matrix (column 0 = noise component; see contract above)
/// * `n_components` - row stride of `component_log_liks`
/// * `n_free_params` - number of free parameters for BIC computation
/// * `params` - EM parameters
#[allow(dead_code)] // unweighted convenience wrapper; in-crate caller goes through `fixed_em_weighted`
pub fn fixed_em(
    component_log_liks: &[f32],
    n_components: usize,
    n_free_params: usize,
    params: &EmParams,
) -> FixedEmResult {
    fixed_em_weighted(
        component_log_liks,
        n_components,
        None,
        None,
        n_free_params,
        params,
    )
}

/// Weighted variant of [`fixed_em`].
///
/// `weights_per_obs` lets the caller treat each observation row as
/// representing `c_m` original observations (fragment clusters); the
/// E-step is unchanged (γ is per-cluster), but the log-likelihood and
/// M-step accumulators multiply by `c_m`:
///
///     log L = Σ_m c_m · log Σ_k π_k L_{m,k}
///     π_k   = (1 / N) Σ_m c_m · γ_{m,k}
///
/// `n_for_bic` is the BIC sample size used in the `n_params · ln(N)`
/// penalty — pass the **original** fragment count, not `n_obs`. Default
/// (`None`) falls back to `n_obs`, giving the standard unweighted BIC.
pub fn fixed_em_weighted(
    component_log_liks: &[f32],
    n_components: usize,
    weights_per_obs: Option<&[f32]>,
    n_for_bic: Option<usize>,
    n_free_params: usize,
    params: &EmParams,
) -> FixedEmResult {
    let n_total = n_components;
    let n_obs = if n_total == 0 {
        0
    } else {
        component_log_liks.len() / n_total
    };
    debug_assert_eq!(
        component_log_liks.len(),
        n_obs * n_total,
        "component_log_liks length must be a multiple of n_components"
    );
    if let Some(w) = weights_per_obs {
        debug_assert_eq!(
            w.len(),
            n_obs,
            "weights_per_obs length must equal n_obs (rows of component_log_liks)"
        );
    }
    if n_obs == 0 {
        return FixedEmResult {
            weights: vec![],
            gamma: vec![],
            n_components: n_total,
            log_lik: 0.0,
            bic: 0.0,
            n_iter: 0,
        };
    }

    // Total weight = N (for BIC + π normalisation). Without weights it's
    // just n_obs.
    let total_weight: f32 = weights_per_obs
        .map(|w| w.iter().sum())
        .unwrap_or(n_obs as f32);
    let n_bic = n_for_bic.unwrap_or(n_obs);

    let mut weights = vec![1.0 / n_total as f32; n_total];
    let mut gamma = vec![0.0_f32; n_obs * n_total];
    let mut log_weights = vec![f32::NEG_INFINITY; n_total];
    let mut prev_ll = f32::NEG_INFINITY;

    let mut iter = 0;
    loop {
        // Precompute log(weights) once per iter so the inner loop avoids
        // recomputing `ln()` per (fragment, component).
        for k in 0..n_total {
            log_weights[k] = if weights[k] > 0.0 {
                weights[k].ln()
            } else {
                f32::NEG_INFINITY
            };
        }

        // E-step. Per-cluster γ is unweighted (conditional on belonging
        // to the cluster); the cluster's multiplicity `c_m` enters as a
        // linear factor on the log-likelihood contribution and on the
        // M-step γ accumulator. With `weights_per_obs = None`, c_m = 1
        // and the math collapses to plain unweighted EM.
        let total_ll: f32 = if n_obs >= E_STEP_PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            gamma
                .par_chunks_mut(n_total)
                .enumerate()
                .map(|(n, gamma_row)| {
                    let log_norm = e_step_one(
                        gamma_row,
                        &component_log_liks[n * n_total..(n + 1) * n_total],
                        &log_weights,
                    );
                    let w = weights_per_obs.map(|w| w[n]).unwrap_or(1.0);
                    w * log_norm
                })
                .sum()
        } else {
            let mut total = 0.0_f32;
            for (n, gamma_row) in gamma.chunks_mut(n_total).enumerate() {
                let log_norm = e_step_one(
                    gamma_row,
                    &component_log_liks[n * n_total..(n + 1) * n_total],
                    &log_weights,
                );
                let w = weights_per_obs.map(|w| w[n]).unwrap_or(1.0);
                total += w * log_norm;
            }
            total
        };

        iter += 1;

        let ll_change = (total_ll - prev_ll).abs();
        if iter > 1 && (ll_change < params.tol || iter >= params.max_iter) {
            let bic = -2.0 * total_ll + n_free_params as f32 * (n_bic as f32).ln();

            return FixedEmResult {
                weights,
                gamma,
                n_components: n_total,
                log_lik: total_ll,
                bic,
                n_iter: iter,
            };
        }

        prev_ll = total_ll;

        // M-step: weighted column sums of γ. With weights_per_obs the
        // per-cluster γ row is scaled by `c_m` before accumulating —
        // i.e. `π_k = (1/N) · Σ_m c_m · γ_{m,k}` where N = total_weight.
        let mut sum_gammas = vec![0.0_f32; n_total];
        if let Some(w) = weights_per_obs {
            for (chunk, &c_m) in gamma.chunks(n_total).zip(w.iter()) {
                for k in 0..n_total {
                    sum_gammas[k] += c_m * chunk[k];
                }
            }
        } else {
            for chunk in gamma.chunks(n_total) {
                for k in 0..n_total {
                    sum_gammas[k] += chunk[k];
                }
            }
        }
        for k in 0..n_total {
            weights[k] = sum_gammas[k] / total_weight;
        }

        // Prune low-weight components (skip noise at k=0)
        for w in weights.iter_mut().skip(1) {
            if *w < params.min_weight {
                *w = 0.0;
            }
        }

        // Renormalize; if everything collapsed, fall back to noise-only.
        let w_sum: f32 = weights.iter().sum();
        if w_sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= w_sum;
            }
        } else {
            weights[0] = 1.0;
        }
    }
}

/// Single-fragment E-step. Returns `log p(x_n) = log Σ_k π_k · p(x_n|k)`
/// and writes posterior γ_{n,·} into `gamma_row`.
#[inline]
fn e_step_one(gamma_row: &mut [f32], cll_row: &[f32], log_weights: &[f32]) -> f32 {
    let n_total = gamma_row.len();
    debug_assert_eq!(cll_row.len(), n_total);
    debug_assert_eq!(log_weights.len(), n_total);

    // Pass 1: log-norm = log Σ_k exp(log_w_k + log_lik_nk)
    let mut log_norm = f32::NEG_INFINITY;
    for k in 0..n_total {
        let lw = log_weights[k];
        if lw.is_finite() {
            log_norm = log_sum_exp(log_norm, lw + cll_row[k]);
        }
    }

    // Pass 2: γ_{n,k} = exp(log_w_k + log_lik_nk − log_norm)
    for k in 0..n_total {
        let lw = log_weights[k];
        gamma_row[k] = if lw.is_finite() {
            (lw + cll_row[k] - log_norm).exp()
        } else {
            0.0
        };
    }

    log_norm
}

/// Log PDF of a Gaussian distribution
fn gaussian_log_pdf(x: f32, mu: f32, sigma: f32) -> f32 {
    if sigma <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * std::f32::consts::TAU.ln()
}

/// Full 1D Gaussian mixture EM with a uniform noise component (unit weights).
///
/// Convenience wrapper around `weighted_gaussian_mixture_em` with all weights = 1.
#[allow(dead_code)]
pub fn gaussian_mixture_em(
    observations: &[f32],
    initial_mus: &[f32],
    initial_sigma: f32,
    domain_length: f32,
    params: &EmParams,
) -> GmmResult {
    let unit_weights = vec![1.0_f32; observations.len()];
    weighted_gaussian_mixture_em(
        observations,
        &unit_weights,
        initial_mus,
        initial_sigma,
        domain_length,
        params,
    )
}

/// Weighted 1D Gaussian mixture EM with a uniform noise component.
///
/// Convenience wrapper around `weighted_gaussian_mixture_em_with_n` that uses
/// `n_for_bic = observations.len()` (textbook BIC convention).
pub fn weighted_gaussian_mixture_em(
    observations: &[f32],
    obs_weights: &[f32],
    initial_mus: &[f32],
    initial_sigma: f32,
    domain_length: f32,
    params: &EmParams,
) -> GmmResult {
    weighted_gaussian_mixture_em_with_n(
        observations,
        obs_weights,
        initial_mus,
        initial_sigma,
        domain_length,
        params,
        observations.len() as f32,
    )
}

/// Weighted 1D Gaussian mixture EM with explicit BIC sample size.
///
/// Same as `weighted_gaussian_mixture_em` except `n_for_bic` is the value used
/// in the BIC penalty `n_params · ln(N)`. Observation weights are renormalized
/// so they sum to `n_for_bic`, which keeps the log-likelihood and the BIC
/// penalty on the same scale — model selection is then invariant to a uniform
/// rescaling of the input weights. This matters when weights are fractional
/// (e.g. Beta-posterior regularized rates) and would otherwise shrink the
/// likelihood relative to the penalty.
///
/// Component 0 is always uniform noise over `[0, domain_length]`.
/// Components 1..K are Gaussians.
pub fn weighted_gaussian_mixture_em_with_n(
    observations: &[f32],
    obs_weights_in: &[f32],
    initial_mus: &[f32],
    initial_sigma: f32,
    domain_length: f32,
    params: &EmParams,
    n_for_bic: f32,
) -> GmmResult {
    let n = observations.len();
    let k = initial_mus.len();
    let n_total = k + 1; // +1 for noise

    if n == 0 || k == 0 {
        return GmmResult {
            weights: vec![],
            mus: vec![],
            sigmas: vec![],
            gamma: vec![],
            bic: 0.0,
        };
    }

    // Renormalize observation weights to sum to `n_for_bic` so that the
    // log-likelihood (which scales with sum of weights) is on the same scale
    // as the BIC penalty `n_params · ln(n_for_bic)`. This makes K-selection
    // invariant to a uniform rescaling of the input weights.
    let input_total: f32 = obs_weights_in.iter().sum();
    let obs_weights: Vec<f32> = if input_total > 0.0 && n_for_bic > 0.0 {
        let scale = n_for_bic / input_total;
        obs_weights_in.iter().map(|w| w * scale).collect()
    } else {
        obs_weights_in.to_vec()
    };
    let obs_weights = obs_weights.as_slice();

    let total_weight: f32 = obs_weights.iter().sum();

    let mut weights = vec![1.0 / n_total as f32; n_total];
    let mut mus = initial_mus.to_vec();
    let mut sigmas = vec![initial_sigma; k];
    let mut gamma = vec![vec![0.0; n_total]; n];
    let mut prev_ll = f32::NEG_INFINITY;
    let mut log_probs = vec![f32::NEG_INFINITY; n_total];

    let noise_log_lik = if domain_length > 0.0 {
        -(domain_length.ln())
    } else {
        f32::NEG_INFINITY
    };

    let sigma_floor = domain_length / (100.0 * k as f32);

    let mut iter = 0;
    loop {
        // E-step
        let mut total_ll = 0.0;

        for i in 0..n {
            let x = observations[i];
            let w = obs_weights[i];

            for lp in log_probs.iter_mut() {
                *lp = f32::NEG_INFINITY;
            }

            // Noise component
            if weights[0] > 0.0 {
                log_probs[0] = weights[0].ln() + noise_log_lik;
            }

            // Gaussian components
            for j in 0..k {
                if weights[j + 1] > 0.0 {
                    log_probs[j + 1] = weights[j + 1].ln() + gaussian_log_pdf(x, mus[j], sigmas[j]);
                }
            }

            let log_norm = log_probs[..n_total]
                .iter()
                .fold(f32::NEG_INFINITY, |acc, &lp| log_sum_exp(acc, lp));

            total_ll += w * log_norm;

            for (kk, lp) in log_probs.iter().enumerate().take(n_total) {
                gamma[i][kk] = (lp - log_norm).exp();
            }
        }

        iter += 1;

        let ll_change = (total_ll - prev_ll).abs();
        if iter > 1 && (ll_change < params.tol || iter >= params.max_iter) {
            let n_params = 3 * k;
            let bic = -2.0 * total_ll + n_params as f32 * n_for_bic.ln();

            return GmmResult {
                weights,
                mus,
                sigmas,
                gamma,
                bic,
            };
        }

        prev_ll = total_ll;

        // M-step: update weights, mus, sigmas
        for j in 0..k {
            let kk = j + 1; // skip noise
            let sum_gamma: f32 = gamma
                .iter()
                .zip(obs_weights.iter())
                .map(|(g, &w)| w * g[kk])
                .sum();

            if sum_gamma < 1e-10 {
                weights[kk] = 0.0;
                continue;
            }

            weights[kk] = sum_gamma / total_weight;

            // Update mu
            let mu_new: f32 = gamma
                .iter()
                .zip(observations.iter())
                .zip(obs_weights.iter())
                .map(|((g, &x), &w)| w * g[kk] * x)
                .sum::<f32>()
                / sum_gamma;
            mus[j] = mu_new;

            // Update sigma
            let var_new: f32 = gamma
                .iter()
                .zip(observations.iter())
                .zip(obs_weights.iter())
                .map(|((g, &x), &w)| w * g[kk] * (x - mu_new).powi(2))
                .sum::<f32>()
                / sum_gamma;
            sigmas[j] = var_new.sqrt().max(sigma_floor);
        }

        // Update noise weight
        let noise_sum: f32 = gamma
            .iter()
            .zip(obs_weights.iter())
            .map(|(g, &w)| w * g[0])
            .sum();
        weights[0] = noise_sum / total_weight;

        // Prune low-weight Gaussian components
        for w in weights.iter_mut().skip(1) {
            if *w < params.min_weight {
                *w = 0.0;
            }
        }

        // Renormalize; if everything collapsed, fall back to noise-only.
        let w_sum: f32 = weights.iter().sum();
        if w_sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= w_sum;
            }
        } else {
            weights[0] = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp() {
        let a = 2.0_f32.ln();
        let b = 3.0_f32.ln();
        let result = log_sum_exp(a, b);
        assert!((result - 5.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_log_sum_exp_neg_inf() {
        assert_eq!(log_sum_exp(f32::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(log_sum_exp(1.0, f32::NEG_INFINITY), 1.0);
    }

    #[test]
    fn test_gmm_single_cluster() {
        // Generate 100 points around mu=50
        let mut obs = Vec::new();
        let mut rng_state = 42u64;
        for _ in 0..100 {
            // Simple LCG for deterministic test
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng_state >> 11) as f32 / (1u64 << 53) as f32;
            // Box-Muller (simplified, using one uniform)
            let x = 50.0 + 5.0 * (2.0 * std::f32::consts::PI * u).cos();
            obs.push(x);
        }

        let params = EmParams {
            max_iter: 100,
            tol: 1e-6,
            min_weight: 0.01,
        };

        let result = gaussian_mixture_em(&obs, &[50.0], 10.0, 100.0, &params);

        assert_eq!(result.mus.len(), 1);
        assert!(
            (result.mus[0] - 50.0).abs() < 10.0,
            "mu should be near 50, got {}",
            result.mus[0]
        );
        assert!(
            result.weights[1] > 0.5,
            "Gaussian weight should dominate, got {}",
            result.weights[1]
        );
    }

    #[test]
    fn test_gmm_two_well_separated_clusters() {
        // Two groups: 50 points around 20, 50 around 80
        let mut obs = Vec::new();
        for i in 0..50 {
            obs.push(20.0 + (i as f32 - 25.0) * 0.5);
        }
        for i in 0..50 {
            obs.push(80.0 + (i as f32 - 25.0) * 0.5);
        }

        let params = EmParams {
            max_iter: 200,
            tol: 1e-6,
            min_weight: 0.01,
        };

        let result = gaussian_mixture_em(&obs, &[20.0, 80.0], 15.0, 100.0, &params);

        assert_eq!(result.mus.len(), 2);
        // Check means are near their true values
        let mut sorted_mus = result.mus.clone();
        sorted_mus.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (sorted_mus[0] - 20.0).abs() < 5.0,
            "first mu should be near 20, got {}",
            sorted_mus[0]
        );
        assert!(
            (sorted_mus[1] - 80.0).abs() < 5.0,
            "second mu should be near 80, got {}",
            sorted_mus[1]
        );
    }

    #[test]
    fn test_fixed_em_basic() {
        // 2 components, 10 observations; flat row-major with stride 2.
        // Component 0 has high likelihood for first 5, component 1 for last 5
        let mut component_log_liks: Vec<f32> = Vec::new();
        for i in 0..10 {
            if i < 5 {
                component_log_liks.extend_from_slice(&[-1.0, -10.0]);
            } else {
                component_log_liks.extend_from_slice(&[-10.0, -1.0]);
            }
        }

        let params = EmParams {
            max_iter: 100,
            tol: 1e-6,
            min_weight: 0.01,
        };

        let result = fixed_em(&component_log_liks, 2, 1, &params);

        assert!(
            (result.weights[0] - 0.5).abs() < 0.1,
            "weight 0 should be ~0.5, got {}",
            result.weights[0]
        );
        assert!(
            (result.weights[1] - 0.5).abs() < 0.1,
            "weight 1 should be ~0.5, got {}",
            result.weights[1]
        );
    }
}
