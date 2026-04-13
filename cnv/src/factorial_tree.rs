//! Factorial tree model for CNV detection from pseudobulk residuals.
//!
//! Decomposes a `[B × S]` block-level signal (genes × samples) into F additive
//! factors, each with its own latent state sequence along the genome:
//!
//!   R[g, i] = Σ_f L[i,f] · μ_f[s_f(g)] + ε[g,i]
//!
//! - F factors, each with K states (e.g. del/neutral/gain)
//! - Factors combine additively — a sample can carry any mix of CNV events
//! - Cell-level evaluation reduces to linear regression onto factor profiles
//!
//! Inference via Gibbs sampling with tree-based state updates:
//! 1. Belief propagation on the coarsening tree per factor (conditioned on others)
//! 2. Conjugate Normal update for per-sample loadings
//! 3. ESS for emission means (neutral pinned at 0)
//! 4. Closed-form noise variance update

use log::info;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};

use mcmc_util::engine::{elliptical_slice_step, EssParam};

use crate::coarsening_tree::{tree_state_inference, CoarseningTree};

// ---------------------------------------------------------------------------
// Config & Result
// ---------------------------------------------------------------------------

/// Configuration for the factorial tree sampler.
#[derive(Debug, Clone)]
pub struct FactorialTreeConfig {
    /// Number of CNV factors (default 3).
    pub n_factors: usize,
    /// Number of CN states per factor (default 3: del/neutral/gain).
    pub n_states: usize,
    /// Include a null factor (constant across genome) that absorbs
    /// per-sample global offsets. Factor index 0 when enabled. (default true)
    pub null_factor: bool,
    /// Off-diagonal transition probability for tree parent→child (default 1e-6).
    pub transition_prob: f32,
    /// Total Gibbs iterations (default 500).
    pub n_iter: usize,
    /// Burn-in iterations (default 200).
    pub warmup: usize,
    /// Random seed.
    pub seed: u64,
    /// Prior variance on emission means (default 1.0).
    pub prior_var_mu: f32,
    /// Prior variance on factor loadings (default 1.0).
    pub prior_var_loading: f32,
}

impl Default for FactorialTreeConfig {
    fn default() -> Self {
        Self {
            n_factors: 3,
            n_states: 3,
            null_factor: true,
            transition_prob: 1e-6,
            n_iter: 500,
            warmup: 200,
            seed: 42,
            prior_var_mu: 1.0,
            prior_var_loading: 1.0,
        }
    }
}

/// Result of the factorial tree sampler.
#[derive(Debug, Clone)]
pub struct FactorialTreeResult {
    /// Per-factor posterior mean emission means: `[F]` vectors of length K.
    pub factor_emission_means: Vec<DVector<f32>>,
    /// Per-factor MAP state path at finest level: `[F]` vectors of length B.
    pub factor_viterbi_paths: Vec<Vec<usize>>,
    /// Per-factor averaged posteriors: `[F]` matrices of shape `[B × K]`.
    pub factor_posteriors: Vec<DMatrix<f32>>,
    /// Per-sample factor loadings (posterior mean): `[S × F_total]`
    /// (includes null factor at index 0 when enabled).
    pub loadings: DMatrix<f32>,
    /// Noise variance (posterior mean).
    pub noise_variance: f32,
    /// Log-likelihood trace (one per post-warmup iteration).
    pub log_likelihoods: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Emission parameter wrapper for ESS
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EmissionTheta(DVector<f32>);

impl EssParam for EmissionTheta {
    fn linear_combine(&self, a: f32, other: &Self, b: f32) -> Self {
        EmissionTheta(self.0.linear_combine(a, &other.0, b))
    }
}

impl EmissionTheta {
    fn new(n_states: usize) -> Self {
        EmissionTheta(DVector::zeros(n_states))
    }

    fn n_states(&self) -> usize {
        self.0.len()
    }

    /// Write emission means into a pre-allocated slice (avoids allocation).
    fn fill_means(&self, out: &mut [f32]) {
        let k = self.n_states();
        let neutral = k / 2;
        out[..k].copy_from_slice(self.0.as_slice());
        out[neutral] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tree-based factorial inference
// ---------------------------------------------------------------------------

/// Fit factorial model using tree-based state inference.
///
/// Uses multi-level coarsening tree for state inference instead of
/// chain HMM. O(nodes × K) per factor per iteration.
///
/// The null factor (index 0 when enabled) absorbs per-sample global offsets.
/// Initialized as the per-sample mean signal.
pub fn fit_tree_factorial(
    tree: &CoarseningTree,
    config: &FactorialTreeConfig,
) -> FactorialTreeResult {
    let n_blocks = tree.n_finest_blocks();
    let n_samples = tree.n_samples();
    let has_null = config.null_factor;
    let n_cnv_factors = config.n_factors;
    let n_total_factors = n_cnv_factors + if has_null { 1 } else { 0 };
    let cnv_offset = if has_null { 1 } else { 0 };
    let k = config.n_states;
    let neutral = k / 2;

    let finest_signal = &tree.block_signals[0];

    info!(
        "Factorial tree: {} blocks × {} samples, {} CNV factors{}, {} states, {} levels, {} iter",
        n_blocks,
        n_samples,
        n_cnv_factors,
        if has_null { " + null" } else { "" },
        k,
        tree.n_levels(),
        config.n_iter,
    );

    let mut rng = SmallRng::seed_from_u64(config.seed);

    let sample_signals: Vec<Vec<f32>> = (0..n_samples)
        .map(|s| (0..n_blocks).map(|b| finest_signal[(b, s)]).collect())
        .collect();

    // --- Initialize ---
    let mut thetas: Vec<EmissionTheta> = (0..n_cnv_factors)
        .map(|f| {
            let mut theta = EmissionTheta::new(k);
            if k >= 3 {
                let scale = 0.3 + 0.05 * f as f32;
                theta.0[0] = -scale;
                theta.0[k - 1] = scale;
            }
            theta
        })
        .collect();

    let mut loadings = DMatrix::<f32>::zeros(n_samples, n_total_factors);
    if has_null {
        for s in 0..n_samples {
            loadings[(s, 0)] = sample_signals[s].iter().sum::<f32>() / n_blocks as f32;
        }
    }
    if n_cnv_factors > 0 {
        for s in 0..n_samples {
            loadings[(s, cnv_offset)] = 1.0;
        }
    }

    let mut sigma_sq = {
        let mut ss = 0.0f32;
        for s in 0..n_samples {
            let null_l = if has_null { loadings[(s, 0)] } else { 0.0 };
            for &val in &sample_signals[s] {
                let r = val - null_l;
                ss += r * r;
            }
        }
        (ss / (n_blocks * n_samples) as f32).max(0.01)
    };

    let mut factor_states: Vec<Vec<usize>> = vec![vec![neutral; n_blocks]; n_cnv_factors];
    let mut factor_posteriors: Vec<DMatrix<f32>> = (0..n_cnv_factors)
        .map(|_| {
            let mut p = DMatrix::zeros(n_blocks, k);
            for b in 0..n_blocks {
                p[(b, neutral)] = 1.0;
            }
            p
        })
        .collect();

    // --- Accumulators ---
    let n_collect = config.n_iter.saturating_sub(config.warmup);
    let mut means_acc: Vec<DVector<f32>> = (0..n_cnv_factors).map(|_| DVector::zeros(k)).collect();
    let mut loadings_acc = DMatrix::<f32>::zeros(n_samples, n_total_factors);
    let mut sigma_sq_acc = 0.0f32;
    let mut posteriors_acc: Vec<DMatrix<f32>> = (0..n_cnv_factors)
        .map(|_| DMatrix::zeros(n_blocks, k))
        .collect();
    let mut ll_trace: Vec<f32> = Vec::with_capacity(n_collect);

    let prior_std_mu = config.prior_var_mu.sqrt();
    let inv_prior_var_loading = 1.0 / config.prior_var_loading;

    let mut factor_profiles: Vec<Vec<f32>> = vec![vec![0.0; n_blocks]; n_cnv_factors];
    // Cached means per factor (avoids DVector allocation in hot loop)
    let mut cached_means: Vec<Vec<f32>> = vec![vec![0.0; k]; n_cnv_factors];
    // Reusable partial residual buffer (hoisted out of Gibbs loop)
    let mut partial_resid = DMatrix::<f32>::zeros(n_blocks, n_samples);

    // --- Gibbs loop ---
    for iter in 0..config.n_iter {
        // Precompute factor means and profiles (no allocation)
        for ci in 0..n_cnv_factors {
            thetas[ci].fill_means(&mut cached_means[ci]);
            for g in 0..n_blocks {
                factor_profiles[ci][g] = cached_means[ci][factor_states[ci][g]];
            }
        }

        // === Step 1: Tree state inference per CNV factor ===
        for ci in 0..n_cnv_factors {
            let fi = ci + cnv_offset;

            // Compute partial residual: signal minus null and other factors
            for s in 0..n_samples {
                for g in 0..n_blocks {
                    let mut r = sample_signals[s][g];
                    if has_null {
                        r -= loadings[(s, 0)];
                    }
                    for ci2 in 0..n_cnv_factors {
                        if ci2 != ci {
                            r -= loadings[(s, ci2 + cnv_offset)] * factor_profiles[ci2][g];
                        }
                    }
                    partial_resid[(g, s)] = r;
                }
            }

            let loadings_f: Vec<f32> = (0..n_samples).map(|s| loadings[(s, fi)]).collect();

            let tree_result = tree_state_inference(
                tree,
                &partial_resid,
                &loadings_f,
                &cached_means[ci],
                sigma_sq,
                config.transition_prob,
            );

            factor_posteriors[ci] = tree_result.posteriors;
            factor_states[ci] = tree_result.map_states;

            // Update this factor's profile immediately after state change
            for g in 0..n_blocks {
                factor_profiles[ci][g] = cached_means[ci][factor_states[ci][g]];
            }
        }

        // === Step 2: Update loadings (conjugate Normal) ===
        for fi in 0..n_total_factors {
            // Precompute den (sum of v_fg^2) — same for all samples
            let den = if has_null && fi == 0 {
                n_blocks as f32 // v_fg = 1.0 for null factor
            } else {
                let ci = fi - cnv_offset;
                factor_profiles[ci].iter().map(|&v| v * v).sum::<f32>()
            };
            let precision = den / sigma_sq + inv_prior_var_loading;

            for s in 0..n_samples {
                let mut num = 0.0f32;
                for g in 0..n_blocks {
                    let v_fg = if has_null && fi == 0 {
                        1.0
                    } else {
                        factor_profiles[fi - cnv_offset][g]
                    };
                    // Partial residual (signal minus other factors)
                    let mut r = sample_signals[s][g];
                    if has_null && fi != 0 {
                        r -= loadings[(s, 0)];
                    }
                    for (ci2, profile) in factor_profiles.iter().enumerate() {
                        let fi2 = ci2 + cnv_offset;
                        if fi2 != fi {
                            r -= loadings[(s, fi2)] * profile[g];
                        }
                    }
                    num += v_fg * r;
                }
                let post_mean = (num / sigma_sq) / precision;
                let post_sd = (1.0 / precision).sqrt();
                let z: f64 = StandardNormal.sample(&mut rng);
                loadings[(s, fi)] = post_mean + post_sd * z as f32;
            }
        }

        // === Step 3: Update emission means via ESS ===
        let log_norm = -0.5 * (sigma_sq.ln() + std::f32::consts::TAU.ln());
        let inv_s2 = 1.0 / sigma_sq;

        for ci in 0..n_cnv_factors {
            let fi = ci + cnv_offset;

            // Precompute partial residual [B × S] for this factor (reuse buffer)
            for s in 0..n_samples {
                for g in 0..n_blocks {
                    let mut r = sample_signals[s][g];
                    if has_null {
                        r -= loadings[(s, 0)];
                    }
                    for ci2 in 0..n_cnv_factors {
                        if ci2 != ci {
                            r -= loadings[(s, ci2 + cnv_offset)] * factor_profiles[ci2][g];
                        }
                    }
                    partial_resid[(g, s)] = r;
                }
            }

            let lnpdf = |theta: &EmissionTheta| -> f32 {
                let mut means_buf = [0.0f32; 16]; // stack buffer, K <= 16
                theta.fill_means(&mut means_buf[..k]);
                let mut ll = 0.0f32;
                for s in 0..n_samples {
                    let l_sf = loadings[(s, fi)];
                    for g in 0..n_blocks {
                        // Partial residual already computed
                        let r = partial_resid[(g, s)];
                        for kk in 0..k {
                            let gamma = factor_posteriors[ci][(g, kk)];
                            if gamma < 1e-10 {
                                continue;
                            }
                            let diff = r - l_sf * means_buf[kk];
                            ll += gamma * (log_norm - 0.5 * diff * diff * inv_s2);
                        }
                    }
                }
                ll
            };

            let cur_ll = lnpdf(&thetas[ci]);
            let prior_sample = {
                let mut v = DVector::<f32>::zeros(k);
                for i in 0..k {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    v[i] = z as f32 * prior_std_mu;
                }
                EmissionTheta(v)
            };
            let (new_theta, _) =
                elliptical_slice_step(&thetas[ci], &prior_sample, &lnpdf, cur_ll, &mut rng);
            thetas[ci] = new_theta;
        }

        // === Step 4: Update noise variance ===
        {
            let mut ss = 0.0f32;
            for s in 0..n_samples {
                for g in 0..n_blocks {
                    let mut predicted = 0.0f32;
                    if has_null {
                        predicted += loadings[(s, 0)];
                    }
                    for ci in 0..n_cnv_factors {
                        predicted += loadings[(s, ci + cnv_offset)] * factor_profiles[ci][g];
                    }
                    let r = sample_signals[s][g] - predicted;
                    ss += r * r;
                }
            }
            sigma_sq = (ss / (n_blocks * n_samples) as f32).max(1e-6);
        }

        let total_ll = -((n_blocks * n_samples) as f32) * 0.5 * (sigma_sq.ln() + 1.0);

        if iter >= config.warmup {
            for ci in 0..n_cnv_factors {
                let mut acc = vec![0.0f32; k];
                thetas[ci].fill_means(&mut acc);
                for kk in 0..k {
                    means_acc[ci][kk] += acc[kk];
                }
                posteriors_acc[ci] += &factor_posteriors[ci];
            }
            loadings_acc += &loadings;
            sigma_sq_acc += sigma_sq;
            ll_trace.push(total_ll);
        }

        if iter % 50 == 0 || iter == config.n_iter - 1 {
            let means_str: Vec<String> = (0..n_cnv_factors)
                .map(|ci| format!("{:?}", &cached_means[ci]))
                .collect();
            info!(
                "Factorial tree iter {}/{}: σ²={:.4}, means=[{}]",
                iter,
                config.n_iter,
                sigma_sq,
                means_str.join("; "),
            );
        }
    }

    // --- Posterior means ---
    let n_coll = n_collect.max(1) as f32;
    let factor_emission_means: Vec<DVector<f32>> = means_acc
        .into_iter()
        .map(|v| v.scale(1.0 / n_coll))
        .collect();
    loadings_acc.scale_mut(1.0 / n_coll);
    let noise_variance = sigma_sq_acc / n_coll;
    for acc in posteriors_acc.iter_mut() {
        acc.scale_mut(1.0 / n_coll);
    }

    let factor_viterbi_paths: Vec<Vec<usize>> = (0..n_cnv_factors)
        .map(|ci| {
            (0..n_blocks)
                .map(|b| {
                    (0..k)
                        .max_by(|&a, &b_| {
                            posteriors_acc[ci][(b, a)]
                                .partial_cmp(&posteriors_acc[ci][(b, b_)])
                                .unwrap()
                        })
                        .unwrap()
                })
                .collect()
        })
        .collect();

    info!(
        "Factorial tree done. {} factors, σ²={:.4}",
        n_cnv_factors, noise_variance,
    );

    FactorialTreeResult {
        factor_emission_means,
        factor_viterbi_paths,
        factor_posteriors: posteriors_acc,
        loadings: loadings_acc,
        noise_variance,
        log_likelihoods: ll_trace,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coarsening_tree::CoarseningTree;

    #[test]
    fn test_tree_all_neutral() {
        let n_genes = 30;
        let n_samples = 5;
        let signal = DMatrix::<f32>::zeros(n_genes, n_samples);
        let chr_bounds: Vec<(Box<str>, usize, usize)> = vec![("chr1".into(), 0, n_genes)];
        let tree = CoarseningTree::build(&signal, &chr_bounds, &[0.5]);

        let config = FactorialTreeConfig {
            n_factors: 2,
            n_states: 3,
            null_factor: false,
            n_iter: 100,
            warmup: 50,
            seed: 42,
            ..Default::default()
        };

        let result = fit_tree_factorial(&tree, &config);

        for f in 0..2 {
            for &state in &result.factor_viterbi_paths[f] {
                assert_eq!(state, 1, "expected neutral for zero signal");
            }
        }
    }

    #[test]
    fn test_tree_null_factor_absorbs_offset() {
        let n_genes = 40;
        let n_samples = 4;
        let mut signal = DMatrix::<f32>::zeros(n_genes, n_samples);

        let offsets = [0.3, -0.2, 0.5, -0.1];
        for s in 0..n_samples {
            for g in 0..n_genes {
                signal[(g, s)] = offsets[s];
            }
        }
        // Add CNV gain in genes 20-30
        for g in 20..30 {
            for s in 0..n_samples {
                signal[(g, s)] += 0.4;
            }
        }

        let chr_bounds: Vec<(Box<str>, usize, usize)> = vec![("chr1".into(), 0, n_genes)];
        let tree = CoarseningTree::build(&signal, &chr_bounds, &[0.5]);

        let config = FactorialTreeConfig {
            n_factors: 1,
            n_states: 3,
            null_factor: true,
            n_iter: 300,
            warmup: 150,
            seed: 42,
            ..Default::default()
        };

        let result = fit_tree_factorial(&tree, &config);

        // Null factor loadings should approximate offsets
        for s in 0..n_samples {
            let null_loading = result.loadings[(s, 0)];
            assert!(
                (null_loading - offsets[s]).abs() < 0.3,
                "null loading for sample {} should be ~{}, got {}",
                s,
                offsets[s],
                null_loading,
            );
        }
    }
}
