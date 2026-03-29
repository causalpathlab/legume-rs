//! MCMC-based sparse regression with pluggable component priors.
//!
//! The [`ComponentPrior`] trait defines how inclusion weights and effect sizes
//! are parameterized (e.g. softmax-normal for SuSiE, spike-and-slab).
//!
//! Two samplers:
//! - [`mcmc_sparse`]: Blackbox log-likelihood — ESS for both inclusion and effects.
//! - [`mcmc_sparse_regression`]: Linear regression `y = X*theta + eps` — ESS for
//!   inclusion, conjugate Gaussian for effects, optional residual variance estimation.

mod prior;
mod regression;
mod sampler;

pub use prior::{BernoulliNormalPrior, ComponentPrior, SoftmaxNormalPrior};
pub use regression::{mcmc_sparse_regression, RegressionConfig};
pub use sampler::mcmc_sparse;

/// One posterior sample.
pub struct SparseSample {
    /// Inclusion probabilities per component: L vectors of length p.
    pub alphas: Vec<Vec<f32>>,
    /// Scalar effect size per component.
    pub betas: Vec<f32>,
    /// Residual variance (NaN for blackbox sampler).
    pub sigma2_eps: f32,
    /// Effect size prior variance (NaN when not estimated).
    pub effect_var: f32,
}

/// Credible set for one component.
pub struct CredibleSet {
    pub component: usize,
    /// Variable indices, sorted by descending posterior-averaged alpha.
    pub indices: Vec<usize>,
    /// Achieved coverage (>= target).
    pub coverage: f32,
}

/// Result of MCMC sparse regression.
pub struct SparseResult {
    pub samples: Vec<SparseSample>,
    /// Posterior inclusion probabilities, length p.
    pub pip: Vec<f32>,
    /// Posterior mean of combined effect sum_l(alpha_l * beta_l), length p.
    pub posterior_mean_beta: Vec<f32>,
    /// Posterior mean of residual variance (NaN for blackbox).
    pub sigma2_eps_mean: f32,
    /// Posterior mean of effect size prior variance (NaN when not estimated).
    pub effect_var_mean: f32,
}

impl SparseResult {
    /// Compute credible sets at the given coverage level (e.g. 0.95).
    pub fn credible_sets(&self, coverage_target: f32) -> Vec<CredibleSet> {
        let Some(first) = self.samples.first() else {
            return vec![];
        };
        let num_comp = first.alphas.len();
        let p = first.alphas[0].len();
        let inv_t = 1.0 / self.samples.len() as f32;

        (0..num_comp)
            .map(|l| {
                let mut alpha_bar = vec![0.0f32; p];
                for sample in &self.samples {
                    for (ab, &a) in alpha_bar.iter_mut().zip(sample.alphas[l].iter()) {
                        *ab += a;
                    }
                }
                for ab in &mut alpha_bar {
                    *ab *= inv_t;
                }

                let mut order: Vec<usize> = (0..p).collect();
                order.sort_unstable_by(|&a, &b| alpha_bar[b].partial_cmp(&alpha_bar[a]).unwrap());

                let mut cum = 0.0f32;
                let mut indices = Vec::new();
                for &j in &order {
                    cum += alpha_bar[j];
                    indices.push(j);
                    if cum >= coverage_target {
                        break;
                    }
                }

                CredibleSet {
                    component: l,
                    indices,
                    coverage: cum,
                }
            })
            .collect()
    }
}

/// PIP_j = (1/T) * sum_t [1 - prod_l(1 - alpha_l^(t)[j])]
pub(crate) fn compute_pip(samples: &[SparseSample], p: usize) -> Vec<f32> {
    let t = samples.len() as f32;
    let mut pip = vec![0.0f32; p];
    for sample in samples {
        for j in 0..p {
            let mut log_excl = 0.0f32;
            for alpha_l in &sample.alphas {
                log_excl += (1.0 - alpha_l[j]).max(1e-15).ln();
            }
            pip[j] += 1.0 - log_excl.exp();
        }
    }
    for v in &mut pip {
        *v /= t;
    }
    pip
}

/// E[sum_l alpha_l * beta_l] averaged over posterior samples.
pub(crate) fn compute_posterior_mean_beta(samples: &[SparseSample], p: usize) -> Vec<f32> {
    let t = samples.len() as f32;
    let mut mean = vec![0.0f32; p];
    for sample in samples {
        for (alpha_l, &beta_l) in sample.alphas.iter().zip(sample.betas.iter()) {
            for j in 0..p {
                mean[j] += alpha_l[j] * beta_l;
            }
        }
    }
    for v in &mut mean {
        *v /= t;
    }
    mean
}

/// Std[sum_l alpha_l * beta_l] across posterior samples, length p.
pub fn compute_posterior_std_beta(samples: &[SparseSample], p: usize) -> Vec<f32> {
    let t = samples.len() as f32;
    let mut mean = vec![0.0f32; p];
    let mut mean_sq = vec![0.0f32; p];
    for sample in samples {
        for j in 0..p {
            let mut theta_j = 0.0f32;
            for (alpha_l, &beta_l) in sample.alphas.iter().zip(sample.betas.iter()) {
                theta_j += alpha_l[j] * beta_l;
            }
            mean[j] += theta_j;
            mean_sq[j] += theta_j * theta_j;
        }
    }
    mean.iter()
        .zip(mean_sq.iter())
        .map(|(&m, &m2)| {
            let mu = m / t;
            ((m2 / t) - mu * mu).max(0.0).sqrt()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::McmcConfig;
    use nalgebra::{DMatrix, DVector};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    /// Helper: generate X ~ N(0,1) of shape (n, p).
    fn random_x(n: usize, p: usize, seed: u64) -> DMatrix<f32> {
        let mut rng = SmallRng::seed_from_u64(seed);
        DMatrix::from_fn(n, p, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32
        })
    }

    /// Helper: generate noise ~ N(0, var).
    fn random_noise(n: usize, var: f32, seed: u64) -> DVector<f32> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let std = var.sqrt();
        DVector::from_fn(n, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32 * std
        })
    }

    // ---- Blackbox sampler tests ----

    #[test]
    fn test_blackbox_single_signal() {
        let n = 200;
        let p = 50;
        let causal = 17;
        let beta_true = 2.0f32;
        let noise_var = 1.0f32;

        let x = random_x(n, p, 42);
        let noise = random_noise(n, noise_var, 43);
        let y = x.column(causal) * beta_true + &noise;

        // Blackbox log-likelihood: -0.5 * ||y - X*theta||^2 / sigma2
        let lnpdf = |theta: &DVector<f32>| -> f32 {
            let pred = &x * theta;
            let diff = &y - &pred;
            -0.5 * diff.dot(&diff) / noise_var
        };

        let prior = SoftmaxNormalPrior::new(1.0, 10.0);
        let config = McmcConfig {
            n_samples: 2_000,
            warmup: 1_000,
            thin: 2,
            seed: 123,
        };

        let result = mcmc_sparse(&lnpdf, p, &prior, &config, 1);

        let max_pip_idx = result
            .pip
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert_eq!(max_pip_idx, causal, "max PIP should be at causal SNP");
        assert!(
            result.pip[causal] > 0.5,
            "PIP at causal should be > 0.5, got {}",
            result.pip[causal]
        );
    }

    // ---- Regression sampler tests ----

    #[test]
    fn test_regression_single_signal() {
        let n = 200;
        let p = 50;
        let causal = 17;
        let beta_true = 2.0f32;
        let noise_var = 1.0f32;

        let x = random_x(n, p, 42);
        let noise = random_noise(n, noise_var, 43);
        let y = x.column(causal) * beta_true + &noise;

        let prior = SoftmaxNormalPrior::new(1.0, 10.0);
        let config = McmcConfig {
            n_samples: 2_000,
            warmup: 1_000,
            thin: 2,
            seed: 123,
        };
        let reg = RegressionConfig {
            estimate_residual_var: false,
            residual_var: noise_var,
            estimate_effect_var: false,
        };

        let result = mcmc_sparse_regression(&x, &y, &prior, &config, 1, &reg);

        let max_pip_idx = result
            .pip
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert_eq!(max_pip_idx, causal, "max PIP should be at causal SNP");
        assert!(
            result.pip[causal] > 0.5,
            "PIP at causal should be > 0.5, got {}",
            result.pip[causal]
        );
        assert!(
            result.posterior_mean_beta[causal] > 0.0,
            "posterior mean beta at causal should be positive"
        );
    }

    #[test]
    fn test_regression_multiple_signals() {
        let n = 300;
        let p = 50;
        let causals = [5, 20, 40];
        let betas_true = [1.5f32, -2.0, 1.0];
        let noise_var = 1.0f32;

        let x = random_x(n, p, 99);
        let mut y = random_noise(n, noise_var, 100);
        for (&j, &b) in causals.iter().zip(betas_true.iter()) {
            y += x.column(j) * b;
        }

        let prior = SoftmaxNormalPrior::new(1.0, 10.0);
        let config = McmcConfig {
            n_samples: 3_000,
            warmup: 2_000,
            thin: 2,
            seed: 456,
        };
        let reg = RegressionConfig {
            estimate_residual_var: false,
            residual_var: noise_var,
            estimate_effect_var: false,
        };

        let result = mcmc_sparse_regression(&x, &y, &prior, &config, 5, &reg);

        // Top 5 PIPs should include all causal SNPs
        let mut pip_order: Vec<(usize, f32)> = result.pip.iter().cloned().enumerate().collect();
        pip_order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<usize> = pip_order.iter().take(5).map(|&(i, _)| i).collect();
        for &j in &causals {
            assert!(
                top5.contains(&j),
                "causal SNP {} should be in top 5 PIPs, top 5 are {:?}",
                j,
                top5
            );
        }
    }

    #[test]
    fn test_regression_residual_variance() {
        let n = 200;
        let p = 20;
        let true_var = 3.0f32;

        let x = random_x(n, p, 77);
        let y = random_noise(n, true_var, 78);

        let prior = SoftmaxNormalPrior::new(0.01, 0.01);
        let config = McmcConfig {
            n_samples: 2_000,
            warmup: 1_000,
            thin: 2,
            seed: 88,
        };
        let reg = RegressionConfig {
            estimate_residual_var: true,
            residual_var: 1.0, // start far from truth
            estimate_effect_var: false,
        };

        let result = mcmc_sparse_regression(&x, &y, &prior, &config, 1, &reg);

        assert!(
            (result.sigma2_eps_mean - true_var).abs() < 1.0,
            "estimated sigma2_eps should be near {}, got {}",
            true_var,
            result.sigma2_eps_mean
        );
    }

    #[test]
    fn test_credible_sets() {
        let n = 200;
        let p = 50;
        let causal = 10;
        let beta_true = 3.0f32;
        let noise_var = 1.0f32;

        let x = random_x(n, p, 55);
        let noise = random_noise(n, noise_var, 56);
        let y = x.column(causal) * beta_true + &noise;

        let prior = SoftmaxNormalPrior::new(1.0, 10.0);
        let config = McmcConfig {
            n_samples: 2_000,
            warmup: 1_000,
            thin: 2,
            seed: 66,
        };
        let reg = RegressionConfig {
            estimate_residual_var: false,
            residual_var: noise_var,
            estimate_effect_var: false,
        };

        let result = mcmc_sparse_regression(&x, &y, &prior, &config, 1, &reg);
        let cs = result.credible_sets(0.95);

        assert_eq!(cs.len(), 1);
        assert!(
            cs[0].indices.contains(&causal),
            "95% CS should contain causal SNP {}, got {:?}",
            causal,
            cs[0].indices
        );
        assert!(cs[0].coverage >= 0.95);
    }

    #[test]
    fn test_softmax_stability() {
        use super::prior::softmax;

        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let s = softmax(&v);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(s[2] > s[1] && s[1] > s[0]);

        // Large values — should not overflow
        let v = DVector::from_vec(vec![1000.0, 1001.0, 1002.0]);
        let s = softmax(&v);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Uniform
        let v = DVector::from_vec(vec![0.0; 4]);
        let s = softmax(&v);
        for &val in &s {
            assert!((val - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_stability() {
        use super::prior::sigmoid;

        let v = DVector::from_vec(vec![0.0, 1.0, -1.0, 10.0, -10.0]);
        let s = sigmoid(&v);
        assert!((s[0] - 0.5).abs() < 1e-6);
        assert!(s[1] > 0.5 && s[1] < 1.0);
        assert!(s[2] < 0.5 && s[2] > 0.0);
        assert!(s[3] > 0.99);
        assert!(s[4] < 0.01);

        // Large values — should not overflow
        let v = DVector::from_vec(vec![100.0, -100.0]);
        let s = sigmoid(&v);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!(s[1].abs() < 1e-6);
    }

    #[test]
    fn test_spike_slab_regression_single_signal() {
        let n = 200;
        let p = 50;
        let causal = 17;
        let beta_true = 2.0f32;
        let noise_var = 1.0f32;

        let x = random_x(n, p, 42);
        let noise = random_noise(n, noise_var, 43);
        let y = x.column(causal) * beta_true + &noise;

        let prior = BernoulliNormalPrior::new(1.0, 10.0);
        let config = McmcConfig {
            n_samples: 2_000,
            warmup: 1_000,
            thin: 2,
            seed: 123,
        };
        let reg = RegressionConfig {
            estimate_residual_var: false,
            residual_var: noise_var,
            estimate_effect_var: false,
        };

        let result = mcmc_sparse_regression(&x, &y, &prior, &config, 1, &reg);

        let max_pip_idx = result
            .pip
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert_eq!(max_pip_idx, causal, "max PIP should be at causal SNP");
        assert!(
            result.pip[causal] > 0.3,
            "PIP at causal should be > 0.3, got {}",
            result.pip[causal]
        );

        let std_beta = compute_posterior_std_beta(&result.samples, p);
        assert!(
            std_beta[causal] > 0.0,
            "posterior std at causal should be positive"
        );
    }
}
