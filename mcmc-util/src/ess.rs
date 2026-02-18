use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;

use crate::chain::McmcChain;
use crate::traits::EssParam;

/// One ESS transition. Returns `(new_params, new_lnpdf)`.
///
/// - `current`: current parameter value f
/// - `prior_sample`: a draw ν from the prior (caller handles Cholesky, etc.)
/// - `lnpdf`: log-likelihood function (just likelihood, not prior)
/// - `cur_lnpdf`: cached log-likelihood at `current`
/// - `rng`: random number generator
pub fn elliptical_slice_step<P: EssParam>(
    current: &P,
    prior_sample: &P,
    lnpdf: &impl Fn(&P) -> f32,
    cur_lnpdf: f32,
    rng: &mut impl Rng,
) -> (P, f32) {
    // 1. Choose ellipse and log-likelihood threshold
    let u: f32 = rng.random();
    let hh = u.ln() + cur_lnpdf;

    // 2. Draw initial proposal angle and define bracket
    let phi: f32 = rng.random_range(0.0..2.0 * PI);
    let mut phi_min = phi - 2.0 * PI;
    let mut phi_max = phi;

    // 3. Slice sampling loop
    let mut angle = phi;
    loop {
        let proposal = current.linear_combine(angle.cos(), prior_sample, angle.sin());
        let new_lnpdf = lnpdf(&proposal);

        if new_lnpdf > hh {
            return (proposal, new_lnpdf);
        }

        // Shrink the bracket
        if angle < 0.0 {
            phi_min = angle;
        } else {
            phi_max = angle;
        }
        angle = rng.random_range(phi_min..phi_max);
    }
}

/// ESS chain runner configuration.
pub struct EssSampler {
    pub n_samples: usize,
    pub warmup: usize,
    pub thin: usize,
    pub seed: u64,
}

impl EssSampler {
    pub fn new(n_samples: usize, warmup: usize) -> Self {
        Self {
            n_samples,
            warmup,
            thin: 1,
            seed: 42,
        }
    }

    /// Run a single ESS chain.
    ///
    /// - `lnpdf`: log-likelihood function
    /// - `prior_draw`: generates a sample from the prior N(0, Σ)
    /// - `init`: initial parameter value
    pub fn run<P: EssParam>(
        &self,
        lnpdf: &impl Fn(&P) -> f32,
        prior_draw: &impl Fn(&mut SmallRng) -> P,
        init: &P,
    ) -> McmcChain<P> {
        let total = self.warmup + self.n_samples * self.thin;
        let mut rng = SmallRng::seed_from_u64(self.seed);

        let mut current = init.clone();
        let mut cur_lnpdf = lnpdf(&current);

        let mut samples = Vec::with_capacity(self.n_samples);
        let mut log_likelihoods = Vec::with_capacity(self.n_samples);

        for i in 0..total {
            let nu = prior_draw(&mut rng);
            let (new, new_ll) = elliptical_slice_step(&current, &nu, lnpdf, cur_lnpdf, &mut rng);
            current = new;
            cur_lnpdf = new_ll;

            if i >= self.warmup && (i - self.warmup).is_multiple_of(self.thin) {
                samples.push(current.clone());
                log_likelihoods.push(cur_lnpdf);
            }
        }

        McmcChain {
            samples,
            log_likelihoods,
        }
    }

    /// Run multiple independent chains in parallel via rayon.
    /// Each chain gets `seed + chain_idx` for reproducibility.
    pub fn run_parallel<P: EssParam + Send + Sync>(
        &self,
        n_chains: usize,
        lnpdf: &(impl Fn(&P) -> f32 + Sync),
        prior_draw: &(impl Fn(&mut SmallRng) -> P + Sync),
        init: &P,
    ) -> Vec<McmcChain<P>> {
        (0..n_chains)
            .into_par_iter()
            .map(|i| {
                let sampler = EssSampler {
                    n_samples: self.n_samples,
                    warmup: self.warmup,
                    thin: self.thin,
                    seed: self.seed.wrapping_add(i as u64),
                };
                sampler.run(lnpdf, prior_draw, init)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use rand_distr::{Distribution, StandardNormal};

    /// Test 1: Gaussian-Gaussian conjugate (1D)
    /// Prior: N(0,1), Likelihood: N(y_obs | f, σ²)
    /// Posterior: N(μ_post, σ²_post)
    #[test]
    fn test_conjugate_gaussian_1d() {
        let y_obs = 3.0f32;
        let sigma_sq = 2.0f32; // likelihood variance

        // Analytic posterior
        let sigma_sq_post = 1.0 / (1.0 + 1.0 / sigma_sq);
        let mu_post = sigma_sq_post * y_obs / sigma_sq;

        let lnpdf = move |f: &DVector<f32>| -> f32 {
            let diff = f[0] - y_obs;
            -0.5 * diff * diff / sigma_sq
        };

        let prior_draw = |rng: &mut SmallRng| -> DVector<f32> {
            DVector::from_fn(1, |_, _| {
                let v: f64 = StandardNormal.sample(rng);
                v as f32
            })
        };

        let init = DVector::from_element(1, 0.0f32);

        let sampler = EssSampler {
            n_samples: 10_000,
            warmup: 2_000,
            thin: 1,
            seed: 123,
        };

        let chain = sampler.run(&lnpdf, &prior_draw, &init);
        let mean = chain.posterior_mean();
        let var = chain.posterior_variance();

        assert!(
            (mean[0] - mu_post).abs() < 0.1,
            "mean: {}, expected: {}",
            mean[0],
            mu_post
        );
        assert!(
            (var[0] - sigma_sq_post).abs() < 0.1,
            "var: {}, expected: {}",
            var[0],
            sigma_sq_post
        );
    }

    /// Test 2: Multivariate Gaussian conjugate (D=5)
    /// Prior: N(0, I_5), Likelihood: N(y_obs | f, σ²·I_5)
    #[test]
    fn test_conjugate_gaussian_5d() {
        let d = 5;
        let sigma_sq = 2.0f32;
        let y_obs = DVector::from_vec(vec![1.0, -2.0, 0.5, 3.0, -1.5]);

        let sigma_sq_post = 1.0 / (1.0 + 1.0 / sigma_sq);

        let lnpdf = {
            let y = y_obs.clone();
            move |f: &DVector<f32>| -> f32 {
                let diff = f - &y;
                -0.5 * diff.dot(&diff) / sigma_sq
            }
        };

        let prior_draw = move |rng: &mut SmallRng| -> DVector<f32> {
            DVector::from_fn(d, |_, _| {
                let v: f64 = StandardNormal.sample(rng);
                v as f32
            })
        };

        let init = DVector::from_element(d, 0.0f32);

        let sampler = EssSampler {
            n_samples: 10_000,
            warmup: 2_000,
            thin: 1,
            seed: 456,
        };

        let chain = sampler.run(&lnpdf, &prior_draw, &init);
        let mean = chain.posterior_mean();

        for j in 0..d {
            let analytic = sigma_sq_post * y_obs[j] / sigma_sq;
            assert!(
                (mean[j] - analytic).abs() < 0.15,
                "dim {}: mean={}, expected={}",
                j,
                mean[j],
                analytic
            );
        }
    }

    /// Test 3: Correlated prior (D=3) with Cholesky
    /// Prior: N(0, Σ), Likelihood: N(y_obs | f, σ²·I)
    /// Posterior: N(Σ_post · y/σ², Σ_post) where Σ_post = (Σ⁻¹ + I/σ²)⁻¹
    #[test]
    fn test_correlated_prior_3d() {
        use nalgebra::DMatrix;

        let d = 3;
        let sigma_sq = 1.5f32;
        let y_obs = DVector::from_vec(vec![2.0, -1.0, 1.5]);

        // Prior covariance with off-diagonal correlations
        #[rustfmt::skip]
        let sigma_prior = DMatrix::from_row_slice(d, d, &[
            1.0, 0.5, 0.2,
            0.5, 1.0, 0.3,
            0.2, 0.3, 1.0,
        ]);

        // Cholesky decomposition of prior covariance
        let chol = sigma_prior.clone().cholesky().unwrap();
        let chol_l = chol.l();

        // Analytic posterior: Σ_post = (Σ⁻¹ + I/σ²)⁻¹
        let sigma_prior_inv = sigma_prior.clone().try_inverse().unwrap();
        let eye = DMatrix::identity(d, d);
        let precision_post = &sigma_prior_inv + &eye / sigma_sq;
        let sigma_post = precision_post.try_inverse().unwrap();
        let mu_post = &sigma_post * &y_obs / sigma_sq;

        let lnpdf = {
            let y = y_obs.clone();
            move |f: &DVector<f32>| -> f32 {
                let diff = f - &y;
                -0.5 * diff.dot(&diff) / sigma_sq
            }
        };

        let prior_draw = {
            let l = chol_l.clone();
            move |rng: &mut SmallRng| -> DVector<f32> {
                let z = DVector::from_fn(d, |_, _| {
                    let v: f64 = StandardNormal.sample(rng);
                    v as f32
                });
                &l * z
            }
        };

        let init = DVector::from_element(d, 0.0f32);

        let sampler = EssSampler {
            n_samples: 20_000,
            warmup: 5_000,
            thin: 1,
            seed: 789,
        };

        let chain = sampler.run(&lnpdf, &prior_draw, &init);
        let mean = chain.posterior_mean();

        for j in 0..d {
            assert!(
                (mean[j] - mu_post[j]).abs() < 0.15,
                "dim {}: mean={}, expected={}",
                j,
                mean[j],
                mu_post[j]
            );
        }
    }

    /// Test 4: Non-conjugate logistic regression
    /// Prior: N(0, I), Likelihood: logistic
    /// Check posterior mean is in the right region (near MAP).
    #[test]
    fn test_logistic_regression() {
        let d = 2;

        // Simple dataset: 4 points
        let x = vec![
            DVector::from_vec(vec![1.0, 0.5]),
            DVector::from_vec(vec![-1.0, -0.3]),
            DVector::from_vec(vec![0.8, -0.7]),
            DVector::from_vec(vec![-0.5, 1.0]),
        ];
        let y = vec![1.0f32, 0.0, 1.0, 0.0]; // labels

        let lnpdf = {
            let x = x.clone();
            let y = y.clone();
            move |f: &DVector<f32>| -> f32 {
                let mut ll = 0.0f32;
                for (xi, &yi) in x.iter().zip(y.iter()) {
                    let eta = f.dot(xi);
                    // log p(y|eta) = y*eta - log(1+exp(eta))
                    let log1pexp = if eta > 20.0 {
                        eta
                    } else if eta < -20.0 {
                        0.0
                    } else {
                        (1.0 + eta.exp()).ln()
                    };
                    ll += yi * eta - log1pexp;
                }
                ll
            }
        };

        let prior_draw = move |rng: &mut SmallRng| -> DVector<f32> {
            DVector::from_fn(d, |_, _| {
                let v: f64 = StandardNormal.sample(rng);
                v as f32
            })
        };

        let init = DVector::from_element(d, 0.0f32);

        let sampler = EssSampler {
            n_samples: 10_000,
            warmup: 2_000,
            thin: 1,
            seed: 101,
        };

        let chain = sampler.run(&lnpdf, &prior_draw, &init);
        let mean = chain.posterior_mean();

        // With this data and N(0,I) prior, the posterior mean should be
        // in the positive region for dim 0 (positive examples have positive x[0]).
        // Just verify we get a reasonable, non-degenerate result.
        assert!(
            mean[0] > 0.0,
            "expected positive posterior mean for dim 0, got {}",
            mean[0]
        );
        assert!(
            mean[0].abs() < 5.0 && mean[1].abs() < 5.0,
            "posterior mean should be moderate: {:?}",
            mean
        );
    }

    /// Test 5: Parallel chains give consistent results
    #[test]
    fn test_parallel_chains() {
        let y_obs = 2.0f32;
        let sigma_sq = 1.0f32;
        let sigma_sq_post = 1.0 / (1.0 + 1.0 / sigma_sq);
        let mu_post = sigma_sq_post * y_obs / sigma_sq;

        let lnpdf = move |f: &DVector<f32>| -> f32 {
            let diff = f[0] - y_obs;
            -0.5 * diff * diff / sigma_sq
        };

        let prior_draw = |rng: &mut SmallRng| -> DVector<f32> {
            DVector::from_fn(1, |_, _| {
                let v: f64 = StandardNormal.sample(rng);
                v as f32
            })
        };

        let init = DVector::from_element(1, 0.0f32);

        let sampler = EssSampler {
            n_samples: 5_000,
            warmup: 1_000,
            thin: 1,
            seed: 222,
        };

        let chains = sampler.run_parallel(4, &lnpdf, &prior_draw, &init);
        assert_eq!(chains.len(), 4);

        // Each chain's posterior mean should be close to analytic
        for (i, chain) in chains.iter().enumerate() {
            let mean = chain.posterior_mean();
            assert!(
                (mean[0] - mu_post).abs() < 0.2,
                "chain {}: mean={}, expected={}",
                i,
                mean[0],
                mu_post
            );
        }
    }

    /// Test 6: Candle Tensor — Gaussian conjugate (D=5)
    /// Same setup as test 2, but using Tensor on CPU.
    #[test]
    fn test_tensor_conjugate_gaussian_5d() {
        use candle_core::{DType, Device, Tensor};

        let device = Device::Cpu;
        let d = 5;
        let sigma_sq = 2.0f32;
        let y_vals = vec![1.0f32, -2.0, 0.5, 3.0, -1.5];

        let sigma_sq_post = 1.0 / (1.0 + 1.0 / sigma_sq);

        let y_tensor = Tensor::from_vec(y_vals.clone(), &[d], &device).unwrap();

        let lnpdf = {
            let y = y_tensor.clone();
            move |f: &Tensor| -> f32 {
                let diff = (f - &y).unwrap();
                let sq = (&diff * &diff).unwrap().sum_all().unwrap();
                let val = (sq.to_scalar::<f32>().unwrap()) * (-0.5 / sigma_sq);
                val
            }
        };

        let prior_draw = {
            let dev = device.clone();
            move |_rng: &mut SmallRng| -> Tensor { Tensor::randn(0f32, 1f32, &[d], &dev).unwrap() }
        };

        let init = Tensor::zeros(&[d], DType::F32, &device).unwrap();

        let sampler = EssSampler {
            n_samples: 10_000,
            warmup: 2_000,
            thin: 1,
            seed: 456,
        };

        let chain = sampler.run(&lnpdf, &prior_draw, &init);

        // Manual posterior mean (no EssParamSummary for Tensor)
        let n = chain.n_samples();
        let stacked = Tensor::stack(&chain.samples, 0).unwrap();
        let mean_tensor = (stacked.sum(0).unwrap() / n as f64).unwrap();
        let mean: Vec<f32> = mean_tensor.to_vec1().unwrap();

        for j in 0..d {
            let analytic = sigma_sq_post * y_vals[j] / sigma_sq;
            assert!(
                (mean[j] - analytic).abs() < 0.15,
                "dim {}: mean={}, expected={}",
                j,
                mean[j],
                analytic
            );
        }
    }

    /// Test 7: Composite parameters via Vec<DVector<f32>>
    /// Two independent blocks: β (D=3) and γ (D=2) with separate likelihoods.
    /// Prior: N(0, I) on each block independently.
    /// Composite log-likelihood = llik_1(β) + llik_2(γ).
    #[test]
    fn test_composite_vec_params() {
        let y1 = DVector::from_vec(vec![1.0f32, -0.5, 2.0]); // observed for block 1
        let y2 = DVector::from_vec(vec![-1.0f32, 3.0]); // observed for block 2
        let sigma_sq_1 = 1.0f32;
        let sigma_sq_2 = 2.0f32;

        // Analytic posteriors (independent conjugate Gaussians)
        let sp1 = 1.0 / (1.0 + 1.0 / sigma_sq_1);
        let sp2 = 1.0 / (1.0 + 1.0 / sigma_sq_2);

        let lnpdf = {
            let y1 = y1.clone();
            let y2 = y2.clone();
            move |params: &Vec<DVector<f32>>| -> f32 {
                let diff1 = &params[0] - &y1;
                let diff2 = &params[1] - &y2;
                -0.5 * diff1.dot(&diff1) / sigma_sq_1 + -0.5 * diff2.dot(&diff2) / sigma_sq_2
            }
        };

        let prior_draw = |rng: &mut SmallRng| -> Vec<DVector<f32>> {
            vec![
                DVector::from_fn(3, |_, _| {
                    let v: f64 = StandardNormal.sample(rng);
                    v as f32
                }),
                DVector::from_fn(2, |_, _| {
                    let v: f64 = StandardNormal.sample(rng);
                    v as f32
                }),
            ]
        };

        let init = vec![
            DVector::from_element(3, 0.0f32),
            DVector::from_element(2, 0.0f32),
        ];

        let sampler = EssSampler {
            n_samples: 10_000,
            warmup: 2_000,
            thin: 1,
            seed: 333,
        };

        let chain = sampler.run(&lnpdf, &prior_draw, &init);
        let n = chain.n_samples() as f32;

        // Compute mean for each block manually
        let mut mean1 = DVector::from_element(3, 0.0f32);
        let mut mean2 = DVector::from_element(2, 0.0f32);
        for sample in &chain.samples {
            mean1 += &sample[0];
            mean2 += &sample[1];
        }
        mean1 /= n;
        mean2 /= n;

        for j in 0..3 {
            let analytic = sp1 * y1[j] / sigma_sq_1;
            assert!(
                (mean1[j] - analytic).abs() < 0.15,
                "block1 dim {}: mean={}, expected={}",
                j,
                mean1[j],
                analytic
            );
        }
        for j in 0..2 {
            let analytic = sp2 * y2[j] / sigma_sq_2;
            assert!(
                (mean2[j] - analytic).abs() < 0.15,
                "block2 dim {}: mean={}, expected={}",
                j,
                mean2[j],
                analytic
            );
        }
    }
}
