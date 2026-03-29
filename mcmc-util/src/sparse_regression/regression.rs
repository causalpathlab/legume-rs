use nalgebra::{DMatrix, DVector, DVectorView};
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal};

use crate::engine::{elliptical_slice_step, McmcConfig, McmcModel};

use super::prior::ComponentPrior;
use super::{compute_pip, compute_posterior_mean_beta, SparseResult, SparseSample};

/// Additional configuration for regression sampler.
pub struct RegressionConfig {
    /// Whether to Gibbs-sample the residual variance.
    pub estimate_residual_var: bool,
    /// Initial (or fixed) residual variance.
    pub residual_var: f32,
    /// Whether to Gibbs-sample the effect size prior variance.
    pub estimate_effect_var: bool,
}

/// Linear regression sparse model for the MCMC engine: y = X * theta + eps.
///
/// ESS for inclusion weights, conjugate Gaussian for effect sizes,
/// optional Gibbs update of residual variance.
pub(crate) struct RegressionModel<'a, P: ComponentPrior> {
    x: &'a DMatrix<f32>,
    y: &'a DVector<f32>,
    prior: P,
    num_components: usize,
    reg: &'a RegressionConfig,
}

/// Internal Gibbs state for regression sampler.
pub(crate) struct RegressionState {
    inclusions: Vec<DVector<f32>>,
    alphas: Vec<Vec<f32>>,
    betas: Vec<f32>,
    sigma2_eps: f32,
    effect_var: f32,
    residual: DVector<f32>,
    xa_buf: DVector<f32>,
}

impl<'a, P> McmcModel for RegressionModel<'a, P>
where
    P: ComponentPrior<Inclusion = DVector<f32>, Effect = DVector<f32>, Theta = DVector<f32>>,
{
    type State = RegressionState;
    type Sample = SparseSample;
    type Result = SparseResult;

    fn init(&self, rng: &mut SmallRng) -> RegressionState {
        let p = self.x.ncols();
        let n = self.x.nrows();
        let inclusions: Vec<DVector<f32>> = (0..self.num_components)
            .map(|_| self.prior.draw_inclusion(p, rng))
            .collect();
        let alphas: Vec<Vec<f32>> = inclusions
            .iter()
            .map(|inc| self.prior.to_alpha(inc))
            .collect();
        RegressionState {
            inclusions,
            alphas,
            betas: vec![0.0f32; self.num_components],
            sigma2_eps: self.reg.residual_var,
            effect_var: self.prior.effect_var(),
            residual: self.y.clone(),
            xa_buf: DVector::from_element(n, 0.0f32),
        }
    }

    fn sweep(&self, state: &mut RegressionState, rng: &mut SmallRng) {
        let p = self.x.ncols();
        let n = self.x.nrows();

        // --- ESS update of inclusion params ---
        for l in 0..self.num_components {
            let beta_l = state.betas[l];
            update_residual(&mut state.residual, self.x, &state.alphas[l], beta_l);

            let r_l = state.residual.clone();
            let s2 = state.sigma2_eps;
            let lnpdf_inc = |inc: &DVector<f32>| -> f32 {
                let a = self.prior.to_alpha(inc);
                let alpha_view = DVectorView::from_slice(&a, a.len());
                let xa = self.x * alpha_view;
                let predicted = &xa * beta_l;
                let diff = &r_l - &predicted;
                -0.5 * diff.dot(&diff) / s2
            };

            let cur_ll = lnpdf_inc(&state.inclusions[l]);
            let prior_sample = self.prior.draw_inclusion(p, rng);
            let (new_inc, _) =
                elliptical_slice_step(&state.inclusions[l], &prior_sample, &lnpdf_inc, cur_ll, rng);

            state.inclusions[l] = new_inc;
            state.alphas[l] = self.prior.to_alpha(&state.inclusions[l]);
            update_residual(&mut state.residual, self.x, &state.alphas[l], -beta_l);
        }

        // --- Conjugate Gaussian update of beta ---
        let effect_var = state.effect_var;
        for l in 0..self.num_components {
            let old_beta = state.betas[l];

            let alpha_view = DVectorView::from_slice(&state.alphas[l], p);
            state.xa_buf.gemv(1.0, self.x, &alpha_view, 0.0);

            state.residual.axpy(old_beta, &state.xa_buf, 1.0);

            let ztz = state.xa_buf.dot(&state.xa_buf);
            let ztr = state.xa_buf.dot(&state.residual);

            let sigma2_post = 1.0 / (ztz / state.sigma2_eps + 1.0 / effect_var);
            let mu_post = sigma2_post * ztr / state.sigma2_eps;

            let normal = Normal::new(mu_post, sigma2_post.sqrt()).unwrap();
            let new_beta: f32 = normal.sample(rng);
            state.betas[l] = new_beta;

            state.residual.axpy(-new_beta, &state.xa_buf, 1.0);
        }

        // --- Conjugate InvGamma update of sigma2_eps ---
        if self.reg.estimate_residual_var {
            let rss = state.residual.dot(&state.residual);
            let a_post = 0.01 + n as f32 / 2.0;
            let b_post = 0.01 + rss / 2.0;
            let gamma = rand_distr::Gamma::new(a_post as f64, (1.0 / b_post) as f64).unwrap();
            state.sigma2_eps = 1.0 / gamma.sample(rng) as f32;
        }

        // --- Conjugate InvGamma update of effect_var ---
        // beta_l ~ N(0, effect_var), so effect_var | betas ~ InvGamma
        if self.reg.estimate_effect_var {
            let l = self.num_components;
            let ss: f32 = state.betas.iter().map(|&b| b * b).sum();
            let a_post = 0.01 + l as f32 / 2.0;
            let b_post = 0.01 + ss / 2.0;
            let gamma = rand_distr::Gamma::new(a_post as f64, (1.0 / b_post) as f64).unwrap();
            state.effect_var = 1.0 / gamma.sample(rng) as f32;
        }
    }

    fn collect(&self, state: &RegressionState) -> SparseSample {
        SparseSample {
            alphas: state.alphas.clone(),
            betas: state.betas.clone(),
            sigma2_eps: state.sigma2_eps,
            effect_var: state.effect_var,
        }
    }

    fn summarize(&self, samples: Vec<SparseSample>) -> SparseResult {
        let p = self.x.ncols();
        let t = samples.len() as f32;
        let pip = compute_pip(&samples, p);
        let posterior_mean_beta = compute_posterior_mean_beta(&samples, p);
        let sigma2_eps_mean = samples.iter().map(|s| s.sigma2_eps).sum::<f32>() / t;
        let effect_var_mean = samples.iter().map(|s| s.effect_var).sum::<f32>() / t;
        SparseResult {
            samples,
            pip,
            posterior_mean_beta,
            sigma2_eps_mean,
            effect_var_mean,
        }
    }
}

/// Convenience wrapper around [`RegressionModel`] + [`run_mcmc`].
pub fn mcmc_sparse_regression<P>(
    x: &DMatrix<f32>,
    y: &DVector<f32>,
    prior: &P,
    config: &McmcConfig,
    num_components: usize,
    reg: &RegressionConfig,
) -> SparseResult
where
    P: ComponentPrior<Inclusion = DVector<f32>, Effect = DVector<f32>, Theta = DVector<f32>>,
{
    let model = RegressionModel {
        x,
        y,
        prior: prior.clone(),
        num_components,
        reg,
    };
    crate::engine::run_mcmc(&model, config)
}

/// residual += X * alpha * beta (sign via beta: pass -beta to subtract)
fn update_residual(residual: &mut DVector<f32>, x: &DMatrix<f32>, alpha: &[f32], beta: f32) {
    let alpha_view = DVectorView::from_slice(alpha, alpha.len());
    let xa = x * alpha_view;
    residual.axpy(beta, &xa, 1.0);
}
