use rand::rngs::SmallRng;

use crate::engine::traits::EssParamSummary;
use crate::engine::{elliptical_slice_step, McmcConfig, McmcModel};

use super::prior::ComponentPrior;
use super::{compute_pip, compute_posterior_mean_beta, SparseResult, SparseSample};

/// Blackbox sparse regression model for the MCMC engine.
///
/// Uses ESS for both inclusion weights and effect sizes. The log-likelihood
/// is a blackbox function of the combined effect theta.
pub(crate) struct SparseModel<P: ComponentPrior, F: Fn(&P::Theta) -> f32> {
    lnpdf: F,
    prior: P,
    p: usize,
    num_components: usize,
}

/// Internal Gibbs state for the blackbox sampler.
pub(crate) struct SparseState<P: ComponentPrior> {
    pub inclusions: Vec<P::Inclusion>,
    pub alphas: Vec<Vec<f32>>,
    pub effects: Vec<P::Effect>,
    pub theta: P::Theta,
}

impl<P: ComponentPrior, F: Fn(&P::Theta) -> f32> McmcModel for SparseModel<P, F> {
    type State = SparseState<P>;
    type Sample = SparseSample;
    type Result = SparseResult;

    fn init(&self, rng: &mut SmallRng) -> SparseState<P> {
        let inclusions: Vec<P::Inclusion> = (0..self.num_components)
            .map(|_| self.prior.draw_inclusion(self.p, rng))
            .collect();
        let alphas: Vec<Vec<f32>> = inclusions
            .iter()
            .map(|inc| self.prior.to_alpha(inc))
            .collect();
        let effects: Vec<P::Effect> = (0..self.num_components)
            .map(|_| self.prior.draw_effect(rng))
            .collect();
        let theta = self.prior.combine(&alphas, &effects);
        SparseState {
            inclusions,
            alphas,
            effects,
            theta,
        }
    }

    fn sweep(&self, state: &mut SparseState<P>, rng: &mut SmallRng) {
        // --- ESS update of inclusion params ---
        for l in 0..self.num_components {
            self.prior.remove_component_inplace(
                &mut state.theta,
                &state.alphas[l],
                &state.effects[l],
            );

            let lnpdf_inc = |inc: &P::Inclusion| -> f32 {
                let alpha_new = self.prior.to_alpha(inc);
                let mut theta_new = state.theta.clone();
                self.prior
                    .add_component_inplace(&mut theta_new, &alpha_new, &state.effects[l]);
                (self.lnpdf)(&theta_new)
            };

            let cur_ll = lnpdf_inc(&state.inclusions[l]);
            let prior_sample = self.prior.draw_inclusion(self.p, rng);
            let (new_inc, _) =
                elliptical_slice_step(&state.inclusions[l], &prior_sample, &lnpdf_inc, cur_ll, rng);

            state.inclusions[l] = new_inc;
            state.alphas[l] = self.prior.to_alpha(&state.inclusions[l]);
            self.prior
                .add_component_inplace(&mut state.theta, &state.alphas[l], &state.effects[l]);
        }

        // --- ESS update of effect sizes ---
        for l in 0..self.num_components {
            self.prior.remove_component_inplace(
                &mut state.theta,
                &state.alphas[l],
                &state.effects[l],
            );

            let alpha_l = &state.alphas[l];
            let lnpdf_eff = |eff: &P::Effect| -> f32 {
                let mut theta_new = state.theta.clone();
                self.prior
                    .add_component_inplace(&mut theta_new, alpha_l, eff);
                (self.lnpdf)(&theta_new)
            };

            let cur_ll = lnpdf_eff(&state.effects[l]);
            let prior_eff = self.prior.draw_effect(rng);
            let (new_eff, _) =
                elliptical_slice_step(&state.effects[l], &prior_eff, &lnpdf_eff, cur_ll, rng);

            state.effects[l] = new_eff;
            self.prior
                .add_component_inplace(&mut state.theta, &state.alphas[l], &state.effects[l]);
        }
    }

    fn collect(&self, state: &SparseState<P>) -> SparseSample {
        SparseSample {
            alphas: state.alphas.clone(),
            betas: state.effects.iter().map(|e| e.as_slice()[0]).collect(),
            sigma2_eps: f32::NAN,
            effect_var: f32::NAN,
        }
    }

    fn summarize(&self, samples: Vec<SparseSample>) -> SparseResult {
        let pip = compute_pip(&samples, self.p);
        let posterior_mean_beta = compute_posterior_mean_beta(&samples, self.p);
        SparseResult {
            samples,
            pip,
            posterior_mean_beta,
            sigma2_eps_mean: f32::NAN,
            effect_var_mean: f32::NAN,
        }
    }
}

/// Blackbox MCMC sparse regression.
///
/// Convenience wrapper around [`SparseModel`] + [`run_mcmc`].
pub fn mcmc_sparse<P: ComponentPrior>(
    lnpdf: &impl Fn(&P::Theta) -> f32,
    p: usize,
    prior: &P,
    config: &McmcConfig,
    num_components: usize,
) -> SparseResult {
    let model = SparseModel {
        lnpdf,
        prior: prior.clone(),
        p,
        num_components,
    };
    crate::engine::run_mcmc(&model, config)
}
