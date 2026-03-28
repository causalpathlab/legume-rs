use nalgebra::DVector;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::engine::{EssParam, EssParamSummary};

/// Prior for one sparse regression component, bundling inclusion and effect.
///
/// The inclusion parameters must have a Gaussian prior so that ESS can sample them.
/// The `Theta` associated type represents the combined effect across all components
/// (e.g. `DVector<f32>` for single-output, `DMatrix<f32>` for multi-output).
pub trait ComponentPrior: Clone + Send + Sync {
    /// Raw inclusion parameter type (must implement EssParam for ESS).
    type Inclusion: EssParam + Clone + Send;
    /// Effect size type per component. EssParamSummary for as_slice() access.
    type Effect: EssParamSummary + Clone + Send;
    /// Combined effect across all components.
    type Theta: Clone + Send;

    /// Draw raw inclusion parameters from the prior.
    fn draw_inclusion(&self, p: usize, rng: &mut impl Rng) -> Self::Inclusion;

    /// Draw effect size from the prior.
    fn draw_effect(&self, rng: &mut impl Rng) -> Self::Effect;

    /// Transform raw inclusion parameters to probabilities (length p).
    fn to_alpha(&self, raw: &Self::Inclusion) -> Vec<f32>;

    /// Combine all components' alphas and effects into the combined theta.
    fn combine(&self, alphas: &[Vec<f32>], effects: &[Self::Effect]) -> Self::Theta;

    /// Remove one component's contribution from theta in-place.
    fn remove_component_inplace(
        &self,
        theta: &mut Self::Theta,
        alpha: &[f32],
        effect: &Self::Effect,
    );

    /// Add one component's contribution to theta in-place.
    fn add_component_inplace(&self, theta: &mut Self::Theta, alpha: &[f32], effect: &Self::Effect);

    /// Prior variance on effect size (used by regression sampler for conjugate updates).
    fn effect_var(&self) -> f32;

    /// Zero-valued theta for initialization.
    fn zero_theta(&self, p: usize) -> Self::Theta;
}

/// SuSiE-style prior: logits ~ N(0, logit_var * I), alpha = softmax(logits).
/// Effect size: beta ~ N(0, effect_var). Single-output (k=1).
#[derive(Clone)]
pub struct SoftmaxNormalPrior {
    pub logit_var: f32,
    pub effect_var: f32,
}

impl SoftmaxNormalPrior {
    pub fn new(logit_var: f32, effect_var: f32) -> Self {
        Self {
            logit_var,
            effect_var,
        }
    }
}

impl ComponentPrior for SoftmaxNormalPrior {
    type Inclusion = DVector<f32>;
    type Effect = DVector<f32>; // length 1 for single-output
    type Theta = DVector<f32>; // length p

    fn draw_inclusion(&self, p: usize, rng: &mut impl Rng) -> DVector<f32> {
        let std = self.logit_var.sqrt();
        DVector::from_fn(p, |_, _| {
            let v: f64 = StandardNormal.sample(rng);
            v as f32 * std
        })
    }

    fn draw_effect(&self, rng: &mut impl Rng) -> DVector<f32> {
        let std = self.effect_var.sqrt();
        let v: f64 = StandardNormal.sample(rng);
        DVector::from_element(1, v as f32 * std)
    }

    fn to_alpha(&self, raw: &DVector<f32>) -> Vec<f32> {
        softmax(raw)
    }

    fn combine(&self, alphas: &[Vec<f32>], effects: &[Self::Effect]) -> DVector<f32> {
        let p = alphas[0].len();
        let mut theta = DVector::from_element(p, 0.0f32);
        for (alpha, effect) in alphas.iter().zip(effects.iter()) {
            let beta = effect[0];
            for (j, &a) in alpha.iter().enumerate() {
                theta[j] += a * beta;
            }
        }
        theta
    }

    fn remove_component_inplace(
        &self,
        theta: &mut DVector<f32>,
        alpha: &[f32],
        effect: &DVector<f32>,
    ) {
        let beta = effect[0];
        for (j, &a) in alpha.iter().enumerate() {
            theta[j] -= a * beta;
        }
    }

    fn add_component_inplace(
        &self,
        theta: &mut DVector<f32>,
        alpha: &[f32],
        effect: &DVector<f32>,
    ) {
        let beta = effect[0];
        for (j, &a) in alpha.iter().enumerate() {
            theta[j] += a * beta;
        }
    }

    fn effect_var(&self) -> f32 {
        self.effect_var
    }

    fn zero_theta(&self, p: usize) -> DVector<f32> {
        DVector::from_element(p, 0.0f32)
    }
}

/// Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x))).
pub(crate) fn softmax(v: &DVector<f32>) -> Vec<f32> {
    let max_v = v.max();
    let mut out: Vec<f32> = v.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f32 = out.iter().sum();
    let inv = 1.0 / sum;
    for x in &mut out {
        *x *= inv;
    }
    out
}
