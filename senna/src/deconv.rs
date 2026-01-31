use crate::embed_common::*;
use candle_util::cli::regression_likelihood::PoissonLikelihood;
use candle_util::sgvb::{
    direct_elbo_loss, GaussianPrior, LinearModelSGVB, SGVBConfig, SusieVar,
};
use candle_core::{Result, Tensor};

/// SuSiE-Poisson deconvolution model.
///
/// Uses Sum of Single Effects (SuSiE) prior for sparse topic-to-annotation mapping.
/// Model: Y[g,t] ~ Poisson(exp(X[g,:] @ θ[:,t]))
/// Prior: θ = Σ_l (α_l ⊙ β_l) where α_l = softmax(logits_l)
pub struct SusieDeconv {
    model: LinearModelSGVB<SusieVar, GaussianPrior>,
    config: SGVBConfig,
}

impl SusieDeconv {
    pub fn new(
        x_ga: Tensor,
        n_topics: usize,
        n_components: usize,
        prior_var: f32,
        num_samples: usize,
        vb: candle_nn::VarBuilder,
    ) -> anyhow::Result<Self> {
        let n_annotations = x_ga.dim(1)?;
        let susie = SusieVar::new(vb.pp("susie"), n_components, n_annotations, n_topics)?;
        let prior = GaussianPrior::new(vb.pp("prior"), prior_var)?;
        let config = SGVBConfig::new(num_samples);
        let model = LinearModelSGVB::from_variational(susie, x_ga, prior, config.clone());
        Ok(Self { model, config })
    }

    /// Compute the SGVB loss for training.
    pub fn loss(&self, y_gt: &Tensor) -> Result<Tensor> {
        let likelihood = PoissonLikelihood::new(y_gt.clone());
        direct_elbo_loss(&self.model, &likelihood, self.config.num_samples)
    }

    /// Get posterior inclusion probabilities (PIPs) for each annotation-topic pair.
    pub fn pip(&self) -> Result<Tensor> {
        self.model.variational.pip()
    }
}
