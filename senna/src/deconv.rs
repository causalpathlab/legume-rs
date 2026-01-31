use crate::embed_common::*;
use candle_util::cli::regression_likelihood::PoissonLikelihood;
use candle_util::sgvb::{
    direct_elbo_loss, BiSusieVar, BlackBoxLikelihood, GaussianPrior, LinearModelSGVB, SGVBConfig,
    SgvbModel, SusieVar,
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

    /// Compute SGVB loss with KL annealing.
    ///
    /// KL annealing gradually increases the KL divergence weight during training,
    /// which helps prevent posterior collapse and improves training stability.
    ///
    /// # Arguments
    /// * `y_gt` - Target data tensor
    /// * `kl_weight` - Weight for KL divergence term (0.0 to 1.0)
    pub fn loss_with_kl_weight(&self, y_gt: &Tensor, kl_weight: f32) -> Result<Tensor> {
        let likelihood = PoissonLikelihood::new(y_gt.clone());
        let sample = self.model.sample(self.config.num_samples)?;
        let llik = likelihood.log_likelihood(&[&sample.eta])?;
        let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };
        // ELBO with KL annealing: log_lik - β*(log_q - log_prior)
        let kl_term = (&sample.log_q - &sample.log_prior)?;
        let elbo = (&llik - kl_term.affine(kl_weight as f64, 0.0)?)?;
        elbo.mean(0)?.neg()
    }

    /// Get posterior inclusion probabilities (PIPs) for each annotation-topic pair.
    pub fn pip(&self) -> Result<Tensor> {
        self.model.variational.pip()
    }
}

/// Bi-directional SuSiE-Poisson deconvolution model.
///
/// Uses Bi-directional Sum of Single Effects (BiSuSiE) prior for sparse
/// annotation-to-topic mapping with sparsity enforced in BOTH dimensions:
/// - Each component selects one annotation (cell type)
/// - Each component selects one topic
///
/// This creates a near-permutation structure where each (annotation, topic)
/// pair is selected by at most a few components, leading to more decisive
/// cell type annotations.
///
/// Model: Y[g,t] ~ Poisson(exp(X[g,:] @ θ[:,t]))
/// Prior: θ[a,t] = Σ_l α_a[l,a] * α_t[l,t] * β[l]
pub struct BiSusieDeconv {
    model: LinearModelSGVB<BiSusieVar, GaussianPrior>,
    config: SGVBConfig,
}

impl BiSusieDeconv {
    /// Create BiSusieDeconv with softmax selection.
    pub fn new(
        x_ga: Tensor,
        n_topics: usize,
        n_components: usize,
        prior_var: f32,
        num_samples: usize,
        vb: candle_nn::VarBuilder,
    ) -> anyhow::Result<Self> {
        let n_annotations = x_ga.dim(1)?;
        let bisusie =
            BiSusieVar::new(vb.pp("bisusie"), n_components, n_annotations, n_topics)?;
        let prior = GaussianPrior::new(vb.pp("prior"), prior_var)?;
        let config = SGVBConfig::new(num_samples);
        let model = LinearModelSGVB::from_variational(bisusie, x_ga, prior, config.clone());
        Ok(Self { model, config })
    }

    /// Compute the SGVB loss for training.
    pub fn loss(&self, y_gt: &Tensor) -> Result<Tensor> {
        let likelihood = PoissonLikelihood::new(y_gt.clone());
        direct_elbo_loss(&self.model, &likelihood, self.config.num_samples)
    }

    /// Compute SGVB loss with KL annealing.
    ///
    /// KL annealing gradually increases the KL divergence weight during training,
    /// which helps prevent posterior collapse and improves training stability.
    ///
    /// # Arguments
    /// * `y_gt` - Target data tensor
    /// * `kl_weight` - Weight for KL divergence term (0.0 to 1.0)
    pub fn loss_with_kl_weight(&self, y_gt: &Tensor, kl_weight: f32) -> Result<Tensor> {
        let likelihood = PoissonLikelihood::new(y_gt.clone());
        let sample = self.model.sample(self.config.num_samples)?;
        let llik = likelihood.log_likelihood(&[&sample.eta])?;
        let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };
        // ELBO with KL annealing: log_lik - β*(log_q - log_prior)
        let kl_term = (&sample.log_q - &sample.log_prior)?;
        let elbo = (&llik - kl_term.affine(kl_weight as f64, 0.0)?)?;
        elbo.mean(0)?.neg()
    }

    /// Get posterior inclusion probabilities (PIPs) for each annotation-topic pair.
    /// Shape: (n_annotations, n_topics)
    pub fn pip(&self) -> Result<Tensor> {
        self.model.variational.pip()
    }

}
