use crate::embed_common::*;
use candle_util::cli::regression_likelihood::VonMisesFisherLikelihood;
use candle_util::sgvb::{
    samples_direct_elbo_loss, BiSusieVar, BlackBoxLikelihood, GaussianPrior, LinearModelSGVB,
    LinearRegressionSGVB, SGVBConfig, SgvbModel, SusieVar,
};
use candle_core::{DType, Result, Tensor};

/// SuSiE deconvolution with von Mises-Fisher likelihood.
///
/// Models directional alignment between topic signatures and annotation patterns.
/// Better suited when comparing normalized vectors where angle matters more than magnitude.
///
/// Model: y_t ~ vMF(normalize(X @ θ_t), κ_t)
/// Prior: θ = Σ_l (α_l ⊙ β_l) where α_l = softmax(logits_l)
pub struct VmfSusieDeconv {
    model_dir: LinearModelSGVB<SusieVar, GaussianPrior>,
    model_kappa: LinearRegressionSGVB<GaussianPrior>,
    y_normalized: Tensor,
    config: SGVBConfig,
}

impl VmfSusieDeconv {
    pub fn new(
        x_ga: Tensor,
        y_normalized: Tensor,
        n_topics: usize,
        n_components: usize,
        prior_var: f32,
        num_samples: usize,
        vb: candle_nn::VarBuilder,
    ) -> anyhow::Result<Self> {
        let n_annotations = x_ga.dim(1)?;
        let n_genes = x_ga.dim(0)?;
        let device = x_ga.device().clone();
        let dtype = DType::F32;

        // Concentration model: intercept-only for per-topic κ
        // (create before moving x_ga)
        let x_intercept = Tensor::ones((n_genes, 1), dtype, &device)?;
        let prior_kappa = GaussianPrior::new(vb.pp("prior_kappa"), 1.0)?;
        let config = SGVBConfig::new(num_samples);
        let model_kappa = LinearRegressionSGVB::new(
            vb.pp("kappa"),
            x_intercept,
            n_topics,
            prior_kappa,
            config.clone(),
        )?;

        // Direction model: SuSiE for sparse selection
        let susie = SusieVar::new(vb.pp("susie"), n_components, n_annotations, n_topics)?;
        let prior_dir = GaussianPrior::new(vb.pp("prior_dir"), prior_var)?;
        let model_dir =
            LinearModelSGVB::from_variational(susie, x_ga, prior_dir, config.clone());

        Ok(Self {
            model_dir,
            model_kappa,
            y_normalized,
            config,
        })
    }

    /// Compute the SGVB loss for training.
    pub fn loss(&self) -> Result<Tensor> {
        let sample_dir = self.model_dir.sample(self.config.num_samples)?;
        let sample_kappa = self.model_kappa.sample(self.config.num_samples)?;
        let likelihood = VonMisesFisherLikelihood::from_normalized(self.y_normalized.clone());
        samples_direct_elbo_loss(&[sample_dir, sample_kappa], &likelihood)
    }

    /// Compute SGVB loss with KL annealing.
    pub fn loss_with_kl_weight(&self, kl_weight: f32) -> Result<Tensor> {
        let sample_dir = self.model_dir.sample(self.config.num_samples)?;
        let sample_kappa = self.model_kappa.sample(self.config.num_samples)?;
        let likelihood = VonMisesFisherLikelihood::from_normalized(self.y_normalized.clone());

        let llik = likelihood.log_likelihood(&[&sample_dir.eta, &sample_kappa.eta])?;
        let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

        // Sum log_prior and log_q across both models
        let log_prior = (&sample_dir.log_prior + &sample_kappa.log_prior)?;
        let log_q = (&sample_dir.log_q + &sample_kappa.log_q)?;

        // ELBO with KL annealing: log_lik - β*(log_q - log_prior)
        let kl_term = (&log_q - &log_prior)?;
        let elbo = (&llik - kl_term.affine(kl_weight as f64, 0.0)?)?;
        elbo.mean(0)?.neg()
    }

    /// Get posterior inclusion probabilities (PIPs) for each annotation-topic pair.
    pub fn pip(&self) -> Result<Tensor> {
        self.model_dir.variational.pip()
    }
}

/// Bi-directional SuSiE deconvolution with von Mises-Fisher likelihood.
///
/// Combines BiSuSiE's bi-directional sparsity with vMF's directional alignment.
/// Best for finding sparse annotation-topic matches based on pattern similarity.
///
/// Model: y_t ~ vMF(normalize(X @ θ_t), κ_t)
/// Prior: θ[a,t] = Σ_l α_a[l,a] * α_t[l,t] * β[l]
pub struct VmfBiSusieDeconv {
    model_dir: LinearModelSGVB<BiSusieVar, GaussianPrior>,
    model_kappa: LinearRegressionSGVB<GaussianPrior>,
    y_normalized: Tensor,
    config: SGVBConfig,
}

impl VmfBiSusieDeconv {
    pub fn new(
        x_ga: Tensor,
        y_normalized: Tensor,
        n_topics: usize,
        n_components: usize,
        prior_var: f32,
        num_samples: usize,
        vb: candle_nn::VarBuilder,
    ) -> anyhow::Result<Self> {
        let n_annotations = x_ga.dim(1)?;
        let n_genes = x_ga.dim(0)?;
        let device = x_ga.device().clone();
        let dtype = DType::F32;

        // Concentration model: intercept-only for per-topic κ
        // (create before moving x_ga)
        let x_intercept = Tensor::ones((n_genes, 1), dtype, &device)?;
        let prior_kappa = GaussianPrior::new(vb.pp("prior_kappa"), 1.0)?;
        let config = SGVBConfig::new(num_samples);
        let model_kappa = LinearRegressionSGVB::new(
            vb.pp("kappa"),
            x_intercept,
            n_topics,
            prior_kappa,
            config.clone(),
        )?;

        // Direction model: BiSuSiE for bi-directional sparse selection
        let bisusie = BiSusieVar::new(vb.pp("bisusie"), n_components, n_annotations, n_topics)?;
        let prior_dir = GaussianPrior::new(vb.pp("prior_dir"), prior_var)?;
        let model_dir =
            LinearModelSGVB::from_variational(bisusie, x_ga, prior_dir, config.clone());

        Ok(Self {
            model_dir,
            model_kappa,
            y_normalized,
            config,
        })
    }

    /// Compute the SGVB loss for training.
    pub fn loss(&self) -> Result<Tensor> {
        let sample_dir = self.model_dir.sample(self.config.num_samples)?;
        let sample_kappa = self.model_kappa.sample(self.config.num_samples)?;
        let likelihood = VonMisesFisherLikelihood::from_normalized(self.y_normalized.clone());
        samples_direct_elbo_loss(&[sample_dir, sample_kappa], &likelihood)
    }

    /// Compute SGVB loss with KL annealing.
    pub fn loss_with_kl_weight(&self, kl_weight: f32) -> Result<Tensor> {
        let sample_dir = self.model_dir.sample(self.config.num_samples)?;
        let sample_kappa = self.model_kappa.sample(self.config.num_samples)?;
        let likelihood = VonMisesFisherLikelihood::from_normalized(self.y_normalized.clone());

        let llik = likelihood.log_likelihood(&[&sample_dir.eta, &sample_kappa.eta])?;
        let llik = if llik.rank() > 1 { llik.sum(1)? } else { llik };

        // Sum log_prior and log_q across both models
        let log_prior = (&sample_dir.log_prior + &sample_kappa.log_prior)?;
        let log_q = (&sample_dir.log_q + &sample_kappa.log_q)?;

        // ELBO with KL annealing: log_lik - β*(log_q - log_prior)
        let kl_term = (&log_q - &log_prior)?;
        let elbo = (&llik - kl_term.affine(kl_weight as f64, 0.0)?)?;
        elbo.mean(0)?.neg()
    }

    /// Get posterior inclusion probabilities (PIPs) for each annotation-topic pair.
    pub fn pip(&self) -> Result<Tensor> {
        self.model_dir.variational.pip()
    }
}
