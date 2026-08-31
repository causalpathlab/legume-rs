#![allow(dead_code)]

use crate::loss::nb_log_likelihood;
use crate::nn::linear::*;
use crate::traits::model::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, Module, VarBuilder};

/// Canonical name for this decoder — used by `DecoderType::as_str`,
/// the `DynDecoderModuleT` registry, and `senna eval-topic` dispatch.
pub const DECODER_NAME: &str = "nbmixture";

/// Topic decoder with NB likelihood and a learned ambient-RNA mixture.
///
/// Generative model:
///   θ_n  = softmax(z_n)                         — topic proportions
///   β_kd = softmax(W_kd, dim=-1)                — topic dictionary
///   α_d  = softmax(log_α_1d, dim=-1)            — ambient gene profile
///   L_n  = Σ_d x_nd                             — library size
///   ρ_n  = sigmoid(a · log L_n + b)             — per-sample ambient fraction
///   π_nd = (1 − ρ_n) · θ_n·β_kd  +  ρ_n · α_d
///   μ_nd = L_n · π_nd
///   y_nd ~ NB(μ_nd, φ_g)
pub struct NbMixtureTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
    /// [1, D] log of per-gene NB inverse dispersion
    log_phi_1d: Tensor,
    /// [1, D] ambient logits, simplex-normalized via log_softmax on use
    log_alpha_1d: Tensor,
    /// [1, 1] slope in ρ = sigmoid(a · log L + b)
    rho_a: Tensor,
    /// [1, 1] bias in ρ = sigmoid(a · log L + b)
    rho_b: Tensor,
    /// Overall weight on the Beta(α_β, β_β) prior over ρ. 0 = disabled.
    rho_prior_weight: f32,
    /// Beta prior shape α_β
    rho_prior_alpha: f32,
    /// Beta prior shape β_β
    rho_prior_beta: f32,
}

impl NbMixtureTopicDecoder {
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = log_softmax_linear(n_topics, n_features, vs.pp("dictionary"))?;

        let log_phi_1d =
            vs.get_with_hints((1, n_features), "log_phi", candle_nn::Init::Const(0.693))?;

        // α init: uniform (logits all 0 → softmax is 1/D for all genes). The
        // likelihood + the KL(p_emp || α) penalty pull it toward the empirical
        // mean profile during training.
        let log_alpha_1d =
            vs.get_with_hints((1, n_features), "log_alpha", candle_nn::Init::Const(0.0))?;

        // ρ = sigmoid(a · log L + b). Init a = −0.5, b = 0 → larger cells get
        // smaller ρ. At L=1000, ρ ≈ 0.03; at L=100, ρ ≈ 0.09; at L=10, ρ ≈ 0.24.
        let rho_a = vs.get_with_hints((1, 1), "rho_a", candle_nn::Init::Const(-0.5))?;
        let rho_b = vs.get_with_hints((1, 1), "rho_b", candle_nn::Init::Const(0.0))?;

        Ok(Self {
            n_features,
            n_topics,
            dictionary,
            log_phi_1d,
            log_alpha_1d,
            rho_a,
            rho_b,
            rho_prior_weight: 0.0,
            rho_prior_alpha: 2.0,
            rho_prior_beta: 18.0,
        })
    }

    /// Configure the Beta(α, β) prior over ρ. `weight=0` disables the prior.
    /// Typical settings: `(weight=10.0, alpha=2.0, beta=18.0)` → pushes
    /// per-cell ρ toward ≈0.1 so α actually absorbs ambient signal.
    pub fn set_rho_prior(&mut self, weight: f32, alpha: f32, beta: f32) {
        self.rho_prior_weight = weight;
        self.rho_prior_alpha = alpha;
        self.rho_prior_beta = beta;
    }

    pub fn rho_prior_weight(&self) -> f32 {
        self.rho_prior_weight
    }
    pub fn rho_prior_alpha(&self) -> f32 {
        self.rho_prior_alpha
    }
    pub fn rho_prior_beta(&self) -> f32 {
        self.rho_prior_beta
    }

    pub fn phi(&self) -> Result<Tensor> {
        self.log_phi_1d.exp()
    }

    pub fn log_phi(&self) -> &Tensor {
        &self.log_phi_1d
    }

    pub fn log_alpha(&self) -> &Tensor {
        &self.log_alpha_1d
    }

    /// Ambient gene profile α as a [1, D] simplex.
    pub fn alpha(&self) -> Result<Tensor> {
        let r = self.log_alpha_1d.rank();
        ops::log_softmax(&self.log_alpha_1d, r - 1)?.exp()
    }

    pub fn rho_a(&self) -> &Tensor {
        &self.rho_a
    }

    pub fn rho_b(&self) -> &Tensor {
        &self.rho_b
    }

    /// The per-cell log Beta(α_β, β_β) prior on ρ_n, or `None` when disabled.
    ///
    /// `log p(ρ) ∝ (α−1) log ρ + (β−1) log(1−ρ)`; the Beta normaliser is
    /// constant and does not affect gradients. One home, because both the
    /// dense and the gene-sliced likelihood add it — and the sliced one must
    /// add it exactly ONCE, not per slice.
    fn rho_prior_llik(&self, rho_n1: &Tensor, one_minus_rho: &Tensor) -> Result<Option<Tensor>> {
        if self.rho_prior_weight <= 0.0 {
            return Ok(None);
        }
        let eps = 1e-6f64;
        let log_rho = (rho_n1 + eps)?.log()?;
        let log_1m_rho = (one_minus_rho + eps)?.log()?;
        let term = ((log_rho * f64::from(self.rho_prior_alpha - 1.0))?
            + (log_1m_rho * f64::from(self.rho_prior_beta - 1.0))?)?;
        Ok(Some((term.squeeze(1)? * f64::from(self.rho_prior_weight))?))
    }

    /// Per-sample ambient fraction ρ_n from library sizes `lib_n1` [N, 1].
    pub fn rho_from_lib(&self, lib_n1: &Tensor) -> Result<Tensor> {
        let log_lib = (lib_n1 + 1e-8)?.log()?;
        let z = log_lib
            .broadcast_mul(&self.rho_a)?
            .broadcast_add(&self.rho_b)?;
        ops::sigmoid(&z)
    }
}

impl NewDecoder for NbMixtureTopicDecoder {
    fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        NbMixtureTopicDecoder::new(n_features, n_topics, vs)
    }
}

impl DecoderModuleT for NbMixtureTopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        // Topic component only. α mixing needs x_nd for the library size,
        // so it lives in forward_with_llik.
        self.dictionary.forward(z_nk)
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        self.dictionary.weight_dk()
    }

    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &Tensor,
        _llik: &LlikFn,
    ) -> Result<(Tensor, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let last_dim = x_nd.rank() - 1;
        let topic_recon_nd = self.dictionary.forward(z_nk)?; // [N, D]
        let alpha_1d = self.alpha()?;

        let lib_n1 = x_nd.sum(last_dim)?.unsqueeze(1)?; // [N, 1]
        let rho_n1 = self.rho_from_lib(&lib_n1)?;
        let one_minus_rho = rho_n1.affine(-1.0, 1.0)?;

        let pi_nd = topic_recon_nd
            .broadcast_mul(&one_minus_rho)?
            .broadcast_add(&alpha_1d.broadcast_mul(&rho_n1)?)?;
        let mu_nd = pi_nd.broadcast_mul(&lib_n1)?;

        let data_llik = nb_log_likelihood(x_nd, &mu_nd, &self.log_phi_1d)?;

        // Added to the log-lik so the training loop's `loss = kl − llik`
        // subtracts it once.
        let llik = match self.rho_prior_llik(&rho_n1, &one_minus_rho)? {
            Some(prior) => (data_llik + prior)?,
            None => data_llik,
        };
        Ok((pi_nd, llik))
    }

    fn llik_is_gene_chunked(&self) -> bool {
        true
    }

    /// The same likelihood as [`Self::forward_with_llik`], taken a gene slice
    /// at a time and never holding `[N, D]`.
    ///
    /// One pass is enough here, unlike the Gaussian head: `pi_d` mixes the
    /// topic rate with the ambient profile per gene, and the topic rate is a
    /// `logsumexp` over TOPICS, so a gene's rate needs no other gene's. Only
    /// the library size and `rho` are global, and both are `[N, 1]` — read off
    /// `x` up front and shared by every slice. The `rho` prior is per cell, so
    /// it is added once at the end rather than per slice.
    fn llik_gene_chunked(&self, z_nk: &Tensor, x_nd: &Tensor, gene_chunk: usize) -> Result<Tensor> {
        let last = x_nd.rank() - 1;
        let chunk = gene_chunk.max(1).min(self.n_features);
        let lib_n1 = x_nd.sum(last)?.unsqueeze(1)?; // [N, 1]
        let rho_n1 = self.rho_from_lib(&lib_n1)?;
        let one_minus_rho = rho_n1.affine(-1.0, 1.0)?;
        // `alpha` stays `[1, D]` and is narrowed before meeting the per-cell
        // `rho`: multiplying them up front would build an `[N, D]` tensor —
        // the very allocation this method exists to avoid — and hold it for
        // the whole loop.
        let alpha_1d = self.alpha()?;
        // The full-width softmax, hoisted so it is not redone per slice.
        let log_w_kd = self.dictionary.log_weight_kd()?;

        let mut data_llik: Option<Tensor> = None;
        for (start, len) in super::gene_slices(self.n_features, chunk) {
            // The topic rate for this slice only. `forward_log_slice` owns the
            // narrow-after-normalise rule, so this cannot slice unnormalised
            // weights by mistake.
            let topic = self
                .dictionary
                .forward_log_slice(z_nk, Some(&log_w_kd), start, len)?
                .exp()?;
            let ambient = alpha_1d.narrow(last, start, len)?.broadcast_mul(&rho_n1)?;
            let pi = topic
                .broadcast_mul(&one_minus_rho)?
                .broadcast_add(&ambient)?;
            let mu = pi.broadcast_mul(&lib_n1)?;
            let x = x_nd.narrow(last, start, len)?;
            let log_phi = self
                .log_phi_1d
                .narrow(1, start, len)?
                .broadcast_as(x.shape())?;
            let part = crate::loss::nb_log_likelihood_elem(&x, &mu, &log_phi)?.sum(last)?;
            data_llik = Some(match data_llik.take() {
                Some(acc) => acc.add(&part)?,
                None => part,
            });
        }
        let data_llik =
            data_llik.ok_or_else(|| candle_core::Error::Msg("no gene slices to score".into()))?;

        match self.rho_prior_llik(&rho_n1, &one_minus_rho)? {
            Some(prior) => data_llik + prior,
            None => Ok(data_llik),
        }
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }

    fn build_ess_llik<'a>(
        &'a self,
        x_nd: &'a Tensor,
        topic_smoothing: f64,
    ) -> Result<EssLlikFn<'a>> {
        let last_dim = x_nd.rank() - 1;
        let log_dict_dk = self.get_dictionary()?.detach();
        let beta_kd = log_dict_dk.t()?.exp()?.contiguous()?;
        let log_phi = self.log_phi_1d.detach();
        let log_alpha_det = self.log_alpha_1d.detach();
        let alpha_1d = {
            let r = log_alpha_det.rank();
            ops::log_softmax(&log_alpha_det, r - 1)?.exp()?
        };
        let rho_a = self.rho_a.detach();
        let rho_b = self.rho_b.detach();

        let lib_n1 = x_nd.sum(last_dim)?.unsqueeze(1)?;
        let log_lib = (&lib_n1 + 1e-8)?.log()?;
        let rho_n1 = ops::sigmoid(&log_lib.broadcast_mul(&rho_a)?.broadcast_add(&rho_b)?)?;
        let one_minus_rho = rho_n1.affine(-1.0, 1.0)?;
        let k = self.dim_latent() as f64;

        Ok(Box::new(move |z_nk: &Tensor| {
            let mut z = ops::softmax(z_nk, 1)?;
            if topic_smoothing > 0.0 {
                z = ((z * (1.0 - topic_smoothing))? + topic_smoothing / k)?;
            }
            let topic_recon = z.matmul(&beta_kd)?;
            let pi = topic_recon
                .broadcast_mul(&one_minus_rho)?
                .broadcast_add(&alpha_1d.broadcast_mul(&rho_n1)?)?;
            let mu = pi.broadcast_mul(&lib_n1)?;
            nb_log_likelihood(x_nd, &mu, &log_phi)
        }))
    }
}

#[cfg(test)]
#[path = "nb_mixture_tests.rs"]
mod chunked_llik_tests;
