#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_loss_functions::nb_log_likelihood;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, Module, VarBuilder};

pub(crate) const DEFAULT_BGM_INIT_PI: f32 = 0.8;

pub(crate) fn logit_f32(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// `log p = logsumexp([log π + log bgm, log(1-π) + log β̃])` on the last axis.
/// `logit_pi` is a scalar (shape `[1]`), shared across all topics and cells.
pub(crate) fn bgm_log_reconstruction(
    log_beta_tilde_nd: &Tensor,
    log_bgm_1d: &Tensor,
    logit_pi: &Tensor,
) -> Result<Tensor> {
    let pi = ops::sigmoid(logit_pi)?; // scalar [1]
    let log_pi = pi.log()?;
    let log_1mpi = pi.affine(-1.0, 1.0)?.log()?;

    let bgm_1d = log_pi.broadcast_add(log_bgm_1d)?; // [1, D]
    let topic_nd = log_1mpi.broadcast_add(log_beta_tilde_nd)?; // [N, D]
                                                               // Broadcast [1, D] to [N, D] before stacking
    let bgm_nd = bgm_1d.broadcast_as(topic_nd.shape())?;

    let stacked = Tensor::stack(&[&bgm_nd, &topic_nd], 2)?;
    stacked.log_sum_exp(2)
}

/////////////////////////
// Topic Model Decoder //
/////////////////////////

pub struct MultinomTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
}

impl MultinomTopicDecoder {
    /// Will create a new topic model decoder with the following parameters:
    /// * `dictionary.weight`
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = log_softmax_linear(n_topics, n_features, vs.pp("dictionary"))?;

        Ok(Self {
            n_features,
            n_topics,
            dictionary,
        })
    }

    pub fn dictionary(&self) -> &SoftmaxLinear {
        &self.dictionary
    }
}

impl NewDecoder for MultinomTopicDecoder {
    fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        MultinomTopicDecoder::new(n_features, n_topics, vs)
    }
}

impl DecoderModuleT for MultinomTopicDecoder {
    /// Input z_nk is already on the probability simplex (from softmax/sparsemax)
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
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
        let log_recon_nd = self.dictionary.forward_log(z_nk)?;
        let recon_nd = log_recon_nd.exp()?;

        // Direct log-space likelihood: llik = Σ_d x_d * log(recon_d)
        let llik = x_nd
            .clamp(0.0, f64::INFINITY)?
            .mul(&log_recon_nd)?
            .sum(x_nd.rank() - 1)?;

        Ok((recon_nd, llik))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

////////////////////////////////////////
// Negative Binomial Topic Decoder    //
////////////////////////////////////////

/// Topic decoder with negative binomial likelihood.
///
/// μ_gn = l_n · softmax(W · z_n)_g
/// x_gn ~ NB(μ_gn, φ_g)
///
/// where l_n is per-cell library size (total counts) and
/// φ_g is per-gene inverse dispersion (learned).
pub struct NbTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
    /// log(φ_g) per-gene inverse dispersion, [1, D]
    log_phi_1d: Tensor,
}

impl NbTopicDecoder {
    /// Create a NB topic decoder.
    /// `log_phi` initialized to ln(2) ≈ 0.69 (moderate dispersion).
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = log_softmax_linear(n_topics, n_features, vs.pp("dictionary"))?;

        let init_val = candle_nn::Init::Const(0.693); // ln(2)
        let log_phi_1d = vs.get_with_hints((1, n_features), "log_phi", init_val)?;

        Ok(Self {
            n_features,
            n_topics,
            dictionary,
            log_phi_1d,
        })
    }

    /// Return per-gene dispersion φ_g as [1, D]
    pub fn phi(&self) -> Result<Tensor> {
        self.log_phi_1d.exp()
    }

    /// Return log(φ_g) as [1, D]
    pub fn log_phi(&self) -> &Tensor {
        &self.log_phi_1d
    }
}

impl NewDecoder for NbTopicDecoder {
    fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        NbTopicDecoder::new(n_features, n_topics, vs)
    }
}

impl DecoderModuleT for NbTopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
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
        // softmax(W · z) gives gene proportions per cell [N, D]
        let log_recon_nd = self.dictionary.forward_log(z_nk)?;
        let recon_nd = log_recon_nd.exp()?;

        // Library size: l_n = Σ_g x_gn per cell [N, 1]
        let lib_size = x_nd.sum(x_nd.rank() - 1)?.unsqueeze(1)?; // [N, 1]

        // μ_gn = l_n · softmax(W · z_n)_g
        let mu_nd = recon_nd.broadcast_mul(&lib_size)?;

        // NB log-likelihood
        let llik = nb_log_likelihood(x_nd, &mu_nd, &self.log_phi_1d)?;

        // Return proportions (not mu) for dictionary extraction
        Ok((recon_nd, llik))
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
        let log_dict_dk = self.get_dictionary()?.detach();
        let beta_kd = log_dict_dk.t()?.exp()?.contiguous()?;
        let log_phi = self.log_phi_1d.detach();
        let lib_n1 = x_nd.sum(x_nd.rank() - 1)?.unsqueeze(1)?;
        let k = self.dim_latent() as f64;

        Ok(Box::new(move |z_nk: &Tensor| {
            let mut z = ops::softmax(z_nk, 1)?;
            if topic_smoothing > 0.0 {
                z = ((z * (1.0 - topic_smoothing))? + topic_smoothing / k)?;
            }
            let mu = z.matmul(&beta_kd)?.broadcast_mul(&lib_n1)?;
            nb_log_likelihood(x_nd, &mu, &log_phi)
        }))
    }
}

//////////////////////////////////////////////////
// Housekeeping-inflated Topic Decoder          //
//////////////////////////////////////////////////

/// Shared BGM state: fixed per-gene background log-profile + frozen scalar π.
/// `log_bgm_1d` is a Var so `init_decoder_dictionary` can overwrite it;
/// `logit_pi` is a Var for the same reason. Both are excluded from Adam.
pub struct BgmState {
    pub log_bgm_1d: Tensor,
    pub logit_pi: Tensor,
}

impl BgmState {
    pub fn new(n_features: usize, vs: &VarBuilder) -> Result<Self> {
        let init_bgm = candle_nn::Init::Const(-(n_features as f64).ln());
        let log_bgm_1d = vs.get_with_hints((1, n_features), "dictionary.bgm_profile", init_bgm)?;
        let logit_pi = vs.get_with_hints(
            (1,),
            "mixture.pi.bgm_profile",
            candle_nn::Init::Const(logit_f32(DEFAULT_BGM_INIT_PI) as f64),
        )?;
        Ok(Self {
            log_bgm_1d,
            logit_pi,
        })
    }

    pub fn pi(&self) -> Result<f32> {
        let v: Vec<f32> = ops::sigmoid(&self.logit_pi)?.to_vec1()?;
        Ok(v[0])
    }
}

/// Multinomial BGM topic decoder: `p_ng = π_n · hk_g + (1-π_n) · Σ_k z_nk β_kg`.
pub struct BgmMultinomTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
    bgm: BgmState,
}

impl BgmMultinomTopicDecoder {
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = log_softmax_linear_nobias(n_topics, n_features, vs.pp("dictionary"))?;
        let bgm = BgmState::new(n_features, &vs)?;
        Ok(Self {
            n_features,
            n_topics,
            dictionary,
            bgm,
        })
    }

    pub(crate) fn dictionary(&self) -> &SoftmaxLinear {
        &self.dictionary
    }

    pub fn pi(&self) -> Result<f32> {
        self.bgm.pi()
    }
}

impl NewDecoder for BgmMultinomTopicDecoder {
    fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        BgmMultinomTopicDecoder::new(n_features, n_topics, vs)
    }
}

impl DecoderModuleT for BgmMultinomTopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        let log_beta_tilde = self.dictionary.forward_log(z_nk)?;
        bgm_log_reconstruction(&log_beta_tilde, &self.bgm.log_bgm_1d, &self.bgm.logit_pi)?.exp()
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
        let log_beta_tilde = self.dictionary.forward_log(z_nk)?;
        let log_p_nd =
            bgm_log_reconstruction(&log_beta_tilde, &self.bgm.log_bgm_1d, &self.bgm.logit_pi)?;
        let recon_nd = log_p_nd.exp()?;
        let llik = x_nd
            .clamp(0.0, f64::INFINITY)?
            .mul(&log_p_nd)?
            .sum(x_nd.rank() - 1)?;
        Ok((recon_nd, llik))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

////////////////////////////////////////////
// Negative Binomial BGM Topic Decoder    //
////////////////////////////////////////////

/// NB topic decoder with housekeeping-inflated mean:
///
/// ```text
/// μ_ng = l_n · (π_n · hk_g + (1 - π_n) · Σ_k z_nk β_kg)
/// x_ng ~ NB(μ_ng, φ_g)
/// ```
///
/// Combines the BGM mixture (fixed hk profile + per-topic π) with
/// per-gene dispersion inherited from `NbTopicDecoder`.
pub struct BgmNbTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
    /// log(φ_g) per-gene inverse dispersion, [1, D]
    log_phi_1d: Tensor,
    bgm: BgmState,
}

impl BgmNbTopicDecoder {
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = log_softmax_linear_nobias(n_topics, n_features, vs.pp("dictionary"))?;
        let log_phi_1d =
            vs.get_with_hints((1, n_features), "log_phi", candle_nn::Init::Const(0.693))?;
        let bgm = BgmState::new(n_features, &vs)?;
        Ok(Self {
            n_features,
            n_topics,
            dictionary,
            log_phi_1d,
            bgm,
        })
    }

    pub fn pi(&self) -> Result<f32> {
        self.bgm.pi()
    }

    pub fn log_phi(&self) -> &Tensor {
        &self.log_phi_1d
    }
}

impl NewDecoder for BgmNbTopicDecoder {
    fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        BgmNbTopicDecoder::new(n_features, n_topics, vs)
    }
}

impl DecoderModuleT for BgmNbTopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        let log_beta_tilde = self.dictionary.forward_log(z_nk)?;
        bgm_log_reconstruction(&log_beta_tilde, &self.bgm.log_bgm_1d, &self.bgm.logit_pi)?.exp()
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
        let log_beta_tilde = self.dictionary.forward_log(z_nk)?;
        let log_p_nd =
            bgm_log_reconstruction(&log_beta_tilde, &self.bgm.log_bgm_1d, &self.bgm.logit_pi)?;
        let recon_nd = log_p_nd.exp()?;

        // μ_nd = l_n · p_nd
        let lib_size = x_nd.sum(x_nd.rank() - 1)?.unsqueeze(1)?; // [N, 1]
        let mu_nd = recon_nd.broadcast_mul(&lib_size)?;

        let llik = nb_log_likelihood(x_nd, &mu_nd, &self.log_phi_1d)?;
        Ok((recon_nd, llik))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
