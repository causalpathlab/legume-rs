#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_loss_functions::nb_log_likelihood;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, Module, VarBuilder};

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

        // Log1p-weighted likelihood: llik = Σ_d log(1 + x_d) * log(recon_d)
        // Compresses dynamic range so marker genes (count ~1) contribute
        // comparably to housekeeping (count ~1000), improving topic
        // differentiation on sparse single-cell data.
        let llik = (x_nd + 1.0)?
            .log()?
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
        Ok((log_recon_nd.exp()?, llik))
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
