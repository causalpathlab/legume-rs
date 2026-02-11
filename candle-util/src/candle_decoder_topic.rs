#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_loss_functions::zi_topic_log_likelihood;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, Module, VarBuilder};

/////////////////////////
// Topic Model Decoder //
/////////////////////////

pub struct TopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
}

impl TopicDecoder {
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

impl DecoderModuleT for TopicDecoder {
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

////////////////////////////////
// Sparse Topic Model Decoder //
////////////////////////////////

pub struct SparseTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SparsemaxLinear,
}

impl SparseTopicDecoder {
    /// Create a sparse topic decoder using sparsemax for gene distributions
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = sparsemax_linear(n_topics, n_features, vs.pp("dictionary"))?;

        Ok(Self {
            n_features,
            n_topics,
            dictionary,
        })
    }

    pub fn dictionary(&self) -> &SparsemaxLinear {
        &self.dictionary
    }
}

impl DecoderModuleT for SparseTopicDecoder {
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
// Zero-Inflated Topic Model Decoder //
////////////////////////////////////////

pub struct ZITopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
    dropout_logit_1d: Tensor, // [1, D] learnable parameter
}

impl ZITopicDecoder {
    /// Create a zero-inflated topic decoder with per-feature dropout gate.
    /// `dropout_logit_1d` is initialized to -2.0 (sigmoid(-2) ≈ 0.12).
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = log_softmax_linear(n_topics, n_features, vs.pp("dictionary"))?;

        let init_val = candle_nn::Init::Const(-2.0);
        let dropout_logit_1d = vs.get_with_hints((1, n_features), "dropout_logit", init_val)?;

        Ok(Self {
            n_features,
            n_topics,
            dictionary,
            dropout_logit_1d,
        })
    }

    /// Return per-feature dropout probabilities sigmoid(dropout_logit_1d) as [1, D]
    pub fn dropout_prob(&self) -> Result<Tensor> {
        ops::sigmoid(&self.dropout_logit_1d)
    }
}

impl DecoderModuleT for ZITopicDecoder {
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
        let llik = zi_topic_log_likelihood(x_nd, &log_recon_nd, &self.dropout_logit_1d)?;
        Ok((recon_nd, llik))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
