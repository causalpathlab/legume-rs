#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{ops, Module, VarBuilder};

pub trait DecoderModule {
    /// A decoder that spits out reconstruction
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor>;

    /// A decoder that spits out reconstruction and log-likelihood
    /// * `z_nk` - latent states
    /// * `x_nd` - observed data to validate with
    /// * `llik` - fn (observed, reconstruction) -> log-likelihood
    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &Tensor,
        llik: &LlikFn,
    ) -> Result<(Tensor, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;
}

////////////////////////////////
// Linear module with Softmax //
////////////////////////////////

pub struct SoftmaxLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl SoftmaxLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
    pub fn weight(&self) -> Result<Tensor> {
        ops::log_softmax(&self.weight, 0)
    }
}

impl Module for SoftmaxLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let log_w = match *x.dims() {
            [b1, b2, _, _] => self.weight()?.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.weight()?.broadcast_left(bsize)?.t()?,
            _ => self.weight()?.t()?,
        };

        match &self.bias {
            None => x.matmul(&log_w.exp()?),
            Some(bias) => x.matmul(&(log_w.broadcast_add(bias)?).exp()?),
        }
    }
}

pub fn softmax_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<SoftmaxLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bs = vb.get_with_hints((out_dim,), "bias", candle_nn::init::ZERO)?;
    Ok(SoftmaxLinear::new(ws, Some(bs)))
}

/////////////////////////
// Topic Model Decoder //
/////////////////////////

pub struct TopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
}

impl TopicDecoder {
    /// Will create a new ETM decoder with the following parameters:
    /// * `dictionary.weight`
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = softmax_linear(n_topics, n_features, vs.pp("dictionary"))?;
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

impl DecoderModule for TopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        let theta_nk = z_nk.exp()?;
        let prob_nd = self.dictionary.forward(&theta_nk)?;
        let log_prob_nd = prob_nd.log()?;
        ops::log_softmax(&log_prob_nd, 1)
    }

    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &Tensor,
        llik: &LlikFn,
    ) -> Result<(Tensor, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let logits_nd = self.forward(z_nk)?;
        let llik = llik(x_nd, &logits_nd)?;
        Ok((logits_nd, llik))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
