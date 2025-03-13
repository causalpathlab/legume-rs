#![allow(dead_code)]

use crate::candle_aux_linear::*;
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
        ops::log_softmax(&prob_nd.log()?, 1)
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
