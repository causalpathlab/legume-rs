#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

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
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        let theta_nk = z_nk.exp()?;
        self.dictionary.forward(&theta_nk)
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        self.dictionary.weight_dk()
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

        // penalize self regression likelihood
        // use std::f64::INFINITY;
        // let eps = 1e-8;
        // let logits_weight_dk = self.dictionary.weight()?;
        // let weight_dk = logits_weight_dk.exp()?.clamp(eps, INFINITY)?;
        // let llik_kk = weight_dk.transpose(0, 1)?.matmul(&logits_weight_dk)?;
        // let penalty = self.mask.mul(&llik_kk)?.sum_all()?;
        // let llik = llik(x_nd, &logits_nd)?.broadcast_sub(&penalty)?;
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
