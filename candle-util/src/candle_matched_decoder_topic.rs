#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

//////////////////////////////////////
// Differential Topic Model Decoder //
//////////////////////////////////////

pub struct MatchedTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: LogSoftmaxLinear,
}

impl MatchedTopicDecoder {
    /// Will create a new ETM decoder with the following parameters:
    /// * `dictionary.weight`
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = log_softmax_linear(n_topics, n_features, vs.pp("dictionary"))?;
        Ok(Self {
            n_features,
            n_topics,
            dictionary,
        })
    }

    pub fn dictionary(&self) -> &LogSoftmaxLinear {
        &self.dictionary
    }
}

impl MatchedDecoderModuleT for MatchedTopicDecoder {
    fn get_dictionary(&self) -> Result<Tensor> {
        self.dictionary.weight()
    }

    fn forward(&self, latent: &MatchedEncoderLatent) -> Result<Tensor> {
        self.dictionary.forward(&latent.logits_theta.exp()?)
    }

    fn forward_with_llik<LlikFn>(
        &self,
        latent: &MatchedEncoderLatent,
        data: MatchedDecoderData,
        llik: &LlikFn,
    ) -> Result<(Tensor, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let recon = self.forward(latent)?;
        // let llik_val = llik(&data.left.add(data.right)?, &recon)?;
        let llik_val = llik(data.left, &recon)?.add(&llik(data.right, &recon)?)?;
        Ok((recon, llik_val))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
