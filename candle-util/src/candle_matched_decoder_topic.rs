#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, Module, VarBuilder};

//////////////////////////////////////
// Differential Topic Model Decoder //
//////////////////////////////////////

pub struct MatchedTopicDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: SoftmaxLinear,
}

impl MatchedTopicDecoder {
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

impl MatchedDecoderModuleT for MatchedTopicDecoder {
    fn get_dictionary(&self) -> Result<Tensor> {
        self.dictionary.weight()
    }

    fn forward(&self, latent: &MatchedEncoderLatent) -> Result<(Tensor, Tensor)> {
        let theta_average_nk = latent.average.exp()?;
        let theta_left_nk = latent.left.exp()?;
        let theta_right_nk = latent.right.exp()?;

        // reconstruct left and right separately
        let shared_nd = self.dictionary.forward(&theta_average_nk)?;
        let left_delta_nd = self.dictionary.forward(&theta_left_nk)?;
        let right_delta_nd = self.dictionary.forward(&theta_right_nk)?;

        let recon_left_nd = ops::log_softmax(&(shared_nd.log()? + left_delta_nd.log()?)?, 1)?;
        let recon_right_nd = ops::log_softmax(&(shared_nd.log()? + right_delta_nd.log()?)?, 1)?;
        Ok((recon_left_nd, recon_right_nd))
    }

    fn forward_with_llik<LlikFn>(
        &self,
        latent: &MatchedEncoderLatent,
        x_nd_pair: (&Tensor, &Tensor),
        llik: &LlikFn,
    ) -> Result<(MatchedDecoderRecon, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let (x_left_nd, x_right_nd) = x_nd_pair;
        let (r_left_nd, r_right_nd) = self.forward(latent)?;
        let llik = (llik(x_left_nd, &r_left_nd)? + llik(x_right_nd, &r_right_nd))?;
        Ok((
            MatchedDecoderRecon {
                left: r_left_nd,
                right: r_right_nd,
            },
            llik,
        ))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
