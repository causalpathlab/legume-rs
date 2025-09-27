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
    dictionary: SoftmaxLinear,
    dictionary_delta: SoftmaxLinear,
}

impl MatchedTopicDecoder {
    /// Will create a new ETM decoder with the following parameters:
    /// * `dictionary.weight`
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = log_softmax_linear(n_topics, n_features, vs.pp("dictionary"))?;
        let dictionary_delta = log_softmax_linear(n_topics, n_features, vs.pp("dictionary.delta"))?;
        Ok(Self {
            n_features,
            n_topics,
            dictionary,
            dictionary_delta,
        })
    }

    pub fn dictionary(&self) -> &SoftmaxLinear {
        &self.dictionary
    }
    pub fn dictionary_delta(&self) -> &SoftmaxLinear {
        &self.dictionary_delta
    }
}

impl MatchedDecoderModuleT for MatchedTopicDecoder {
    fn get_dictionary(&self) -> Result<Tensor> {
        self.dictionary.weight()
    }

    fn forward(&self, latent: &MatchedEncoderLatent) -> Result<MatchedDecoderRecon> {
        Ok(MatchedDecoderRecon {
            x_left: self.dictionary.forward(&latent.logits_theta_left.exp()?)?,
            x_right: self.dictionary.forward(&latent.logits_theta_left.exp()?)?,
        })
    }

    fn forward_with_llik<LlikFn>(
        &self,
        latent: &MatchedEncoderLatent,
        data: MatchedDecoderData,
        llik: &LlikFn,
    ) -> Result<(MatchedDecoderRecon, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        if let (Some(delta_left), Some(delta_right)) = (data.delta_left, data.delta_right) {
            let x_left = self
                .dictionary_delta
                .forward(&latent.logits_theta_left.exp()?)?;
            let x_right = self
                .dictionary_delta
                .forward(&latent.logits_theta_left.exp()?)?;

            let llik_val = llik(&delta_left, &x_left)?.add(&llik(&delta_right, &x_right)?)?;

            // // let recon = self.dictionary().forward(&latent.logits_theta.exp()?)?;
            // // let recon_delta = self
            // //     .dictionary_delta()
            // //     .forward(&latent.logits_theta.exp()?)?;
            // let llik_val = llik(&data.left, &recon)?
            //     .add(&llik(&data.right, &recon)?)?
            //     .add(&llik(&delta_left, &recon_delta)?)?
            //     .add(&llik(&delta_right, &recon_delta)?)?;
            // Ok((recon, llik_val))
            let recon = self.forward(latent)?;
            // let llik_val =
            //     llik(&data.left, &recon.x_left)?.add(&llik(&data.right, &recon.x_right)?)?;
            Ok((recon, llik_val))
        } else {
            let recon = self.forward(latent)?;
            let llik_val =
                llik(&data.left, &recon.x_left)?.add(&llik(&data.right, &recon.x_right)?)?;
            Ok((recon, llik_val))
        }
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
