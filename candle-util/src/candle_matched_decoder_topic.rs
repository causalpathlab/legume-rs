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

    fn forward(&self, latent: &MatchedEncoderLatent) -> Result<MatchedDecoderRecon> {
        let theta_marginal = latent.marginal.exp()?;
        let theta_border = latent.border.exp()?;

        let marginal = self.dictionary.forward(&theta_marginal)?.log()?;
        let border = self.dictionary.forward(&theta_border)?.log()?;

        Ok(MatchedDecoderRecon { marginal, border })
    }

    fn forward_with_llik<LlikFn>(
        &self,
        latent: &MatchedEncoderLatent,
        x_data: MatchedDecoderData,
        llik: &LlikFn,
    ) -> Result<(MatchedDecoderRecon, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let recon = self.forward(latent)?;

        let (xm_l, xm_r) = (x_data.marginal_left, x_data.marginal_right);

        let llik_val = if let (Some(xb_l), Some(xb_r)) = (x_data.border_left, x_data.border_right) {
            (llik(xm_l, &recon.marginal)? + llik(xm_r, &recon.marginal)?)?
                + (&llik(xb_l, &recon.border)? + &llik(xb_r, &recon.border)?)?
        } else {
            llik(x_data.marginal_left, &recon.marginal)?
                + llik(x_data.marginal_right, &recon.marginal)?
        }?;

        Ok((recon, llik_val))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
