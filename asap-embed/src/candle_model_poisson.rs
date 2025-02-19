#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::Module;

///////////////////////////
// Poisson Topic Decoder //
///////////////////////////

pub struct PoissonDecoder {
    n_features: usize,
    n_topics: usize,
    dictionary: NonNegLinear,
}

impl PoissonDecoder {
    pub fn new(n_features: usize, n_topics: usize, vs: candle_nn::VarBuilder) -> Result<Self> {
        let dictionary = non_neg_linear(n_topics, n_features, vs)?;
        Ok(Self {
            n_features,
            n_topics,
            dictionary,
        })
    }

    pub fn dictionary(&self) -> &NonNegLinear {
        &self.dictionary
    }
}

impl DecoderModule for PoissonDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        let theta_nk = z_nk.exp()?;
        self.dictionary.forward(&theta_nk)
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
        let rate_nd = self.forward(z_nk)?;
        let llik = llik(x_nd, &rate_nd)?;
        Ok((rate_nd, llik))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
