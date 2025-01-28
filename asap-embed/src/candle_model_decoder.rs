use crate::candle_loss_functions::{gaussian_kl_loss, topic_likelihood};
use candle_core::{Result, Tensor};
use candle_nn::{ops, Linear, ModuleT, Sequential, VarBuilder};

/////////////////////
// Decoder for ETM //
/////////////////////

#[allow(dead_code)]
pub struct ETMDecoder {
    n_features: usize,
    n_topics: usize,
    beta: Linear,
}

#[allow(dead_code)]
impl candle_nn::ModuleT for ETMDecoder {
    fn forward_t(&self, z_nk: &Tensor, train: bool) -> Result<Tensor> {
	let theta_nk = z_nk.exp()?;
        let beta_kd = self.beta.forward_t(&theta_nk, train)?;
        Ok(ops::log_softmax(&beta_kd, 1)?)
    }
}

#[allow(dead_code)]
impl ETMDecoder {
    pub fn num_features(&self) -> usize {
        self.n_features
    }

    pub fn num_topics(&self) -> usize {
        self.n_topics
    }

    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let beta = candle_nn::linear(n_topics, n_features, vs.pp("dec.beta"))?;
        Ok(Self {
            n_features,
            n_topics,
            beta,
        })
    }
}
