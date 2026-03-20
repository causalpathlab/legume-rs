#![allow(dead_code)]

use crate::candle_aux_linear::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

/////////////////////////
// Topic Model Decoder //
/////////////////////////

pub struct MultimodalTopicDecoder {
    n_features: Vec<usize>,
    n_topics: usize,
    dictionary: Vec<SoftmaxLinear>,
}

impl MultimodalTopicDecoder {
    /// Will create a new topic model decoder with the following parameters:
    /// * `dictionary.weight`
    pub fn new(n_features: &[usize], n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let dictionary = n_features
            .iter()
            .enumerate()
            .map(|(m, &d)| log_softmax_linear(n_topics, d, vs.pp(format!("dictionary.{}", m))))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            n_features: n_features.to_vec(),
            n_topics,
            dictionary,
        })
    }

    pub fn dictionary(&self) -> &Vec<SoftmaxLinear> {
        &self.dictionary
    }
}

impl MultimodalDecoderModuleT for MultimodalTopicDecoder {
    fn get_dictionary(&self) -> Result<Vec<Tensor>> {
        self.dictionary
            .iter()
            .map(|x| x.weight_dk())
            .collect::<Result<Vec<Tensor>>>()
    }

    /// Input z_nk is already on the probability simplex (from softmax/sparsemax)
    fn forward(&self, z_nk: &Tensor) -> Result<Vec<Tensor>> {
        self.dictionary.iter().map(|x| x.forward(z_nk)).collect()
    }

    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd_vec: &[Tensor],
        _llik: &LlikFn,
    ) -> Result<(Vec<Tensor>, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        // Compute log-space reconstructions for each modality
        let log_recon_vec: Vec<Tensor> = self
            .dictionary
            .iter()
            .map(|x| x.forward_log(z_nk))
            .collect::<Result<Vec<_>>>()?;

        let recon_vec: Vec<Tensor> = log_recon_vec
            .iter()
            .map(|x| x.exp())
            .collect::<Result<Vec<_>>>()?;

        let llik_vec = x_nd_vec
            .iter()
            .zip(&log_recon_vec)
            .map(|(x, log_recon)| -> Result<Tensor> {
                let ret = x
                    .clamp(0.0, f64::INFINITY)?
                    .mul(log_recon)?
                    .sum(x.rank() - 1)?;
                ret.unsqueeze(ret.rank())
            })
            .collect::<Result<Vec<Tensor>>>()?;

        let k = llik_vec[0].rank();
        let llik = Tensor::cat(&llik_vec, k - 1)?.sum(k - 1)?;

        Ok((recon_vec, llik))
    }

    fn dim_obs(&self) -> &[usize] {
        &self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
