#![allow(dead_code)]

use crate::candle_aux_layers::StackLayers;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Linear, ModuleT, VarBuilder};

pub struct LogSoftmaxEncoder {
    n_features: usize,
    n_topics: usize,
    bn_x: BatchNorm,
    fc: StackLayers<Linear>,
    bn_z: BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
}

impl EncoderModuleT for LogSoftmaxEncoder {
    fn forward_t(&self, x_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_params(x_nd, train)?;
        let z_nk = self.reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;
        Ok((
            ops::log_softmax(&z_nk, 1)?,
            gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?,
        ))
    }
    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

impl LogSoftmaxEncoder {
    pub fn preprocess_input(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let log1p_nd = (x_nd + 1.)?.log()?;
        let x_n = log1p_nd.sum_keepdim(1)?;
        let log1p_nd = (log1p_nd.broadcast_div(&x_n)? * 1e4)?;
        self.bn_x.forward_t(&log1p_nd, train)
    }

    pub fn latent_params(&self, x_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let bn_nd = self.preprocess_input(x_nd, train)?;
        let fc_nl = self.fc.forward_t(&bn_nd, train)?;
        let bn_nl = self.bn_z.forward_t(&fc_nl, train)?;
        let z_mean_nk = self.z_mean.forward_t(&bn_nl, train)?;
        let z_lnvar_nk = self.z_lnvar.forward_t(&bn_nl, train)?.clamp(-8., 8.)?;
        Ok((z_mean_nk, z_lnvar_nk))
    }

    ///
    /// z = mu + sigma * eps
    /// where eps ~ N(0, 1)
    ///
    /// # Arguments
    /// * `z_mean` - mean of Gaussian distribution
    /// * `z_lnvar` - log variance of Gaussian distribution
    pub fn reparameterize(&self, z_mean: &Tensor, z_lnvar: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            let eps = Tensor::randn_like(&z_mean, 0., 1.)?;
            z_mean + (z_lnvar * 0.5)?.exp()? * eps
        } else {
            Ok(z_mean.clone())
        }
    }

    /// Will create a new non-negative encoder module
    /// with these variables:
    ///
    /// * `nn.enc.fc.{}.weight` where {} is the layer index
    /// * `nn.enc.z.mean.weight`
    /// * `nn.enc.z.lnvar.weight`
    pub fn new(
        n_features: usize,
        n_topics: usize,
        layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        let config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };
        let bn_x = candle_nn::batch_norm(n_features, config, vs.pp("nn.enc.bn_x"))?;

        // (1) data -> fc
        let mut fc = StackLayers::<Linear>::new();
        let mut prev_dim = n_features;
        for (j, &next_dim) in layers.iter().enumerate() {
            let _name = format!("nn.enc.fc.{}", j);
            fc.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Relu,
            );
            prev_dim = next_dim;
        }

        let bn_z = candle_nn::batch_norm(prev_dim, config, vs.pp("nn.enc.bn_z"))?;

        // (2) fc -> K
        let z_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.lnvar"))?;

        Ok(Self {
            n_features,
            n_topics,
            bn_x,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
        })
    }
}
