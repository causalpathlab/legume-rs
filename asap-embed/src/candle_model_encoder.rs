#![allow(dead_code)]

use crate::candle_aux_layers::StackLayers;
use crate::candle_loss_functions::gaussian_kl_loss;
use candle_core::{Result, Tensor};
use candle_nn::{ops, sequential, BatchNorm, Linear, ModuleT, Sequential, VarBuilder};

pub trait EncoderModuleT {
    /// An encoder that spits out two results (latent inference, KL loss)
    ///
    /// # Arguments
    /// * `x_nd` - input data (n x d)
    /// * `train` - whether to use dropout/batchnorm or not
    ///
    /// # Returns `(z_nk, kl_loss_n)`
    /// * `z_nk` - latent inference (n x k)
    /// * `kl_loss_n` - KL loss (n x 1)
    fn forward_t(&self, x_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)>;

    /// An encoder that spits out two results (latent inference, KL loss)
    ///
    /// # Arguments
    /// * `x1_nd` - foreground input data (n x d)
    /// * `x0_nd` - background input data (n x d)
    /// * `train` - whether to use dropout/batchnorm or not
    ///
    /// # Returns `(z_nk, kl_loss_n)`
    /// * `z_nk` - latent inference (n x k)
    /// * `kl_loss_n` - KL loss (n x 1)
    fn adjusted_forward_t(
        &self,
        x1_nd: &Tensor,
        x0_nd: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor)>;

    fn dim_obs(&self) -> usize;
    fn dim_latent(&self) -> usize;
}

pub struct NonNegEncoder {
    n_features: usize,
    n_topics: usize,
    bn: BatchNorm,
    fc: StackLayers<Linear>,
    z_mean: Linear,
    z_lnvar: Linear,
}

impl EncoderModuleT for NonNegEncoder {
    fn adjusted_forward_t(
        &self,
        x1_nd: &Tensor,
        _x0: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        todo!("z1 - z0");
    }

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

impl NonNegEncoder {
    pub fn batch_norm_input(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let eps = 1e-4;
        let x_log1p_nd = (x_nd + 1.)?.log()?;
        let x_norm_nd = x_log1p_nd.broadcast_div(&(x_log1p_nd.sum(0)? + eps)?)?;
        self.bn.forward_t(&x_norm_nd, train)
    }

    pub fn latent_params(&self, x_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let batch_nd = self.batch_norm_input(x_nd, train)?;
        let x_nl = self.fc.forward_t(&batch_nd, train)?;
        let z_mean_nk = self.z_mean.forward_t(&x_nl, train)?;
        let z_lnvar_nk = self.z_lnvar.forward_t(&x_nl, train)?;
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
        // (0) batch norm
        let config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: false,
            momentum: 0.1,
        };
        let bn = candle_nn::batch_norm(n_features, config, vs.pp("nn.enc.bn"))?;

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

        // (2) fc -> K
        let z_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.lnvar"))?;

        Ok(Self {
            n_features,
            n_topics,
            bn,
            fc,
            z_mean,
            z_lnvar,
        })
    }
}
