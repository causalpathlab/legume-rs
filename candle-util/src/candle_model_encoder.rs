#![allow(dead_code)]

use crate::candle_aux_layers::StackLayers;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Embedding, Linear, ModuleT, VarBuilder};

pub struct LogSoftmaxEncoder {
    n_features: usize,
    n_topics: usize,
    d_emb: usize,
    emb_x: Embedding,
    fc: StackLayers<Linear>,
    bn_z: BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
}

impl EncoderModuleT for LogSoftmaxEncoder {
    fn forward_t(&self, x_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_gaussian_params(x_nd, train)?;
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
    ///
    /// embedding matrix
    pub fn get_embedding_matrix(&self) -> Tensor {
        self.emb_x.embeddings().clone()
    }

    /// Preprocess: x -> embedding
    pub fn preprocess_input(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        debug_assert_eq!(x_nd.dims().len(), 2);
        debug_assert!(x_nd.min_all()?.to_scalar::<f32>()? >= 0_f32);
        // let maxval = x_nd.max_all()?.to_scalar::<f32>()?;
        // let x_nd = x_nd.clamp(0_f32, maxval)?;

        // 1. Discretize data after log1p transformation
        let d_input = self.n_features as f64;

        let log1p_nd = (x_nd + 1.)?.log()?;
        let x_n = log1p_nd.sum_keepdim(1)?;

        let log1p_nd =
            (log1p_nd.broadcast_div(&x_n)? * d_input)?.to_dtype(candle_core::DType::U32)?;

        // 2. log1p intensity embedding: n x d -> n x d x d
        let emb_ndd = self.emb_x.forward_t(&log1p_nd, train)?;
        emb_ndd.sum(emb_ndd.dims().len() - 1)
    }

    ///
    /// Evaluate latent Gaussian parameters: mu and log_var
    /// z ~ (mu(x), log_var(x))
    pub fn latent_gaussian_params(&self, x_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let min_mean = -(self.n_features as f64).sqrt(); // stabilize
        let max_mean = (self.n_features as f64).sqrt(); // mean
        let min_lv = -8.; // and log variance
        let max_lv = 8.; //

        let bn_nd = self.preprocess_input(x_nd, train)?;
        let fc_nl = self.fc.forward_t(&bn_nd, train)?;
        let bn_nl = self.bn_z.forward_t(&fc_nl, train)?;
        let z_mean_nk = self
            .z_mean
            .forward_t(&bn_nl, train)?
            .clamp(min_mean, max_mean)?;
        let z_lnvar_nk = self
            .z_lnvar
            .forward_t(&bn_nl, train)?
            .clamp(min_lv, max_lv)?;
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
        let bn_config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        debug_assert!(layers.len() > 0);

        let d_emb = layers[0];

        let emb_x = candle_nn::embedding(n_features, d_emb, vs.pp("nn.embed_x"))?;

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

        let bn_z = candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn_z"))?;

        // (2) fc -> K
        let z_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.lnvar"))?;

        Ok(Self {
            n_features,
            n_topics,
            d_emb,
            emb_x,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
        })
    }
}
