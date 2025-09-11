use crate::candle_aux_layers::*;
use crate::candle_aux_linear::*;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{BatchNorm, Embedding, Linear, Module, ModuleT, VarBuilder, ops};

pub struct LogSoftmaxEncoder {
    n_features: usize,
    n_topics: usize,
    n_vocab: usize,
    feature_module: AggregateLinear,
    emb_x: Embedding,
    emb_logx: Embedding,
    fc: StackLayers<Linear>,
    bn_z: BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
}

impl EncoderModuleT for LogSoftmaxEncoder {
    fn forward_t(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_gaussian_params(x_nd, x0_nd, train)?;
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
    fn discretize_whitened_tensor(&self, x_nd: &Tensor) -> Result<Tensor> {
        use candle_core::DType::U32;

        let eps = 1e-4;
        let n_vocab = self.n_vocab as f64;
        let d = x_nd.dims().len();
        let min_val = x_nd.min_keepdim(d - 1)?;
        let max_val = x_nd.max_keepdim(d - 1)?;
        let div_val = ((max_val - &min_val)? + eps)?;

        let x_nd = x_nd.broadcast_sub(&min_val)?.broadcast_div(&div_val)?;

        (x_nd * n_vocab)?.floor()?.to_dtype(U32)
    }

    fn composite_embedding(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let int_x_nd = self.discretize_whitened_tensor(x_nd)?;
        let logx_nd = (x_nd + 1.)?.log()?;
        let int_logx_nd = self.discretize_whitened_tensor(&logx_nd)?;

        self.emb_x.forward_t(&int_x_nd, train)? + self.emb_logx.forward_t(&int_logx_nd, train)?
    }

    fn modularized_embedding(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let x_nm = self.feature_module.forward(x_nd)?;
        self.composite_embedding(&x_nm, train)
    }

    fn preprocess_input(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        debug_assert_eq!(x_nd.dims().len(), 2);
        debug_assert!(x_nd.min_all()?.to_scalar::<f32>()? >= 0_f32);

        let emb_ndd = self.modularized_embedding(x_nd, train)?;
        let last_dim = emb_ndd.dims().len();

        if let Some(x0_nd) = x0_nd {
            let emb0_ndd = self.modularized_embedding(x0_nd, train)?;
            (emb_ndd - &emb0_ndd)?.sum(last_dim - 1)
        } else {
            emb_ndd.sum(last_dim - 1)
        }
    }

    ///
    /// Evaluate latent Gaussian parameters: mu and log_var
    /// z ~ (mu(x), log_var(x))
    fn latent_gaussian_params(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let min_mean = -(self.n_features as f64).sqrt(); // stabilize
        let max_mean = (self.n_features as f64).sqrt(); // mean
        let min_lv = -8.; // and log variance
        let max_lv = 8.; //

        let bn_nd = self.preprocess_input(x_nd, x0_nd, train)?;
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
    fn reparameterize(&self, z_mean: &Tensor, z_lnvar: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            let eps = Tensor::randn_like(z_mean, 0., 1.)?;
            z_mean + (z_lnvar * 0.5)?.exp()? * eps
        } else {
            Ok(z_mean.clone())
        }
    }

    /// Will create a new non-negative encoder module
    /// with these variables:
    ///
    /// * `nn.embed_x` for intensity embedding
    /// * `nn.enc.fc.{}.weight` where {} is the layer index
    /// * `nn.enc.z.mean.weight`
    /// * `nn.enc.z.lnvar.weight`
    ///
    /// # Arguments
    /// * `n_features` - the number of features
    /// * `n_topics` - the number of topics (latent factors)
    /// * `n_vocab` - the size of intensity vocabulary
    /// * `d_emb` - vocabulary embedding dim
    /// * `layers` - fully connected layers, each with the dim
    /// * `vs` - variable builder
    pub fn new(
        n_features: usize,
        n_topics: usize,
        n_vocab: usize,
        d_emb: usize,
        layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        let bn_config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        debug_assert!(!layers.is_empty());

        let n_modules = layers[0];
        let feature_module = aggregate_linear(n_features, n_modules, vs.pp("feature.module"))?;

        let emb_x = candle_nn::embedding(n_vocab, d_emb, vs.pp("nn.embed_x"))?;
        let emb_logx = candle_nn::embedding(n_vocab, d_emb, vs.pp("nn.embed_logx"))?;

        // (1) data -> fc
        let mut fc = StackLayers::<Linear>::new();
        let mut prev_dim = n_modules;
        // let mut prev_dim = n_features;
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
            n_vocab,
            feature_module,
            emb_x,
            emb_logx,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
        })
    }
}
