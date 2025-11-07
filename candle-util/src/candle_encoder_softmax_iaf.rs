use crate::candle_aux_layers::*;
use crate::candle_aux_linear::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Embedding, Linear, Module, ModuleT, VarBuilder};

pub struct LogSoftmaxIAFEncoder {
    n_features: usize,
    n_topics: usize,
    n_vocab: usize,
    n_modules: usize,
    feature_module: AggregateLinear,
    emb_x: Embedding,
    emb_logx: Embedding,
    fc: StackLayers<Linear>,
    bn_z: BatchNorm,
    iaf_layers: IAFLayers<Linear, StackLayers<Linear>>,
}

impl EncoderModuleT for LogSoftmaxIAFEncoder {
    fn forward_t(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_nk, kl) = self.latent_gaussian_with_kl(x_nd, x0_nd, train)?;
        Ok((ops::log_softmax(&z_nk, z_nk.rank() - 1)?, kl))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

impl LogSoftmaxIAFEncoder {
    pub fn num_feature_modules(&self) -> usize {
        self.n_modules
    }

    fn discretize_whitened_tensor(&self, x_nd: &Tensor) -> Result<Tensor> {
        let n_vocab = self.n_vocab as f64;
        let d = x_nd.rank();
        let min_val = x_nd.min_keepdim(d - 1)?;
        let max_val = x_nd.max_keepdim(d - 1)?;
        let div_val = ((max_val - &min_val)? + 1.0)?;

        let x_nd = x_nd.broadcast_sub(&min_val)?.broadcast_div(&div_val)?;

        Ok((x_nd * (n_vocab - 1.0))?
            .floor()?
            .to_dtype(candle_core::DType::U32)?)
    }

    fn composite_embedding(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let int_x_nd = self.discretize_whitened_tensor(x_nd)?;
        let logx_nd = (x_nd + 1.)?.log()?;
        let int_logx_nd = self.discretize_whitened_tensor(&logx_nd)?;
        self.emb_x.forward_t(&int_x_nd, train)? + self.emb_logx.forward_t(&int_logx_nd, train)?
    }

    fn modularized_composite_embedding(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        // This is important to make every data points to have the same scale
        let denom_n1 = x_nd.sum_keepdim(x_nd.rank() - 1)?;
        let x_nd = (x_nd.broadcast_div(&denom_n1)? * (self.n_features as f64))?;
        // group features into modules
        let x_nm = self.feature_module.forward(&x_nd)?;
        self.composite_embedding(&x_nm, train)
    }

    pub fn feature_module_membership(&self) -> Result<Tensor> {
        self.feature_module.membership()
    }

    fn preprocess_input(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        debug_assert_eq!(x_nd.dims().len(), 2);
        debug_assert!(x_nd.min_all()?.to_scalar::<f32>()? >= 0_f32);

        let emb_nmk = self.modularized_composite_embedding(x_nd, train)?;
        let last_dim = emb_nmk.rank();

        if let Some(x0_nd) = x0_nd {
            let emb0_nmk = self.modularized_composite_embedding(x0_nd, train)?;
            (emb_nmk - &emb0_nmk)?.mean(last_dim - 1)
        } else {
            emb_nmk.mean(last_dim - 1)
        }
    }

    ///
    /// Evaluate latent Gaussian parameters: mu and log_var
    /// * `z ~ (mu(x), exp(log_var(x)))`
    /// * `kl` divergence
    ///
    fn latent_gaussian_with_kl(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let xx_nm = self.preprocess_input(x_nd, x0_nd, train)?;
        let fc_nl = self.fc.forward_t(&xx_nm, train)?;
        let bn_nl = self.bn_z.forward_t(&fc_nl, train)?;
        self.iaf_layers.flow(&bn_nl, train)
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
    pub fn new(args: LogSoftmaxIAFEncoderArgs, vb: VarBuilder) -> Result<Self> {
        let bn_config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        debug_assert!(!args.layers.is_empty());

        let feature_module =
            aggregate_linear(args.n_features, args.n_modules, vb.pp("feature.module"))?;

        let emb_x = candle_nn::embedding(args.n_vocab, args.d_vocab_emb, vb.pp("nn.embed_x"))?;
        let emb_logx =
            candle_nn::embedding(args.n_vocab, args.d_vocab_emb, vb.pp("nn.embed_logx"))?;

        // (1) data -> fc
        let interm_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = args.n_modules;
        let fc_dim = *args.layers.last().unwrap();
        let fc = stack_relu_linear(in_dim, fc_dim, &interm_dims, vb.pp("nn.enc.fc"))?;

        let bn_z = candle_nn::batch_norm(fc_dim, bn_config, vb.pp("nn.enc.bn_z"))?;

        // (2) fc -> K
        let iaf_layers = iaf_stack_linear(
            fc_dim,
            args.n_topics,
            args.n_transforms,
            &interm_dims,
            vb.pp("iaf"),
        )?;

        Ok(Self {
            n_features: args.n_features,
            n_topics: args.n_topics,
            n_vocab: args.n_vocab,
            n_modules: args.n_modules,
            feature_module,
            emb_x,
            emb_logx,
            fc,
            bn_z,
            iaf_layers,
        })
    }
}

pub struct LogSoftmaxIAFEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub n_modules: usize,
    pub n_vocab: usize,
    pub d_vocab_emb: usize,
    pub layers: &'a [usize],
    pub n_transforms: usize,
}
