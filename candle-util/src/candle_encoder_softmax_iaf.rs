use crate::candle_aux_layers::*;
use crate::candle_aux_linear::*;
use crate::candle_model_traits::*;
use crate::candle_value_transform::anscombe_residual;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Linear, Module, ModuleT, VarBuilder};

pub struct LogSoftmaxIAFEncoder {
    n_topics: usize,
    n_modules: usize,
    feature_module: AggregateLinear,
    fc: StackLayers<Linear>,
    bn_z: BatchNorm,
    iaf_layers: IAFLayers<Linear, StackLayers<Linear>>,
    /// `[1, D]` non-trainable mean rate; broadcast inside
    /// `anscombe_residual` as a multiplicative count-rate divisor.
    feature_mean: Option<Tensor>,
}

impl EncoderModuleT for LogSoftmaxIAFEncoder {
    /// Returns (log_prob, kl) where log_prob is log-probabilities on the simplex
    fn forward_t(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_nk, kl) = self.latent_gaussian_with_kl(x_nd, x0_nd, train)?;
        let log_prob = ops::log_softmax(&z_nk, z_nk.rank() - 1)?;
        Ok((log_prob, kl))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

impl LogSoftmaxIAFEncoder {
    pub fn num_feature_modules(&self) -> usize {
        self.n_modules
    }

    pub fn feature_module_membership(&self) -> Result<Tensor> {
        self.feature_module.membership()
    }

    fn preprocess_input(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        _train: bool,
    ) -> Result<Tensor> {
        let r_nd = anscombe_residual(x_nd, x0_nd, self.feature_mean.as_ref())?;
        self.feature_module.forward(&r_nd)
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
    ///
    /// # Arguments
    /// * `args` - encoder arguments
    /// * `vb` - variable builder
    pub fn new(args: LogSoftmaxIAFEncoderArgs, vb: VarBuilder) -> Result<Self> {
        let bn_config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        debug_assert!(!args.layers.is_empty());

        let feature_module =
            aggregate_linear_hard(args.n_features, args.n_modules, vb.pp("feature.module"))?;

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

        let feature_mean = match args.feature_mean {
            Some(slice) => {
                debug_assert_eq!(slice.len(), args.n_features);
                Some(Tensor::from_slice(slice, (1, args.n_features), vb.device())?)
            }
            None => None,
        };

        Ok(Self {
            n_topics: args.n_topics,
            n_modules: args.n_modules,
            feature_module,
            fc,
            bn_z,
            iaf_layers,
            feature_mean,
        })
    }
}

pub struct LogSoftmaxIAFEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub n_modules: usize,
    pub layers: &'a [usize],
    pub n_transforms: usize,
    /// Optional per-feature mean rate `μ_d` (length = n_features).
    pub feature_mean: Option<&'a [f32]>,
}
