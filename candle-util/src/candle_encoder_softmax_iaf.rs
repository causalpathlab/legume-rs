use crate::candle_aux_layers::*;
use crate::candle_aux_linear::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Linear, Module, ModuleT, VarBuilder};

pub struct LogSoftmaxIAFEncoder {
    n_features: usize,
    n_topics: usize,
    n_modules: usize,
    use_sparsemax: bool,
    feature_module: AggregateLinear,
    fc: StackLayers<Linear>,
    bn_z: BatchNorm,
    iaf_layers: IAFLayers<Linear, StackLayers<Linear>>,
}

impl EncoderModuleT for LogSoftmaxIAFEncoder {
    /// Returns (prob, kl) where prob is on the probability simplex
    fn forward_t(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_nk, kl) = self.latent_gaussian_with_kl(x_nd, x0_nd, train)?;
        let prob = if self.use_sparsemax {
            sparsemax(&z_nk)?
        } else {
            ops::softmax(&z_nk, z_nk.rank() - 1)?
        };
        Ok((prob, kl))
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
        debug_assert_eq!(x_nd.dims().len(), 2);
        debug_assert!(x_nd.min_all()?.to_scalar::<f32>()? >= 0_f32);

        let lx_nd = (x_nd + 1.)?.log()?;
        let denom_n1 = lx_nd.sum_keepdim(lx_nd.rank() - 1)?;
        let h_nm = self
            .feature_module
            .forward(&(lx_nd.broadcast_div(&denom_n1)? * (self.n_features as f64))?)?;

        match x0_nd {
            Some(x0) => {
                let lx0_nd = (x0 + 1.)?.log()?;
                let denom0 = lx0_nd.sum_keepdim(lx0_nd.rank() - 1)?;
                let x0_nm = self
                    .feature_module
                    .forward(&(lx0_nd.broadcast_div(&denom0)? * (self.n_features as f64))?)?;
                h_nm - x0_nm
            }
            None => Ok(h_nm),
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

        // Sparsemax and IAF are mutually exclusive
        debug_assert!(
            !(args.use_sparsemax && args.n_transforms > 0),
            "Sparsemax and IAF are mutually exclusive. Use sparsemax OR iaf_trans, not both."
        );

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

        Ok(Self {
            n_features: args.n_features,
            n_topics: args.n_topics,
            n_modules: args.n_modules,
            use_sparsemax: args.use_sparsemax,
            feature_module,
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
    pub layers: &'a [usize],
    pub use_sparsemax: bool,
    pub n_transforms: usize,
}
