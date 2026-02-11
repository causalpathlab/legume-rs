use crate::candle_aux_layers::*;
use crate::candle_aux_linear::*;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Linear, Module, ModuleT, VarBuilder};

pub struct LogSoftmaxEncoder {
    n_features: usize,
    n_topics: usize,
    n_modules: usize,
    use_sparsemax: bool,
    feature_module: AggregateLinear,
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

        let prob = if self.use_sparsemax {
            sparsemax(&z_nk)?
        } else {
            ops::log_softmax(&z_nk, 1)?.exp()?
        };

        Ok((prob, gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

impl LogSoftmaxEncoder {
    pub fn num_feature_modules(&self) -> usize {
        self.n_modules
    }

    pub fn set_use_sparsemax(&mut self, use_sparsemax: bool) {
        self.use_sparsemax = use_sparsemax;
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
    /// z ~ (mu(x), log_var(x))
    fn latent_gaussian_params(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let min_lv = -10.; // clamp log variance
        let max_lv = 10.; //

        let xx_nm = self.preprocess_input(x_nd, x0_nd, train)?;
        let fc_nl = self.fc.forward_t(&xx_nm, train)?;
        let bn_nl = self.bn_z.forward_t(&fc_nl, train)?;

        let z_mean_nk = self.z_mean.forward_t(&bn_nl, train)?;
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
    ///
    /// # Arguments
    /// * `args` - encoder arguments
    /// * `vb` - variable builder
    pub fn new(args: LogSoftmaxEncoderArgs, vb: VarBuilder) -> Result<Self> {
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
        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = args.n_modules;
        let out_dim = *args.layers.last().unwrap();
        let fc = stack_relu_linear(in_dim, out_dim, &fc_dims, vb.pp("nn.enc.fc"))?;

        let bn_z = candle_nn::batch_norm(out_dim, bn_config, vb.pp("nn.enc.bn_z"))?;

        // (2) fc -> K
        let z_mean = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.lnvar"))?;

        Ok(Self {
            n_features: args.n_features,
            n_topics: args.n_topics,
            n_modules: args.n_modules,
            use_sparsemax: args.use_sparsemax,
            feature_module,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
        })
    }
}

pub struct LogSoftmaxEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub n_modules: usize,
    pub layers: &'a [usize],
    pub use_sparsemax: bool,
}
