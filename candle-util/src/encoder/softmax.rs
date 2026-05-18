use crate::nn::layers::*;
use crate::nn::batch_norm;
use crate::loss::{gaussian_kl_loss, gaussian_reparameterize};
use crate::traits::model::*;
use crate::candle_value_transform::anscombe_residual;
use candle_core::{Device, Result, Tensor};
use candle_nn::{ops, Linear, ModuleT, VarBuilder, VarMap};

pub struct LogSoftmaxEncoder {
    n_topics: usize,
    fc: StackLayers<Linear>,
    bn_z: batch_norm::BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
    /// Per-feature mean rate `μ_d` as a `[1, D]` non-trainable tensor.
    /// Composed with the per-cell batch null inside `anscombe_residual`
    /// as a joint multiplicative count-rate divisor — same correction
    /// the indexed encoder applies via gather.
    feature_mean: Option<Tensor>,
}

impl EncoderModuleT for LogSoftmaxEncoder {
    fn forward_t(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_gaussian_params(x_nd, x0_nd, train)?;
        let z_nk = gaussian_reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;

        let log_prob = ops::log_softmax(&z_nk, 1)?;

        Ok((log_prob, gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

impl LogSoftmaxEncoder {
    fn preprocess_input(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        _train: bool,
    ) -> Result<Tensor> {
        anscombe_residual(x_nd, x0_nd, self.feature_mean.as_ref())
    }

    ///
    /// Evaluate latent Gaussian parameters: mu and log_var
    /// z ~ (mu(x), log_var(x))
    pub fn latent_gaussian_params(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let clamp_lo = -8.;
        let clamp_hi = 8.;

        let xx_nd = self.preprocess_input(x_nd, x0_nd, train)?;
        let fc_nl = self.fc.forward_t(&xx_nd, train)?;
        let bn_nl = self.bn_z.forward_t(&fc_nl, train)?;

        let z_mean_nk = self
            .z_mean
            .forward_t(&bn_nl, train)?
            .clamp(clamp_lo, clamp_hi)?;
        let z_lnvar_nk = self
            .z_lnvar
            .forward_t(&bn_nl, train)?
            .clamp(clamp_lo, clamp_hi)?;

        Ok((z_mean_nk, z_lnvar_nk))
    }

    /// Will create a new non-negative encoder module
    ///
    /// # Arguments
    /// * `args` - encoder arguments
    /// * `vb` - variable builder
    pub fn new(args: LogSoftmaxEncoderArgs, varmap: &VarMap, vb: VarBuilder) -> Result<Self> {
        let bn_config = batch_norm::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        debug_assert!(!args.layers.is_empty());

        // data [N, D] -> fc stack -> final_hidden
        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = args.n_features;
        let out_dim = *args.layers.last().unwrap();
        let fc = stack_relu_linear(in_dim, out_dim, &fc_dims, vb.pp("nn.enc.fc"))?;

        let bn_z = batch_norm::batch_norm(out_dim, bn_config, varmap, vb.pp("nn.enc.bn_z"))?;

        // fc -> K
        let z_mean = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.lnvar"))?;

        let feature_mean = match args.feature_mean {
            Some(slice) => {
                debug_assert_eq!(
                    slice.len(),
                    args.n_features,
                    "feature_mean length must equal n_features"
                );
                Some(host_to_row_tensor(slice, vb.device())?)
            }
            None => None,
        };

        Ok(Self {
            n_topics: args.n_topics,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
            feature_mean,
        })
    }
}

/// Move a host `[D]` slice to a `[1, D]` device tensor for row-broadcast.
fn host_to_row_tensor(slice: &[f32], dev: &Device) -> Result<Tensor> {
    let d = slice.len();
    Tensor::from_slice(slice, (1, d), dev)
}

pub struct LogSoftmaxEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub layers: &'a [usize],
    /// Optional per-feature mean rate `μ_d` (length = n_features).
    /// Composed with the per-cell batch null inside `anscombe_residual`
    /// as a multiplicative count-rate divisor before Anscombe — gives
    /// the dense path the same gene-mean correction the indexed
    /// encoder gets via `values_mean`.
    pub feature_mean: Option<&'a [f32]>,
}
