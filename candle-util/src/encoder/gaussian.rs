//! Dense **Gaussian (scVI-style) encoder**.
//!
//! Structurally identical to [`super::softmax::LogSoftmaxEncoder`] — same
//! value transform, FC stack, batch-norm, and `z_mean`/`z_lnvar` heads — but
//! [`EncoderModuleT::forward_t`] returns the **raw reparameterized Gaussian
//! latent `z`** instead of `log_softmax(z)`. The latent is unconstrained
//! continuous factors, not topic proportions.
//!
//! Because `z` is not on the simplex, it must be paired with a decoder that
//! maps `z → gene distribution` (scVI: `softmax_d(z·W) → μ = library·π`, NB),
//! NOT the simplex topic decoders (whose `logsumexp_k(log θ_k + log β_kd)`
//! mixture assumes `θ` sums to 1).

use crate::loss::{gaussian_kl_loss, gaussian_reparameterize};
use crate::nn::batch_norm;
use crate::nn::layers::*;
use crate::traits::model::*;
use crate::value_transform::anscombe_residual;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, ModuleT, VarBuilder, VarMap};

pub struct GaussianEncoder {
    n_latent: usize,
    fc: StackLayers<Linear>,
    bn_z: batch_norm::BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
    /// Per-feature mean rate `μ_d` as a `[1, D]` non-trainable tensor; the
    /// divisive gene-mean correction inside `anscombe_residual`.
    feature_mean: Option<Tensor>,
}

impl EncoderModuleT for GaussianEncoder {
    /// Returns `(z, KL)` where `z` is the reparameterized **Gaussian** latent
    /// (no simplex projection). At eval (`train = false`) `z` is the posterior
    /// mean `z_mean`.
    fn forward_t(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_gaussian_params(x_nd, x0_nd, train)?;
        let z_nk = gaussian_reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;
        Ok((z_nk, gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?))
    }

    fn dim_latent(&self) -> usize {
        self.n_latent
    }
}

impl GaussianEncoder {
    fn preprocess_input(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        _train: bool,
    ) -> Result<Tensor> {
        anscombe_residual(x_nd, x0_nd, self.feature_mean.as_ref())
    }

    /// Latent Gaussian parameters `(z_mean, z_lnvar)` — identical to the
    /// log-softmax encoder; only the downstream projection differs.
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

    pub fn new(args: GaussianEncoderArgs, varmap: &VarMap, vb: VarBuilder) -> Result<Self> {
        let bn_config = batch_norm::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        debug_assert!(!args.layers.is_empty());

        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = args.n_features;
        let out_dim = *args.layers.last().unwrap();
        let fc = stack_relu_linear(in_dim, out_dim, &fc_dims, vb.pp("nn.enc.fc"))?;

        let bn_z = batch_norm::batch_norm(out_dim, bn_config, varmap, vb.pp("nn.enc.bn_z"))?;

        let z_mean = candle_nn::linear(out_dim, args.n_latent, vb.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(out_dim, args.n_latent, vb.pp("nn.enc.z.lnvar"))?;

        let feature_mean = match args.feature_mean {
            Some(slice) => {
                debug_assert_eq!(
                    slice.len(),
                    args.n_features,
                    "feature_mean length must equal n_features"
                );
                // `[1, D]` row tensor for broadcast inside `anscombe_residual`.
                Some(Tensor::from_slice(slice, (1, slice.len()), vb.device())?)
            }
            None => None,
        };

        Ok(Self {
            n_latent: args.n_latent,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
            feature_mean,
        })
    }
}

pub struct GaussianEncoderArgs<'a> {
    pub n_features: usize,
    /// Continuous latent dimension (number of factors).
    pub n_latent: usize,
    pub layers: &'a [usize],
    /// Optional per-feature mean rate `μ_d` (length = n_features).
    pub feature_mean: Option<&'a [f32]>,
}
