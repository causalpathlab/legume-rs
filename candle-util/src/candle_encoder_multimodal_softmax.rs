use crate::candle_aux_layers::*;
use crate::candle_aux_linear::*;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Linear, Module, ModuleT, VarBuilder};

pub struct LogSoftmaxMultimodalEncoder {
    n_features: Vec<usize>,
    n_topics: usize,
    feature_module: Vec<AggregateLinear>,
    fc: Vec<StackLayers<Linear>>,
    bn_z: Vec<BatchNorm>,
    z_mean: Vec<Linear>,
    z_lnvar: Vec<Linear>,
}

impl MultimodalEncoderModuleT for LogSoftmaxMultimodalEncoder {
    /// Returns (log_prob, kl) where log_prob is log-probabilities on the simplex
    fn forward_t(
        &self,
        x_nd_vec: &[Tensor],
        x0_nd_vec: &[Option<Tensor>],
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_nk, kl) = self.latent_gaussian_with_kl(x_nd_vec, x0_nd_vec, train)?;
        let log_prob = ops::log_softmax(&z_nk, z_nk.rank() - 1)?;
        Ok((log_prob, kl))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

impl LogSoftmaxMultimodalEncoder {
    fn preprocess_input(
        &self,
        x_nd_vec: &[Tensor],
        x0_nd_vec: &[Option<Tensor>],
        _train: bool,
    ) -> Result<Vec<Tensor>> {
        x_nd_vec
            .iter()
            .zip(x0_nd_vec)
            .enumerate()
            .map(|(m, (x_nd, x0_nd))| {
                let lx_nd = (x_nd + 1.)?.log()?;
                let denom_n1 = lx_nd.sum_keepdim(lx_nd.rank() - 1)?;
                let h_nm = self.feature_module[m]
                    .forward(&(lx_nd.broadcast_div(&denom_n1)? * (self.n_features[m] as f64))?)?;

                if let Some(x0_nd) = x0_nd {
                    let lx0_nd = (x0_nd + 1.)?.log()?;
                    let denom0 = lx0_nd.sum_keepdim(lx0_nd.rank() - 1)?;
                    let x0_nm = self.feature_module[m].forward(
                        &(lx0_nd.broadcast_div(&denom0)? * (self.n_features[m] as f64))?,
                    )?;
                    h_nm - x0_nm
                } else {
                    Ok(h_nm)
                }
            })
            .collect()
    }

    ///
    /// Evaluate latent Gaussian parameters: mu and log_var
    /// * `z ~ (mu(x), exp(log_var(x)))`
    /// * `kl` divergence
    ///
    fn latent_gaussian_with_kl(
        &self,
        x_nd_vec: &[Tensor],
        x0_nd_vec: &[Option<Tensor>],
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let hh_vec = self
            .preprocess_input(x_nd_vec, x0_nd_vec, train)?
            .into_iter()
            .enumerate()
            .map(|(m, xx_nm)| -> Result<Tensor> {
                let fc_nl = self.fc[m].forward_t(&xx_nm, train)?;
                self.bn_z[m].forward_t(&fc_nl, train)
            })
            .collect::<Result<Vec<_>>>()?;

        let z_kl_vec = hh_vec
            .into_iter()
            .enumerate()
            .map(|(m, hh)| -> Result<(Tensor, Tensor)> {
                let z_mean_nk = self.z_mean[m].forward(&hh)?;
                let z_lnvar_nk = self.z_lnvar[m].forward(&hh)?;
                let z = self.reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;
                let kl = gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?;

                Ok((z, kl))
            })
            .collect::<Result<Vec<_>>>()?;

        let z_aux_dim = z_kl_vec[0].0.rank();
        let kl_aux_dim = z_kl_vec[0].1.rank();

        let z = Tensor::cat(
            &z_kl_vec
                .iter()
                .map(|(z, _)| -> Result<_> { z.unsqueeze(z_aux_dim) })
                .collect::<Result<Vec<_>>>()?,
            z_aux_dim,
        )?
        .sum(z_aux_dim)?;

        let kl = Tensor::cat(
            &z_kl_vec
                .iter()
                .map(|(_, kl)| -> Result<_> { kl.unsqueeze(kl_aux_dim) })
                .collect::<Result<Vec<_>>>()?,
            kl_aux_dim,
        )?
        .sum(kl_aux_dim)?;

        Ok((z, kl))
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
    pub fn new(args: LogSoftmaxMultimodalEncoderArgs, vb: VarBuilder) -> Result<Self> {
        debug_assert!(!args.layers.is_empty());

        let bn_config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        let n_features = args.n_features.clone();
        let n_modalities = n_features.len();

        let feature_module = n_features
            .iter()
            .enumerate()
            .map(|(i, &in_dim)| {
                aggregate_linear_hard(
                    in_dim,
                    args.n_modules,
                    vb.pp(format!("feature.module_{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // (1) data -> fc
        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = args.n_modules;
        let out_dim = *args.layers.last().unwrap();

        let fc = (0..n_modalities)
            .map(|i| {
                stack_relu_linear(in_dim, out_dim, &fc_dims, vb.pp(format!("nn.enc.fc_{}", i)))
            })
            .collect::<Result<Vec<_>>>()?;

        let bn_z = (0..n_modalities)
            .map(|i| candle_nn::batch_norm(out_dim, bn_config, vb.pp(format!("nn.enc.bn_z_{}", i))))
            .collect::<Result<Vec<_>>>()?;

        // (2) fc -> K
        let z_mean = (0..n_modalities)
            .map(|i| {
                candle_nn::linear(
                    out_dim,
                    args.n_topics,
                    vb.pp(format!("nn.enc.z.mean_{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let z_lnvar = (0..n_modalities)
            .map(|i| {
                candle_nn::linear(
                    out_dim,
                    args.n_topics,
                    vb.pp(format!("nn.enc.z.lnvar_{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            n_features: args.n_features,
            n_topics: args.n_topics,
            feature_module,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
        })
    }
}

pub struct LogSoftmaxMultimodalEncoderArgs<'a> {
    pub n_features: Vec<usize>,
    pub n_topics: usize,
    pub n_modules: usize,
    pub layers: &'a [usize],
}
