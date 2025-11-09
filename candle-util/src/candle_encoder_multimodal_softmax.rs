use crate::candle_aux_layers::*;
use crate::candle_aux_linear::*;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Embedding, Linear, Module, ModuleT, VarBuilder};

pub struct LogSoftmaxMultimodalEncoder {
    n_features: Vec<usize>,
    n_topics: usize,
    n_vocab: usize,
    // n_modules: usize,
    feature_module: Vec<AggregateLinear>,
    emb_x: Vec<Embedding>,
    emb_logx: Vec<Embedding>,
    fc: Vec<StackLayers<Linear>>,
    bn_z: Vec<BatchNorm>,
    z_mean: Vec<Linear>,
    z_lnvar: Vec<Linear>,
}

impl MultimodalEncoderModuleT for LogSoftmaxMultimodalEncoder {
    fn forward_t(
        &self,
        x_nd_vec: &[Tensor],
        x0_nd_vec: &[Option<Tensor>],
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_nk, kl) = self.latent_gaussian_with_kl(x_nd_vec, x0_nd_vec, train)?;
        Ok((ops::log_softmax(&z_nk, z_nk.rank() - 1)?, kl))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

impl LogSoftmaxMultimodalEncoder {
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

    fn modularized_composite_embedding(
        &self,
        x_nd: &Tensor,
        m: usize,
        train: bool,
    ) -> Result<Tensor> {
        // This is important to make every data points to have the same scale
        let denom_n1 = x_nd.sum_keepdim(x_nd.rank() - 1)?;
        let x_nd = (x_nd.broadcast_div(&denom_n1)? * (self.n_features[m] as f64))?;
        // group features into modules
        let x_nm = self.feature_module[m].forward(&x_nd)?;
        // combine two types of embedding results
        let int_x_nm = self.discretize_whitened_tensor(&x_nm)?;
        let logx_nm = (x_nm + 1.)?.log()?;
        let int_logx_nm = self.discretize_whitened_tensor(&logx_nm)?;
        self.emb_x[m].forward_t(&int_x_nm, train)? + self.emb_logx[m].forward_t(&int_logx_nm, train)
    }

    fn preprocess_input(
        &self,
        x_nd_vec: &[Tensor],
        x0_nd_vec: &[Option<Tensor>],
        train: bool,
    ) -> Result<Vec<Tensor>> {
        x_nd_vec
            .iter()
            .zip(x0_nd_vec)
            .enumerate()
            .map(|(m, (x_nd, x0_nd))| {
                let emb_nmk = self.modularized_composite_embedding(x_nd, m, train)?;
                let last_dim = emb_nmk.rank();
                if let Some(x0_nd) = x0_nd {
                    let emb0_nmk = self.modularized_composite_embedding(x0_nd, m, train)?;
                    (emb_nmk - &emb0_nmk)?.mean(last_dim - 1)
                } else {
                    emb_nmk.mean(last_dim - 1)
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
                aggregate_linear(
                    in_dim,
                    args.n_modules,
                    vb.pp(format!("feature.module_{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let emb_x = (0..n_modalities)
            .into_iter()
            .map(|i| {
                candle_nn::embedding(
                    args.n_vocab,
                    args.d_vocab_emb,
                    vb.pp(format!("nn.embed_x_{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let emb_logx = (0..n_modalities)
            .into_iter()
            .map(|i| {
                candle_nn::embedding(
                    args.n_vocab,
                    args.d_vocab_emb,
                    vb.pp(format!("nn.embed_logx_{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // (1) data -> fc
        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = args.n_modules;
        let out_dim = *args.layers.last().unwrap();

        let fc = (0..n_modalities)
            .into_iter()
            .map(|i| {
                stack_relu_linear(in_dim, out_dim, &fc_dims, vb.pp(format!("nn.enc.fc_{}", i)))
            })
            .collect::<Result<Vec<_>>>()?;

        let bn_z = (0..n_modalities)
            .into_iter()
            .map(|i| candle_nn::batch_norm(out_dim, bn_config, vb.pp(format!("nn.enc.bn_z_{}", i))))
            .collect::<Result<Vec<_>>>()?;

        // (2) fc -> K
        let z_mean = (0..n_modalities)
            .into_iter()
            .map(|i| {
                candle_nn::linear(
                    out_dim,
                    args.n_topics,
                    vb.pp(format!("nn.enc.z.mean_{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let z_lnvar = (0..n_modalities)
            .into_iter()
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
            n_vocab: args.n_vocab,
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

pub struct LogSoftmaxMultimodalEncoderArgs<'a> {
    pub n_features: Vec<usize>,
    pub n_topics: usize,
    pub n_modules: usize,
    pub n_vocab: usize,
    pub d_vocab_emb: usize,
    pub layers: &'a [usize],
}
