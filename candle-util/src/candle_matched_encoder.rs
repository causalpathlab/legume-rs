use crate::candle_aux_layers::*;
use crate::candle_aux_linear::*;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_model_traits::*;

use candle_core::{Result, Tensor};
use candle_nn::linear;
use candle_nn::{ops, BatchNorm, Embedding, Linear, Module, ModuleT, VarBuilder};

pub struct MatchedEncoder {
    n_features: usize,
    n_topics: usize,
    n_vocab: usize,
    feature_module: AggregateLinear,
    emb_x: Embedding,
    emb_logx: Embedding,
    coord_proj: Linear,
    fc_expr: StackLayers<Linear>,
    fc_coord: StackLayers<Linear>,
    bn_expr: BatchNorm,
    bn_coord: BatchNorm,
    z_expr_mean: Linear,
    z_expr_lnvar: Linear,
    z_coord_mean: Linear,
    z_coord_lnvar: Linear,
}

impl MatchedEncoderModuleT for MatchedEncoder {
    ///
    /// Returns:
    /// 1. latent z
    /// 2. kl-divergence
    ///
    fn forward_t(&self, data: MatchedEncoderData, train: bool) -> Result<MatchedEncoderLatent> {
        let latent_states_with_coordinates = |value_nm: &Tensor,
                                              coord_nc: Option<&Tensor>|
         -> Result<(Tensor, Tensor)> {
            if let Some(coord_nc) = coord_nc {
                let (z_expr_mean_nk, z_expr_lnvar_nk) =
                    self.latent_gaussian_params_expr(value_nm, train)?;

                let proj_nm = self.coord_proj.forward_t(coord_nc, train)?;
                let (z_coord_mean_nk, z_coord_lnvar_nk) =
                    self.latent_gaussian_params_coord(&proj_nm, train)?;

                let z = self
                    .reparameterize(&z_expr_mean_nk, &z_expr_lnvar_nk, train)?
                    .add(&self.reparameterize(&z_coord_mean_nk, &z_coord_lnvar_nk, train)?)?;
                let kl = gaussian_kl_loss(&z_expr_mean_nk, &z_expr_lnvar_nk)?
                    .add(&gaussian_kl_loss(&z_coord_mean_nk, &z_coord_lnvar_nk)?)?;

                Ok((z, kl))
            } else {
                let (z_mean_nk, z_lnvar_nk) = self.latent_gaussian_params_expr(value_nm, train)?;

                let z_nk = self.reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;
                let kl = gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?;
                Ok((z_nk, kl))
            }
        };

        let left_nmk = self.modularized_value_embedding(data.left, train)?;
        let right_nmk = self.modularized_value_embedding(data.right, train)?;
        let left_nm = left_nmk.sum(left_nmk.rank() - 1)?;
        let right_nm = right_nmk.sum(right_nmk.rank() - 1)?;

        let (z_left_nk, kl_left) = latent_states_with_coordinates(&left_nm, data.aux_left)?;
        let (z_right_nk, kl_right) = latent_states_with_coordinates(&right_nm, data.aux_right)?;

        let kl_div = kl_left.add(&kl_right)?;
        let logits_theta = ops::log_softmax(&z_left_nk.add(&z_right_nk)?, z_left_nk.rank() - 1)?;

        Ok(MatchedEncoderLatent {
            logits_theta,
            kl_div,
        })
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }
}

pub struct MatchedEncoderArg<'a> {
    pub dim_feature: usize,
    pub dim_latent: usize,
    pub dim_coord: usize,
    pub n_vocab_emb: usize,
    pub dim_emb: usize,
    pub num_feature_modules: usize,
    pub layers: &'a [usize],
}

impl MatchedEncoder {
    /// will first aggregate `d` features into `m` modules
    /// then embed each element in `k` different ways
    /// * input: `x_nd` - `n x d` tensor
    /// * output: `emb_nmk` - `n x m x k` embedding results
    fn modularized_value_embedding(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let x_nm = self.feature_module.forward(x_nd)?;
        Ok(self.value_embedding(&x_nm, train)?.detach())
    }

    fn value_embedding(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let logx = self.discretize_whitened_tensor(&(x + 1e-4)?.log()?)?;
        let x = self.discretize_whitened_tensor(x)?;
        self.emb_x
            .forward_t(&x, train)?
            .add(&self.emb_logx.forward_t(&logx, train)?)
    }

    fn discretize_whitened_tensor(&self, x_nd: &Tensor) -> Result<Tensor> {
        let eps = 1e-4;
        let n_vocab = self.n_vocab as f64;
        let d = x_nd.rank();

        let min_val = x_nd.min_keepdim(d - 1)?;
        let max_val = x_nd.max_keepdim(d - 1)?;
        let div_val = ((max_val - &min_val)? + eps)?;

        let x_nd = x_nd.broadcast_sub(&min_val)?.broadcast_div(&div_val)?;

        Ok((x_nd * n_vocab)?
            .floor()?
            .to_dtype(candle_core::DType::U8)?
            .detach())
    }

    pub fn new(args: MatchedEncoderArg, vs: VarBuilder) -> Result<Self> {
        let bn_config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        debug_assert!(!args.layers.is_empty());

        let feature_module = aggregate_linear(
            args.dim_feature,
            args.num_feature_modules,
            vs.pp("feature.module"),
        )?;

        let emb_x = candle_nn::embedding(args.n_vocab_emb, args.dim_emb, vs.pp("nn.embed_x"))?;
        let emb_logx =
            candle_nn::embedding(args.n_vocab_emb, args.dim_emb, vs.pp("nn.embed_logx"))?;

        let coord_proj = linear(
            args.dim_coord,
            args.num_feature_modules,
            vs.pp("coordinate.projection"),
        )?;

        // (1) data -> fc
        let mut fc_expr = StackLayers::<Linear>::new();
        let mut fc_coord = StackLayers::<Linear>::new();

        let mut prev_dim = args.num_feature_modules;
        for (j, &next_dim) in args.layers.iter().enumerate() {
            let _name = format!("nn.enc.fc.left.{}", j);
            fc_expr.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Relu,
            );
            let _name = format!("nn.enc.fc.coord.{}", j);
            fc_coord.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Relu,
            );
            prev_dim = next_dim;
        }

        let bn_expr = candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn.left"))?;
        let bn_coord = candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn.coord"))?;

        // (2) fc -> K
        let z_expr_mean =
            candle_nn::linear(prev_dim, args.dim_latent, vs.pp("nn.enc.left.z.mean"))?;
        let z_expr_lnvar =
            candle_nn::linear(prev_dim, args.dim_latent, vs.pp("nn.enc.left.z.lnvar"))?;
        let z_coord_mean =
            candle_nn::linear(prev_dim, args.dim_latent, vs.pp("nn.enc.coord.z.mean"))?;
        let z_coord_lnvar =
            candle_nn::linear(prev_dim, args.dim_latent, vs.pp("nn.enc.coord.z.lnvar"))?;

        Ok(Self {
            n_features: args.dim_feature,
            n_topics: args.dim_latent,
            n_vocab: args.n_vocab_emb,
            feature_module,
            emb_x,
            emb_logx,
            coord_proj,
            fc_expr,
            fc_coord,
            bn_expr,
            bn_coord,
            z_expr_mean,
            z_expr_lnvar,
            z_coord_mean,
            z_coord_lnvar,
        })
    }

    ///
    /// Evaluate latent Gaussian parameters: mu and log_var
    /// z ~ (mu(x), log_var(x))
    ///
    fn latent_gaussian_params(
        &self,
        emb_nm: &Tensor,
        fc: &StackLayers<Linear>,
        bn: &BatchNorm,
        z_mean: &Linear,
        z_lnvar: &Linear,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let min_mean = -(self.n_features as f64).sqrt(); // stabilize
        let max_mean = (self.n_features as f64).sqrt(); // mean
        let min_lv = -8.; // and log variance
        let max_lv = 8.; //

        let fc_nl = fc.forward_t(emb_nm, train)?;
        let bn_nl = bn.forward_t(&fc_nl, train)?;

        let z_mean_nk = z_mean.forward_t(&bn_nl, train)?.clamp(min_mean, max_mean)?;
        let z_lnvar_nk = z_lnvar.forward_t(&bn_nl, train)?.clamp(min_lv, max_lv)?;
        Ok((z_mean_nk, z_lnvar_nk))
    }

    fn latent_gaussian_params_expr(
        &self,
        emb_nm: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        self.latent_gaussian_params(
            emb_nm,
            &self.fc_expr,
            &self.bn_expr,
            &self.z_expr_mean,
            &self.z_expr_lnvar,
            train,
        )
    }

    fn latent_gaussian_params_coord(
        &self,
        proj_nm: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        self.latent_gaussian_params(
            proj_nm,
            &self.fc_coord,
            &self.bn_coord,
            &self.z_coord_mean,
            &self.z_coord_lnvar,
            train,
        )
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
}

pub struct MatchedEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub n_modules: usize,
    pub n_vocab: usize,
    pub d_vocab_emb: usize,
    pub layers: &'a [usize],
}
