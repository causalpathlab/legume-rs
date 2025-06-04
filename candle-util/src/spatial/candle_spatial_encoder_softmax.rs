// #![allow(dead_code)]

use crate::candle_aux_layers::StackLayers;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_spatial_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Embedding, Linear, ModuleT, VarBuilder};

pub struct SpatialLogSoftmaxEncoder {
    n_features: usize,
    n_topics: usize,
    n_coords: usize,
    n_vocab: usize,
    min_coord: f64,
    max_coord: f64,
    // d_emb: usize,
    emb_x: Embedding,    // data embedding
    emb_logx: Embedding, // log data embedding
    // emb_s: Embedding,    // spatial embedding
    // coord_mapping: Linear,
    coord_expander: Linear,
    fc: StackLayers<Linear>,
    bn_z: BatchNorm,
    // bn_spatial_z: BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
    // z_spatial_mean: Linear,
    // z_spatial_lnvar: Linear,
}

impl SpatialEncoderModuleT for SpatialLogSoftmaxEncoder {
    fn forward_t(
        &self,
        x_nd: &Tensor,
        s_nc: Option<&Tensor>,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_gaussian_params(x_nd, s_nc, x0_nd, train)?;
        let z_nk = self.reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;
        Ok((
            ops::log_softmax(&z_nk, 1)?,
            gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?,
        ))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_spatial(&self) -> usize {
        self.n_coords
    }
}

impl SpatialLogSoftmaxEncoder {
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
        n_coords: usize,
        n_vocab: usize,
        min_coord: f64,
        max_coord: f64,
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

        debug_assert!(layers.len() > 0);

        let emb_x = candle_nn::embedding(n_vocab, d_emb, vs.pp("nn.embed_x"))?;
        let emb_logx = candle_nn::embedding(n_vocab, d_emb, vs.pp("nn.embed_logx"))?;
        // let emb_s = candle_nn::embedding(n_vocab, d_emb, vs.pp("nn.embed_s"))?;

        // (1) data -> fc
        let mut fc = StackLayers::<Linear>::new();
        let mut prev_dim = n_features;
        for (j, &next_dim) in layers.iter().enumerate() {
            let _name = format!("nn.enc.fc.{}", j);
            fc.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Relu,
            );
            prev_dim = next_dim;
        }

        let coord_expander = candle_nn::linear(n_coords, d_emb, vs.pp("coordinate.expander"))?;

        // let coord_mapping = candle_nn::linear(n_coords, prev_dim, vs.pp("coordinate.mapping"))?;

        let bn_z = candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn_z"))?;
        // let bn_spatial_z =
        //     candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn_z_spatial"))?;

        // (2) fc -> K
        let z_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.lnvar"))?;
        // let z_spatial_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.spatial.mean"))?;
        // let z_spatial_lnvar =
        //     candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.spatial.lnvar"))?;

        Ok(Self {
            n_features,
            n_topics,
            n_coords,
            n_vocab,
            min_coord,
            max_coord,
            // d_emb,
            emb_x,
            emb_logx,
            // emb_s,
            // coord_mapping,
            coord_expander,
            fc,
            bn_z,
            // bn_spatial_z,
            z_mean,
            z_lnvar,
            // z_spatial_mean,
            // z_spatial_lnvar,
        })
    }

    fn discretize_whitened_data(&self, x_nd: &Tensor) -> Result<Tensor> {
        use candle_core::DType::U32;

        let n_vocab = self.n_vocab as f64;
        let d = x_nd.dims().len();
        let min_val = x_nd.min_keepdim(d - 1)?;
        let max_val = x_nd.max_keepdim(d - 1)?;
        let div_val = ((max_val - &min_val)? + 1.)?;

        let x_nd = x_nd.broadcast_sub(&min_val)?.broadcast_div(&div_val)?;

        (x_nd * n_vocab)?.floor()?.to_dtype(U32)
    }

    // fn discretize_coords(&self, s_nc: &Tensor) -> Result<Tensor> {
    //     use candle_core::DType::U32;
    //     let n_vocab = self.n_vocab as f64;
    //     let div_val = self.max_coord;
    //     let s_nc = s_nc.clamp(self.min_coord, self.max_coord)?;
    //     ((s_nc / div_val)? * n_vocab)?.floor()?.to_dtype(U32)
    // }

    pub fn preprocess_input_data(
        &self,
        x_nd: &Tensor,
        s_nc: Option<&Tensor>,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        debug_assert_eq!(x_nd.dims().len(), 2);
        debug_assert!(x_nd.min_all()?.to_scalar::<f32>()? >= 0_f32);

        // 1. Discretize data after log1p transformation
        let int_x_nd = self.discretize_whitened_data(&x_nd)?;
        let logx_nd = (x_nd + 1.)?.log()?;
        let int_logx_nd = self.discretize_whitened_data(&logx_nd)?;

        // 2. log1p intensity embedding: n x d -> n x d x k
        // Note: Works like log(x) - beta * log(x0), but we worry
        // less about how to tune the parameter beta.
        let emb_ndk = (self.emb_x.forward_t(&int_x_nd, train)?
            + self.emb_logx.forward_t(&int_logx_nd, train)?)?;

        let pool_embedding = |emb_ndk: &Tensor, s_nc: Option<&Tensor>| -> Result<Tensor> {
            let last_dim = emb_ndk.dims().len() - 1;

            if let Some(s_nc) = s_nc {
                // aggregate the last embedding dimension by the
                // importance weights learned from spatial coordinates
                let div = (self.max_coord - self.min_coord).max(1.);
                let s_nc = ((s_nc - self.min_coord)? / div)?;

                let agg_nd = self.coord_expander.forward_t(&s_nc, train)?;

                let agg_n1d = agg_nd.unsqueeze(1)?;

                // let agg_n1d = ops::log_softmax(&agg_nd, 1)?.exp()?.unsqueeze(1)?;

                emb_ndk.broadcast_add(&agg_n1d)?.sum(last_dim)
            } else {
                emb_ndk.sum(last_dim)
            }
        };

        match x0_nd {
            Some(x0_nd) => {
                let int_x0_nd = self.discretize_whitened_data(&x0_nd)?;
                let logx0_nd = (x0_nd + 1.)?.log()?;
                let int_logx0_nd = self.discretize_whitened_data(&logx0_nd)?;

                let emb0_ndk = (self.emb_x.forward_t(&int_x0_nd, train)?
                    + self.emb_logx.forward_t(&int_logx0_nd, train)?)?;

                pool_embedding(&(emb_ndk - &emb0_ndk)?, s_nc)
            }
            None => pool_embedding(&emb_ndk, s_nc),
        }
    }

    ///
    /// z = mu + sigma * eps
    /// where eps ~ N(0, 1)
    ///
    /// # Arguments
    /// * `z_mean` - mean of Gaussian distribution
    /// * `z_lnvar` - log variance of Gaussian distribution
    pub fn reparameterize(&self, z_mean: &Tensor, z_lnvar: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            let eps = Tensor::randn_like(&z_mean, 0., 1.)?;
            z_mean + (z_lnvar * 0.5)?.exp()? * eps
        } else {
            Ok(z_mean.clone())
        }
    }

    ///
    /// Evaluate latent Gaussian parameters: mu and log_var
    /// z ~ (mu(x), log_var(x))
    pub fn latent_gaussian_params(
        &self,
        x_nd: &Tensor,
        s_nc: Option<&Tensor>,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let min_mean = -(self.n_features as f64).sqrt(); // stabilize
        let max_mean = (self.n_features as f64).sqrt(); // mean
        let min_lv = -8.; // and log variance
        let max_lv = 8.; //

        let bn_nd = self.preprocess_input_data(x_nd, s_nc, x0_nd, train)?;

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

        // if let Some(s_nc) = s_nc {
        //     let s_nl = self.coord_mapping.forward_t(&s_nc, train)?;
        //     let bn_nl = self.bn_spatial_z.forward_t(&s_nl, train)?;

        //     let z_spatial_mean_nk = self
        //         .z_spatial_mean
        //         .forward_t(&bn_nl, train)?
        //         .clamp(min_mean, max_mean)?;
        //     let z_spatial_lnvar_nk = self
        //         .z_spatial_lnvar
        //         .forward_t(&bn_nl, train)?
        //         .clamp(min_lv, max_lv)?;

        //     Ok((
        //         (z_mean_nk + z_spatial_mean_nk)?,
        //         (z_lnvar_nk + z_spatial_lnvar_nk)?,
        //     ))
        // } else {

        // }
    }
}
