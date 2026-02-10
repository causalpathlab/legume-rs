// #![allow(dead_code)]

use crate::candle_aux_layers::StackLayers;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_spatial_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Linear, ModuleT, VarBuilder};

pub struct SpatialLogSoftmaxEncoder {
    n_features: usize,
    n_topics: usize,
    n_coords: usize,
    min_coord: f64,
    max_coord: f64,
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
    /// # Arguments
    /// * `n_features` - the number of features
    /// * `n_topics` - the number of topics (latent factors)
    /// * `n_coords` - the number of spatial coordinates
    /// * `min_coord` - minimum coordinate value for normalization
    /// * `max_coord` - maximum coordinate value for normalization
    /// * `layers` - fully connected layers, each with the dim
    /// * `vs` - variable builder
    pub fn new(
        n_features: usize,
        n_topics: usize,
        n_coords: usize,
        min_coord: f64,
        max_coord: f64,
        layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        let bn_config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        debug_assert!(!layers.is_empty());

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

        let coord_expander = candle_nn::linear(n_coords, n_features, vs.pp("coordinate.expander"))?;

        let bn_z = candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn_z"))?;

        // (2) fc -> K
        let z_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.z.lnvar"))?;

        Ok(Self {
            n_features,
            n_topics,
            n_coords,
            min_coord,
            max_coord,
            coord_expander,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
        })
    }

    pub fn preprocess_input_data(
        &self,
        x_nd: &Tensor,
        s_nc: Option<&Tensor>,
        x0_nd: Option<&Tensor>,
        _train: bool,
    ) -> Result<Tensor> {
        debug_assert_eq!(x_nd.dims().len(), 2);
        debug_assert!(x_nd.min_all()?.to_scalar::<f32>()? >= 0_f32);

        let h_nd = (x_nd + 1.)?.log()?;

        let h_nd = match x0_nd {
            Some(x0) => (h_nd - (x0 + 1.)?.log()?)?,
            None => h_nd,
        };

        match s_nc {
            Some(s_nc) => {
                let div = (self.max_coord - self.min_coord).max(1.);
                let s_nc = ((s_nc - self.min_coord)? / div)?;
                let coord_nd = self.coord_expander.forward_t(&s_nc, false)?;
                h_nd.add(&coord_nd)
            }
            None => Ok(h_nd),
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
