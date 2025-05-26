#![allow(dead_code)]

use candle_core::{Result, Tensor};

pub trait SpatialEncoderModuleT {
    /// An encoder that spits out two results (latent inference, KL loss)
    ///
    /// # Arguments
    /// * `x_nd` - input data (n x d)
    /// * `s_nc` - input spatial coordinate data (n x c)
    /// * `x0_nd` - null data (n x d)
    /// * `train` - whether to use dropout/batchnorm or not
    ///
    /// # Returns `(z_nk, kl_loss_n)`
    /// * `z_nk` - latent inference (n x k)
    /// * `kl_loss_n` - KL loss (n x 1)
    fn forward_t(
        &self,
        x_nd: &Tensor,
        s_nc: Option<&Tensor>,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)>;

    fn dim_obs(&self) -> usize;

    fn dim_spatial(&self) -> usize;

    fn dim_latent(&self) -> usize;
}
