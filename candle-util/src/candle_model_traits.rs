#![allow(dead_code)]

use candle_core::{Result, Tensor};

pub trait EncoderModuleT {
    /// An encoder that spits out two results (latent inference, KL loss)
    ///
    /// # Arguments
    /// * `x_nd` - input data (n x d)
    /// * `x0_nd` - null data (n x d)
    /// * `train` - whether to use dropout/batchnorm or not
    ///
    /// # Returns `(z_nk, kl_loss_n)`
    /// * `z_nk` - latent inference (n x k)
    /// * `kl_loss_n` - KL loss (n x 1)
    fn forward_t(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;
}

pub trait DecoderModuleT {
    /// A decoder that spits out reconstruction
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor>;

    /// Get a representative dictionary matrix
    fn get_dictionary(&self) -> Result<Tensor>;

    /// A decoder that spits out reconstruction and log-likelihood
    /// * `z_nk` - latent states
    /// * `x_nd` - observed data to validate with
    /// * `llik` - fn (observed, reconstruction) -> log-likelihood
    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &Tensor,
        llik: &LlikFn,
    ) -> Result<(Tensor, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;
}

pub struct MatchedEncoderData<'a> {
    pub marginal_left: &'a Tensor,
    pub marginal_right: &'a Tensor,
    pub neigh_left: Option<&'a Tensor>,
    pub neigh_right: Option<&'a Tensor>,
}

pub struct MatchedDecoderData<'a> {
    pub marginal_left: &'a Tensor,
    pub marginal_right: &'a Tensor,
    pub border_left: Option<&'a Tensor>,
    pub border_right: Option<&'a Tensor>,
}

pub trait MatchedEncoderModuleT {
    /// An encoder that spits out two results (latent inference, KL loss)
    fn forward_t(&self, data: MatchedEncoderData, train: bool) -> Result<MatchedEncoderLatent>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;
}

pub struct MatchedEncoderLatent {
    pub marginal: Tensor,
    pub border: Tensor,
    pub kl_div: Tensor,
}

pub struct MatchedDecoderRecon {
    pub marginal: Tensor,
    pub border: Tensor,
}

pub trait MatchedDecoderModuleT {
    /// A decoder that spits out reconstruction
    fn forward(&self, latent: &MatchedEncoderLatent) -> Result<MatchedDecoderRecon>;

    /// Get a representative dictionary matrix
    fn get_dictionary(&self) -> Result<Tensor>;

    /// A decoder that spits out reconstruction and log-likelihood
    /// * `latent` - latent states by `DiffEncoderModuleT`
    /// * `x_nd` - observed data to validate with
    /// * `x0_nd` - matched data to validate with
    /// * `llik` - fn (observed, reconstruction) -> log-likelihood
    fn forward_with_llik<LlikFn>(
        &self,
        latent: &MatchedEncoderLatent,
        x_nd_pair: MatchedDecoderData,
        llik: &LlikFn,
    ) -> Result<(MatchedDecoderRecon, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;
}
