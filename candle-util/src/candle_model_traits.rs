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

    fn dim_latent(&self) -> usize;
}

pub trait MultimodalEncoderModuleT {
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
        x_nd_vec: &[Tensor],
        x0_nd_vec: &[Option<Tensor>],
        train: bool,
    ) -> Result<(Tensor, Tensor)>;

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

pub trait MultimodalDecoderModuleT {
    /// A decoder that spits out reconstruction
    fn forward(&self, z_nk: &Tensor) -> Result<Vec<Tensor>>;

    /// Get a representative dictionary matrix
    fn get_dictionary(&self) -> Result<Vec<Tensor>>;

    /// A decoder that spits out reconstruction and log-likelihood
    /// * `z_nk` - latent states
    /// * `x_nd` - observed data to validate with
    /// * `llik` - fn (observed, reconstruction) -> log-likelihood
    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &[Tensor],
        llik: &LlikFn,
    ) -> Result<(Vec<Tensor>, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    fn dim_obs(&self) -> &[usize];

    fn dim_latent(&self) -> usize;
}

pub struct MatchedEncoderData<'a> {
    pub left: &'a Tensor,
    pub right: &'a Tensor,
    pub aux_left: Option<&'a Tensor>,
    pub aux_right: Option<&'a Tensor>,
}

pub struct MatchedDecoderData<'a> {
    pub left: &'a Tensor,
    pub right: &'a Tensor,
    pub delta_left: Option<&'a Tensor>,
    pub delta_right: Option<&'a Tensor>,
}

pub trait MatchedEncoderModuleT {
    /// An encoder that spits out two results (latent inference, KL loss)
    fn forward_t(&self, data: MatchedEncoderData, train: bool) -> Result<MatchedEncoderLatent>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;
}

pub struct MatchedEncoderLatent {
    pub logits_theta_left: Tensor,
    pub logits_theta_right: Tensor,
    pub kl_div: Tensor,
}

pub struct MatchedDecoderRecon {
    pub x_left: Tensor,
    pub x_right: Tensor,
}

pub trait MatchedDecoderModuleT {
    /// A decoder that spits out reconstruction
    fn forward(&self, latent: &MatchedEncoderLatent) -> Result<MatchedDecoderRecon>;

    /// Get a representative dictionary matrix
    fn get_dictionary(&self) -> Result<Tensor>;

    /// A decoder that spits out reconstruction and log-likelihood
    fn forward_with_llik<LlikFn>(
        &self,
        latent: &MatchedEncoderLatent,
        x_pair: MatchedDecoderData,
        llik: &LlikFn,
    ) -> Result<(MatchedDecoderRecon, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;
}
