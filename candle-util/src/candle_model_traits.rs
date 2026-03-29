use candle_core::{Result, Tensor};
use candle_nn::ops;

/// Boxed log-likelihood closure for ESS evaluation.
/// Maps `z_nk [N, K]` to per-sample log-likelihood `[N]`.
pub type EssLlikFn<'a> = Box<dyn Fn(&Tensor) -> Result<Tensor> + 'a>;

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

pub trait JointEncoderModuleT {
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

    /// Build a lightweight log-likelihood closure for ESS evaluation.
    ///
    /// Returns a boxed closure `|z_nk [N,K]| -> llik [N]` that uses
    /// **detached** snapshots of decoder weights. No gradient graph is built,
    /// and no redundant softmax recomputation occurs across ESS iterations.
    ///
    /// Default implementation pre-computes the dictionary and uses multinomial
    /// log-likelihood. Override for decoders with different likelihoods (e.g. NB).
    fn build_ess_llik<'a>(&'a self, x_nd: &'a Tensor) -> Result<EssLlikFn<'a>> {
        use crate::candle_loss_functions::topic_likelihood;

        let log_dict_dk = self.get_dictionary()?.detach();
        let beta_kd = log_dict_dk.t()?.exp()?.contiguous()?;
        let x_clamped = x_nd.clamp(0.0, f64::INFINITY)?;

        Ok(Box::new(move |z_nk: &Tensor| {
            let z = ops::log_softmax(z_nk, 1)?.exp()?;
            let recon = z.matmul(&beta_kd)?;
            topic_likelihood(&x_clamped, &recon)
        }))
    }
}

/// Compute joint multinomial log-likelihood from per-modality log-reconstructions.
/// Returns (recon_vec, total_llik) where total_llik sums across modalities.
pub fn joint_multinomial_llik(
    log_recon_vec: Vec<Tensor>,
    x_nd_vec: &[Tensor],
) -> Result<(Vec<Tensor>, Tensor)> {
    let recon_vec: Vec<Tensor> = log_recon_vec
        .iter()
        .map(|x| x.exp())
        .collect::<Result<Vec<_>>>()?;

    let llik_vec = x_nd_vec
        .iter()
        .zip(&log_recon_vec)
        .map(|(x, log_recon)| -> Result<Tensor> {
            let ret = x
                .clamp(0.0, f64::INFINITY)?
                .mul(log_recon)?
                .sum(x.rank() - 1)?;
            ret.unsqueeze(ret.rank())
        })
        .collect::<Result<Vec<Tensor>>>()?;

    let k = llik_vec[0].rank();
    let llik = Tensor::cat(&llik_vec, k - 1)?.sum(k - 1)?;
    Ok((recon_vec, llik))
}

pub trait JointDecoderModuleT {
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
