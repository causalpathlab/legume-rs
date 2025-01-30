use candle_core::{Result, Tensor};
use candle_nn::{ops, Linear, Module, VarBuilder};

pub trait DecoderModule {
    /// A decoder that spits out reconstruction
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor>;

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

#[allow(dead_code)]
pub struct ETMDecoder {
    n_features: usize,
    n_topics: usize,
    beta: Linear,
}

impl DecoderModule for ETMDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        let theta_nk = z_nk.exp()?;
        let beta_kd = self.beta.forward(&theta_nk)?;
        Ok(ops::log_softmax(&beta_kd, 1)?)
    }

    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &Tensor,
        llik: &LlikFn,
    ) -> Result<(Tensor, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let theta_nk = z_nk.exp()?;
        let beta_kd = self.beta.forward(&theta_nk)?;
        let x_hat_nd = ops::softmax(&beta_kd, 1)?;
        let llik = llik(x_nd, &x_hat_nd)?;
        Ok((x_hat_nd, llik))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

#[allow(dead_code)]
impl ETMDecoder {
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let beta = candle_nn::linear(n_topics, n_features, vs.pp("dec.beta"))?;
        Ok(Self {
            n_features,
            n_topics,
            beta,
        })
    }
}
