use candle_core::{Result, Tensor};
use candle_nn::{ops, sequential, BatchNorm, Linear, ModuleT, Sequential, VarBuilder};

/// KL divergence loss between two Gaussian distributions
///
/// -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
#[allow(dead_code)]
pub fn kl_loss(z_mean: &Tensor, z_lnvar: &Tensor) -> Result<Tensor> {
    let z_var = z_lnvar.exp()?;
    (z_var - 1. + z_mean.powf(2.)? - z_lnvar)?.sum(1)? * 0.5
}

/// Topic model log-likelihood of multinomial data
///
/// llik(i) = sum_w x(i,w) * log pr(i,w)
///
#[allow(dead_code)]
pub fn topic_likelihood(x_nd: &Tensor, logits_nd: &Tensor) -> Result<Tensor> {
    x_nd.mul(&logits_nd)?.sum(1)
}

/////////////////////////
// Emedded Topic Model //
/////////////////////////

#[allow(dead_code)]
pub struct ETM {
    pub encoder: ETMEncoder,
    pub decoder: ETMDecoder,
}

#[allow(dead_code)]
impl candle_nn::ModuleT for ETM {
    fn forward_t(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let z_nk = self.encoder.forward_t(x_nd, train)?;
        let logits_nd = self.decoder.forward_t(&z_nk, train)?;
        topic_likelihood(x_nd, &logits_nd)
    }
}

#[allow(dead_code)]
impl ETM {
    pub fn new(
        n_features: usize,
        n_topics: usize,
        enc_layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        let encoder = ETMEncoder::new(n_features, n_topics, enc_layers, vs.clone())?;
        let decoder = ETMDecoder::new(n_features, n_topics, vs.clone())?;
        Ok(Self { encoder, decoder })
    }
}

/////////////////////
// Encoder for ETM //
/////////////////////
#[allow(dead_code)]
pub struct ETMEncoder {
    n_features: usize,
    n_topics: usize,
    bn: BatchNorm,
    fc: Sequential,
    z_mean: Linear,
    z_lnvar: Linear,
}

#[allow(dead_code)]
impl candle_nn::ModuleT for ETMEncoder {
    fn forward_t(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_params(x_nd, train)?;
        let z_nk = self.reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;
        Ok(ops::log_softmax(&z_nk, 1)?)
    }
}

#[allow(dead_code)]
impl ETMEncoder {
    pub fn num_features(&self) -> usize {
        self.n_features
    }

    pub fn num_topics(&self) -> usize {
        self.n_topics
    }

    pub fn batch_norm_input(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let x_log1p_nd = (x_nd + 1.)?.log()?;
        let x_norm_nd = x_log1p_nd.broadcast_div(&x_log1p_nd.sum(0)?)?;
        self.bn.forward_t(&x_norm_nd, train)
    }

    pub fn latent_params(&self, x_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let batch_nd = self.batch_norm_input(x_nd, train)?;
        let x_nl = self.fc.forward_t(&batch_nd, train)?;
        let z_mean_nk = self.z_mean.forward_t(&x_nl, train)?;
        let z_lnvar_nk = self.z_lnvar.forward_t(&x_nl, train)?;
        Ok((z_mean_nk, z_lnvar_nk))
    }

    pub fn kl_loss(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_params(x_nd, train)?;
        kl_loss(&z_mean_nk, &z_lnvar_nk)
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

    pub fn new(
        n_features: usize,
        n_topics: usize,
        layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        // (0) batch norm
        let bn = candle_nn::batch_norm(
            n_features,
            candle_nn::BatchNormConfig {
                eps: 1e-4,
                remove_mean: true,
                affine: true,
                momentum: 0.1,
            },
            vs.pp("enc.bn"),
        )?;

        // (1) data -> fc
        let mut fc = sequential::seq();
        let mut prev_dim = n_features;
        for (j, &next_dim) in layers.iter().enumerate() {
            fc = fc.add(candle_nn::linear(
                prev_dim,
                next_dim,
                vs.pp(format!("enc.fc.{}", j)),
            )?);
            prev_dim = next_dim;
        }

        // (2) fc -> K
        let z_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("enc.z.lnvar"))?;

        Ok(Self {
            n_features,
            n_topics,
            bn,
            fc,
            z_mean,
            z_lnvar,
        })
    }
}

/////////////////////
// Decoder for ETM //
/////////////////////

#[allow(dead_code)]
pub struct ETMDecoder {
    n_features: usize,
    n_topics: usize,
    beta: Linear,
}

#[allow(dead_code)]
impl candle_nn::ModuleT for ETMDecoder {
    fn forward_t(&self, z_nk: &Tensor, train: bool) -> Result<Tensor> {
        let beta_kd = self.beta.forward_t(z_nk, train)?;
        Ok(ops::log_softmax(&beta_kd, 1)?)
    }
}

#[allow(dead_code)]
impl ETMDecoder {
    pub fn num_features(&self) -> usize {
        self.n_features
    }

    pub fn num_topics(&self) -> usize {
        self.n_topics
    }

    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let beta = candle_nn::linear(n_topics, n_features, vs.pp("dec.beta"))?;
        Ok(Self {
            n_features,
            n_topics,
            beta,
        })
    }
}
