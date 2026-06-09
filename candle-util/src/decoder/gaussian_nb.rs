//! **scVI-style NB decoder** for a Gaussian latent.
//!
//! Maps the unconstrained latent `z` to a gene distribution by a linear
//! decoder + gene-axis softmax, then a negative-binomial likelihood scaled by
//! the library size:
//!
//! ```text
//! logits_nd = z_nk · W + b          # W: [n_latent, D]
//! π_nd      = softmax_d(logits_nd)   # gene distribution (sums to 1 over D)
//! μ_nd      = library_n · π_nd
//! llik      = NB(x_nd | μ_nd, φ_d)
//! ```
//!
//! Pairs with [`crate::encoder::GaussianEncoder`]. The gene-axis softmax (not a
//! simplex mixture of per-topic distributions) is what makes a *Gaussian*,
//! unconstrained `z` valid here — unlike the topic decoders.

use crate::loss::nb_log_likelihood;
use crate::traits::model::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, Linear, Module, VarBuilder};

pub struct GaussianNbDecoder {
    n_features: usize,
    n_latent: usize,
    /// Linear factor loadings `z → gene logits`; weight is `[D, n_latent]`.
    decoder: Linear,
    /// `[1, D]` log per-gene NB inverse dispersion.
    log_phi_1d: Tensor,
}

impl GaussianNbDecoder {
    pub fn new(n_features: usize, n_latent: usize, vs: VarBuilder) -> Result<Self> {
        let decoder = candle_nn::linear(n_latent, n_features, vs.pp("gauss_decoder"))?;
        let log_phi_1d =
            vs.get_with_hints((1, n_features), "log_phi", candle_nn::Init::Const(0.693))?;
        Ok(Self {
            n_features,
            n_latent,
            decoder,
            log_phi_1d,
        })
    }

    /// `log π_nd = log_softmax_d(z·W + b)`.
    fn log_pi(&self, z_nk: &Tensor) -> Result<Tensor> {
        let logits_nd = self.decoder.forward(z_nk)?; // [N, D]
        ops::log_softmax(&logits_nd, logits_nd.rank() - 1)
    }
}

impl NewDecoder for GaussianNbDecoder {
    fn new(n_features: usize, n_latent: usize, vs: VarBuilder) -> Result<Self> {
        GaussianNbDecoder::new(n_features, n_latent, vs)
    }
}

impl DecoderModuleT for GaussianNbDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        self.log_pi(z_nk)?.exp()
    }

    /// Factor loadings `[D, n_latent]` — the analogue of the topic dictionary.
    fn get_dictionary(&self) -> Result<Tensor> {
        Ok(self.decoder.weight().clone())
    }

    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &Tensor,
        _llik: &LlikFn,
    ) -> Result<(Tensor, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let last = x_nd.rank() - 1;
        // `softmax` directly — no need to `log_softmax` then `exp` back.
        let logits_nd = self.decoder.forward(z_nk)?; // [N, D]
        let pi_nd = ops::softmax(&logits_nd, last)?;
        let lib_n1 = x_nd.sum_keepdim(last)?; // [N, 1]
        let mu_nd = pi_nd.broadcast_mul(&lib_n1)?;
        let llik = nb_log_likelihood(x_nd, &mu_nd, &self.log_phi_1d)?;
        Ok((pi_nd, llik))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_latent
    }

    /// ESS log-likelihood closure for the Gaussian latent: multinomial
    /// `Σ_d x_d · log π_d` with `π = softmax_d(z·W + b)` on detached weights.
    /// Overrides the simplex-`θ` default (which would `softmax(z)` first).
    fn build_ess_llik<'a>(
        &'a self,
        x_nd: &'a Tensor,
        _topic_smoothing: f64,
    ) -> Result<EssLlikFn<'a>> {
        // `[n_latent, D]`, contiguous — transposed once here, not per call.
        let w_kd = self.decoder.weight().detach().t()?.contiguous()?;
        let bias_d = self.decoder.bias().map(Tensor::detach);
        let x_pos = x_nd.clamp(0.0, f64::INFINITY)?;

        Ok(Box::new(move |z_nk: &Tensor| {
            let logits = z_nk.matmul(&w_kd)?; // [N, D]
            let logits = match &bias_d {
                Some(b) => logits.broadcast_add(&b.unsqueeze(0)?)?,
                None => logits,
            };
            let log_pi = ops::log_softmax(&logits, logits.rank() - 1)?;
            x_pos.mul(&log_pi)?.sum(x_pos.rank() - 1)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{GaussianEncoder, GaussianEncoderArgs};
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_gaussian_encoder_decoder_smoke() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let (n, d, h, k) = (4usize, 6usize, 8usize, 3usize);

        let enc = GaussianEncoder::new(
            GaussianEncoderArgs {
                n_features: d,
                n_latent: k,
                layers: &[h],
                feature_mean: None,
            },
            &varmap,
            vb.pp("enc"),
        )
        .unwrap();
        let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();

        let x = Tensor::rand(0f32, 5f32, (n, d), &dev).unwrap();
        let (z, kl) = enc.forward_t(&x, None, true).unwrap();
        // Raw Gaussian latent — NOT projected to the simplex.
        assert_eq!(z.dims(), &[n, k]);
        assert_eq!(kl.dims(), &[n]);

        let noop = |_a: &Tensor, _b: &Tensor| Ok(_a.clone());
        let (pi, llik) = dec.forward_with_llik(&z, &x, &noop).unwrap();
        assert_eq!(pi.dims(), &[n, d]);
        assert_eq!(llik.dims(), &[n]);

        // π is a gene distribution: each cell sums to 1 over genes.
        for s in pi.sum(1).unwrap().to_vec1::<f32>().unwrap() {
            assert!((s - 1.0).abs() < 1e-4, "pi row sum {s} != 1");
        }
        // NB likelihood is finite.
        for v in llik.to_vec1::<f32>().unwrap() {
            assert!(v.is_finite(), "llik {v} not finite");
        }
    }
}
