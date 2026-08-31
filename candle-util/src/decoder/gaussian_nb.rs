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

    /// The per-gene logit offset `b` in `π = softmax_d(z·W + b)`.
    ///
    /// Exposed so a scorer can rebuild the same rate outside the module: the
    /// loadings already ship as `dictionary.parquet`, but `b` lives only in
    /// the checkpoint, and without it the reconstruction is off by a per-gene
    /// factor.
    #[must_use]
    pub fn feature_bias(&self) -> Option<Tensor> {
        self.decoder.bias().cloned()
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

    fn llik_is_gene_chunked(&self) -> bool {
        true
    }

    /// Per-cell NB log-likelihood **without ever holding an `[N, D]` tensor**.
    ///
    /// [`forward_with_llik`](DecoderModuleT::forward_with_llik) has to return
    /// `π` itself, so training legitimately materialises `[N, D]`. Inference
    /// only wants the scalar per cell, and the whole chain behind it —
    /// logits, softmax, μ, and the dozen temporaries inside
    /// [`crate::loss::nb_log_likelihood_elem`] — is a sum over genes. So it
    /// is taken in gene slices: the weight is narrowed to `[chunk, K]` and
    /// nothing wider than `[N, chunk]` is ever allocated. At
    /// whole-transcriptome D that is the difference between a couple of
    /// hundred MB and tens of GB per block.
    ///
    /// Two passes over the slices, because the softmax denominator is over
    /// ALL genes: the first accumulates `log Σ_d exp(logit_d)` in the
    /// streaming max/sumexp form, the second the likelihood terms. The
    /// logits are recomputed rather than stored — that is the trade being
    /// made, compute for memory.
    fn llik_gene_chunked(&self, z_nk: &Tensor, x_nd: &Tensor, gene_chunk: usize) -> Result<Tensor> {
        let chunk = gene_chunk.max(1).min(self.n_features);
        let last = x_nd.rank() - 1;
        let lib_n1 = x_nd.sum_keepdim(last)?; // [N, 1]
        let w_dk = self.decoder.weight();
        let bias_d = self.decoder.bias();

        // Logits for one gene slice, as `[N, chunk]`.
        let logits_of = |start: usize, len: usize| -> Result<Tensor> {
            let w = w_dk.narrow(0, start, len)?; // [chunk, K]
            let l = z_nk.matmul(&w.t()?)?; // [N, chunk]
            match bias_d {
                Some(b) => l.broadcast_add(&b.narrow(0, start, len)?.unsqueeze(0)?),
                None => Ok(l),
            }
        };

        // Pass 1: running max and Σexp, so the denominator never needs the
        // full logit row in memory.
        let mut running_max: Option<Tensor> = None;
        let mut running_sum: Option<Tensor> = None;
        let mut start = 0;
        while start < self.n_features {
            let len = chunk.min(self.n_features - start);
            let logits = logits_of(start, len)?;
            let m = logits.max_keepdim(last)?; // [N, 1]
            let (new_max, sum) = match (running_max.take(), running_sum.take()) {
                (Some(pm), Some(ps)) => {
                    let new_max = pm.maximum(&m)?;
                    let rescaled = ps.mul(&pm.sub(&new_max)?.exp()?)?;
                    let add = logits.broadcast_sub(&new_max)?.exp()?.sum_keepdim(last)?;
                    (new_max, rescaled.add(&add)?)
                }
                _ => {
                    let sum = logits.broadcast_sub(&m)?.exp()?.sum_keepdim(last)?;
                    (m, sum)
                }
            };
            running_max = Some(new_max);
            running_sum = Some(sum);
            start += len;
        }
        let max_n1 = running_max.expect("at least one gene slice");
        let log_denom = running_sum
            .expect("at least one gene slice")
            .log()?
            .add(&max_n1)?; // [N, 1]

        // Pass 2: the likelihood terms, one slice at a time.
        let mut llik: Option<Tensor> = None;
        let mut start = 0;
        while start < self.n_features {
            let len = chunk.min(self.n_features - start);
            let logits = logits_of(start, len)?;
            let pi = logits.broadcast_sub(&log_denom)?.exp()?; // [N, chunk]
            let mu = pi.broadcast_mul(&lib_n1)?;
            let x = x_nd.narrow(last, start, len)?;
            let log_phi = self
                .log_phi_1d
                .narrow(1, start, len)?
                .broadcast_as(x.shape())?;
            let part = crate::loss::nb_log_likelihood_elem(&x, &mu, &log_phi)?.sum(last)?;
            llik = Some(match llik.take() {
                Some(acc) => acc.add(&part)?,
                None => part,
            });
            start += len;
        }
        llik.ok_or_else(|| candle_core::Error::Msg("no gene slices to score".into()))
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

#[cfg(test)]
mod chunked_llik_tests {
    use super::*;
    use crate::loss::nb_log_likelihood;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    /// The chunked scorer exists only to save memory, so it has to agree with
    /// the dense path it replaces — for every chunk width, including ones that
    /// do not divide the gene count evenly.
    #[test]
    fn chunked_llik_matches_the_dense_path() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let (n, d, k) = (5usize, 23usize, 4usize);
        let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();

        let z = Tensor::rand(-1f32, 1f32, (n, k), &dev).unwrap();
        let x = Tensor::rand(0f32, 9f32, (n, d), &dev).unwrap();

        // The dense reference: exactly what `forward_with_llik` computes.
        let (_, dense) = dec
            .forward_with_llik(&z, &x, &|_, _| unreachable!())
            .unwrap();
        let dense: Vec<f32> = dense.to_vec1().unwrap();

        for chunk in [1usize, 2, 7, 23, 100] {
            let got: Vec<f32> = dec
                .llik_gene_chunked(&z, &x, chunk)
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(got.len(), n, "chunk {chunk}");
            for (i, (g, e)) in got.iter().zip(&dense).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-3 * e.abs().max(1.0),
                    "chunk {chunk}, cell {i}: chunked {g} vs dense {e}"
                );
            }
        }
    }

    /// A cell with no counts still has a well-defined score, and the streaming
    /// max/sumexp must not produce NaN for it.
    #[test]
    fn an_empty_cell_scores_finitely() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let (n, d, k) = (2usize, 9usize, 3usize);
        let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();
        let z = Tensor::rand(-1f32, 1f32, (n, k), &dev).unwrap();
        let x = Tensor::zeros((n, d), DType::F32, &dev).unwrap();
        let got: Vec<f32> = dec.llik_gene_chunked(&z, &x, 4).unwrap().to_vec1().unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "got {got:?}");
    }

    /// The reference against which the dense path itself is defined, so a
    /// change to either is caught rather than silently agreed upon.
    #[test]
    fn the_dense_path_is_the_nb_likelihood_of_its_own_rate() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let (n, d, k) = (3usize, 11usize, 2usize);
        let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();
        let z = Tensor::rand(-1f32, 1f32, (n, k), &dev).unwrap();
        let x = Tensor::rand(0f32, 4f32, (n, d), &dev).unwrap();

        let (pi, llik) = dec
            .forward_with_llik(&z, &x, &|_, _| unreachable!())
            .unwrap();
        let lib = x.sum_keepdim(1).unwrap();
        let mu = pi.broadcast_mul(&lib).unwrap();
        let want: Vec<f32> = nb_log_likelihood(&x, &mu, &dec.log_phi_1d)
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = llik.to_vec1().unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{g} vs {w}");
        }
    }
}
