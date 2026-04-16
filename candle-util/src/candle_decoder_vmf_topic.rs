use crate::candle_model_traits::*;
use crate::sgvb::likelihood::{l2_normalize_dim, suggest_kappa_init, vmf_log_normalizer};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ops, VarBuilder};

/// Von Mises-Fisher topic decoder with per-topic concentration.
///
/// ```text
/// p(x|z) = Σ_k z_k · C_d(κ_k) · exp(κ_k · xᵀμ_k)
/// ```
///
/// Cap log(κ) at ln(500) ≈ 6.21 — beyond this concentration is extreme
/// for high-D data and the Bessel normalizer becomes numerically fragile.
const MAX_LOG_KAPPA: f64 = 6.21;

pub struct VmfTopicDecoder {
    n_features: usize,
    n_topics: usize,
    weight_kd: Tensor,
    /// Per-topic log(κ_k) concentration [K].
    log_kappa_k: Tensor,
}

impl VmfTopicDecoder {
    pub fn log_kappa(&self) -> &Tensor {
        &self.log_kappa_k
    }

    /// Clamped κ_k values as a Vec<f64>.
    fn clamped_kappa_vec(&self) -> Result<Vec<f64>> {
        let vals: Vec<f64> = self
            .log_kappa_k
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F64)?
            .to_vec1::<f64>()?;
        Ok(vals.iter().map(|v| v.min(MAX_LOG_KAPPA).exp()).collect())
    }

    /// Per-topic κ values.
    pub fn kappa_vec(&self) -> Result<Vec<f64>> {
        self.clamped_kappa_vec()
    }

    fn mu_kd(&self) -> Result<Tensor> {
        l2_normalize_dim(&self.weight_kd, 1)
    }

    /// Per-topic log normalizer C_d(κ_k) as [1, K] f32 tensor on the given device.
    fn log_normalizer_1k(&self, dev: &Device) -> Result<Tensor> {
        let kappas = self.clamped_kappa_vec()?;
        let log_cs: Vec<f32> = kappas
            .iter()
            .map(|&k| vmf_log_normalizer(self.n_features, k) as f32)
            .collect();
        Tensor::new(&log_cs[..], dev)?.unsqueeze(0)
    }
}

impl NewDecoder for VmfTopicDecoder {
    fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let weight_kd = vs.get_with_hints(
            (n_topics, n_features),
            "dictionary.weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;

        let kappa_init = suggest_kappa_init(n_features).ln();
        let log_kappa_k =
            vs.get_with_hints((n_topics,), "log_kappa", candle_nn::Init::Const(kappa_init))?;

        Ok(Self {
            n_features,
            n_topics,
            weight_kd,
            log_kappa_k,
        })
    }
}

impl DecoderModuleT for VmfTopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        let mu_kd = self.mu_kd()?;
        z_nk.exp()?.matmul(&mu_kd)
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        self.mu_kd()?.t()
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
        let dev = x_nd.device();
        let mu_kd = self.mu_kd()?;
        let x_log = (x_nd + 1.0)?.log()?;
        let x_hat = l2_normalize_dim(&x_log, 1)?;
        let sim_nk = x_hat.matmul(&mu_kd.t()?)?; // [N, K]

        // κ_k [1, K] broadcast over N
        let kappa_1k = self
            .log_kappa_k
            .clamp(f64::NEG_INFINITY, MAX_LOG_KAPPA)?
            .exp()?
            .unsqueeze(0)?;
        let log_c_1k = self.log_normalizer_1k(dev)?; // [1, K]

        // log p(x|z) = logsumexp_k [ log z_k + κ_k·sim_nk + log C_d(κ_k) ]
        let log_topic_nk = (z_nk + sim_nk.broadcast_mul(&kappa_1k)?)?.broadcast_add(&log_c_1k)?;
        let llik_n = log_topic_nk.log_sum_exp(1)?;

        let recon_nd = z_nk.exp()?.matmul(&mu_kd)?;

        Ok((recon_nd, llik_n))
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }

    fn build_ess_llik<'a>(
        &'a self,
        x_nd: &'a Tensor,
        topic_smoothing: f64,
    ) -> Result<EssLlikFn<'a>> {
        let mu_kd = self.mu_kd()?.detach();
        let x_log = (x_nd + 1.0)?.log()?;
        let x_hat = l2_normalize_dim(&x_log, 1)?;
        let sim_nk = x_hat.matmul(&mu_kd.t()?)?.detach(); // [N, K]

        // Detached per-topic κ and log C as [1, K]
        let kappas = self.clamped_kappa_vec()?;
        let kappa_1k: Vec<f32> = kappas.iter().map(|&k| k as f32).collect();
        let log_c_1k: Vec<f32> = kappas
            .iter()
            .map(|&k| vmf_log_normalizer(self.n_features, k) as f32)
            .collect();
        let dev = x_nd.device().clone();
        let kappa_t = Tensor::new(&kappa_1k[..], &dev)?.unsqueeze(0)?;
        let log_c_t = Tensor::new(&log_c_1k[..], &dev)?.unsqueeze(0)?;
        let k = self.n_topics as f64;

        Ok(Box::new(move |z_nk: &Tensor| -> Result<Tensor> {
            let mut z = ops::softmax(z_nk, 1)?;
            if topic_smoothing > 0.0 {
                z = ((z * (1.0 - topic_smoothing))? + topic_smoothing / k)?;
            }
            let log_z = z.log()?;
            let log_topic_nk =
                (log_z + sim_nk.broadcast_mul(&kappa_t)?)?.broadcast_add(&log_c_t)?;
            log_topic_nk.log_sum_exp(1)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_vmf_decoder_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let n_features = 50;
        let n_topics = 5;
        let n_samples = 10;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let decoder = VmfTopicDecoder::new(n_features, n_topics, vs)?;

        let dict = decoder.get_dictionary()?;
        assert_eq!(dict.dims(), &[n_features, n_topics]);

        // Dictionary columns are unit vectors
        let norms = dict.sqr()?.sum(0)?.sqrt()?;
        let norms_vec: Vec<f32> = norms.to_vec1()?;
        for norm in &norms_vec {
            assert!((norm - 1.0).abs() < 1e-4, "norm = {}", norm);
        }

        let z = Tensor::randn(0.0f32, 1.0, (n_samples, n_topics), &dev)?;
        let recon = decoder.forward(&z)?;
        assert_eq!(recon.dims(), &[n_samples, n_features]);

        let x = Tensor::rand(0.0f32, 1.0, (n_samples, n_features), &dev)?;
        let dummy_llik = |_x: &Tensor, _r: &Tensor| -> Result<Tensor> { unreachable!() };
        let (recon2, llik) = decoder.forward_with_llik(&z, &x, &dummy_llik)?;
        assert_eq!(recon2.dims(), &[n_samples, n_features]);
        assert_eq!(llik.dims(), &[n_samples]);

        Ok(())
    }

    #[test]
    fn test_vmf_kappa_per_topic() -> Result<()> {
        let dev = Device::Cpu;
        let n_topics = 5;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let decoder = VmfTopicDecoder::new(100, n_topics, vs)?;

        let kappas = decoder.kappa_vec()?;
        assert_eq!(kappas.len(), n_topics);
        for &k in &kappas {
            assert!(k > 0.0, "kappa should be positive, got {}", k);
        }

        // All initialized to same value
        let first = kappas[0];
        for &k in &kappas[1..] {
            assert!((k - first).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_vmf_llik_finite() -> Result<()> {
        let dev = Device::Cpu;
        let n_features = 50;
        let n_topics = 3;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let decoder = VmfTopicDecoder::new(n_features, n_topics, vs)?;

        let z = Tensor::full(-((n_topics as f32).ln()), (5, n_topics), &dev)?;
        let x = Tensor::rand(0.0f32, 1.0, (5, n_features), &dev)?;
        let dummy = |_x: &Tensor, _r: &Tensor| -> Result<Tensor> { unreachable!() };
        let (_, llik) = decoder.forward_with_llik(&z, &x, &dummy)?;
        let llik_vec: Vec<f32> = llik.to_vec1()?;
        for &v in &llik_vec {
            assert!(v.is_finite(), "llik should be finite, got {}", v);
        }

        Ok(())
    }
}
