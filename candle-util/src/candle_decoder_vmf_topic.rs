use crate::candle_decoder_topic::BgmState;
use crate::candle_model_traits::*;
use crate::sgvb::likelihood::{l2_normalize_dim, suggest_kappa_init, vmf_log_normalizer};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ops, VarBuilder};

/// Cap log(κ) at ln(500) ≈ 6.21 — beyond this concentration is extreme
/// for high-D data and the Bessel normalizer becomes numerically fragile.
const MAX_LOG_KAPPA: f64 = 6.21;

/// Shared vMF core: unit-norm dictionary μ_kd + per-topic log(κ_k).
struct VmfCore {
    n_features: usize,
    weight_kd: Tensor,
    log_kappa_k: Tensor,
}

impl VmfCore {
    fn new(n_features: usize, n_topics: usize, vs: &VarBuilder) -> Result<Self> {
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
            weight_kd,
            log_kappa_k,
        })
    }

    fn mu_kd(&self) -> Result<Tensor> {
        l2_normalize_dim(&self.weight_kd, 1)
    }

    fn clamped_kappa_vec(&self) -> Result<Vec<f64>> {
        let vals: Vec<f64> = self
            .log_kappa_k
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F64)?
            .to_vec1::<f64>()?;
        Ok(vals.iter().map(|v| v.min(MAX_LOG_KAPPA).exp()).collect())
    }

    fn log_normalizer_1k(&self, dev: &Device) -> Result<Tensor> {
        let kappas = self.clamped_kappa_vec()?;
        let log_cs: Vec<f32> = kappas
            .iter()
            .map(|&k| vmf_log_normalizer(self.n_features, k) as f32)
            .collect();
        Tensor::new(&log_cs[..], dev)?.unsqueeze(0)
    }

    fn clamped_kappa_1k(&self) -> Result<Tensor> {
        self.log_kappa_k
            .clamp(f64::NEG_INFINITY, MAX_LOG_KAPPA)?
            .exp()?
            .unsqueeze(0)
    }
}

/// Von Mises-Fisher topic decoder with per-topic concentration.
///
/// ```text
/// p(x|z) = Σ_k z_k · C_d(κ_k) · exp(κ_k · xᵀμ_k)
/// ```
pub struct VmfTopicDecoder {
    n_topics: usize,
    core: VmfCore,
}

impl VmfTopicDecoder {
    pub fn log_kappa(&self) -> &Tensor {
        &self.core.log_kappa_k
    }

    pub fn kappa_vec(&self) -> Result<Vec<f64>> {
        self.core.clamped_kappa_vec()
    }
}

impl NewDecoder for VmfTopicDecoder {
    fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let core = VmfCore::new(n_features, n_topics, &vs)?;
        Ok(Self { n_topics, core })
    }
}

impl DecoderModuleT for VmfTopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        let mu_kd = self.core.mu_kd()?;
        z_nk.exp()?.matmul(&mu_kd)
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        self.core.mu_kd()?.t()
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
        let mu_kd = self.core.mu_kd()?;
        let x_log = (x_nd + 1.0)?.log()?;
        let x_hat = l2_normalize_dim(&x_log, 1)?;
        let sim_nk = x_hat.matmul(&mu_kd.t()?)?;

        let kappa_1k = self.core.clamped_kappa_1k()?;
        let log_c_1k = self.core.log_normalizer_1k(dev)?;

        let log_topic_nk = (z_nk + sim_nk.broadcast_mul(&kappa_1k)?)?.broadcast_add(&log_c_1k)?;
        let llik_n = log_topic_nk.log_sum_exp(1)?;

        let recon_nd = z_nk.exp()?.matmul(&mu_kd)?;
        Ok((recon_nd, llik_n))
    }

    fn dim_obs(&self) -> usize {
        self.core.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }

    fn build_ess_llik<'a>(
        &'a self,
        x_nd: &'a Tensor,
        topic_smoothing: f64,
    ) -> Result<EssLlikFn<'a>> {
        let mu_kd = self.core.mu_kd()?.detach();
        let x_log = (x_nd + 1.0)?.log()?;
        let x_hat = l2_normalize_dim(&x_log, 1)?;
        let sim_nk = x_hat.matmul(&mu_kd.t()?)?.detach();

        let kappas = self.core.clamped_kappa_vec()?;
        let kappa_1k: Vec<f32> = kappas.iter().map(|&k| k as f32).collect();
        let log_c_1k: Vec<f32> = kappas
            .iter()
            .map(|&k| vmf_log_normalizer(self.core.n_features, k) as f32)
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

///////////////////////////////////////////////
// Housekeeping-Inflated vMF Topic Decoder   //
///////////////////////////////////////////////

/// vMF topic decoder with housekeeping-inflated mixture of two vMFs:
/// `π_n · vMF(x; μ_null, κ_null) + (1-π_n) · Σ_k z_nk · vMF(x; μ_k, κ_k)`.
/// `μ_null = L2norm(exp(log_hk))` from the fixed hk profile; `κ_null` is frozen.
pub struct BgmVmfTopicDecoder {
    n_topics: usize,
    core: VmfCore,
    bgm: BgmState,
    log_kappa_null: Tensor,
}

impl BgmVmfTopicDecoder {
    pub fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        let core = VmfCore::new(n_features, n_topics, &vs)?;
        let bgm = BgmState::new(n_features, &vs)?;
        let log_kappa_null = vs.get_with_hints(
            (1,),
            "mixture.log_kappa_null.bgm_profile",
            candle_nn::Init::Const(10.0_f64.ln()),
        )?;
        Ok(Self {
            n_topics,
            core,
            bgm,
            log_kappa_null,
        })
    }

    pub fn pi(&self) -> Result<f32> {
        self.bgm.pi()
    }

    /// `[1, D]` unit-norm null direction from the fixed hk profile.
    fn mu_null_1d(&self) -> Result<Tensor> {
        let bgm_exp = self.bgm.log_bgm_1d.exp()?;
        l2_normalize_dim(&bgm_exp, 1)
    }

    fn kappa_null_f64(&self) -> Result<f64> {
        let v: Vec<f64> = self
            .log_kappa_null
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F64)?
            .to_vec1::<f64>()?;
        Ok(v[0].min(MAX_LOG_KAPPA).exp())
    }
}

impl NewDecoder for BgmVmfTopicDecoder {
    fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self> {
        BgmVmfTopicDecoder::new(n_features, n_topics, vs)
    }
}

impl DecoderModuleT for BgmVmfTopicDecoder {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
        let mu_kd = self.core.mu_kd()?;
        z_nk.exp()?.matmul(&mu_kd)
    }

    fn get_dictionary(&self) -> Result<Tensor> {
        self.core.mu_kd()?.t()
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
        let x_log = (x_nd + 1.0)?.log()?;
        let x_hat = l2_normalize_dim(&x_log, 1)?;

        let mu_kd = self.core.mu_kd()?;
        let sim_nk = x_hat.matmul(&mu_kd.t()?)?;
        let kappa_1k = self.core.clamped_kappa_1k()?;
        let log_c_1k = self.core.log_normalizer_1k(dev)?;
        let log_topic_nk = (z_nk + sim_nk.broadcast_mul(&kappa_1k)?)?.broadcast_add(&log_c_1k)?;
        let log_topic_n = log_topic_nk.log_sum_exp(1)?;

        let mu_null_1d = self.mu_null_1d()?;
        let sim_null_n1 = x_hat.matmul(&mu_null_1d.t()?)?;
        let kappa_null = self.kappa_null_f64()?;
        let log_c_null = vmf_log_normalizer(self.core.n_features, kappa_null) as f32;
        let log_null_n = sim_null_n1
            .affine(kappa_null, log_c_null as f64)?
            .squeeze(1)?;

        let pi = ops::sigmoid(&self.bgm.logit_pi)?;
        let log_pi_n = pi.log()?.squeeze(0)?;
        let log_1mpi_n = pi.affine(-1.0, 1.0)?.log()?.squeeze(0)?;

        let bgm_term_n = (log_pi_n + log_null_n)?;
        let topic_term_n = (log_1mpi_n + log_topic_n)?;
        let stacked = Tensor::stack(&[&bgm_term_n, &topic_term_n], 1)?;
        let llik_n = stacked.log_sum_exp(1)?;

        let recon_nd = z_nk.exp()?.matmul(&mu_kd)?;
        Ok((recon_nd, llik_n))
    }

    fn dim_obs(&self) -> usize {
        self.core.n_features
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
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
