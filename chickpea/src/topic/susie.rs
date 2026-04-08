use candle_util::candle_core::{DType, Result, Tensor};
use candle_util::candle_nn::{self, ops, VarBuilder};

const NEG_INF_FILL: f64 = -1e9;
const GATE_EPS: f64 = 1e-4;

/// Single Effect Regression (SER) component.
struct SingleEffect {
    logit: Tensor, // [G, C] selection logits
    mu: Tensor,    // [G, 1] effect size mean
    lnvar: Tensor, // [G, 1] effect size log-variance
}

/// SuSiE (Sum of Single Effects) for gene-peak linkage M[G, C_max].
///
/// Each gene has a learned inclusion gate π_g ∈ (ε, 1-ε) that scales the
/// linkage effect. Null genes learn gate≈0 (no causal peaks).
/// Within included genes, L SER components each select one peak via softmax.
pub struct SuSiE {
    components: Vec<SingleEffect>,
    gate_logit: Tensor, // [G, 1] per-gene inclusion gate
    neg_inf_mask: Option<Tensor>,
    log_cg: Tensor, // [G] precomputed log(valid peaks per gene) for categorical KL
    pub n_genes: usize,
    pub c_max: usize,
}

impl SuSiE {
    pub fn new(
        n_genes: usize,
        c_max: usize,
        n_components: usize,
        mask: Option<Tensor>,
        vs: VarBuilder,
    ) -> Result<Self> {
        let v0: f64 = 1.0; // initial effect size scale
        let mut components = Vec::with_capacity(n_components);
        for l in 0..n_components {
            let vs_l = vs.pp(format!("ser_{}", l));

            let logit =
                vs_l.get_with_hints((n_genes, c_max), "logit", candle_nn::Init::Const(0.0))?;

            let mu_init = candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.1,
            };
            let mu = vs_l.get_with_hints((n_genes, 1), "mu", mu_init)?;

            let lnvar =
                vs_l.get_with_hints((n_genes, 1), "lnvar", candle_nn::Init::Const(v0.ln()))?;

            components.push(SingleEffect { logit, mu, lnvar });
        }

        let gate_logit =
            vs.get_with_hints((n_genes, 1), "gate_logit", candle_nn::Init::Const(0.0))?;

        let dev = components[0].logit.device();

        let neg_inf_mask = match mask {
            Some(ref m) => Some(((Tensor::ones_like(m)? - m)? * NEG_INF_FILL)?),
            None => None,
        };

        // Precompute per-gene log(C_g) for categorical KL prior
        let log_cg = match &neg_inf_mask {
            Some(neg_inf) => {
                let valid = neg_inf.ge(NEG_INF_FILL / 10.0)?.to_dtype(DType::F32)?;
                valid.sum(1)?.clamp(1.0, f64::MAX)?.log()?
            }
            None => Tensor::new(vec![(c_max as f32).ln(); n_genes], dev)?,
        };

        Ok(Self {
            components,
            gate_logit,
            neg_inf_mask,
            log_cg,
            n_genes,
            c_max,
        })
    }

    /// Per-gene inclusion gate π_g ∈ (ε, 1-ε), smoothed sigmoid.
    fn gate(&self) -> Result<Tensor> {
        let sig = ops::sigmoid(&self.gate_logit)?;
        (sig * (1.0 - 2.0 * GATE_EPS))? + GATE_EPS
    }

    fn log_alpha(&self, comp: &SingleEffect) -> Result<Tensor> {
        match &self.neg_inf_mask {
            Some(neg_inf) => ops::log_softmax(&(&comp.logit + neg_inf)?, 1),
            None => ops::log_softmax(&comp.logit, 1),
        }
    }

    fn alpha(&self, comp: &SingleEffect) -> Result<Tensor> {
        self.log_alpha(comp)?.exp()
    }

    /// Forward: returns gated M[G, C_max] = π_g · Σ_l α_gl · μ_gl.
    pub fn forward(&self) -> Result<Tensor> {
        let raw = self.posterior_mean_raw()?;
        raw.broadcast_mul(&self.gate()?)
    }

    /// KL divergence: per-component categorical + Gaussian, plus gate Bernoulli KL.
    pub fn kl(&self, prior_var: f64, gate_prior: f64) -> Result<Tensor> {
        let dev = self.components[0].logit.device();
        let mut total = Tensor::new(0f32, dev)?;
        let ln_prior_var = prior_var.ln();

        /* Per-component KL */
        for comp in &self.components {
            let log_alpha = self.log_alpha(comp)?;
            let alpha = log_alpha.exp()?;

            let neg_entropy = (&alpha * &log_alpha)?.sum(1)?;
            let cat_kl = (neg_entropy + &self.log_cg)?.sum_all()?;

            let var_g = comp.lnvar.exp()?;
            let mu_sq = comp.mu.powf(2.)?;
            let scaled = ((&var_g + &mu_sq)? / prior_var)?;
            let shifted = ((scaled - 1.0)? - &comp.lnvar)?;
            let gauss_kl = ((shifted + ln_prior_var)?.sum_all()? * 0.5)?;

            total = ((&total + &cat_kl)? + &gauss_kl)?;
        }

        /* Gate Bernoulli KL: Σ_g [π log(π/π₀) + (1-π) log((1-π)/(1-π₀))] */
        let pi = self.gate()?;
        let one_minus_pi = (1.0 - &pi)?;
        let log_p0 = gate_prior.ln();
        let log_1mp0 = (1.0 - gate_prior).ln();
        let term1 = (&pi * (pi.log()? - log_p0)?)?;
        let term2 = (&one_minus_pi * (one_minus_pi.log()? - log_1mp0)?)?;
        let gate_kl = (term1 + term2)?.sum_all()?;
        total = (&total + &gate_kl)?;

        Ok(total)
    }

    /// PIP: π_g · [1 - Π_l (1 - α_gl[p])].
    pub fn pip(&self) -> Result<Tensor> {
        let dev = self.components[0].logit.device();
        let ones = Tensor::ones((self.n_genes, self.c_max), DType::F32, dev)?;
        let mut log_complement = Tensor::zeros((self.n_genes, self.c_max), DType::F32, dev)?;

        for comp in &self.components {
            let alpha = self.alpha(comp)?;
            log_complement = (log_complement + (&ones - &alpha)?.log()?)?;
        }

        let raw_pip = (&ones - log_complement.exp()?)?;
        raw_pip.broadcast_mul(&self.gate()?)
    }

    /// Ungated posterior mean: Σ_l α_gl · μ_gl.
    fn posterior_mean_raw(&self) -> Result<Tensor> {
        let dev = self.components[0].logit.device();
        let mut total = Tensor::zeros((self.n_genes, self.c_max), DType::F32, dev)?;

        for comp in &self.components {
            let alpha = self.alpha(comp)?;
            total = (total + alpha.broadcast_mul(&comp.mu)?)?;
        }

        Ok(total)
    }

    /// Gated posterior mean.
    pub fn posterior_mean(&self) -> Result<Tensor> {
        self.posterior_mean_raw()?.broadcast_mul(&self.gate()?)
    }

    /// Gated posterior variance.
    pub fn posterior_var(&self) -> Result<Tensor> {
        let dev = self.components[0].logit.device();
        let mut total = Tensor::zeros((self.n_genes, self.c_max), DType::F32, dev)?;

        for comp in &self.components {
            let alpha = self.alpha(comp)?;
            total = (total + alpha.broadcast_mul(&comp.lnvar.exp()?)?)?;
        }

        let gate = self.gate()?;
        total.broadcast_mul(&(&gate * &gate)?)
    }
}
