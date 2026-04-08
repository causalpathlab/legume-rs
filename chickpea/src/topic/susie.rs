use candle_util::candle_core::{DType, Result, Tensor};
use candle_util::candle_nn::{self, ops, VarBuilder};

const NEG_INF_FILL: f64 = -1e9;

/// Single Effect Regression (SER) component.
struct SingleEffect {
    logit: Tensor, // [G, C] selection logits
    mu: Tensor,    // [G, 1] effect size mean
    lnvar: Tensor, // [G, 1] effect size log-variance
}

/// SuSiE (Sum of Single Effects) for gene-peak linkage M[G, C_max].
///
/// Decomposes M as a sum of L SER components, each selecting one peak per gene.
/// Sparsity is structural (softmax forces each component to pick one peak).
/// Regularized by exact per-component KL: categorical on selection + Gaussian on effects.
pub struct SuSiE {
    components: Vec<SingleEffect>,
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
            neg_inf_mask,
            log_cg,
            n_genes,
            c_max,
        })
    }

    /// Log-softmax selection probabilities, respecting mask.
    fn log_alpha(&self, comp: &SingleEffect) -> Result<Tensor> {
        match &self.neg_inf_mask {
            Some(neg_inf) => ops::log_softmax(&(&comp.logit + neg_inf)?, 1),
            None => ops::log_softmax(&comp.logit, 1),
        }
    }

    /// Softmax selection probabilities, respecting mask.
    fn alpha(&self, comp: &SingleEffect) -> Result<Tensor> {
        self.log_alpha(comp)?.exp()
    }

    /// Forward: returns deterministic M[G, C_max] = Σ_l α_gl · μ_gl (posterior mean).
    pub fn forward(&self) -> Result<Tensor> {
        self.posterior_mean()
    }

    /// Exact SuSiE KL divergence, summed over L components and G genes.
    ///
    /// Per component l, per gene g:
    /// - Categorical: KL(Cat(α_l[g,:]) || Uniform(1/C_g)) where C_g = unmasked peaks
    /// - Gaussian:    KL(N(μ_l[g], exp(lnvar_l[g])) || N(0, prior_var))
    ///
    /// Returns a scalar.
    pub fn kl(&self, prior_var: f64) -> Result<Tensor> {
        let dev = self.components[0].logit.device();
        let mut total = Tensor::new(0f32, dev)?;
        let ln_prior_var = prior_var.ln();

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

        Ok(total)
    }

    /// PIP: 1 - Π_l (1 - α_gl[p]).
    pub fn pip(&self) -> Result<Tensor> {
        let dev = self.components[0].logit.device();
        let ones = Tensor::ones((self.n_genes, self.c_max), DType::F32, dev)?;
        let mut log_complement = Tensor::zeros((self.n_genes, self.c_max), DType::F32, dev)?;

        for comp in &self.components {
            let alpha = self.alpha(comp)?;
            log_complement = (log_complement + (&ones - &alpha)?.log()?)?;
        }

        &ones - log_complement.exp()?
    }

    /// Posterior mean: Σ_l α_gl · μ_gl.
    pub fn posterior_mean(&self) -> Result<Tensor> {
        let dev = self.components[0].logit.device();
        let mut total = Tensor::zeros((self.n_genes, self.c_max), DType::F32, dev)?;

        for comp in &self.components {
            let alpha = self.alpha(comp)?;
            total = (total + alpha.broadcast_mul(&comp.mu)?)?;
        }

        Ok(total)
    }

    /// Posterior variance: Σ_l α_gl · exp(lnvar_gl).
    pub fn posterior_var(&self) -> Result<Tensor> {
        let dev = self.components[0].logit.device();
        let mut total = Tensor::zeros((self.n_genes, self.c_max), DType::F32, dev)?;

        for comp in &self.components {
            let alpha = self.alpha(comp)?;
            total = (total + alpha.broadcast_mul(&comp.lnvar.exp()?)?)?;
        }

        Ok(total)
    }
}
