use candle_util::candle_core::{DType, Result, Tensor};
use candle_util::candle_nn::{self, ops, VarBuilder};

/// Single Effect Regression (SER) component.
struct SingleEffect {
    logit: Tensor, // [G, C] selection logits
    mu: Tensor,    // [G, 1] effect size mean
    lnvar: Tensor, // [G, 1] effect size log-variance
}

/// SuSiE (Sum of Single Effects) for gene-peak linkage M[G, C_max].
///
/// Decomposes M as a sum of L SER components, each selecting one peak per gene.
/// Sparsity is structural (softmax forces each component to pick one peak),
/// no KL penalty needed.
pub struct SuSiE {
    components: Vec<SingleEffect>,
    neg_inf_mask: Option<Tensor>, // [G, C] pre-computed: (1-mask)*(-1e9), or None
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

        // Pre-compute neg_inf mask once: (1 - mask) * (-1e9)
        let neg_inf_mask = match mask {
            Some(ref m) => Some(((Tensor::ones_like(m)? - m)? * (-1e9f64))?),
            None => None,
        };

        Ok(Self {
            components,
            neg_inf_mask,
            n_genes,
            c_max,
        })
    }

    /// Compute softmax selection probabilities for a component, respecting mask.
    fn alpha(&self, comp: &SingleEffect) -> Result<Tensor> {
        match &self.neg_inf_mask {
            Some(neg_inf) => ops::softmax(&(&comp.logit + neg_inf)?, 1),
            None => ops::softmax(&comp.logit, 1),
        }
    }

    /// Forward: returns M[G, C_max] = Σ_l α_gl · effect_gl.
    pub fn forward(&self, train: bool) -> Result<Tensor> {
        let dev = self.components[0].logit.device();
        let mut total = Tensor::zeros((self.n_genes, self.c_max), DType::F32, dev)?;

        for comp in &self.components {
            let alpha = self.alpha(comp)?;

            let effect = if train {
                let eps = Tensor::randn(0f32, 1f32, (self.n_genes, 1), dev)?;
                (&comp.mu + ((&comp.lnvar * 0.5)?.exp()? * eps)?)?
            } else {
                comp.mu.clone()
            };

            total = (total + alpha.broadcast_mul(&effect)?)?;
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
