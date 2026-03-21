use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::candle_loss_functions::approx_lgamma;

/// Trait for bipartite decoder likelihood functions.
///
/// Given a score matrix S [D_l, N_l] (from BipartiteDecoder::forward_scores)
/// and observed data A [D_l, N_l], compute the total log-likelihood as a
/// scalar tensor.
pub trait BipartiteLikelihood {
    fn compute_llik(&self, scores: &Tensor, a_dn: &Tensor) -> Result<Tensor>;
}

/// Block model multinomial: single softmax over the entire D×N matrix.
///
/// log p(f,c) = S_{fc} - logsumexp(S)
///
/// Treats the count matrix as one multinomial where the probability of
/// each (feature, cell) pair is proportional to exp(z_f · R · z_c).
pub struct BlockModelMultinomial;

impl BipartiteLikelihood for BlockModelMultinomial {
    fn compute_llik(&self, scores: &Tensor, a_dn: &Tensor) -> Result<Tensor> {
        let flat = scores.flatten_all()?;
        let log_p = candle_nn::ops::log_softmax(&flat, 0)?;
        let flat_a = a_dn.flatten_all()?;
        (flat_a * log_p)?.sum_all()
    }
}

/// Symmetric multinomial: softmax over rows + softmax over columns.
pub struct SymmetricMultinomial;

impl BipartiteLikelihood for SymmetricMultinomial {
    fn compute_llik(&self, scores: &Tensor, a_dn: &Tensor) -> Result<Tensor> {
        let log_p_col = candle_nn::ops::log_softmax(scores, 0)?;
        let llik_col = (a_dn * &log_p_col)?.sum_all()?;
        let log_p_row = candle_nn::ops::log_softmax(scores, 1)?;
        let llik_row = (a_dn * &log_p_row)?.sum_all()?;
        &llik_col + &llik_row
    }
}

/// Poisson likelihood: scores are log-rates.
///
/// llik = Σ (x · s - exp(s))  where s = clamp(scores, -20, 20)
pub struct PoissonLikelihood;

impl BipartiteLikelihood for PoissonLikelihood {
    fn compute_llik(&self, scores: &Tensor, a_dn: &Tensor) -> Result<Tensor> {
        let s = scores.clamp(-20.0, 20.0)?;
        (a_dn * &s - s.exp()?)?.sum_all()
    }
}

/// Gaussian likelihood: scores are means.
///
/// llik = -0.5 · Σ (x - s)²
pub struct GaussianLikelihood;

impl BipartiteLikelihood for GaussianLikelihood {
    fn compute_llik(&self, scores: &Tensor, a_dn: &Tensor) -> Result<Tensor> {
        (a_dn - scores)?.powf(2.0)?.sum_all()? * (-0.5)
    }
}

/// Negative binomial likelihood: scores are log-mu, with learned per-feature
/// dispersion parameter phi.
///
/// NB(x; μ, φ): Var(X) = μ + μ²/φ
///
/// log p(x | μ, φ) = lgamma(x + φ) - lgamma(φ) - lgamma(x + 1)
///                  + φ·log(φ/(φ+μ)) + x·log(μ/(φ+μ))
pub struct NbLikelihood {
    /// Per-feature log-dispersion [D_l, 1], lives in VarMap.
    pub log_phi: Tensor,
}

impl BipartiteLikelihood for NbLikelihood {
    fn compute_llik(&self, scores: &Tensor, a_dn: &Tensor) -> Result<Tensor> {
        let log_mu = scores.clamp(-20.0, 20.0)?;
        let log_phi = self.log_phi.clamp(-10.0, 10.0)?;
        let mu = log_mu.exp()?;
        let phi = log_phi.exp()?;

        let phi_plus_mu = phi.broadcast_add(&mu)?;
        let log_phi_plus_mu = phi_plus_mu.log()?;

        let term_phi = phi.broadcast_mul(&log_phi.broadcast_sub(&log_phi_plus_mu)?)?;
        let term_x = a_dn.mul(&log_mu.broadcast_sub(&log_phi_plus_mu)?)?;

        let x_plus_phi = a_dn.broadcast_add(&phi)?;
        let lgamma_term = approx_lgamma(&x_plus_phi)?
            .broadcast_sub(&approx_lgamma(&phi)?)?
            .sub(&approx_lgamma(&(a_dn + 1.0)?)?)?;

        (lgamma_term + term_phi + term_x)?.sum_all()
    }
}

/// Bipartite decoder: reconstructs count matrix A [D_l × N_l] from
/// feature embeddings z_f [D_l, K] and cell embeddings z_c [N_l, K].
///
/// Score: S_{fc} = Σ_k z_fk · r_k · z_ck + b_f + b_c   (diagonal R)
///    or: S_{fc} = z_f^T · R · z_c + b_f + b_c           (full R)
///
/// The decoder operates at coarsened resolution (D_l, N_l) while
/// encoders operate at full resolution. This allows encoder persistence
/// across coarsening levels — only the decoder changes.
pub struct BipartiteDecoder {
    /// Topic interaction weights [K] (diagonal) or [K, K] (full)
    relation: Tensor,
    /// Whether relation is diagonal (K,) or full (K, K)
    diagonal: bool,
    /// Feature bias [D_l] — baseline feature popularity
    feature_bias: Tensor,
    /// Cell bias [N_l] — baseline cell library size effect
    cell_bias: Tensor,
}

impl BipartiteDecoder {
    fn make_biases(n_features: usize, n_cells: usize, vb: &VarBuilder) -> Result<(Tensor, Tensor)> {
        let feature_bias =
            vb.get_with_hints((1, n_features), "feature_bias", candle_nn::Init::Const(0.0))?;
        let cell_bias =
            vb.get_with_hints((1, n_cells), "cell_bias", candle_nn::Init::Const(0.0))?;
        Ok((feature_bias, cell_bias))
    }

    /// Create a new bipartite decoder with diagonal R (NMF-like).
    pub fn new_diagonal(
        n_topics: usize,
        n_features: usize,
        n_cells: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let relation = vb.get_with_hints(n_topics, "relation", candle_nn::Init::Const(1.0))?;
        let (feature_bias, cell_bias) = Self::make_biases(n_features, n_cells, &vb)?;
        Ok(Self {
            relation,
            diagonal: true,
            feature_bias,
            cell_bias,
        })
    }

    /// Create a new bipartite decoder with full R [K, K].
    pub fn new_full(
        n_topics: usize,
        n_features: usize,
        n_cells: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let relation = vb.get_with_hints(
            (n_topics, n_topics),
            "relation",
            candle_nn::Init::Const(0.0),
        )?;
        let (feature_bias, cell_bias) = Self::make_biases(n_features, n_cells, &vb)?;
        Ok(Self {
            relation,
            diagonal: false,
            feature_bias,
            cell_bias,
        })
    }

    /// Compute the score matrix S [D_l, N_l].
    ///
    /// # Arguments
    /// * `log_z_f` - Feature embeddings [D_l, K] (log-softmax from encoder)
    /// * `log_z_c` - Cell embeddings [N_l, K] (log-softmax from encoder)
    ///
    /// Note: D_l, N_l are the coarsened dimensions (decoder resolution).
    /// The embeddings come from mapping full-resolution encoder outputs
    /// through the coarsening.
    pub fn forward_scores(&self, log_z_f: &Tensor, log_z_c: &Tensor) -> Result<Tensor> {
        let z_f = log_z_f.exp()?; // [D_l, K]
        let z_c = log_z_c.exp()?; // [N_l, K]

        let scores = if self.diagonal {
            // Diagonal R: score = (z_f * r) · z_c^T
            // z_f [D_l, K] * r [K] → [D_l, K], then matmul z_c^T [K, N_l] → [D_l, N_l]
            let z_f_scaled = z_f.broadcast_mul(&self.relation)?; // [D_l, K]
            z_f_scaled.matmul(&z_c.t()?)? // [D_l, N_l]
        } else {
            // Full R: score = z_f · R · z_c^T
            let z_f_r = z_f.matmul(&self.relation)?; // [D_l, K]
            z_f_r.matmul(&z_c.t()?)? // [D_l, N_l]
        };

        // Add biases
        let scores = scores
            .broadcast_add(&self.feature_bias.t()?)? // + b_f [D_l, 1]
            .broadcast_add(&self.cell_bias)?; // + b_c [1, N_l]

        Ok(scores)
    }

    /// Compute log-likelihood using a generic likelihood function.
    pub fn forward_llik<L: BipartiteLikelihood + ?Sized>(
        &self,
        log_z_f: &Tensor,
        log_z_c: &Tensor,
        a_dn: &Tensor,
        likelihood: &L,
    ) -> Result<Tensor> {
        let scores = self.forward_scores(log_z_f, log_z_c)?;
        likelihood.compute_llik(&scores, a_dn)
    }

    /// Get the relation parameters for inspection.
    pub fn relation(&self) -> &Tensor {
        &self.relation
    }

    /// Whether the relation is diagonal.
    pub fn is_diagonal(&self) -> bool {
        self.diagonal
    }
}
