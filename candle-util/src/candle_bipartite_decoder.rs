use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

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
    /// Create a new bipartite decoder with diagonal R (NMF-like).
    ///
    /// score(f,c) = Σ_k z_fk · r_k · z_ck + b_f + b_c
    pub fn new_diagonal(
        n_topics: usize,
        n_features: usize,
        n_cells: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let relation = vb.get_with_hints(
            n_topics,
            "relation",
            candle_nn::Init::Const(1.0), // init to 1 → starts as pure dot product
        )?;

        let feature_bias =
            vb.get_with_hints((1, n_features), "feature_bias", candle_nn::Init::Const(0.0))?;

        let cell_bias =
            vb.get_with_hints((1, n_cells), "cell_bias", candle_nn::Init::Const(0.0))?;

        Ok(Self {
            relation,
            diagonal: true,
            feature_bias,
            cell_bias,
        })
    }

    /// Create a new bipartite decoder with full R [K, K].
    ///
    /// score(f,c) = z_f^T · R · z_c + b_f + b_c
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

        let feature_bias =
            vb.get_with_hints((1, n_features), "feature_bias", candle_nn::Init::Const(0.0))?;

        let cell_bias =
            vb.get_with_hints((1, n_cells), "cell_bias", candle_nn::Init::Const(0.0))?;

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

    /// Compute symmetric multinomial log-likelihoods.
    ///
    /// Returns (llik_col, llik_row):
    /// - llik_col = Σ_{f,c} A_{fc} · log p(f|c) — column-wise multinomial
    /// - llik_row = Σ_{f,c} A_{fc} · log p(c|f) — row-wise multinomial
    pub fn forward_symmetric_llik(
        &self,
        log_z_f: &Tensor,
        log_z_c: &Tensor,
        a_dn: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let scores = self.forward_scores(log_z_f, log_z_c)?; // [D_l, N_l]

        // Column-wise: p(f|c) = softmax over features (dim 0)
        let log_p_col = candle_nn::ops::log_softmax(&scores, 0)?;
        let llik_col = (a_dn * &log_p_col)?.sum_all()?;

        // Row-wise: p(c|f) = softmax over cells (dim 1)
        let log_p_row = candle_nn::ops::log_softmax(&scores, 1)?;
        let llik_row = (a_dn * &log_p_row)?.sum_all()?;

        Ok((llik_col, llik_row))
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
