use crate::candle_indexed_model_traits::*;
use crate::candle_model_traits::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, VarBuilder};

/// Marker-guided encoder using Product-of-Experts.
///
/// Combines two pathways additively in logit space before log_softmax:
///   z_combined = z_data + λ * z_marker
///
/// - Data expert: existing `LogSoftmaxEncoder` producing Gaussian z → logits
/// - Marker expert: `log1p(x) @ M_da @ B_ak` — deterministic topic signal
/// - λ: learnable scalar weight (initialized small)
///
/// KL loss comes from the data expert only (marker expert is deterministic).
pub struct MarkerGuidedEncoder<E: EncoderModuleT> {
    base_encoder: E,
    marker_da: Tensor,     // [D, A] fixed
    marker_linear: Tensor, // [A, K] learnable B_ak
    marker_weight: Tensor, // scalar λ, learnable
}

impl<E: EncoderModuleT> MarkerGuidedEncoder<E> {
    /// Create a marker-guided encoder wrapping a base encoder.
    ///
    /// # Arguments
    /// * `base_encoder` - the data expert (e.g., LogSoftmaxEncoder)
    /// * `marker_da` - [D, A] fixed binary marker membership matrix
    /// * `marker_weight_init` - initial value for λ (e.g., 0.1)
    /// * `vs` - variable builder for learnable parameters
    pub fn new(
        base_encoder: E,
        marker_da: Tensor,
        marker_weight_init: f64,
        vs: VarBuilder,
    ) -> Result<Self> {
        let n_annots = marker_da.dim(1)?;
        let n_topics = base_encoder.dim_latent();

        let marker_linear = vs.get_with_hints(
            (n_annots, n_topics),
            "marker_expert.linear",
            candle_nn::init::ZERO,
        )?;

        let marker_weight = vs.get_with_hints(
            1,
            "marker_expert.weight",
            candle_nn::Init::Const(marker_weight_init),
        )?;

        Ok(Self {
            base_encoder,
            marker_da,
            marker_linear,
            marker_weight,
        })
    }

    /// Compute marker expert logits: log1p(x) @ M_da @ B_ak → [N, K]
    fn marker_expert_logits(&self, x_nd: &Tensor) -> Result<Tensor> {
        let lx_nd = (x_nd + 1.)?.log()?; // log1p(x) [N, D]
        let score_na = lx_nd.matmul(&self.marker_da)?; // [N, A]
        score_na.matmul(&self.marker_linear) // [N, K]
    }
}

impl<E: EncoderModuleT> EncoderModuleT for MarkerGuidedEncoder<E> {
    fn forward_t(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        // Data expert: returns (log_z_nk, kl)
        let (log_z_data, kl) = self.base_encoder.forward_t(x_nd, x0_nd, train)?;

        // Marker expert: deterministic logits
        let z_marker = self.marker_expert_logits(x_nd)?;

        // Combine in logit space: exp(log_z_data) is on simplex,
        // but we want to combine in logit space before softmax.
        // log_z_data is already log_softmax output. Convert back to logits
        // (up to constant), add marker signal, re-softmax.
        // Since log_softmax(x) = x - logsumexp(x), and we want to add
        // λ * z_marker to the logits: log_softmax(x + λ*z_marker)
        // We can use log_z_data as logits (they differ from raw logits by a constant
        // which gets cancelled by log_softmax anyway).
        let lambda = &self.marker_weight;
        let combined = (&log_z_data + lambda.broadcast_mul(&z_marker)?)?;
        let log_prob = ops::log_softmax(&combined, 1)?;

        Ok((log_prob, kl))
    }

    fn dim_latent(&self) -> usize {
        self.base_encoder.dim_latent()
    }
}

/// Marker-guided indexed encoder using Product-of-Experts.
///
/// Same concept as `MarkerGuidedEncoder` but for indexed input.
/// The marker expert operates on the indexed (sparse) input by
/// selecting rows of M_da corresponding to the union indices.
pub struct MarkerGuidedIndexedEncoder<E: IndexedEncoderT> {
    base_encoder: E,
    marker_da: Tensor,     // [D, A] fixed
    marker_linear: Tensor, // [A, K] learnable B_ak
    marker_weight: Tensor, // scalar λ, learnable
    n_features: usize,
}

impl<E: IndexedEncoderT> MarkerGuidedIndexedEncoder<E> {
    pub fn new(
        base_encoder: E,
        marker_da: Tensor,
        marker_weight_init: f64,
        n_features: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        let n_annots = marker_da.dim(1)?;
        let n_topics = base_encoder.dim_latent();

        let marker_linear = vs.get_with_hints(
            (n_annots, n_topics),
            "marker_expert.linear",
            candle_nn::init::ZERO,
        )?;

        let marker_weight = vs.get_with_hints(
            1,
            "marker_expert.weight",
            candle_nn::Init::Const(marker_weight_init),
        )?;

        Ok(Self {
            base_encoder,
            marker_da,
            marker_linear,
            marker_weight,
            n_features,
        })
    }

    /// Compute marker expert logits from indexed input.
    /// Uses M_da sliced at union_indices: M_sa = M_da[union_indices, :]
    fn marker_expert_logits_indexed(
        &self,
        union_indices: &Tensor,
        indexed_x: &Tensor,
    ) -> Result<Tensor> {
        // log1p(indexed_x) [N, S]
        let lx_ns = (indexed_x + 1.)?.log()?;
        // Normalize by total features (not just S) to be consistent
        let denom_n1 = lx_ns.sum_keepdim(lx_ns.rank() - 1)?;
        let lx_ns = (lx_ns.broadcast_div(&denom_n1)? * (self.n_features as f64))?;
        // M_sa = M_da[union_indices, :] → [S, A]
        let marker_sa = self.marker_da.index_select(union_indices, 0)?;
        // score_na = lx_ns [N, S] @ marker_sa [S, A] → [N, A]
        let score_na = lx_ns.matmul(&marker_sa)?;
        // z_marker = score_na [N, A] @ B_ak [A, K] → [N, K]
        score_na.matmul(&self.marker_linear)
    }
}

impl<E: IndexedEncoderT> IndexedEncoderT for MarkerGuidedIndexedEncoder<E> {
    fn forward_indexed_t(
        &self,
        union_indices: &Tensor,
        indexed_x: &Tensor,
        indexed_x_null: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (log_z_data, kl) =
            self.base_encoder
                .forward_indexed_t(union_indices, indexed_x, indexed_x_null, train)?;

        let z_marker = self.marker_expert_logits_indexed(union_indices, indexed_x)?;

        let lambda = &self.marker_weight;
        let combined = (&log_z_data + lambda.broadcast_mul(&z_marker)?)?;
        let log_prob = ops::log_softmax(&combined, 1)?;

        Ok((log_prob, kl))
    }

    fn dim_latent(&self) -> usize {
        self.base_encoder.dim_latent()
    }
}
