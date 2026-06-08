//! Embedded topic decoder with **negative-binomial masked imputation**.
//!
//! The training head of the masked-imputation topic model. Same ETM
//! factorization as [`crate::decoder::EmbeddedTopicDecoder`]
//! (`β = softmax_d(α·ρᵀ)`, `ρ` shared with the encoder), but instead of a
//! multinomial reconstruction it scores a **negative-binomial** likelihood on
//! the **held-out (masked)** genes only:
//!
//! ```text
//! μ_gn = residual_gn · ℓ_n · (θ_n · β)_g          (residual = per-cell μ_residual offset)
//! x_gn ~ NB(μ_gn, φ_g)                            (φ_g per-gene dispersion)
//! llik = Σ_{g ∈ masked} log NB(x_gn | μ_gn, φ_g)
//! ```
//!
//! No KL / no variational posterior is involved — `θ` is the encoder's point
//! estimate — so the objective has no posterior-collapse pressure. Modelling
//! observed counts as `NB(residual · ℓ · θβ)` keeps `β` batch-free (the
//! per-cell `residual` absorbs the batch effect, matching the collapse model
//! `E[y] = μ_residual · μ_adjusted`).

use crate::loss::{log_sigmoid, nb_log_likelihood_elem};
use candle_core::{Result, Tensor};
use candle_nn::{ops, VarBuilder};

/// Per-cell minibatch target for [`EmbeddedNbTopicDecoder::impute_masked_nb`].
/// All tensors are at the cell's top-K positions, in `indices` order.
pub struct MaskedNbTarget<'a> {
    /// `[N, K]` u32 per-cell gene ids (the cell's top-K).
    pub indices: &'a Tensor,
    /// `[N, K]` per-cell μ_residual at `indices` (batch offset); `None` ⇒ no
    /// batch offset (factor 1).
    pub residual: Option<&'a Tensor>,
    /// `[N, K]` observed counts at `indices` (NB target).
    pub values: &'a Tensor,
    /// `[N, 1]` per-cell library size.
    pub lib: &'a Tensor,
    /// `[N, K]` 1 = masked (scored), 0 = visible.
    pub mask: &'a Tensor,
}

/// Per-cell minibatch target for
/// [`EmbeddedNbTopicDecoder::impute_masked_contrastive`]. All `[N, K]` at the
/// cell's encoder top-K positions, in `indices` order.
pub struct MaskedContrastiveTarget<'a> {
    /// `[N, K]` u32 per-cell gene ids (positives — the cell's top-K).
    pub indices: &'a Tensor,
    /// `[N, K]` observed counts at `indices` (used for the count weight).
    pub values: &'a Tensor,
    /// `[N, K]` 1 = masked (scored), 0 = visible.
    pub mask: &'a Tensor,
    /// `[N, K]` u32 `count^α`-sampled negative gene ids.
    pub neg_indices: &'a Tensor,
}

/// NB embedded-topic decoder for masked imputation.
pub struct EmbeddedNbTopicDecoder {
    n_features: usize,
    n_topics: usize,
    /// `α [K, H]` topic embeddings (learnable, decoder scope).
    topic_embeddings: Tensor,
    /// `ρ [D, H]` gene symbol embeddings — the **same** handle as the encoder
    /// (ETM tying); gradients from either path land on the same `Var`.
    feature_embeddings: Tensor,
    /// `log φ_g [1, D]` per-gene NB inverse dispersion (learnable).
    log_phi_1d: Tensor,
    /// `b_g [1, D]` per-gene bias for the **contrastive** head only
    /// (self-normalization, mirroring bge's per-feature bias). Learnable,
    /// zero-init. Deliberately kept **out of `β`** — `get_dictionary` /
    /// `full_logits_kd` never read it, exactly as `log_phi` is NB-only.
    contrastive_bias_1d: Tensor,
}

impl EmbeddedNbTopicDecoder {
    /// Construct with a shared feature-embedding handle
    /// (`encoder.feature_embeddings().clone()`). `α` is Kaiming-init in `vs`;
    /// `log φ` starts at ln(2) ≈ 0.69 (moderate dispersion).
    pub fn new(n_topics: usize, feature_embeddings: Tensor, vs: VarBuilder) -> Result<Self> {
        let dims = feature_embeddings.dims();
        if dims.len() != 2 {
            candle_core::bail!(
                "EmbeddedNbTopicDecoder: feature_embeddings must be 2-D [D, H], got {:?}",
                dims
            );
        }
        let n_features = dims[0];
        let embedding_dim = dims[1];

        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let topic_embeddings =
            vs.get_with_hints((n_topics, embedding_dim), "topic.embeddings", init_ws)?;
        let log_phi_1d =
            vs.get_with_hints((1, n_features), "log_phi", candle_nn::Init::Const(0.693))?;
        let contrastive_bias_1d = vs.get_with_hints(
            (1, n_features),
            "contrastive_bias",
            candle_nn::Init::Const(0.0),
        )?;

        Ok(Self {
            n_features,
            n_topics,
            topic_embeddings,
            feature_embeddings,
            log_phi_1d,
            contrastive_bias_1d,
        })
    }

    pub fn topic_embeddings(&self) -> &Tensor {
        &self.topic_embeddings
    }
    pub fn feature_embeddings(&self) -> &Tensor {
        &self.feature_embeddings
    }
    pub fn log_phi(&self) -> &Tensor {
        &self.log_phi_1d
    }
    pub fn contrastive_bias(&self) -> &Tensor {
        &self.contrastive_bias_1d
    }
    pub fn phi(&self) -> Result<Tensor> {
        self.log_phi_1d.exp()
    }
    pub fn dim_obs(&self) -> usize {
        self.n_features
    }
    pub fn dim_latent(&self) -> usize {
        self.n_topics
    }

    /// Full `[K, D]` pre-softmax logits `α · ρᵀ` (for the anchor-prior CE).
    pub fn full_logits_kd(&self) -> Result<Tensor> {
        self.topic_embeddings.matmul(&self.feature_embeddings.t()?)
    }

    /// Full `[D, K]` log-β = `log_softmax_d(α·ρᵀ)` — for dictionary output.
    pub fn get_dictionary(&self) -> Result<Tensor> {
        let logits_kd = self.full_logits_kd()?;
        let log_beta_kd = ops::log_softmax(&logits_kd, logits_kd.rank() - 1)?;
        log_beta_kd.transpose(0, 1)?.contiguous()
    }

    /// Per-topic log-partition `log Z_k = logsumexp_d(logits_kd)` as `[1, 1, K]`,
    /// from precomputed `[K, D]` logits (see [`Self::full_logits_kd`]). The
    /// `[K, D]` product is the dominant decoder cost, so the caller computes it
    /// once per minibatch and shares it between this partition and the
    /// anchor-prior CE rather than recomputing it inside each.
    pub fn log_partition_from_logits(full_kd: &Tensor) -> Result<Tensor> {
        let k = full_kd.dim(0)?;
        let m = full_kd.max_keepdim(1)?; // [K, 1]
        let lse = (full_kd.broadcast_sub(&m)?.exp()?.sum_keepdim(1)? + 1e-20)?.log()?; // [K,1]
        let logz_k1 = (lse + m)?; // [K, 1]
        logz_k1.reshape((1, 1, k))
    }

    /// Masked NB imputation log-likelihood, summed over masked positions →
    /// `[N]`.
    ///
    /// * `log_theta_nk` — `[N, K_topics]` encoder log-proportions.
    /// * `target` — the per-cell minibatch target (see [`MaskedNbTarget`]).
    /// * `logz_11k` — `[1, 1, K]` per-topic log-partition from
    ///   [`Self::log_partition_from_logits`] (caller-hoisted; see there).
    pub fn impute_masked_nb(
        &self,
        log_theta_nk: &Tensor,
        target: &MaskedNbTarget<'_>,
        logz_11k: &Tensor,
    ) -> Result<Tensor> {
        let MaskedNbTarget {
            indices,
            residual: residual_nk,
            values: values_nk,
            lib: lib_n1,
            mask: mask_nk,
        } = *target;

        let n = indices.dim(0)?;
        let k = indices.dim(1)?;
        let t = self.n_topics;

        let theta_nt = log_theta_nk.exp()?; // [N, T]
        let flat = indices.flatten_all()?; // [N*K]

        // β_kg at the cell's genes, properly normalized over the full vocab.
        let rho_g = self.feature_embeddings.index_select(&flat, 0)?; // [N*K, H]
        let logits = rho_g
            .matmul(&self.topic_embeddings.t()?)? // [N*K, T]
            .reshape((n, k, t))?; // [N, K, T]
        let beta_nkt = logits.broadcast_sub(logz_11k)?.exp()?; // [N, K, T]

        // (θ·β)_nk = Σ_t θ_nt · β_nkt
        let theta_n1t = theta_nt.reshape((n, 1, t))?;
        let theta_beta_nk = beta_nkt.broadcast_mul(&theta_n1t)?.sum(2)?; // [N, K]

        // μ = residual · ℓ · θβ
        let mu_nk = match residual_nk {
            Some(r) => theta_beta_nk.mul(r)?.broadcast_mul(lib_n1)?,
            None => theta_beta_nk.broadcast_mul(lib_n1)?,
        };

        // φ at the cell's genes
        let log_phi_nk = self
            .log_phi_1d
            .squeeze(0)? // [D]
            .index_select(&flat, 0)?
            .reshape((n, k))?; // [N, K]

        let nb_nk = nb_log_likelihood_elem(values_nk, &mu_nk, &log_phi_nk)?; // [N, K]
        nb_nk.mul(mask_nk)?.sum(1) // [N]
    }

    /// Contrastive masked-imputation loss (scalar). Alternative head to
    /// [`Self::impute_masked_nb`]: rather than an NB likelihood on the masked
    /// genes, score each masked-present gene against negatives with a
    /// log-sigmoid (bge-style) objective — importing the contrastive embedding
    /// geometry into the joint gene+topic head. Self-normalizing via the
    /// per-gene `contrastive_bias`, so no `[K, D]` partition is needed. The
    /// dictionary `β = softmax(α·ρᵀ)` (see [`Self::get_dictionary`]) is
    /// unchanged — the bias never enters it.
    ///
    /// * `log_theta_nk` — `[N, T]` encoder log-proportions (T = topics).
    /// * `target` — per-cell positives, counts, mask, and `count^α`-sampled
    ///   negatives (see [`MaskedContrastiveTarget`]).
    /// * `count_alpha` — positives are weighted by `value^count_alpha`, so
    ///   counts re-enter as the contrast weight; loss is normalized by Σweight
    ///   (mirrors the bge NCE reduction).
    /// * `fisher_weights_d` — `[D]` per-gene NB-Fisher weight (the same
    ///   housekeeping-downweighting vector the shortlist uses). Multiplied into
    ///   the positive weight so the contrastive objective is not dominated by
    ///   ubiquitous high-count (housekeeping) genes.
    pub fn impute_masked_contrastive(
        &self,
        log_theta_nk: &Tensor,
        target: &MaskedContrastiveTarget<'_>,
        count_alpha: f64,
        fisher_weights_d: &Tensor,
    ) -> Result<Tensor> {
        let MaskedContrastiveTarget {
            indices,
            values,
            mask,
            neg_indices,
        } = *target;

        let n = indices.dim(0)?;
        let k = indices.dim(1)?;
        let h = self.feature_embeddings.dim(1)?;

        // Cell projection into gene space: θ̃ = θ·α  [N,T]·[T,H] → [N,H].
        let theta_nt = log_theta_nk.exp()?;
        let theta_tilde = theta_nt.matmul(&self.topic_embeddings)?; // [N, H]

        let flat = indices.flatten_all()?; // [N*K]
        let bias_d = self.contrastive_bias_1d.squeeze(0)?; // [D]
                                                           // Score a gene set against the cell projection: θ̃_n · ρ_g + b_g → [N,K].
        let score = |ids: &Tensor| -> Result<Tensor> {
            let f = ids.flatten_all()?;
            let rho = self
                .feature_embeddings
                .index_select(&f, 0)?
                .reshape((n, k, h))?;
            let b = bias_d.index_select(&f, 0)?.reshape((n, k))?;
            theta_tilde
                .unsqueeze(1)?
                .broadcast_mul(&rho)?
                .sum(2)?
                .add(&b)
        };

        let pos = score(indices)?; // present (masked) genes
        let neg_score = score(neg_indices)?; // count^α-sampled negatives

        let per_pos = log_sigmoid(&pos)?
            .add(&log_sigmoid(&neg_score.neg()?)?)?
            .neg()?; // [N, K]

        // Positive weight: w = fisher_g · value^α · mask. The per-gene NB-Fisher
        // weight downweights housekeeping genes, so the contrastive objective is
        // not dominated by — and does not collapse onto — ubiquitous high-count
        // genes. Only masked positions are scored.
        let fisher_g = fisher_weights_d.index_select(&flat, 0)?.reshape((n, k))?; // [N, K]
        let w = values.powf(count_alpha)?.mul(mask)?.mul(&fisher_g)?; // [N, K]
        let w_sum = w.sum_all()?.to_scalar::<f32>()?.max(1e-8);
        per_pos.mul(&w)?.sum_all()? / (w_sum as f64)
    }
}
