//! Embedded topic decoder with **negative-binomial masked imputation**.
//!
//! The training head of the masked-imputation topic model. Same ETM
//! factorization as a dense embedded topic decoder
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

use crate::batched_dot::batched_matvec;
use crate::loss::nb_log_likelihood_elem;
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

/// Minibatch target for the **dense** masked heads, laid out over the whole
/// gene axis rather than a per-row context.
///
/// This is the canonical masked-training shape: the encoder reads a bounded
/// context, and the decoder is scored over a prediction space that does not
/// depend on that budget, so a gene the encoder never saw — including one with
/// a zero count — still teaches the dictionary.
pub struct MaskedDenseTarget<'a> {
    /// `[N, D]` observed counts over the full gene axis.
    pub values: &'a Tensor,
    /// `[N, D]` per-row μ_residual (batch offset); `None` ⇒ no offset. Training
    /// scores batch-free target rows and passes `None`; cell-level scoring,
    /// where the counts are batch-mixed, passes the per-cell offset.
    pub residual: Option<&'a Tensor>,
    /// `[N, 1]` per-row library size over the full row.
    pub lib: &'a Tensor,
    /// `[N, D]` 1 = scored, 0 = withheld (the encoder's visible genes).
    pub mask: &'a Tensor,
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
    /// `log π_g [1, D]` per-gene log-background, added to every topic's logits
    /// and **pinned** at the data's gene marginal (see [`pin_background`]).
    ///
    /// Centering `α` over the topic axis makes each gene's total log-mass a
    /// conserved quantity, so a gene abundant in *every* cell has nowhere to
    /// put that abundance: the only escapes are a permanently dead
    /// "background" topic or diffuse `θ`. `log π_g` is that home. It must stay
    /// frozen — a learnable per-gene bias shifts all `K` topics equally on a
    /// gene, which is exactly the shared direction centering removes, and the
    /// optimizer would reinstate it through this parameter.
    log_pi_1d: Tensor,
}

/// Name of the pinned background var inside a decoder's scope.
pub const BACKGROUND_VAR: &str = "log_pi";

/// Pin a decoder's background at `log_pi_1d` `[1, D]` by writing the var named
/// `{prefix}.log_pi` in `varmap`. The decoder holds a handle into that var, so
/// the value takes effect immediately and round-trips through `VarMap::save`;
/// the trainer excludes every `*.log_pi` var from the optimizer.
pub fn pin_background(varmap: &candle_nn::VarMap, prefix: &str, log_pi_1d: &Tensor) -> Result<()> {
    let name = format!("{prefix}.{BACKGROUND_VAR}");
    let tbl = varmap.data().lock().unwrap();
    let var = tbl
        .get(&name)
        .ok_or_else(|| candle_core::Error::Msg(format!("no var `{name}` to pin")))?;
    var.set(&log_pi_1d.to_device(var.device())?.to_dtype(var.dtype())?)
}

/// `log π_g` from a per-gene mean expression `mean_d` (any non-negative scale):
/// the normalized marginal, floored so an unobserved gene keeps a finite
/// background.
pub fn log_background_from_mean(mean_d: &[f32], device: &candle_core::Device) -> Result<Tensor> {
    let total: f64 = mean_d.iter().map(|&m| f64::from(m.max(0.0))).sum();
    let d = mean_d.len().max(1) as f64;
    let floor = 1e-3 / d;
    let log_pi: Vec<f32> = mean_d
        .iter()
        .map(|&m| {
            let p = if total > 0.0 {
                f64::from(m.max(0.0)) / total
            } else {
                1.0 / d
            };
            p.max(floor).ln() as f32
        })
        .collect();
    Tensor::from_vec(log_pi, (1, mean_d.len()), device)
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
        // Uniform until pinned: a constant is a no-op under the gene-axis
        // log_softmax, so an unpinned decoder is exactly background-free.
        let log_pi_1d = vs.get_with_hints(
            (1, n_features),
            BACKGROUND_VAR,
            candle_nn::Init::Const(-(n_features as f64).ln()),
        )?;

        Ok(Self {
            n_features,
            n_topics,
            topic_embeddings,
            feature_embeddings,
            log_phi_1d,
            log_pi_1d,
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
    /// Pinned per-gene log-background `[1, D]`.
    pub fn log_background(&self) -> &Tensor {
        &self.log_pi_1d
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

    /// Full `[K, D]` pre-softmax logits `(α - ᾱ) · ρᵀ + log π_g`.
    ///
    /// `α` is centered over the topic axis first. The raw loading `α · ρᵀ` is
    /// dominated by a shared "abundance" direction (the mean archetype `ᾱ`) that
    /// ranks the same genes top in *every* topic; because each row is normalized
    /// independently, raising all `K` topics on those genes is a direction the
    /// whole dictionary descends at once, and nothing opposes it — the
    /// off-diagonal response of one topic to another is exactly zero.
    ///
    /// Subtracting `ᾱ` is a projection, so `Σ_k (α_k - ᾱ)·ρ_g = 0` holds for
    /// every gene at every step and for all parameter values: each gene's total
    /// log-mass becomes a conserved quantity and one topic can only gain on a
    /// gene at another's expense (`∂/∂S_jg` of topic `k` is `δ_jk - 1/K`, which
    /// is strictly negative off-diagonal). There is no coefficient to balance
    /// and no way to switch it off.
    ///
    /// This is the single chokepoint for the `[K, D]` logits — the dictionary,
    /// the log-partition and the training likelihood all read it — so the
    /// trained model and the dictionary written to disk stay consistent.
    pub fn full_logits_kd(&self) -> Result<Tensor> {
        self.centered_topic_embeddings()?
            .matmul(&self.feature_embeddings.t()?)?
            .broadcast_add(&self.log_pi_1d)
    }

    /// `α - ᾱ` `[K, H]`: the topic embeddings with the mean archetype removed.
    /// Every consumer of `α` — the full dictionary logits *and* the per-cell
    /// rate at the sampled genes — must go through this, because the
    /// log-partition is taken over the centered logits and the numerator has to
    /// be the same quantity or `β` stops summing to one.
    fn centered_topic_embeddings(&self) -> Result<Tensor> {
        let alpha_mean_1h = self.topic_embeddings.mean_keepdim(0)?;
        self.topic_embeddings.broadcast_sub(&alpha_mean_1h)
    }

    /// Full `[D, K]` log-β = `log_softmax_d((α - ᾱ)·ρᵀ + log π)` — for dictionary output.
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
        Self::log_partition_k1(full_kd)?.reshape((1, 1, k))
    }

    /// Per-topic log-partition as `[K, 1]` — the natural shape for the dense
    /// `[K, D]` normalization. [`Self::log_partition_from_logits`] is the same
    /// quantity reshaped for the indexed head's `[N, K, T]` broadcast.
    pub fn log_partition_k1(full_kd: &Tensor) -> Result<Tensor> {
        let m = full_kd.max_keepdim(1)?; // [K, 1]
        let lse = (full_kd.broadcast_sub(&m)?.exp()?.sum_keepdim(1)? + 1e-20)?.log()?; // [K,1]
        lse + m
    }

    /// Per-row mixture rate `p_ng = Σ_t θ_nt · β_{t,g}` over the **whole** gene
    /// axis, `[N, D]`, so `Σ_g p_ng = 1` exactly.
    ///
    /// One `[N,K] × [K,D]` gemm against the normalized dictionary. The indexed
    /// sibling ([`Self::mixture_rate_nk`]) instead gathers `[N·K, T]` logits,
    /// which is the cheaper shape only while the scored set is a small
    /// per-row context.
    pub fn mixture_rate_nd(&self, log_theta_nk: &Tensor, full_kd: &Tensor) -> Result<Tensor> {
        let logz_k1 = Self::log_partition_k1(full_kd)?; // [K, 1]
        let beta_kd = full_kd.broadcast_sub(&logz_k1)?.exp()?; // [K, D]
        log_theta_nk.exp()?.matmul(&beta_kd) // [N, D]
    }

    /// Per-cell mixture rate `p_nk = Σ_t θ_nt · β_{t,g}` at the cell's top-K
    /// genes, with β normalized over the full vocab (so `Σ_g p_g = 1`). The
    /// shared core of both masked-impute heads; the NB and multinomial
    /// likelihoods differ only in how they score this rate. `[N, K]`.
    ///
    /// The logits are **gathered** from the caller's `full_kd` (see
    /// [`Self::full_logits_kd`]) rather than recomputed, so the numerator and
    /// the partition are one quantity by construction.
    fn mixture_rate_nk(
        &self,
        log_theta_nk: &Tensor,
        indices: &Tensor,
        full_kd: &Tensor,
    ) -> Result<Tensor> {
        let n = indices.dim(0)?;
        let k = indices.dim(1)?;
        let t = self.n_topics;

        let theta_nt = log_theta_nk.exp()?; // [N, T]
        let flat = indices.flatten_all()?; // [N*K]

        let logz_11k = Self::log_partition_from_logits(full_kd)?; // [1, 1, T]
        let logits = full_kd
            .t()?
            .contiguous()? // [D, T]
            .index_select(&flat, 0)? // [N*K, T]
            .reshape((n, k, t))?; // [N, K, T]
        let beta_nkt = logits.broadcast_sub(&logz_11k)?.exp()?; // [N, K, T]

        // Mixture rate `Σ_t β·θ` as a gemm — see `candle_util::batched_dot`.
        batched_matvec(&beta_nkt, &theta_nt) // [N, K]
    }

    /// Masked NB imputation log-likelihood, summed over masked positions →
    /// `[N]`.
    ///
    /// * `log_theta_nk` — `[N, K_topics]` encoder log-proportions.
    /// * `target` — the per-cell minibatch target (see [`MaskedNbTarget`]).
    /// * `full_kd` — `[K, D]` logits from [`Self::full_logits_kd`]
    ///   (caller-hoisted: it is the dominant decoder cost and the anchor-prior
    ///   CE shares it).
    pub fn impute_masked_nb(
        &self,
        log_theta_nk: &Tensor,
        target: &MaskedNbTarget<'_>,
        full_kd: &Tensor,
    ) -> Result<Tensor> {
        let MaskedNbTarget {
            indices,
            residual: residual_nk,
            values: values_nk,
            lib: lib_n1,
            mask: mask_nk,
        } = *target;

        let (n, k) = (indices.dim(0)?, indices.dim(1)?);
        let theta_beta_nk = self.mixture_rate_nk(log_theta_nk, indices, full_kd)?; // [N, K]

        // φ at the cell's genes
        let flat = indices.flatten_all()?; // [N*K]
        let log_phi_nk = self
            .log_phi_1d
            .squeeze(0)? // [D]
            .index_select(&flat, 0)?
            .reshape((n, k))?; // [N, K]

        nb_score(
            values_nk,
            &theta_beta_nk,
            residual_nk,
            lib_n1,
            &log_phi_nk,
            mask_nk,
        )
    }

    /// Masked **multinomial** (categorical) imputation log-likelihood, summed
    /// over masked positions → `[N]`. The MLM-faithful sibling of
    /// [`Self::impute_masked_nb`]: it reuses the identical mixture rate
    /// `p_g = Σ_t θ_t · β_{t,g}` (β normalized over the full vocab, so
    /// `Σ_g p_g = 1`) but scores it as full-vocab categorical cross-entropy
    /// `Σ_{g∈mask} y_g · log p_g` — exactly BERT's MLM loss — instead of a
    /// per-gene NB. Depth-invariant: no library-size, no dispersion `φ`, no
    /// batch `residual` (those shape the NB *counts*; the multinomial models
    /// only relative composition). Sharing `p_g` with the NB head makes an
    /// ELBO-vs-masked comparison differ *only* in the objective, not the
    /// likelihood family.
    pub fn impute_masked_multinomial(
        &self,
        log_theta_nk: &Tensor,
        target: &MaskedNbTarget<'_>,
        full_kd: &Tensor,
    ) -> Result<Tensor> {
        let p_nk = self.mixture_rate_nk(log_theta_nk, target.indices, full_kd)?; // [N, K]
                                                                                 // Categorical cross-entropy at masked positions: Σ y_g · log p_g.
        multinomial_score(target.values, &p_nk, target.mask)
    }

    ////////////////////////////////////////////////////
    // Dense heads — scored over the whole gene axis   //
    ////////////////////////////////////////////////////

    /// Masked NB imputation log-likelihood over the full gene axis → `[N]`.
    ///
    /// The dense sibling of [`Self::impute_masked_nb`]: same likelihood, same
    /// dictionary, but the scored positions come from the caller's `[N, D]`
    /// mask instead of a per-row context, so a gene with a zero count is a
    /// first-class observation rather than one the top-K dropped.
    pub fn impute_dense_nb(
        &self,
        log_theta_nk: &Tensor,
        target: &MaskedDenseTarget<'_>,
        full_kd: &Tensor,
    ) -> Result<Tensor> {
        let rate_nd = self.mixture_rate_nd(log_theta_nk, full_kd)?;
        let log_phi_nd = self.log_phi_1d.broadcast_as(rate_nd.shape())?;
        nb_score(
            target.values,
            &rate_nd,
            target.residual,
            target.lib,
            &log_phi_nd,
            target.mask,
        )
    }

    /// Masked multinomial imputation log-likelihood over the full gene axis →
    /// `[N]`. The dense sibling of [`Self::impute_masked_multinomial`].
    pub fn impute_dense_multinomial(
        &self,
        log_theta_nk: &Tensor,
        target: &MaskedDenseTarget<'_>,
        full_kd: &Tensor,
    ) -> Result<Tensor> {
        let rate_nd = self.mixture_rate_nd(log_theta_nk, full_kd)?;
        multinomial_score(target.values, &rate_nd, target.mask)
    }
}

/////////////////////////////////////////////////////////
// Shared scorers — one definition per likelihood       //
/////////////////////////////////////////////////////////

/// `Σ_{scored} log NB(y | residual · ℓ · rate, φ)`, summed over the last axis.
///
/// Shape-agnostic so the indexed `[N, K]` and dense `[N, D]` heads cannot drift
/// apart: they differ in which positions they score, never in how.
fn nb_score(
    values: &Tensor,
    rate: &Tensor,
    residual: Option<&Tensor>,
    lib_n1: &Tensor,
    log_phi: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    // μ = residual · ℓ · rate
    let mu = match residual {
        Some(r) => rate.mul(r)?.broadcast_mul(lib_n1)?,
        None => rate.broadcast_mul(lib_n1)?,
    };
    let elem = nb_log_likelihood_elem(values, &mu, log_phi)?;
    elem.mul(mask)?.sum(elem.rank() - 1)
}

/// `Σ_{scored} y · log p`, summed over the last axis.
fn multinomial_score(values: &Tensor, rate: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let ll = (values * (rate + 1e-20)?.log()?)?;
    ll.mul(mask)?.sum(ll.rank() - 1)
}

#[cfg(test)]
#[path = "masked_etm_tests.rs"]
mod masked_etm_tests;
