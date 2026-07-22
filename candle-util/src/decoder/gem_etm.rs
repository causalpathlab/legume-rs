//! Splice-aware embedded-topic decoder for `faba gem-encoder` — the
//! `u + δ → s` head.
//!
//! Two log-dictionaries built from ONE topic embedding `α`, over a **gene**-keyed
//! feature embedding `ρ` shared with the encoder (ETM tying). The **nascent
//! (unspliced) track is the base**, and the mature track is derived from it:
//!
//! ```text
//! ρ^u_g = ρ_g                        nascent  (base)
//! ρ^s_g = ρ_g + δ_g                  mature   = nascent + splice-ratio offset
//!
//! logits^τ[t,g] = b_g + ⟨α_t, ρ^τ_g⟩
//! β^τ[t,g]      = softmax_g( logits^τ[t,·] )        Σ_g β^τ[t,g] = 1
//! p^τ[n,g]      = Σ_t θ[n,t] · β^τ[t,g]             Σ_g p^τ[n,g] = 1
//! ```
//!
//! Each track is normalized over the **full gene vocabulary separately**, so
//! `p^s` and `p^u` are each a distribution over genes — which is what makes the
//! nascent/mature comparison compositional rather than depth-driven.
//!
//! # Why the per-gene bias `b_g` exists
//!
//! Without it, nothing in the model can hold the **background** — the mean
//! expression profile shared by every cell — so a *factor* gets conscripted to
//! hold it, and since the background dominates the likelihood that factor takes
//! essentially the whole simplex. Measured on a six-sample fit before `b_g`
//! existed: effective latent rank **1.0–1.7 out of K = 20** in every
//! configuration tried (NB, multinomial, nonzero-only, hard clamp, soft
//! clamp). The collapse is not tied to a particular factor — one run pinned the
//! first stick fully OFF and still collapsed onto a later one.
//!
//! The asymmetry that creates the problem: the ENCODER divides the per-gene mean
//! out of its input ([`crate::value_transform::anscombe_residual`]'s `mu_f`), so
//! it sees each cell's deviation from background, while the decoder is scored on
//! the absolute composition, background included. `b_g` is where that background
//! belongs. It costs `G` parameters against `ρ`'s `G × H`.
//!
//! None of this is new — it is [`crate::nn::linear::log_softmax_linear`]'s
//! `logit_bias`, whose docs already record that removing it leaves a
//! **multinomial** decoder near-degenerate (≈7× worse train likelihood) while NB
//! tolerates its absence because per-gene dispersion absorbs baseline instead.
//! This decoder shipped without it *and* defaults to multinomial, i.e. the exact
//! case that measurement warns about. The `_nobias` escape hatch there is scoped
//! to the indexed-topic decoder, "where top-K masking already breaks per-gene
//! baseline absorption"; that reasoning does not transfer here, because this
//! decoder normalizes over the **full** gene vocabulary
//! ([`Self::log_partition_from_logits`]) rather than over each cell's top-K.
//!
//! **One bias, shared by both tracks — deliberately.** Separate `b^u_g`, `b^s_g`
//! would make their difference a per-gene splice ratio competing directly with
//! `⟨α_t, δ_g⟩`, draining `δ` of the very signal it exists to carry. Sharing
//! `b_g` cancels it exactly out of the track difference (see
//! [`Self::get_dictionary`] and the `log_dictionary_difference_*` test), so the
//! `δ` estimand below is untouched by its presence.
//!
//! # Why the base is nascent: where `δ`'s gradient comes from
//!
//! Only the DERIVED track's likelihood can move `δ`, because the base track's
//! dictionary does not contain it. Writing `g^τ = ∂L/∂ρ^τ` for each track's own
//! gradient:
//!
//! ```text
//! nascent-base (this):  ∂L/∂ρ = g^u + g^s     ∂L/∂δ = g^s        mature only
//! mature-base:          ∂L/∂ρ = g^u + g^s     ∂L/∂δ = −g^u       nascent only
//! symmetric (ρ ± δ/2):  ∂L/∂ρ = g^u + g^s     ∂L/∂δ = ½(g^s − g^u)
//! ```
//!
//! `ρ` collects both tracks under every choice; only `δ`'s composition changes.
//! Nascent counts come from intronic reads — sparse, and scored on nonzero
//! positions only — so nascent-base is the arrangement that routes the NOISY
//! track's influence into `ρ`, where the mature track also pulls and damps it,
//! while giving the delicate rank-`H` `δ` nothing but well-measured signal.
//!
//! This was measured, not assumed. Flipping to mature-base on the six-sample
//! fit halved the splice-ratio check — `r = 0.144, 0.178` against `0.331–0.361`
//! across five separate nascent-base configurations — which is exactly what the
//! table predicts once `∂L_mature/∂δ ≡ 0`. The flip did raise the latent's
//! effective rank (3.14 → 5.36, 7.34), but that comparison rests on ONE
//! nascent-base run against two mature-base runs that themselves differ by 37 %,
//! so it trades a replicated, validated estimand for an unreplicated diagnostic.
//!
//! The symmetric split is not a compromise worth taking either: it hands `δ`
//! half the clean mature gradient plus half the nascent noise, diluting the
//! signal rather than adding information.
//!
//! **This is a parameterization choice, not an output one.** Which track is the
//! base says nothing about what gets written: `{out}.feature_embedding.parquet`
//! emits the SPLICED program `ρ + δ` (see `faba::gem_encoder::run`), because
//! that is what `faba annotate`, marker projection, and `faba gem` interop read.
//!
//! # The steady-state reading of `δ`
//!
//! The RNA-velocity ODE for gene `g` is
//!
//! ```text
//! du/dt = α_g(t)  − β_g·u          transcription − splicing
//! ds/dt = β_g·u   − γ_g·s          splicing      − degradation
//! ```
//!
//! At steady state `s* = (β_g/γ_g)·u`, i.e. `log s* = log u + log(β_g/γ_g)`.
//! Subtracting the two dictionaries above gives exactly that form:
//!
//! ```text
//! log β^s[t,g] − log β^u[t,g] = ⟨α_t, δ_g⟩ − ( logZ^s[t] − logZ^u[t] )
//! ```
//!
//! so **`⟨α_t, δ_g⟩` is the model's `log(β_g/γ_g)`** — the log
//! splicing-to-degradation ratio, made *topic-resolved* (processing rates are
//! cell-state dependent) and *rank-H* (genes share splice-ratio programs rather
//! than each carrying a free scalar).
//!
//! **What `δ` is not.** `β_g/γ_g` is splicing over *degradation*, so a large
//! `δ_g` says the gene sits mature-heavy at equilibrium — which happens either
//! because it splices quickly OR because its mature mRNA is stable and decays
//! slowly. The model cannot separate those, and nothing downstream should read
//! `δ` as a splicing rate on its own. The steady-state relation is therefore
//! structural, not an interpretation laid on top: `δ` is the only parameter that
//! can express it, which is what makes the `u→s` masking mode
//! ([`crate::vae::masked_gem`]) fit it.
//!
//! **Identifiability.** Both tracks are separately softmax-normalized, so a
//! shift `δ_g → δ_g + c` (constant across genes) moves every `⟨α_t, δ_g⟩` by one
//! per-topic amount that `logZ^s[t]` absorbs. `δ` is identified only up to that
//! per-topic constant; the caller's ridge selects the minimum-norm
//! representative. Its null, `δ = 0`, is exactly "mature composition equals
//! nascent composition" — no differential processing.
//!
//! **Note on the `gem` convention.** `faba gem` uses the OPPOSITE base
//! (`unspliced e_f = β_g + δ_g`, see `faba::gem::run`), so its `δ_g` and this
//! one point in opposite directions and must not be compared without accounting
//! for it.

use crate::loss::nb_log_likelihood_elem;
use candle_core::{Result, Tensor};
use candle_nn::{ops, VarBuilder};

/// Which splice track a scoring call refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    /// Unspliced / nascent — the **base** (`ρ_g`).
    Nascent,
    /// Spliced / mature — base + splice-ratio offset (`ρ_g + δ_g`).
    Mature,
}

impl Track {
    /// Row-name suffix this track carries in gem-format feature axes
    /// (`{gene}/count/{spliced|unspliced}`).
    #[must_use]
    pub fn row_suffix(self) -> &'static str {
        match self {
            Track::Nascent => "/count/unspliced",
            Track::Mature => "/count/spliced",
        }
    }
}

/// Per-cell minibatch target for one track's masked-imputation score.
/// All tensors are at the cell's top-K **gene** positions, in `indices` order.
pub struct GemMaskedTarget<'a> {
    /// `[N, K]` u32 per-cell gene ids (the cell's top-K genes).
    pub indices: &'a Tensor,
    /// `[N, K]` per-cell μ_residual at `indices` (batch offset); `None` ⇒ no
    /// batch offset (factor 1). Ignored by the multinomial score.
    pub residual: Option<&'a Tensor>,
    /// `[N, K]` observed counts of THIS track at `indices`.
    pub values: &'a Tensor,
    /// `[N, 1]` this track's per-cell library size. Ignored by the
    /// multinomial score.
    pub lib: &'a Tensor,
    /// `[N, K]` 1 = masked (scored), 0 = not scored.
    pub mask: &'a Tensor,
    /// `[N, K]` per-gene NB-Fisher weight, or `None` for an unweighted loss.
    ///
    /// **Multinomial only.** Multiplies the per-position log-likelihood so a
    /// gene carrying little information about cell state contributes less to
    /// `β`'s gradient even once it is inside a cell's top-K. This mirrors
    /// senna, where the weight appears in the multinomial-family path
    /// (`traits::indexed::forward_indexed_with_log_beta`) and NOT in
    /// `MaskedNbTarget` — because NB already has per-gene information weighting
    /// through the learnable dispersion `φ_g`, while multinomial has no
    /// per-gene variance parameter and takes every count at face value.
    ///
    /// It is a deliberate BIAS, not a correction: the weighted objective has
    /// expectation `Σ_g w_g λ_g log p_g`, whose simplex maximizer is
    /// `p̂_g ∝ w_g·λ_g` rather than `λ_g`. So `β` estimates a Fisher-TILTED
    /// composition, and a dictionary that looks more lineage-specific under it
    /// is partly guaranteed rather than discovered. The NB path is left as a
    /// clean MLE for that reason.
    pub values_weight: Option<&'a Tensor>,
}

/// Splice-aware ETM decoder: one `α`, one gene-keyed `ρ`, one splice-ratio `δ`.
pub struct GemEtmDecoder {
    n_genes: usize,
    n_topics: usize,
    /// `α [K, H]` topic embeddings (learnable, decoder scope).
    topic_embeddings: Tensor,
    /// `ρ [G, H]` **NASCENT (unspliced)** gene embeddings — the base, and the
    /// **same** handle as the encoder (ETM tying); gradients from either path
    /// land on the same `Var`.
    ///
    /// Nascent-base is the measured choice, not a convention: it is what leaves
    /// only the DERIVED track's likelihood able to move `δ`. Flipping it halved
    /// the splice-ratio check (r = 0.144 / 0.178 against 0.331–0.361 across five
    /// nascent-base configs), and `{out}.model.json` records it as
    /// `"delta_base": "unspliced"`.
    feature_embeddings: Tensor,
    /// `δ [G, H]` splice-ratio offset — the **same** handle as the encoder.
    /// The MATURE embedding is `ρ + δ`; see [`GemEtmDecoder::track_embedding`].
    ///
    /// NOTE the sign is the OPPOSITE of `faba gem`'s: there `δ_gem = unspliced −
    /// spliced`, here `δ = spliced − unspliced = −δ_gem`. The two write
    /// same-named `delta_feature_embedding.parquet` files that are NOT
    /// comparable without negating one.
    delta_embeddings: Tensor,
    /// `b [1, G]` per-gene log-background, SHARED by both tracks (see the module
    /// header). Same name, shape and zero-init as
    /// [`crate::nn::linear::log_softmax_linear`]'s `logit_bias`, which is where
    /// this pattern is established and measured.
    ///
    /// Unpenalized: it is the nuisance parameter we WANT to absorb the mean.
    logit_bias: Tensor,
    /// `log φ [2, G]` per-track, per-gene NB inverse dispersion (learnable).
    /// Row 0 = nascent, row 1 = mature; the two tracks are genuinely
    /// differently over-dispersed, so they do not share a row.
    log_phi_2g: Tensor,
}

impl GemEtmDecoder {
    /// Construct with shared encoder handles (`encoder.feature_embeddings()`
    /// and `encoder.delta_embeddings()`, both cloned). `α` is Kaiming-init in
    /// `vs`; `log φ` starts at ln(2) ≈ 0.693 (moderate dispersion).
    pub fn new(
        n_topics: usize,
        feature_embeddings: Tensor,
        delta_embeddings: Tensor,
        vs: VarBuilder,
    ) -> Result<Self> {
        let dims = feature_embeddings.dims();
        if dims.len() != 2 {
            candle_core::bail!(
                "GemEtmDecoder: feature_embeddings must be 2-D [G, H], got {:?}",
                dims
            );
        }
        if delta_embeddings.dims() != dims {
            candle_core::bail!(
                "GemEtmDecoder: delta_embeddings {:?} must match feature_embeddings {:?}",
                delta_embeddings.dims(),
                dims
            );
        }
        let n_genes = dims[0];
        let embedding_dim = dims[1];

        let topic_embeddings = vs.get_with_hints(
            (n_topics, embedding_dim),
            "topic.embeddings",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let log_phi_2g =
            vs.get_with_hints((2, n_genes), "log_phi", candle_nn::Init::Const(0.693))?;
        let logit_bias = vs.get_with_hints((1, n_genes), "logit_bias", candle_nn::init::ZERO)?;

        Ok(Self {
            n_genes,
            n_topics,
            topic_embeddings,
            feature_embeddings,
            delta_embeddings,
            logit_bias,
            log_phi_2g,
        })
    }

    #[must_use]
    pub fn topic_embeddings(&self) -> &Tensor {
        &self.topic_embeddings
    }

    /// This track's gene embedding: `ρ` for nascent, `ρ + δ` for mature.
    fn track_embedding(&self, track: Track) -> Result<Tensor> {
        match track {
            Track::Nascent => Ok(self.feature_embeddings.clone()),
            Track::Mature => &self.feature_embeddings + &self.delta_embeddings,
        }
    }

    /// Full `[K, G]` pre-softmax logits `b_g + α · ρ^τᵀ` for one track.
    ///
    /// This `[K, G]` product is the dominant decoder cost, so the trainer
    /// computes it once per minibatch per track and shares it between
    /// [`Self::log_partition_from_logits`] and any penalty that needs it.
    pub fn full_logits_kg(&self, track: Track) -> Result<Tensor> {
        self.topic_embeddings
            .matmul(&self.track_embedding(track)?.t()?)? // [K, G]
            .broadcast_add(&self.logit_bias) // + b_g, same for every topic
    }

    /// Full `[G, K]` log-β = `log_softmax_g(b_g + α · ρ^τᵀ)` for one track —
    /// the dictionary written to disk.
    pub fn get_dictionary(&self, track: Track) -> Result<Tensor> {
        let logits_kg = self.full_logits_kg(track)?;
        let log_beta_kg = ops::log_softmax(&logits_kg, logits_kg.rank() - 1)?;
        log_beta_kg.transpose(0, 1)?.contiguous()
    }

    /// Per-topic log-partition `log Z_t = logsumexp_g(logits[t,·])` as
    /// `[1, 1, K]`, from precomputed `[K, G]` logits. The sum runs over the
    /// FULL gene vocabulary, which is what puts every gene in the autograd
    /// graph on every step regardless of any cell's top-K.
    pub fn log_partition_from_logits(full_kg: &Tensor) -> Result<Tensor> {
        let k = full_kg.dim(0)?;
        let m = full_kg.max_keepdim(1)?; // [K, 1]
        let lse = (full_kg.broadcast_sub(&m)?.exp()?.sum_keepdim(1)? + 1e-20)?.log()?; // [K,1]
        (lse + m)?.reshape((1, 1, k))
    }

    /// Per-gene NB inverse dispersion `log φ [G]` for one track.
    fn log_phi_track(&self, track: Track) -> Result<Tensor> {
        let row = match track {
            Track::Nascent => 0,
            Track::Mature => 1,
        };
        self.log_phi_2g.narrow(0, row, 1)?.squeeze(0)
    }

    /// Per-track per-gene dispersion `φ` as `[G, 2]` (col 0 nascent, col 1
    /// mature) — the on-disk layout.
    pub fn phi_g2(&self) -> Result<Tensor> {
        self.log_phi_2g.exp()?.transpose(0, 1)?.contiguous()
    }

    /// Per-cell mixture rate `p^τ[n,k] = Σ_t θ[n,t] · β^τ[t,g_k]` at the cell's
    /// top-K genes, with `β^τ` normalized over the **full** gene vocabulary (so
    /// `Σ_g p^τ_g = 1`). Shared core of both scoring heads. `[N, K]`.
    fn mixture_rate_nk(
        &self,
        log_theta_nt: &Tensor,
        indices: &Tensor,
        track: Track,
        logz_11k: &Tensor,
    ) -> Result<Tensor> {
        let n = indices.dim(0)?;
        let k = indices.dim(1)?;
        let t = self.n_topics;

        let theta_nt = log_theta_nt.exp()?; // [N, T]
        let flat = indices.flatten_all()?; // [N*K]

        // ρ^τ at the cell's genes, then β normalized over the full vocab.
        let rho_g = self.track_embedding(track)?.index_select(&flat, 0)?; // [N*K, H]
        let logits = rho_g
            .matmul(&self.topic_embeddings.t()?)? // [N*K, T]
            // + b_g at the cell's genes, the same value for every topic. MUST
            // match `full_logits_kg`, since `logz_11k` is computed from that.
            .broadcast_add(&self.logit_bias.reshape((self.n_genes, 1))?.index_select(&flat, 0)?)? // [N*K, 1]
            .reshape((n, k, t))?; // [N, K, T]
        let beta_nkt = logits.broadcast_sub(logz_11k)?.exp()?; // [N, K, T]

        let theta_n1t = theta_nt.reshape((n, 1, t))?;
        beta_nkt.broadcast_mul(&theta_n1t)?.sum(2) // [N, K]
    }

    /// Masked **negative-binomial** imputation log-likelihood for one track,
    /// summed over masked positions → `[N]`.
    ///
    /// `μ = residual · ℓ^τ · p^τ`, scored as `NB(x^τ | μ, φ^τ_g)`. The library
    /// `ℓ^τ` is that track's own, so the nascent track is not scaled by the
    /// (much larger) mature depth.
    pub fn impute_masked_nb(
        &self,
        log_theta_nt: &Tensor,
        target: &GemMaskedTarget<'_>,
        track: Track,
        logz_11k: &Tensor,
    ) -> Result<Tensor> {
        let (n, k) = (target.indices.dim(0)?, target.indices.dim(1)?);
        let rate_nk = self.mixture_rate_nk(log_theta_nt, target.indices, track, logz_11k)?;

        let mu_nk = match target.residual {
            Some(r) => rate_nk.mul(r)?.broadcast_mul(target.lib)?,
            None => rate_nk.broadcast_mul(target.lib)?,
        };

        let flat = target.indices.flatten_all()?; // [N*K]
        let log_phi_nk = self
            .log_phi_track(track)? // [G]
            .index_select(&flat, 0)?
            .reshape((n, k))?; // [N, K]

        let nb_nk = nb_log_likelihood_elem(target.values, &mu_nk, &log_phi_nk)?; // [N, K]
        // NOTE no Fisher weight here — `φ_g` already carries per-gene
        // information weighting, so this stays a clean MLE (see `values_weight`).
        nb_nk.mul(target.mask)?.sum(1) // [N]
    }

    /// Masked **multinomial** imputation log-likelihood for one track, summed
    /// over masked positions → `[N]`.
    ///
    /// `Σ_{g∈mask} x^τ_g · log p^τ_g`, with `p^τ` normalized over the full
    /// vocab. Depth-invariant: no library, no dispersion, no batch residual —
    /// those shape counts, and this scores composition only. Because the two
    /// tracks are normalized independently, this makes the objective a
    /// function of the nascent and mature **compositions**, the closest thing
    /// to modelling the splice ratio directly.
    pub fn impute_masked_multinomial(
        &self,
        log_theta_nt: &Tensor,
        target: &GemMaskedTarget<'_>,
        track: Track,
        logz_11k: &Tensor,
    ) -> Result<Tensor> {
        let p_nk = self.mixture_rate_nk(log_theta_nt, target.indices, track, logz_11k)?;
        let ll_nk = (target.values * (p_nk + 1e-20)?.log()?)?; // [N, K]
        // `Σ w_g · x_g · log p_g` when a weight is supplied — exactly senna's
        // `traits::indexed::forward_indexed_with_log_beta` form. Multinomial has
        // no per-gene variance parameter, so this is where the weighting has to
        // go if it goes anywhere.
        let w = match target.values_weight {
            Some(w) => target.mask.mul(w)?,
            None => target.mask.clone(),
        };
        ll_nk.mul(&w)?.sum(1) // [N]
    }
}

#[cfg(test)]
#[path = "gem_etm_tests.rs"]
mod tests;
