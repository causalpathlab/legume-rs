use crate::encoder::pooled::{PooledGeneEncoder, PooledGeneEncoderArgs};
use crate::encoder::scatter_pool;
use crate::loss::{gaussian_kl_loss, gaussian_reparameterize};
use crate::traits::indexed::*;
use crate::value_transform::anscombe_lite;
use candle_core::{Result, Tensor};
use candle_nn::{ops, Linear, ModuleT, VarBuilder, VarMap};

use crate::nn::soft_clamp::{soft_clamp, MASKED_LOGIT_CLAMP};

/// Indexed embedding encoder over packed top-K input.
///
/// Consumes `(indices [N, K], values [N, K], values_null [N, K]?)` directly:
/// gathers feature embeddings by id, weights them by Anscombe-stabilized
/// values, and pools across the K positions per cell. No `[N, S]` is ever
/// materialized.
pub struct IndexedEmbeddingEncoder {
    n_features: usize,
    n_topics: usize,
    embedding_dim: usize,
    /// The shared trunk: the attention query (only with `attn_pool`, so
    /// sum-pool users neither allocate nor persist `attn.query` and old
    /// safetensors still load), the FC stack, the batch-norm and the `z_mean`
    /// head. The dense masked read is its [`PooledGeneEncoder::pool`]; the
    /// indexed reads pool here and hand the trunk their `[N, H]`.
    pooled: PooledGeneEncoder,
    z_lnvar: Linear,
}

/// Floor on per-module coverage when it is used as a divisor.
///
/// Bounds `∂u/∂numerator` at `1/EPS_COVERAGE`; see the note in
/// [`IndexedEmbeddingEncoder::module_pool`]. Only binds for modules a cell has
/// effectively not observed, where `u` carries no information anyway.
const EPS_COVERAGE: f32 = 1e-2;

pub struct IndexedEmbeddingEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub embedding_dim: usize,
    pub layers: &'a [usize],
    /// When true, allocate the `attn.query` parameter and use single-query
    /// attention pooling on the masked forward path
    /// ([`IndexedEmbeddingEncoder::forward_indexed_masked`]). Sum-pool users
    /// (dense ELBO / pinto) set this `false` so no `attn.query` var is
    /// registered — keeping their safetensors unchanged.
    pub attn_pool: bool,
    /// Number of learned gene modules `M`. They both compose the feature side
    /// and pool the context, so this is the one knob for either.
    ///
    /// `0` disables it entirely: no new var, unchanged FC input width, byte-identical
    /// safetensors. Callers that do not want it pass `0`.
    pub n_gene_modules: usize,
    /// Rank of a LoRA residual on the feature table (`0`: none). The table
    /// becomes [`crate::feature_embedding::FeatureEmbedding::Lora`]; the owner
    /// seeds its base and keeps it out of the optimizer. Not with modules.
    pub lora_rank: usize,
}

impl IndexedEmbeddingEncoder {
    pub fn new(args: IndexedEmbeddingEncoderArgs, varmap: &VarMap, vb: VarBuilder) -> Result<Self> {
        debug_assert!(!args.layers.is_empty());

        let features = std::sync::Arc::new(if args.lora_rank > 0 {
            if args.n_gene_modules > 0 {
                candle_core::bail!(
                    "a LoRA residual on the feature table does not compose with gene modules"
                );
            }
            crate::feature_embedding::FeatureEmbedding::new_lora(
                args.n_features,
                args.embedding_dim,
                args.lora_rank,
                vb.clone(),
            )?
        } else {
            crate::feature_embedding::FeatureEmbedding::new(
                args.n_features,
                args.n_gene_modules,
                args.embedding_dim,
                vb.clone(),
            )?
        });

        // The trunk: FC stack (embedding_dim + 2M) -> ... -> final_hidden, BN,
        // and the `z_mean` head. The module branch appends `log u` and
        // `log1p(cov)` to the pool, hence 2M; at M = 0 the layer shapes are
        // unchanged. Registered under this same `vb`, so the var names are what
        // they have always been.
        let pooled = PooledGeneEncoder::new(
            std::sync::Arc::clone(&features),
            PooledGeneEncoderArgs {
                layers: args.layers,
                out_dim: args.n_topics,
                attn_pool: args.attn_pool,
                in_dim_extra: 2 * args.n_gene_modules,
            },
            varmap,
            vb.clone(),
        )?;
        let out_dim = *args.layers.last().unwrap();
        let z_lnvar = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.lnvar"))?;

        Ok(Self {
            n_features: args.n_features,
            n_topics: args.n_topics,
            embedding_dim: args.embedding_dim,
            pooled,
            z_lnvar,
        })
    }

    /// Number of learned gene modules `M` (`0` when the branch is disabled).
    pub fn n_gene_modules(&self) -> usize {
        self.features().n_modules()
    }

    /// The module dictionary `[M, H]`, or `None` without modules.
    pub fn module_dictionary(&self) -> Option<&Tensor> {
        self.features().dictionary()
    }

    /// Every feature's module membership `[D, M]`, or `None` when the encoder
    /// has no modules.
    ///
    /// Membership is a parameter, so this is a read rather than a derivation:
    /// what a caller exports is what the model trains with.
    pub fn feature_module_membership(&self) -> Result<Option<Tensor>> {
        self.features().membership()
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn n_topics(&self) -> usize {
        self.n_topics
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// The feature side, for a caller that needs to compose rows itself.
    pub fn features(&self) -> &crate::feature_embedding::FeatureEmbedding {
        self.pooled.features()
    }

    /// A shared handle on the feature side.
    ///
    /// A decoder tied to this encoder must hold THIS, not a composed table:
    /// with modules the table is a computed value, so a held copy would freeze
    /// the feature rows at their initialization while the parameters beneath
    /// them keep moving.
    #[must_use]
    pub fn features_shared(&self) -> std::sync::Arc<crate::feature_embedding::FeatureEmbedding> {
        self.pooled.features_shared()
    }

    /// The composed `[D, H]` table.
    ///
    /// Forming it is the cost the module parameterization exists to avoid, so
    /// this belongs to export and to callers that genuinely need every row at
    /// once, not to a per-step path.
    pub fn feature_embeddings(&self) -> Result<Tensor> {
        self.features().full()
    }

    /// Pool packed top-K input into `[N, H]`.
    ///
    /// 1. v_norm = anscombe_lite(values, values_null, values_mean)  → [N, K]
    ///    (both nulls applied as multiplicative count-rate corrections
    ///    in the same divisive step before Anscombe — see [`anscombe_lite`])
    /// 2. E_nkh  = feature_embeddings.index_select(idx_flat)        → [N, K, H]
    /// 3. h_nh   = Σ_k v_norm[i, k] · E_nkh[i, k, :]                → [N, H]
    ///
    /// `values_null` (per-cell μ_residual) and `values_mean` (per-gene
    /// μ_d) compose into the count-rate "clean" value
    /// `values / (null · mean)` — the cell's biological-deviation rate.
    /// That clean rate is variance-stabilised by `2·sqrt(clean + 3/8)`
    /// (Anscombe) and used as a per-slot scalar gate on ρ. The output is
    /// **always non-negative**, so genes at typical expression still
    /// contribute their full ρ-row magnitude to the pool — the cell
    /// signature is the value-weighted sum across all top-K slots, not
    /// just the slots that deviate from baseline.
    fn preprocess_indexed(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
    ) -> Result<Tensor> {
        let n = indices.dim(0)?;
        let k = indices.dim(1)?;
        let h = self.embedding_dim;

        let flat_idx = indices.flatten_all()?; // [N*K]
        let e_nk_h = self.features().gather(&flat_idx)?.reshape((n, k, h))?; // [N, K, H]

        // Per-slot Anscombe scalar gate on ρ — broadcast across H.
        // `anscombe_lite` divides by (batch null × per-gene mean) then
        // applies `2·sqrt(clean + 3/8)`. The subtractive `a_y − a_null`
        // form (f41754c) zeros out baseline slots and collapses the pool
        // under per-cell LN on batch-corrected data; divisive Anscombe
        // keeps every slot's magnitude information and avoids that.
        let a_nk = anscombe_lite(values, values_null, values_mean)?; // [N, K]
        let v_nkh = e_nk_h.broadcast_mul(&a_nk.unsqueeze(2)?)?; // [N, K, H]
        v_nkh.sum(1) // [N, H]
    }
}

impl IndexedEmbeddingEncoder {
    /// Pool packed top-K into `[N, H]` using **only the visible slots**.
    ///
    /// The same value-gated attention pool as [`Self::preprocess_indexed`]'s
    /// tokens (`a_nk · ρ`) scored against a learned query, PMA-style: scores
    /// for masked / padding slots (`visible_mask [N, K]` == 0) are driven to
    /// −∞ so those genes are excluded from the softmax — the raw `a_nk` gate
    /// is *not* zeroed, so a masked gene's value never leaks into the pool.
    /// Used by the masked-imputation topic model.
    ///
    /// Computed by [`crate::encoder::scatter_pool`], which re-associates the
    /// three sums so no `[N, K, H]` block of gathered rows is ever formed: the
    /// query meets the feature side once (`ρq [D]`), the scores gather from
    /// that vector, and the softmax weights scatter onto `[N, D]` before a
    /// single gemm against ρ. Identical arithmetic in a different order, so
    /// `K` no longer sizes the backward.
    fn preprocess_indexed_masked(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        visible_mask: &Tensor,
    ) -> Result<Tensor> {
        let h = self.embedding_dim;

        // Per-slot Anscombe scalar gate on ρ. Note: `a_nk` uses the raw values
        // for ALL slots; masking is applied to the attention scores (below),
        // not by zeroing the gate — so a masked gene is excluded from pooling
        // but its value isn't leaked.
        let a_nk = anscombe_lite(values, values_null, values_mean)?; // [N, K]

        let attn_query = self
            .pooled
            .attn_query()
            .expect("forward_indexed_masked requires an attn_pool encoder");
        // ρq over every feature, once per minibatch. The query is a parameter,
        // so this moves at every step: it is recomputed, never cached.
        let rq_d = scatter_pool::query_over_features(self.features(), attn_query)?; // [D]
        let scores_nk = scatter_pool::attention_scores_from_vector(
            &a_nk,
            indices,
            &rq_d,
            visible_mask,
            1.0 / (h as f64).sqrt(),
        )?;
        // The softmax weights sum to 1, so the pooled vector is depth-normalized.
        let attn_nk = ops::softmax(&scores_nk, 1)?; // [N, K]
        let pooled_nh = scatter_pool::pool_by_scatter(&attn_nk, &a_nk, indices, self.features())?; // [N, H]

        // A row with no visible slot (empty cell, or all real genes masked at
        // high mask_fraction) has an all-−∞ score row, so softmax degenerates to
        // a uniform average over the padding slots. Zero that pool out so such
        // rows get a defined (bias-driven) empty-cell representation instead of
        // pad-gene content.
        let has_visible_n1 = visible_mask
            .sum_keepdim(1)?
            .gt(0.0)?
            .to_dtype(pooled_nh.dtype())?; // [N, 1]
        let pooled_nh = pooled_nh.broadcast_mul(&has_visible_n1)?; // [N, H]

        // Module branch. Disabled ⇒ return exactly what this function always returned.
        let Some(mem) = self.features().gather_membership(&indices.flatten_all()?)? else {
            return Ok(pooled_nh);
        };
        let (u_nm, cov_nm) = self.module_pool(&mem, &a_nk, visible_mask)?;

        // Plain `log`, NOT centered. Expression is multiplicative so a log belongs
        // here, but a linear layer downstream can already form any log-ratio
        // `log u_j − log u_k`, so centering would add no representational power — it
        // would only delete the overall-level direction, which is mostly (not purely)
        // depth. `cov` is likewise uncentered: its absolute level ("3 of 5 members
        // seen") IS the reliability signal it exists to carry.
        let feats_n2m = Tensor::cat(&[&(u_nm + 1e-6)?.log()?, &(cov_nm + 1.0)?.log()?], 1)?; // [N, 2M]
        Tensor::cat(&[&pooled_nh, &feats_n2m.broadcast_mul(&has_visible_n1)?], 1)
        // [N, H + 2M]
    }

    /// Per-module level `u [N, M]` and coverage `cov [N, M]` for one minibatch.
    ///
    /// `cov[n,j] = Σ_{g∈ctx} m[g,j]` over the slots this cell actually observed, and
    /// `u[n,j]` is the membership-weighted mean of the value gate over the *same*
    /// slots.
    ///
    /// **The denominator is the whole point.** Dividing by the total over all context
    /// genes would make `u[n,j]` shrink whenever module `j`'s members happen to be
    /// missing — exactly the cross-dataset fragility this branch exists to remove.
    /// Dividing by `cov[n,j]` instead makes `u` a mean over the members that *were*
    /// captured, so losing members costs variance, not level.
    fn module_pool(
        &self,
        mem_flat: &Tensor,
        a_nk: &Tensor,
        visible_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (n, k) = a_nk.dims2()?;
        let m = self.n_gene_modules();
        // Membership is a parameter now, so this is a gather rather than a
        // similarity: the grouping the pool uses IS the grouping that composes
        // the feature rows, not a second reading of them.
        let mem_nkm = mem_flat.reshape((n, k, m))?; // [N, K, M]

        // Restrict to observed slots: a masked gene must not contribute to either the
        // level or the coverage, or the branch leaks the value being imputed.
        let mem_vis = mem_nkm.broadcast_mul(&visible_mask.unsqueeze(2)?)?; // [N, K, M]
        let cov_nm = mem_vis.sum(1)?; // [N, M]

        // Divide by a FLOORED coverage. Membership is a sparsemax, so `cov` reaches
        // exactly zero: a module none of this cell's visible slots belongs to has no
        // coverage at all, and everything between that and O(1) is reachable too —
        // one slot holding a sliver of mass is as real as fifty holding all of it.
        // Dividing by it directly leaves `∂u/∂numerator = 1/cov` unbounded, and the
        // downstream `log(u + ε)` multiplies by another `1/(u+ε)`, so a barely-present
        // module would dominate the gradient of the whole branch. The floor bounds
        // that at `1/EPS_COVERAGE` and costs nothing where a module is genuinely
        // present, since real coverage is O(1) or larger.
        let u_nm = mem_vis
            .broadcast_mul(&a_nk.unsqueeze(2)?)?
            .sum(1)?
            .div(&cov_nm.clamp(EPS_COVERAGE, f32::INFINITY)?)?; // [N, M]

        Ok((u_nm, cov_nm))
    }

    /// Diagnostic read-out of the module branch: `(u [N, M], cov [N, M])`, or `None`
    /// when it is disabled.
    ///
    /// Not used by training — the objective is deliberately unchanged by this branch.
    /// It exists so the per-module load histogram can be logged without a second
    /// forward pass. Watch it from the first epoch: without a load-balancing penalty
    /// A module carrying no mass is dead in a strong sense: sparsemax gives it
    /// exact zeros, and its Jacobian is zero there, so no gradient reaches it
    /// and it cannot recover on its own. That is what this read-out is for, and
    /// why capacity is added by splitting a loaded module rather than appending.
    pub fn module_activity(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        visible_mask: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let Some(mem) = self.features().gather_membership(&indices.flatten_all()?)? else {
            return Ok(None);
        };
        let a_nk = anscombe_lite(values, values_null, values_mean)?;
        self.module_pool(&mem, &a_nk, visible_mask).map(Some)
    }

    /// Pool EVERY feature of a dense minibatch row, zeros included, using only
    /// the genes the row's mask leaves visible.
    ///
    /// The window-free sibling of [`Self::preprocess_indexed_masked`]. There is
    /// no context and no padding: `visible_nd` is the whole gene axis, and the
    /// genes it hides are the ones the decoder is scored on. The gate is
    /// [`anscombe_residual`] — the transform the dense encoders already use —
    /// so a zero-count gene enters the softmax with a below-mean gate rather
    /// than not entering it at all. That is the model change; the pooling
    /// itself is [`crate::encoder::dense_pool`], which agrees with stage 1's
    /// scatter pool exactly when the visible mask is a cell's support.
    fn preprocess_dense_masked(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        mean_1d: Option<&Tensor>,
        visible_nd: &Tensor,
    ) -> Result<Tensor> {
        // The module branch pools by membership over CONTEXT SLOTS; with no
        // context there are no slots to pool over, and composing a `[D, M]`
        // membership per row is a different operator, not the same one at
        // another width. Refuse rather than invent one.
        if self.features().n_modules() > 0 {
            candle_core::bail!(
                "the window-free (dense) masked encoder has no gene-module branch: modules pool \
                 a cell by membership over its context slots, and without a context window there \
                 are no slots. Train with --gene-modules 0."
            );
        }
        self.pooled.pool(x_nd, x0_nd, mean_1d, Some(visible_nd))
    }

    /// The `z_mean` head under the masked clamp — the contract of every simplex
    /// head, in one place.
    fn clamped_head(&self, bn_nl: &Tensor, train: bool) -> Result<Tensor> {
        soft_clamp(&self.pooled.head(bn_nl, train)?, MASKED_LOGIT_CLAMP)
    }

    /// Clamped per-topic logits from a pooled `[N, H]` — the pre-activation
    /// every simplex head maps to `log θ`.
    fn logits_from_pool(&self, pooled: &Tensor, train: bool) -> Result<Tensor> {
        let bn_nl = self.pooled.trunk(pooled, train)?;
        self.clamped_head(&bn_nl, train)
    }

    /// Shared masked-encoder trunk: visible-pool → FC → BN → `bn_nl [N, L]`.
    /// All three masked heads (softmax / stick-breaking / Gaussian) branch off
    /// this — only the final projection differs — so the pooling + FC + BN wiring
    /// lives in exactly one place.
    fn masked_hidden(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        visible_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let h_nh = self.preprocess_indexed_masked(
            indices,
            values,
            values_null,
            values_mean,
            visible_mask,
        )?;
        self.pooled.trunk(&h_nh, train)
    }

    /// Clamped per-topic logits `z_mean [N, K]` from the masked trunk — the
    /// pre-activation the simplex heads map to `log θ`.
    fn masked_logits(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        visible_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let bn_nl = self.masked_hidden(
            indices,
            values,
            values_null,
            values_mean,
            visible_mask,
            train,
        )?;
        self.clamped_head(&bn_nl, train)
    }

    /// Deterministic masked-encoder forward → `log θ [N, K_topics]`.
    ///
    /// Pools the **visible** genes, runs the shared trunk, and returns
    /// `log_softmax(z_mean)` — **no reparameterization, no KL**. This is the
    /// masked-imputation topic model's encoder: `θ` is a point estimate, so
    /// there is no posterior-collapse pressure.
    pub fn forward_indexed_masked(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        visible_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let z_mean_nk = self.masked_logits(
            indices,
            values,
            values_null,
            values_mean,
            visible_mask,
            train,
        )?;
        ops::log_softmax(&z_mean_nk, 1)
    }

    /// Deterministic **stick-breaking** masked-encoder forward → `log θ [N,K]`.
    ///
    /// Same shared trunk as [`Self::forward_indexed_masked`]; only the final
    /// simplex map differs — the pre-activation logits go through
    /// [`crate::vae::stick_breaking_log_simplex`] instead of `log_softmax`.
    /// Ordered, exchangeability-broken topics with a self-pruning tail, still a
    /// point estimate (no reparameterization, no KL).
    pub fn forward_indexed_masked_stick(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        visible_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let z_mean_nk = self.masked_logits(
            indices,
            values,
            values_null,
            values_mean,
            visible_mask,
            train,
        )?;
        crate::vae::stick_breaking_log_simplex(&z_mean_nk)
    }

    /// **Gaussian** masked-encoder forward → `z [N, K]`, the unconstrained
    /// latent. Visible-pooled like [`Self::forward_indexed_masked`] with the
    /// softmax left off, and deterministic: no reparameterized draw and no KL.
    /// The masked-imputation objective is the regularizer on this path, as it
    /// is for the simplex heads; a KL toward `N(0, I)` on top of it pulled `z`
    /// to zero and every row's θ to uniform at the default weight.
    pub fn forward_indexed_masked_gaussian(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        visible_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        self.masked_logits(
            indices,
            values,
            values_null,
            values_mean,
            visible_mask,
            train,
        )
    }

    /// Clamped per-topic logits `z_mean [N, K]` from the DENSE masked trunk.
    fn dense_masked_logits(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        mean_1d: Option<&Tensor>,
        visible_nd: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let pooled = self.preprocess_dense_masked(x_nd, x0_nd, mean_1d, visible_nd)?;
        self.logits_from_pool(&pooled, train)
    }

    /// Window-free masked forward → `log θ [N, K_topics]`, the dense sibling of
    /// [`Self::forward_indexed_masked`].
    pub fn forward_dense_masked(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        mean_1d: Option<&Tensor>,
        visible_nd: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let z_mean_nk = self.dense_masked_logits(x_nd, x0_nd, mean_1d, visible_nd, train)?;
        ops::log_softmax(&z_mean_nk, 1)
    }

    /// Window-free **stick-breaking** masked forward → `log θ [N, K]`.
    pub fn forward_dense_masked_stick(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        mean_1d: Option<&Tensor>,
        visible_nd: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let z_mean_nk = self.dense_masked_logits(x_nd, x0_nd, mean_1d, visible_nd, train)?;
        crate::vae::stick_breaking_log_simplex(&z_mean_nk)
    }

    /// Window-free **Gaussian** masked forward → the unconstrained `z [N, K]`.
    pub fn forward_dense_masked_gaussian(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        mean_1d: Option<&Tensor>,
        visible_nd: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        self.dense_masked_logits(x_nd, x0_nd, mean_1d, visible_nd, train)
    }

    /// Compute latent Gaussian parameters from packed indexed input.
    pub fn latent_gaussian_params_indexed(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let h_nh = self.preprocess_indexed(indices, values, values_null, values_mean)?;
        let bn_nl = self.pooled.trunk(&h_nh, train)?;

        let z_mean_nk = self.clamped_head(&bn_nl, train)?;
        let z_lnvar_nk = soft_clamp(&self.z_lnvar.forward_t(&bn_nl, train)?, MASKED_LOGIT_CLAMP)?;

        Ok((z_mean_nk, z_lnvar_nk))
    }
}

impl IndexedEncoderT for IndexedEmbeddingEncoder {
    fn forward_indexed_t(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_mean_nk, z_lnvar_nk) =
            self.latent_gaussian_params_indexed(indices, values, values_null, values_mean, train)?;

        let z_nk = gaussian_reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;
        let log_prob = ops::log_softmax(&z_nk, 1)?;

        Ok((log_prob, gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Hand-build a tiny encoder and verify `preprocess_indexed` pools
    /// packed top-K input into a finite `[N, H]`. The value transform is
    /// the learned intensity-embedding gate (random-init tables), so this
    /// checks shape + finiteness rather than an exact host computation —
    /// the gate's binning/lookup is unit-tested in `candle_value_transform`.
    #[test]
    fn test_preprocess_indexed_shape() {
        let device = Device::Cpu;
        let n_features = 6;
        let embedding_dim = 4;

        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let layers = vec![embedding_dim, embedding_dim];
        let enc = IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features,
                n_topics: 2,
                embedding_dim,
                layers: &layers,
                attn_pool: false,
                n_gene_modules: 0,
                lora_rank: 0,
            },
            &varmap,
            vb,
        )
        .unwrap();

        // Two cells: cell 0 selects features {0,1}, cell 1 selects {2,3}.
        let indices = Tensor::from_vec(vec![0u32, 1, 2, 3], (2, 2), &device).unwrap();
        let values = Tensor::from_vec(vec![4.0f32, 9.0, 16.0, 25.0], (2, 2), &device).unwrap();

        let h = enc
            .preprocess_indexed(&indices, &values, None, None)
            .unwrap();
        assert_eq!(h.dims(), &[2, embedding_dim]);
        for row in h.to_vec2::<f32>().unwrap() {
            for v in row {
                assert!(v.is_finite(), "non-finite pooled value {v}");
            }
        }
    }

    /// Build an encoder and report `(sorted var names, fc-input width)`.
    fn var_signature(n_gene_modules: usize) -> (Vec<String>, usize) {
        let device = Device::Cpu;
        let embedding_dim = 4;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let layers = vec![8, 5];
        IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features: 6,
                n_topics: 2,
                embedding_dim,
                layers: &layers,
                attn_pool: true,
                n_gene_modules,
                lora_rank: 0,
            },
            &varmap,
            vb.pp("enc"),
        )
        .unwrap();

        let data = varmap.data().lock().unwrap();
        let mut names: Vec<String> = data.keys().cloned().collect();
        names.sort();
        // First FC layer: [hidden, in_dim].
        let fc_in = data["enc.nn.enc.fc.relu_linear_stack.0.weight"].dims()[1];
        (names, fc_in)
    }

    /// `n_gene_modules = 0` must leave the checkpoint exactly as it was before the
    /// module branch existed: same vars, same FC input width. This is the
    /// compatibility gate — every model trained before this feature has to keep
    /// loading, and `VarMap::load` errors on any shape mismatch.
    #[test]
    fn module_branch_off_is_indistinguishable() {
        let (names, fc_in) = var_signature(0);
        assert_eq!(
            fc_in, 4,
            "at M = 0 the FC stack must take exactly embedding_dim"
        );
        assert!(
            !names.iter().any(|n| n.contains("modules")),
            "M = 0 must register no module var; got {names:?}"
        );
    }

    /// `M > 0` replaces the per-feature table with the membership and the
    /// dictionary, and widens the FC input by `2M` — `log u` and `log1p(cov)`
    /// per module, nothing else. A checkpoint therefore cannot be loaded across
    /// this switch, which is the intended cost of composing the feature side.
    #[test]
    fn module_branch_adds_one_var_and_widens_by_two_m() {
        let m = 3;
        let (off, fc_off) = var_signature(0);
        let (on, fc_on) = var_signature(m);

        assert_eq!(
            fc_on,
            fc_off + 2 * m,
            "FC input must grow by 2M (level + coverage per module)"
        );

        // With modules the feature side IS the module pair, so the free table
        // is replaced rather than supplemented.
        let added: Vec<&String> = on.iter().filter(|n| !off.contains(n)).collect();
        assert_eq!(
            added,
            vec!["enc.modules.logits", "enc.modules.mu"],
            "the branch must add exactly the membership and the dictionary"
        );
        let dropped: Vec<&String> = off.iter().filter(|n| !on.contains(n)).collect();
        assert_eq!(
            dropped,
            vec!["enc.feature.embeddings"],
            "and must drop the per-feature table it composes away"
        );
        assert!(
            off.iter()
                .filter(|n| *n != "enc.feature.embeddings")
                .all(|n| on.contains(n)),
            "enabling modules must leave every other var alone"
        );
    }

    /// The module read-out must be finite, correctly shaped, and — the property the
    /// whole branch exists for — **masked slots must not contribute**, or the encoder
    /// leaks the value being imputed.
    #[test]
    fn module_activity_respects_the_visible_mask() {
        let device = Device::Cpu;
        let embedding_dim = 4;
        let m = 3;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let layers = vec![8, 5];
        let enc = IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features: 6,
                n_topics: 2,
                embedding_dim,
                layers: &layers,
                attn_pool: true,
                n_gene_modules: m,
                lora_rank: 0,
            },
            &varmap,
            vb.pp("enc"),
        )
        .unwrap();

        let indices = Tensor::from_vec(vec![0u32, 1, 2, 3], (2, 2), &device).unwrap();
        let values = Tensor::from_vec(vec![4.0f32, 9.0, 16.0, 25.0], (2, 2), &device).unwrap();
        // Cell 0 sees both slots; cell 1 sees neither.
        let vis = Tensor::from_vec(vec![1.0f32, 1.0, 0.0, 0.0], (2, 2), &device).unwrap();

        let (u, cov) = enc
            .module_activity(&indices, &values, None, None, &vis)
            .unwrap()
            .expect("module branch is enabled");
        assert_eq!(u.dims(), &[2, m]);
        assert_eq!(cov.dims(), &[2, m]);

        let cov_v = cov.to_vec2::<f32>().unwrap();
        let u_v = u.to_vec2::<f32>().unwrap();
        // Coverage sums to the number of VISIBLE slots, since membership is a
        // distribution over modules for each observed gene.
        assert!(
            (cov_v[0].iter().sum::<f32>() - 2.0).abs() < 1e-4,
            "cell 0 saw 2 slots; got {:?}",
            cov_v[0]
        );
        for (j, &c) in cov_v[1].iter().enumerate() {
            assert!(
                c.abs() < 1e-6,
                "fully masked cell must have zero coverage in module {j}: {c}"
            );
        }
        for row in u_v.iter().chain(cov_v.iter()) {
            for v in row {
                assert!(v.is_finite(), "non-finite module read-out {v}");
            }
        }
    }

    /// Gradients through the module branch must stay bounded for a cell that barely
    /// observed a module. Coverage is never exactly zero (membership is a softmax), but
    /// it can land near 1e-6, and dividing by that unfloored would let an *absent*
    /// module dominate the gradient of the whole branch.
    #[test]
    fn module_gradients_stay_finite_and_bounded() {
        let device = Device::Cpu;
        let embedding_dim = 4;
        let m = 6;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let layers = vec![8, 5];
        let enc = IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features: 6,
                n_topics: 2,
                embedding_dim,
                layers: &layers,
                attn_pool: true,
                n_gene_modules: m,
                lora_rank: 0,
            },
            &varmap,
            vb.pp("enc"),
        )
        .unwrap();

        let indices = Tensor::from_vec(vec![0u32, 1, 2, 3], (2, 2), &device).unwrap();
        let values = Tensor::from_vec(vec![4.0f32, 9.0, 16.0, 25.0], (2, 2), &device).unwrap();
        // Cell 1 has a single visible slot, so most of its modules are effectively
        // unobserved — the regime where an unfloored divisor blows up.
        let vis = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 0.0], (2, 2), &device).unwrap();

        let pooled = enc
            .preprocess_indexed_masked(&indices, &values, None, None, &vis)
            .unwrap();
        assert_eq!(pooled.dims(), &[2, embedding_dim + 2 * m]);

        let loss = pooled.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let vars = varmap.all_vars();
        let dictionary = vars
            .iter()
            .find(|v| v.dims() == [m, embedding_dim])
            .expect("module dictionary var");
        let g = grads
            .get(dictionary)
            .expect("the dictionary must receive gradient");
        for row in g.to_vec2::<f32>().unwrap() {
            for v in row {
                assert!(v.is_finite(), "non-finite dictionary gradient {v}");
                assert!(
                    v.abs() < 1e5,
                    "dictionary gradient {v} is exploding — check the coverage floor"
                );
            }
        }
    }

    /// A disabled branch must report `None` rather than an empty tensor, so callers
    /// cannot silently log a zero-width histogram and conclude nothing collapsed.
    #[test]
    fn module_activity_is_none_when_disabled() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let layers = vec![8, 5];
        let enc = IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features: 6,
                n_topics: 2,
                embedding_dim: 4,
                layers: &layers,
                attn_pool: true,
                n_gene_modules: 0,
                lora_rank: 0,
            },
            &varmap,
            vb.pp("enc"),
        )
        .unwrap();

        let indices = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        let values = Tensor::from_vec(vec![4.0f32, 9.0], (1, 2), &device).unwrap();
        let vis = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device).unwrap();
        assert!(enc
            .module_activity(&indices, &values, None, None, &vis)
            .unwrap()
            .is_none());
    }
    /// The membership the encoder pools over and the membership it exports are
    /// now the same parameter, so what a reader gets is what the model used.
    /// This pins that they agree row for row, and that the rows are
    /// distributions with the exact zeros sparsemax is chosen for.
    #[test]
    fn the_pooled_and_exported_membership_are_one_parameter() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, candle_core::DType::F32, &dev);
        let (d, h, m) = (5usize, 4usize, 3usize);
        let layers = vec![h, h];
        let enc = IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features: d,
                n_topics: 2,
                embedding_dim: h,
                layers: &layers,
                attn_pool: false,
                n_gene_modules: m,
                lora_rank: 0,
            },
            &vm,
            vb,
        )
        .unwrap();

        let exported: Vec<Vec<f32>> = enc
            .feature_module_membership()
            .unwrap()
            .expect("modules are on")
            .to_vec2()
            .unwrap();
        assert_eq!(exported.len(), d);
        for row in &exported {
            let total: f32 = row.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "row {row:?}");
        }

        let ids = Tensor::from_vec(vec![4u32, 1], 2, &dev).unwrap();
        let pooled: Vec<Vec<f32>> = enc
            .features()
            .gather_membership(&ids)
            .unwrap()
            .expect("modules are on")
            .to_vec2()
            .unwrap();
        for (row, &g) in pooled.iter().zip([4usize, 1].iter()) {
            assert_eq!(row, &exported[g], "the pool reads a different membership");
        }
    }

    /// Stage 2's gate: pooling in module space must equal gathering composed
    /// rows and pooling them. The cheap path is only allowed to exist because
    /// these agree, and the whole reason the context cap can be lifted is that
    /// the dictionary factors out of the per-slot sum.
    #[test]
    fn module_space_pooling_equals_the_dense_route() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, candle_core::DType::F32, &dev);
        let (d, h, m, n, k) = (6usize, 4usize, 3usize, 2usize, 3usize);
        let layers = vec![h, h];
        let enc = IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features: d,
                n_topics: 2,
                embedding_dim: h,
                layers: &layers,
                attn_pool: false,
                n_gene_modules: m,
                lora_rank: 0,
            },
            &vm,
            vb,
        )
        .unwrap();

        let indices = Tensor::from_vec(vec![0u32, 3, 3, 1, 4, 0], (n, k), &dev).unwrap();
        let values =
            Tensor::from_vec(vec![4.0f32, 9.0, 1.0, 16.0, 2.0, 25.0], (n, k), &dev).unwrap();
        let pooled = enc
            .preprocess_indexed(&indices, &values, None, None)
            .unwrap();

        // Reference: compose every row, select the slots, gate, sum over k.
        let rows = crate::fast_index::gather_rows(
            &enc.features().full().unwrap(),
            &indices.flatten_all().unwrap(),
        )
        .unwrap()
        .reshape((n, k, h))
        .unwrap();
        let a_nk = anscombe_lite(&values, None, None).unwrap();
        let want = rows
            .broadcast_mul(&a_nk.unsqueeze(2).unwrap())
            .unwrap()
            .sum(1)
            .unwrap();

        let got: Vec<Vec<f32>> = pooled.to_vec2().unwrap();
        let want: Vec<Vec<f32>> = want.to_vec2().unwrap();
        for (g, w) in got.iter().zip(&want) {
            for (a, b) in g.iter().zip(w.iter()) {
                assert!((a - b).abs() < 1e-4, "module space {g:?} vs dense {w:?}");
            }
        }
    }

    /// Build the masked encoder on a fixed parameter fill and return the
    /// first six values of `forward_indexed_masked` for the whole fixture,
    /// flattened.
    fn masked_forward_head(n_gene_modules: usize) -> Vec<f32> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let layers = vec![8, 5];
        let enc = IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features: 9,
                n_topics: 3,
                embedding_dim: 4,
                layers: &layers,
                attn_pool: true,
                n_gene_modules,
                lora_rank: 0,
            },
            &varmap,
            vb,
        )
        .unwrap();

        // The CPU RNG cannot be seeded, so every parameter is overwritten with
        // a fixed, name-ordered ramp: the fixture is reproducible across runs
        // and across builds, which is what a pinned output needs.
        {
            let data = varmap.data().lock().unwrap();
            let mut names: Vec<String> = data.keys().cloned().collect();
            names.sort();
            for (j, name) in names.iter().enumerate() {
                let v = &data[name];
                let vals: Vec<f32> = if name.ends_with("running_var") {
                    vec![1.0; v.elem_count()]
                } else if name.ends_with("running_mean") {
                    vec![0.0; v.elem_count()]
                } else if name.ends_with("bias") {
                    // Zero, so the trunk does not drown the pool in its own
                    // offsets: with biases the output barely moved when the
                    // pooling changed, and the pin would have had no teeth.
                    vec![0.0; v.elem_count()]
                } else {
                    (0..v.elem_count())
                        .map(|i| 0.8 * ((i as f32) * 0.61 + j as f32 * 1.7).sin())
                        .collect()
                };
                v.set(&Tensor::from_vec(vals, v.dims().to_vec(), &device).unwrap())
                    .unwrap();
            }
        }

        // Cell 0 names gene 2 twice, cell 1 has two masked slots, cell 2 is
        // fully masked — the three cases the pooling has to get right.
        let indices =
            Tensor::from_vec(vec![2u32, 5, 2, 0, 1, 3, 6, 4, 0, 1, 2, 3], (3, 4), &device).unwrap();
        let values = Tensor::from_vec(
            vec![
                4.0f32, 9.0, 1.0, 16.0, 2.0, 25.0, 7.0, 3.0, 11.0, 5.0, 8.0, 6.0,
            ],
            (3, 4),
            &device,
        )
        .unwrap();
        let visible = Tensor::from_vec(
            vec![
                1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            (3, 4),
            &device,
        )
        .unwrap();

        enc.forward_indexed_masked(&indices, &values, None, None, &visible, false)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[..6]
            .to_vec()
    }

    /// The re-associated pooling is an identity, so the encoder's output must
    /// not move. These literals were recorded from this same fixture BEFORE
    /// `preprocess_indexed_masked` was rewritten, on the commit whose pooling
    /// gathered an `[N, K, H]` block — that path no longer exists to compare
    /// against, so the numbers stand in for it.
    #[test]
    fn masked_forward_is_unchanged_by_the_rewrite() {
        let want_free = [
            -0.7122408f32,
            -4.017215,
            -0.71038854,
            -0.738572,
            -4.091118,
            -0.6822394,
        ];
        let want_modules = [
            -0.7417681f32,
            -3.241218,
            -0.7244054,
            -0.75712043,
            -3.1009138,
            -0.72159404,
        ];
        for (m, want) in [(0usize, want_free), (2, want_modules)] {
            let got = masked_forward_head(m);
            println!("M = {m}: {got:?}");
            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "M = {m}: log theta [{i}] moved: {a} vs the recorded {b}"
                );
            }
        }
    }
}
