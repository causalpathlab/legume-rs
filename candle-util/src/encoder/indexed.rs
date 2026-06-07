use crate::data::indexed::SparseEdgeBatch;
use crate::loss::{gaussian_kl_loss, gaussian_reparameterize};
use crate::nn::batch_norm;
use crate::nn::gcn::GcnBlock;
use crate::nn::layers::*;
use crate::traits::indexed::*;
use crate::value_transform::anscombe_lite;
use candle_core::{Result, Tensor};
use candle_nn::{ops, Linear, ModuleT, VarBuilder, VarMap};

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
    feature_embeddings: Tensor, // [D, H] learnable
    /// Optional γ-gated GCN block applied to the per-slot value-gated
    /// embedding `[N, K, H]` before pooling. Present iff
    /// `IndexedEmbeddingEncoderArgs::use_gcn` was true at construction.
    gcn: Option<GcnBlock>,
    fc: StackLayers<Linear>,
    bn_z: batch_norm::BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
    /// Learned attention query `q [1, H]` for the masked-path attention pool
    /// (PMA-style single-head). Only used by [`Self::forward_indexed_masked`];
    /// the legacy sum-pool path ([`Self::preprocess_indexed`]) ignores it.
    /// `None` unless `attn_pool` was set at construction — so sum-pool users
    /// (dense/legacy/pinto) neither allocate nor persist this var, and old
    /// safetensors (lacking `attn.query`) still load.
    attn_query: Option<Tensor>,
}

pub struct IndexedEmbeddingEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub embedding_dim: usize,
    pub layers: &'a [usize],
    /// When true, construct a [`GcnBlock`] on the per-slot `[N, K, H]`
    /// representation. The caller is responsible for providing the
    /// per-minibatch [`SparseEdgeBatch`] at forward time; when no edge
    /// batch is supplied the GCN branch is bypassed and the legacy
    /// sum-pool path is taken.
    pub use_gcn: bool,
    /// When true, allocate the `attn.query` parameter and use single-query
    /// attention pooling on the masked forward path
    /// ([`IndexedEmbeddingEncoder::forward_indexed_masked`]). Sum-pool users
    /// (dense ELBO / pinto) set this `false` so no `attn.query` var is
    /// registered — keeping their safetensors unchanged.
    pub attn_pool: bool,
}

impl IndexedEmbeddingEncoder {
    pub fn new(args: IndexedEmbeddingEncoderArgs, varmap: &VarMap, vb: VarBuilder) -> Result<Self> {
        let bn_config = batch_norm::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        debug_assert!(!args.layers.is_empty());

        // Feature embeddings: [D, H]
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let feature_embeddings = vb.get_with_hints(
            (args.n_features, args.embedding_dim),
            "feature.embeddings",
            init_ws,
        )?;

        let gcn = if args.use_gcn {
            Some(GcnBlock::new(args.embedding_dim, vb.pp("nn.enc.gcn"))?)
        } else {
            None
        };

        // FC stack: embedding_dim -> ... -> final_hidden
        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = args.embedding_dim;
        let out_dim = *args.layers.last().unwrap();
        let fc = stack_relu_linear(in_dim, out_dim, &fc_dims, vb.pp("nn.enc.fc"))?;

        let bn_z = batch_norm::batch_norm(out_dim, bn_config, varmap, vb.pp("nn.enc.bn_z"))?;

        let z_mean = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.lnvar"))?;

        let attn_query = if args.attn_pool {
            Some(vb.get_with_hints((1, args.embedding_dim), "attn.query", init_ws)?)
        } else {
            None
        };

        Ok(Self {
            n_features: args.n_features,
            n_topics: args.n_topics,
            embedding_dim: args.embedding_dim,
            feature_embeddings,
            gcn,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
            attn_query,
        })
    }

    /// Whether this encoder owns a [`GcnBlock`]. Callers use this to
    /// decide whether to supply per-minibatch sparse edges.
    pub fn has_gcn(&self) -> bool {
        self.gcn.is_some()
    }

    /// Current per-dim γ ∈ ℝ^H from the GCN block, when wired. Returns
    /// `None` otherwise. Used for per-epoch training instrumentation —
    /// caller derives summary stats (L2, max-abs, mean) for logging.
    pub fn gcn_gamma_vec(&self) -> Result<Option<Vec<f32>>> {
        self.gcn.as_ref().map(|g| g.gamma_vec()).transpose()
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

    /// Access the learnable feature embedding table [D, H].
    pub fn feature_embeddings(&self) -> &Tensor {
        &self.feature_embeddings
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
        sparse_edges: Option<&SparseEdgeBatch>,
    ) -> Result<Tensor> {
        let n = indices.dim(0)?;
        let k = indices.dim(1)?;
        let h = self.embedding_dim;

        let flat_idx = indices.flatten_all()?; // [N*K]
        let e_nk_h = self
            .feature_embeddings
            .index_select(&flat_idx, 0)?
            .reshape((n, k, h))?; // [N, K, H]

        // Per-slot Anscombe scalar gate on ρ — broadcast across H.
        // `anscombe_lite` divides by (batch null × per-gene mean) then
        // applies `2·sqrt(clean + 3/8)`. The subtractive `a_y − a_null`
        // form (f41754c) zeros out baseline slots and collapses the pool
        // under per-cell LN on batch-corrected data; divisive Anscombe
        // keeps every slot's magnitude information and avoids that.
        let a_nk = anscombe_lite(values, values_null, values_mean)?; // [N, K]
        let v_nkh = e_nk_h.broadcast_mul(&a_nk.unsqueeze(2)?)?; // [N, K, H]

        // γ-gated sparse GCN diffusion. Block is identity at init
        // (γ=0) so downstream FC+BN sees the no-graph training
        // distribution; γ grows only as the likelihood needs the graph.
        let v_pooled_input = match (&self.gcn, sparse_edges) {
            (Some(gcn), Some(edges)) => gcn.forward(&v_nkh, edges)?,
            _ => v_nkh,
        };
        v_pooled_input.sum(1) // [N, H]
    }
}

impl IndexedEmbeddingEncoder {
    /// Pool packed top-K into `[N, H]` using **only the visible slots**.
    ///
    /// Builds the same value-gated tokens as [`Self::preprocess_indexed`]
    /// (`a_nk · ρ`), then pools by **single-query attention** (PMA-style)
    /// rather than a plain sum: attention scores for masked / padding slots
    /// (`visible_mask [N, K]` == 0) are driven to −∞ so those genes are
    /// excluded from the softmax — the raw `a_nk` gate is *not* zeroed, so a
    /// masked gene's value never leaks into the pool. Used by the masked-
    /// imputation topic model. No GCN branch on this path.
    fn preprocess_indexed_masked(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        visible_mask: &Tensor,
    ) -> Result<Tensor> {
        let n = indices.dim(0)?;
        let k = indices.dim(1)?;
        let h = self.embedding_dim;

        let flat_idx = indices.flatten_all()?;
        let e_nk_h = self
            .feature_embeddings
            .index_select(&flat_idx, 0)?
            .reshape((n, k, h))?;

        // Value-gated token per slot (expression × symbol embedding). Note:
        // `a_nk` uses the raw values for ALL slots; masking is applied to the
        // attention scores (below), not by zeroing the gate — so a masked
        // gene is excluded from pooling but its value isn't leaked.
        let a_nk = anscombe_lite(values, values_null, values_mean)?; // [N, K]
        let content_nkh = e_nk_h.broadcast_mul(&a_nk.unsqueeze(2)?)?; // [N, K, H]

        // Single-query attention pool (PMA-style, O(K·H)): scores = ⟨token, q⟩/√H,
        // visible-masked to −∞, softmax over K, then weighted sum. The softmax
        // weights sum to 1, so the pooled vector is depth-normalized.
        let attn_query = self
            .attn_query
            .as_ref()
            .expect("forward_indexed_masked requires an attn_pool encoder");
        let scale = (h as f64).sqrt();
        let scores_nk = content_nkh
            .broadcast_mul(&attn_query.reshape((1, 1, h))?)?
            .sum(2)?
            .affine(1.0 / scale, 0.0)?; // [N, K]
        let neg_inf = visible_mask.affine(-1.0, 1.0)?.affine(-1e9, 0.0)?; // (1−vis)·(−1e9)
        let attn_nk = ops::softmax(&(scores_nk + neg_inf)?, 1)?; // [N, K]
        let pooled_nh = content_nkh.broadcast_mul(&attn_nk.unsqueeze(2)?)?.sum(1)?; // [N, H]

        // A row with no visible slot (empty cell, or all real genes masked at
        // high mask_fraction) has an all-−∞ score row, so softmax degenerates to
        // a uniform average over the padding slots. Zero that pool out so such
        // rows get a defined (bias-driven) empty-cell representation instead of
        // pad-gene content.
        let has_visible_n1 = visible_mask
            .sum_keepdim(1)?
            .gt(0.0)?
            .to_dtype(pooled_nh.dtype())?; // [N, 1]
        pooled_nh.broadcast_mul(&has_visible_n1) // [N, H]
    }

    /// Deterministic masked-encoder forward → `log θ [N, K_topics]`.
    ///
    /// Pools the **visible** genes, runs FC + BN, and returns
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
        let h_nh = self.preprocess_indexed_masked(
            indices,
            values,
            values_null,
            values_mean,
            visible_mask,
        )?;
        let fc_nl = self.fc.forward_t(&h_nh, train)?;
        let bn_nl = self.bn_z.forward_t(&fc_nl, train)?;
        let z_mean_nk = self.z_mean.forward_t(&bn_nl, train)?.clamp(-8.0, 8.0)?;
        ops::log_softmax(&z_mean_nk, 1)
    }

    /// Compute latent Gaussian parameters from packed indexed input.
    pub fn latent_gaussian_params_indexed(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        sparse_edges: Option<&SparseEdgeBatch>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let clamp_lo = -8.;
        let clamp_hi = 8.;

        let h_nh =
            self.preprocess_indexed(indices, values, values_null, values_mean, sparse_edges)?;
        let fc_nl = self.fc.forward_t(&h_nh, train)?;
        let bn_nl = self.bn_z.forward_t(&fc_nl, train)?;

        let z_mean_nk = self
            .z_mean
            .forward_t(&bn_nl, train)?
            .clamp(clamp_lo, clamp_hi)?;
        let z_lnvar_nk = self
            .z_lnvar
            .forward_t(&bn_nl, train)?
            .clamp(clamp_lo, clamp_hi)?;

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
        sparse_edges: Option<&SparseEdgeBatch>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_gaussian_params_indexed(
            indices,
            values,
            values_null,
            values_mean,
            sparse_edges,
            train,
        )?;

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
                use_gcn: false,
                attn_pool: false,
            },
            &varmap,
            vb,
        )
        .unwrap();

        // Two cells: cell 0 selects features {0,1}, cell 1 selects {2,3}.
        let indices = Tensor::from_vec(vec![0u32, 1, 2, 3], (2, 2), &device).unwrap();
        let values = Tensor::from_vec(vec![4.0f32, 9.0, 16.0, 25.0], (2, 2), &device).unwrap();

        let h = enc
            .preprocess_indexed(&indices, &values, None, None, None)
            .unwrap();
        assert_eq!(h.dims(), &[2, embedding_dim]);
        for row in h.to_vec2::<f32>().unwrap() {
            for v in row {
                assert!(v.is_finite(), "non-finite pooled value {v}");
            }
        }
    }
}
