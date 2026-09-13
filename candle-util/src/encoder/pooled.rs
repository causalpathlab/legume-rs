//! The **pooled-gene trunk**: attention-pool a dense row over every gene
//! through a gene-embedding table, then FC → batch-norm → a linear head.
//!
//! This is the masked encoder's window-free read
//! ([`super::indexed::IndexedEmbeddingEncoder`] delegates its dense path here),
//! lifted out so a second caller can build the same trunk on a table it
//! already has — a frozen dictionary from another fit — and train only the
//! query, the FC and the head. The var names are the masked encoder's, so a
//! masked checkpoint loads unchanged.
//!
//! ```text
//! a_nd    = anscombe_residual(x, x0, mean)     [N, D]   the gate
//! rq_d    = ρ q                                [D]
//! attn_nd = softmax_d(a_nd · rq_d / √H + mask) [N, D]   sums to 1: depth-free
//! pool_nh = (attn_nd · a_nd) ρ                 [N, H]
//! out     = head(bn(fc(pool_nh)))              [N, out] raw
//! ```

use crate::encoder::{dense_pool, scatter_pool};
use crate::feature_embedding::FeatureEmbedding;
use crate::nn::batch_norm;
use crate::nn::layers::*;
use candle_core::{Result, Tensor};
use candle_nn::{ops, Linear, ModuleT, VarBuilder, VarMap};

pub struct PooledGeneEncoderArgs<'a> {
    /// FC widths; the last is the trunk width `L` the head reads.
    pub layers: &'a [usize],
    /// Width of the head's output.
    pub out_dim: usize,
    /// Register the attention query. Without it [`PooledGeneEncoder::pool`]
    /// is unavailable; the indexed sum-pool read needs only the trunk.
    pub attn_pool: bool,
    /// Extra FC input width beyond `H` (the masked encoder's module branch
    /// appends `2M` columns to the pool). `0` for a plain trunk.
    pub in_dim_extra: usize,
}

pub struct PooledGeneEncoder {
    features: std::sync::Arc<FeatureEmbedding>,
    /// `q [1, H]`; `None` without `attn_pool`.
    attn_query: Option<Tensor>,
    fc: StackLayers<Linear>,
    bn_z: batch_norm::BatchNorm,
    head: Linear,
}

impl PooledGeneEncoder {
    /// Build the trunk on `features`, registering `attn.query`, `nn.enc.fc`,
    /// `nn.enc.bn_z` and `nn.enc.z.mean` under `vb`.
    pub fn new(
        features: std::sync::Arc<FeatureEmbedding>,
        args: PooledGeneEncoderArgs,
        varmap: &VarMap,
        vb: VarBuilder,
    ) -> Result<Self> {
        debug_assert!(!args.layers.is_empty());
        let embedding_dim = features.embedding_dim();
        let bn_config = batch_norm::BatchNormConfig::default();
        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = embedding_dim + args.in_dim_extra;
        let out_dim = *args.layers.last().unwrap();
        let fc = stack_relu_linear(in_dim, out_dim, &fc_dims, vb.pp("nn.enc.fc"))?;
        let bn_z = batch_norm::batch_norm(out_dim, bn_config, varmap, vb.pp("nn.enc.bn_z"))?;
        let head = candle_nn::linear(out_dim, args.out_dim, vb.pp("nn.enc.z.mean"))?;
        let attn_query = if args.attn_pool {
            Some(vb.get_with_hints(
                (1, embedding_dim),
                "attn.query",
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?)
        } else {
            None
        };
        Ok(Self {
            features,
            attn_query,
            fc,
            bn_z,
            head,
        })
    }

    pub fn features(&self) -> &FeatureEmbedding {
        &self.features
    }

    #[must_use]
    pub fn features_shared(&self) -> std::sync::Arc<FeatureEmbedding> {
        std::sync::Arc::clone(&self.features)
    }

    /// The attention query, when this trunk pools.
    pub fn attn_query(&self) -> Option<&Tensor> {
        self.attn_query.as_ref()
    }

    /// Attention-pool a dense `[N, D]` row over every gene → `[N, H]`.
    ///
    /// `visible_nd`, when given, is `1` where the pool may look; `None` is
    /// every gene visible. A row with nothing visible pools to zero rather
    /// than to a uniform average over genes it was told not to read.
    pub fn pool(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        mean_1d: Option<&Tensor>,
        visible_nd: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.features.n_modules() > 0 {
            candle_core::bail!(
                "the dense pooled-gene read has no gene-module branch: modules pool a cell by \
                 membership over its context slots, and without a context window there are no \
                 slots. Build with n_gene_modules = 0."
            );
        }
        let Some(attn_query) = self.attn_query.as_ref() else {
            candle_core::bail!("pooling needs an attention query: build with attn_pool = true");
        };
        let scale = 1.0 / (self.features.embedding_dim() as f64).sqrt();
        // The gate over every gene, then ρq once per call: the query is a
        // parameter, so it is recomputed, never cached.
        let a_nd = crate::value_transform::anscombe_residual(x_nd, x0_nd, mean_1d)?; // [N, D]
        let rq_d = scatter_pool::query_over_features(&self.features, attn_query)?; // [D]
        let scores_nd = dense_pool::attention_scores_dense(&a_nd, &rq_d, visible_nd, scale)?;
        // The softmax weights sum to 1, so the pooled vector is depth-normalized.
        let attn_nd = ops::softmax(&scores_nd, 1)?; // [N, D]
        let pooled_nh = dense_pool::pool_dense(&attn_nd, &a_nd, &self.features)?; // [N, H]
        match visible_nd {
            Some(v) => {
                let has_visible_n1 = v.sum_keepdim(1)?.gt(0.0)?.to_dtype(pooled_nh.dtype())?;
                pooled_nh.broadcast_mul(&has_visible_n1)
            }
            None => Ok(pooled_nh),
        }
    }

    /// `pool → FC → BN` → `[N, L]`.
    pub fn trunk(&self, pooled: &Tensor, train: bool) -> Result<Tensor> {
        let fc_nl = self.fc.forward_t(pooled, train)?;
        self.bn_z.forward_t(&fc_nl, train)
    }

    /// The raw linear head on the trunk → `[N, out]`.
    pub fn head(&self, bn_nl: &Tensor, train: bool) -> Result<Tensor> {
        self.head.forward_t(bn_nl, train)
    }

    /// `pool → trunk → head`, raw `[N, out]`.
    pub fn forward(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        mean_1d: Option<&Tensor>,
        visible_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let pooled = self.pool(x_nd, x0_nd, mean_1d, visible_nd)?;
        self.head(&self.trunk(&pooled, train)?, train)
    }
}

#[cfg(test)]
#[path = "pooled_tests.rs"]
mod pooled_tests;
