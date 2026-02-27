use crate::candle_aux_layers::*;
use crate::candle_indexed_model_traits::*;
use crate::candle_loss_functions::gaussian_kl_loss;
use candle_core::{Result, Tensor};
use candle_nn::{ops, BatchNorm, Linear, ModuleT, VarBuilder};

/// Indexed embedding encoder.
///
/// Replaces the dense `AggregateLinear [D×M]` matmul with embedding lookup
/// on selected features. The feature embeddings [D, H] are indexed via
/// `index_select` to [S, H], then combined with normalized input values
/// via matmul [N, S] × [S, H] → [N, H].
pub struct IndexedEmbeddingEncoder {
    n_features: usize,
    n_topics: usize,
    embedding_dim: usize,
    feature_embeddings: Tensor, // [D, H] learnable
    fc: StackLayers<Linear>,
    bn_z: BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
}

pub struct IndexedEmbeddingEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub embedding_dim: usize,
    pub layers: &'a [usize],
}

impl IndexedEmbeddingEncoder {
    pub fn new(args: IndexedEmbeddingEncoderArgs, vb: VarBuilder) -> Result<Self> {
        let bn_config = candle_nn::BatchNormConfig {
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

        // FC stack: embedding_dim -> ... -> final_hidden
        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = args.embedding_dim;
        let out_dim = *args.layers.last().unwrap();
        let fc = stack_relu_linear(in_dim, out_dim, &fc_dims, vb.pp("nn.enc.fc"))?;

        let bn_z = candle_nn::batch_norm(out_dim, bn_config, vb.pp("nn.enc.bn_z"))?;

        let z_mean = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.lnvar"))?;

        Ok(Self {
            n_features: args.n_features,
            n_topics: args.n_topics,
            embedding_dim: args.embedding_dim,
            feature_embeddings,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
        })
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

    /// Preprocess indexed input into embedding space.
    ///
    /// 1. E_sh = feature_embeddings.index_select(union_indices, 0) → [S, H]
    /// 2. lx_ns = log1p(indexed_x) → [N, S]
    /// 3. Normalize: lx / sum(lx) * nnz → [N, S]
    /// 4. h_nh = normalized @ E_sh → [N, H]
    fn preprocess_indexed(
        &self,
        union_indices: &Tensor,
        indexed_x: &Tensor,
        indexed_x_null: Option<&Tensor>,
    ) -> Result<Tensor> {
        // 1. Look up embeddings for selected features
        let e_sh = self.feature_embeddings.index_select(union_indices, 0)?; // [S, H]

        // 2. log1p normalization — scale by n_features (constant) like dense encoder
        let lx_ns = (indexed_x + 1.0)?.log()?; // [N, S]
        let denom = lx_ns.sum_keepdim(1)?; // [N, 1]
        let normalized = (lx_ns.broadcast_div(&denom)? * (self.n_features as f64))?; // [N, S]

        // 3. Matmul: [N, S] × [S, H] → [N, H]
        let h_nh = normalized.matmul(&e_sh)?;

        // 4. Optional null subtraction
        match indexed_x_null {
            Some(x0) => {
                let lx0 = (x0 + 1.0)?.log()?;
                let denom0 = lx0.sum_keepdim(1)?;
                let normalized0 = (lx0.broadcast_div(&denom0)? * (self.n_features as f64))?;
                let h0_nh = normalized0.matmul(&e_sh)?;
                h_nh - h0_nh
            }
            None => Ok(h_nh),
        }
    }

    fn reparameterize(&self, z_mean: &Tensor, z_lnvar: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            let eps = Tensor::randn_like(z_mean, 0., 1.)?;
            z_mean + (z_lnvar * 0.5)?.exp()? * eps
        } else {
            Ok(z_mean.clone())
        }
    }
}

impl IndexedEncoderT for IndexedEmbeddingEncoder {
    fn forward_indexed_t(
        &self,
        union_indices: &Tensor,
        indexed_x: &Tensor,
        indexed_x_null: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let min_lv = -10.;
        let max_lv = 10.;

        let h_nh = self.preprocess_indexed(union_indices, indexed_x, indexed_x_null)?;
        let fc_nl = self.fc.forward_t(&h_nh, train)?;
        let bn_nl = self.bn_z.forward_t(&fc_nl, train)?;

        let z_mean_nk = self.z_mean.forward_t(&bn_nl, train)?;
        let z_lnvar_nk = self
            .z_lnvar
            .forward_t(&bn_nl, train)?
            .clamp(min_lv, max_lv)?;

        let z_nk = self.reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;
        let log_prob = ops::log_softmax(&z_nk, 1)?;

        Ok((log_prob, gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
