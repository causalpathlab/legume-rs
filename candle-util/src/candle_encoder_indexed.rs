use crate::candle_aux_layers::*;
use crate::candle_batch_norm;
use crate::candle_indexed_model_traits::*;
use crate::candle_loss_functions::{gaussian_kl_loss, gaussian_reparameterize};
use crate::candle_value_transform::anscombe_lite;
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
    fc: StackLayers<Linear>,
    bn_z: candle_batch_norm::BatchNorm,
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
    pub fn new(args: IndexedEmbeddingEncoderArgs, varmap: &VarMap, vb: VarBuilder) -> Result<Self> {
        let bn_config = candle_batch_norm::BatchNormConfig {
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

        let bn_z = candle_batch_norm::batch_norm(out_dim, bn_config, varmap, vb.pp("nn.enc.bn_z"))?;

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
    /// With `values_mean` supplied (per-gene μ_d gathered at indices),
    /// housekeeping genes expressed at typical levels divide out to ≈1
    /// → Anscombe(1) ≈ 2.35, a constant absorbed by `bn_z`. Markers
    /// expressed at unusual levels survive as fold-change deviations.
    fn preprocess_indexed(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
    ) -> Result<Tensor> {
        let v_norm = anscombe_lite(values, values_null, values_mean)?; // [N, K]

        let n = indices.dim(0)?;
        let k = indices.dim(1)?;
        let h = self.embedding_dim;

        let flat_idx = indices.flatten_all()?; // [N*K]
        let e_nk_h = self
            .feature_embeddings
            .index_select(&flat_idx, 0)?
            .reshape((n, k, h))?; // [N, K, H]

        // Value-weighted pool: broadcast v_norm[N, K] → [N, K, 1] and reduce
        // along K. `e_nk_h` is contiguous from index_select+reshape.
        let v = v_norm.unsqueeze(2)?; // [N, K, 1]
        let weighted = e_nk_h.broadcast_mul(&v)?; // [N, K, H]
        weighted.sum(1) // [N, H]
    }
}

impl IndexedEmbeddingEncoder {
    /// Compute latent Gaussian parameters from packed indexed input.
    pub fn latent_gaussian_params_indexed(
        &self,
        indices: &Tensor,
        values: &Tensor,
        values_null: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let clamp_lo = -8.;
        let clamp_hi = 8.;

        let h_nh = self.preprocess_indexed(indices, values, values_null, values_mean)?;
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

    /// Hand-build a tiny encoder, feed two cells with disjoint indices and
    /// known values, and verify the Anscombe-lite + weighted-sum pool
    /// matches a from-scratch host computation.
    #[test]
    fn test_anscombe_lite_pool() {
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
            },
            &varmap,
            vb,
        )
        .unwrap();

        // Two cells: cell 0 selects features {0,1}, cell 1 selects {2,3}.
        let indices = Tensor::from_vec(vec![0u32, 1, 2, 3], (2, 2), &device).unwrap();
        let values = Tensor::from_vec(vec![4.0f32, 9.0, 16.0, 25.0], (2, 2), &device).unwrap();

        // Reference: anscombe-lite (no null, no per-gene mean) →
        // bare Anscombe of values, then weighted-sum pool against the
        // encoder's own embedding table.
        let e_dh: Vec<Vec<f32>> = enc.feature_embeddings.to_vec2().unwrap();
        let raw = [[4.0f32, 9.0], [16.0, 25.0]];
        let mut v_norm = [[0f32; 2]; 2];
        for (i, row) in raw.iter().enumerate() {
            for (k, &x) in row.iter().enumerate() {
                v_norm[i][k] = 2.0 * (x + 0.375).sqrt();
            }
        }
        let mut h_ref = vec![vec![0f32; embedding_dim]; 2];
        for (i, row) in v_norm.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                let feat = if i == 0 { k } else { 2 + k };
                for h in 0..embedding_dim {
                    h_ref[i][h] += v * e_dh[feat][h];
                }
            }
        }

        let h = enc
            .preprocess_indexed(&indices, &values, None, None)
            .unwrap();
        let h_vec: Vec<Vec<f32>> = h.to_vec2().unwrap();
        for (i, row) in h_vec.iter().enumerate() {
            for (hh, &got) in row.iter().enumerate() {
                let want = h_ref[i][hh];
                assert!(
                    (got - want).abs() < 1e-4,
                    "row {i} dim {hh}: got {got} want {want}"
                );
            }
        }
    }
}
