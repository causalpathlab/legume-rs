use crate::nn::layers::*;
use crate::nn::batch_norm;
use crate::candle_cell_grouped_data_loader::CellGroupedMinibatchData;
use crate::candle_indexed_model_traits::*;
use crate::candle_loss_functions::{gaussian_kl_loss, gaussian_reparameterize};
use crate::candle_value_transform::{ValueEmbedding, ValueEmbeddingConfig};
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{ops, Linear, ModuleT, VarBuilder, VarMap};

/// Hierarchical cell→PB pooling encoder.
///
/// Two value-weighted ρ pools feed a shared latent head:
///   - **Foreground** — each PB's member cells, two-level pooled
///     (gene→cell value-weighted sum, then cell→PB segment sum). Per-cell
///     library-size normalization (`y / s_c`) is the DC-Poisson degree
///     correction; summing across cells then accumulates evidence.
///   - **Background** — the PB-level `μ_residual` (or observed) profile,
///     single-level value-weighted pool.
///
/// Both pools apply NB-Fisher multiplicative weighting so housekeeping
/// genes are down-weighted, and mask padded top-K slots by `value > 0`.
/// The two `[N, H]` pools are concatenated `[N, 2H]` into the FC/latent
/// head. ρ `[D, H]` is the **same** tensor the decoder ties to (ETM).
///
/// # Value transform
///
/// The per-slot weighting on ρ is the learned **intensity embedding**: the
/// normalized value is binned at two scales (linear + log1p), each bin
/// looked up in an `[n_value_bins, H]` table, the two summed and sigmoid'd into
/// a per-dimension **gate** on ρ — times the `fisher · mask` scalar. The
/// fixed Anscombe scalar transform has been retired here.
pub struct CellEmbeddedEncoder {
    n_features: usize,
    n_topics: usize,
    embedding_dim: usize,
    feature_embeddings: Tensor, // [D, H] learnable, shared with decoder
    value_embedding: ValueEmbedding,
    fc: StackLayers<Linear>, // 2H -> final_hidden
    bn_z: batch_norm::BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
}

pub struct CellEmbeddedEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub embedding_dim: usize,
    pub layers: &'a [usize],
    /// The learned intensity-embedding value transform — the only value
    /// transform (the fixed Anscombe scalar has been retired here).
    pub value_embedding: ValueEmbeddingConfig,
}

impl CellEmbeddedEncoder {
    pub fn new(args: CellEmbeddedEncoderArgs, varmap: &VarMap, vb: VarBuilder) -> Result<Self> {
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

        let value_embedding = ValueEmbedding::new(
            args.value_embedding,
            args.embedding_dim,
            vb.pp("nn.enc.value"),
        )?;

        // FC stack: 2H -> ... -> final_hidden (the [fg, bg] concat is 2H).
        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let in_dim = 2 * args.embedding_dim;
        let out_dim = *args.layers.last().unwrap();
        let fc = stack_relu_linear(in_dim, out_dim, &fc_dims, vb.pp("nn.enc.fc"))?;

        let bn_z = batch_norm::batch_norm(out_dim, bn_config, varmap, vb.pp("nn.enc.bn_z"))?;

        let z_mean = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.mean"))?;
        let z_lnvar = candle_nn::linear(out_dim, args.n_topics, vb.pp("nn.enc.z.lnvar"))?;

        Ok(Self {
            n_features: args.n_features,
            n_topics: args.n_topics,
            embedding_dim: args.embedding_dim,
            feature_embeddings,
            value_embedding,
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

    /// Per-slot learned gate on ρ — `[*, K, H]`, broadcastable against
    /// `E [*, K, H]`: the intensity-embedding gate of the normalized value
    /// times the `fisher · mask` scalar.
    ///
    /// `norm_values` is the per-sample-normalized value (FG: `y / s_c`;
    /// BG: per-PB `1e4`-normalized). `raw_values` is only consulted for the
    /// `value > 0` padded-slot mask.
    fn slot_weight(
        &self,
        norm_values: &Tensor,
        raw_values: &Tensor,
        fisher: &Tensor,
    ) -> Result<Tensor> {
        let mask = raw_values.gt(0f64)?.to_dtype(DType::F32)?; // [*, K]
        let gate = self.value_embedding.gate(norm_values)?; // [*, K, H]
        let scalar = fisher.mul(&mask)?.unsqueeze(D::Minus1)?; // [*, K, 1]
        gate.broadcast_mul(&scalar) // [*, K, H]
    }

    /// Foreground pool — two-level gene→cell→PB → `[N, H]`.
    ///
    /// 1. per-cell library-size normalization `y / s_c`
    /// 2. `w = slot_weight(...)` — intensity-embedding gate × fisher × mask
    /// 3. `h_m = Σ_k w[m,k] ⊙ ρ[fg_indices[m,k], :]`              → [M, H]
    /// 4. segment-sum member cells into their PB row via `index_add` → [N, H]
    fn preprocess_cells(&self, mb: &CellGroupedMinibatchData) -> Result<Tensor> {
        let n_pb = mb.bg_indices.dim(0)?;
        let m = mb.fg_indices.dim(0)?;
        let k = mb.fg_indices.dim(1)?;
        let h = self.embedding_dim;

        // DC-Poisson degree correction: divide by the per-cell library size.
        let norm = mb.fg_values.broadcast_div(&mb.fg_size_factor)?;
        let w = self.slot_weight(&norm, &mb.fg_values, &mb.fg_fisher)?;

        let flat = mb.fg_indices.flatten_all()?;
        let e = self
            .feature_embeddings
            .index_select(&flat, 0)?
            .reshape((m, k, h))?; // [M, K_fg, H]
        let h_m = e.broadcast_mul(&w)?.sum(1)?; // [M, H]

        // cell→PB segment pool. Sum, not mean: per-cell size-factor
        // normalization already removed depth, so summing accumulates
        // evidence (DC-Poisson semantics). PBs with no member cells stay 0.
        let zeros = Tensor::zeros((n_pb, h), DType::F32, h_m.device())?;
        let pooled = zeros.index_add(&mb.cell_to_pb, &h_m, 0)?; // [N_pb, H]
                                                                // The FG pool sums only an S-cell random subsample of each PB;
                                                                // `fg_pb_rescale = |PB|/S` ([N_pb, 1]) upscales the segment-sum to an
                                                                // unbiased estimate of the full-PB sum (1.0 when sampled in full).
        pooled.broadcast_mul(&mb.fg_pb_rescale)
    }

    /// Background pool — single-level PB-profile value-weighted pool → `[N, H]`.
    /// Same Fisher-multiplicative housekeeping treatment as the foreground.
    fn preprocess_bg(&self, mb: &CellGroupedMinibatchData) -> Result<Tensor> {
        let n = mb.bg_indices.dim(0)?;
        let k = mb.bg_indices.dim(1)?;
        let h = self.embedding_dim;

        // Normalize each PB's top-K to ~unit total so its intensity bins
        // land on the same scale as the FG path's per-cell normalization.
        let denom = mb.bg_values.sum_keepdim(1)?.clamp(1.0, f64::INFINITY)?;
        let norm = mb.bg_values.broadcast_div(&denom)?;
        let w = self.slot_weight(&norm, &mb.bg_values, &mb.bg_fisher)?;

        let flat = mb.bg_indices.flatten_all()?;
        let e = self
            .feature_embeddings
            .index_select(&flat, 0)?
            .reshape((n, k, h))?; // [N, K_bg, H]
        e.broadcast_mul(&w)?.sum(1) // [N, H]
    }

    /// Latent Gaussian params from the `[fg, bg]` concat `[N, 2H]`.
    pub fn latent_gaussian_params(
        &self,
        mb: &CellGroupedMinibatchData,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let clamp_lo = -8.;
        let clamp_hi = 8.;

        let fg_nh = self.preprocess_cells(mb)?; // [N, H]
        let bg_nh = self.preprocess_bg(mb)?; // [N, H]
        let h_nh = Tensor::cat(&[&fg_nh, &bg_nh], 1)?; // [N, 2H]

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

impl CellEncoderT for CellEmbeddedEncoder {
    fn forward_cells_t(
        &self,
        mb: &CellGroupedMinibatchData,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_mean_nk, z_lnvar_nk) = self.latent_gaussian_params(mb, train)?;
        let z_nk = gaussian_reparameterize(&z_mean_nk, &z_lnvar_nk, train)?;
        let log_prob = ops::log_softmax(&z_nk, 1)?;
        Ok((log_prob, gaussian_kl_loss(&z_mean_nk, &z_lnvar_nk)?))
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }
}
