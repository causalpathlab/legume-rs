use super::common::{expand_delta_for_block, process_blocks};
use crate::embed_common::*;

use candle_core::{Device, Tensor};
use candle_util::data::csc_columns_to_indexed_samples;
use candle_util::traits::*;

/// Packed top-K representation consumed by the masked encoder.
///
/// Holds the per-cell `(indices, values)` the encoder reads; a single
/// host pass over the sparse columns is enough. Optional `values_mean`
/// carries the per-feature count-rate divisor gathered at the same
/// per-cell positions. Clone is cheap — Tensor is Arc-buffered.
#[derive(Clone)]
pub(crate) struct IndexedPack {
    /// [N, K] u32 — per-cell top-K feature ids
    pub indices: Tensor,
    /// [N, K] f32 — per-cell values in `indices` order
    pub values: Tensor,
    /// [N, K] f32 — per-gene mean expression rate `μ_d` gathered at
    /// `indices` (encoder side; multiplicative count-rate divisor).
    pub values_mean: Option<Tensor>,
}

/// Optional per-gene context for the packers. Each slice is length `D`
/// (training-D order); the packer gathers it at the per-cell top-K
/// positions to produce the `[N, K]` tensor that flows into the encoder.
#[derive(Clone, Copy, Default)]
pub(crate) struct PerGeneContext<'a> {
    pub feature_mean: Option<&'a [f32]>,
}

/// Build an [`IndexedPack`] directly from a sparse `[D, N]` CSC matrix —
/// columns are cells. Skips the dense `[N, D]` materialization the
/// `dense_*` helpers need; only the stored nonzeros are visited.
/// `gene_remap` (`Some(new_to_train)`) maps held-out → training gene
/// indices for `predict` on a differing gene set; the top-K is built over
/// the remapped, training-space ids. `None` when the CSC is already on the
/// training axis (fit-time eval).
pub(crate) fn csc_to_indexed(
    x_dn: &nalgebra_sparse::CscMatrix<f32>,
    context_size: usize,
    shortlist_weights: &[f32],
    gene_remap: Option<&[Option<usize>]>,
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<IndexedPack> {
    let k = context_size.min(x_dn.nrows());
    let samples = csc_columns_to_indexed_samples(x_dn, shortlist_weights, context_size, gene_remap);
    let all_top_k: Vec<(Vec<u32>, Vec<f32>)> =
        samples.into_iter().map(|s| (s.indices, s.values)).collect();
    pack_top_k_to_indexed(&all_top_k, k, ctx, dev)
}

/// Pack per-cell top-K `(indices, values)` into the `[N, K]` tensors of
/// an [`IndexedPack`].
fn pack_top_k_to_indexed(
    all_top_k: &[(Vec<u32>, Vec<f32>)],
    k: usize,
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<IndexedPack> {
    let n_batch = all_top_k.len();

    // Pack per-cell (indices, values) into [N, K]. Short rows (when D < K)
    // get padded with (idx=0, val=0.0); the matching zero value makes the
    // gather + weighted-sum a no-op.
    let mut idx_buf = vec![0u32; n_batch * k];
    let mut val_buf = vec![0.0f32; n_batch * k];
    let mut base_buf = ctx.feature_mean.map(|_| vec![0.0f32; n_batch * k]);
    for (row, (indices, values)) in all_top_k.iter().enumerate() {
        let off = row * k;
        let take = indices.len().min(k);
        idx_buf[off..off + take].copy_from_slice(&indices[..take]);
        val_buf[off..off + take].copy_from_slice(&values[..take]);
        if let (Some(buf), Some(b)) = (base_buf.as_mut(), ctx.feature_mean) {
            for (kk, &feat) in indices[..take].iter().enumerate() {
                buf[off + kk] = b[feat as usize];
            }
        }
    }

    let indices =
        Tensor::from_vec(idx_buf, (n_batch, k), dev)?.to_dtype(candle_core::DType::U32)?;
    let values = Tensor::from_vec(val_buf, (n_batch, k), dev)?;
    let values_mean = base_buf
        .map(|buf| Tensor::from_vec(buf, (n_batch, k), dev))
        .transpose()?;

    Ok(IndexedPack {
        indices,
        values,
        values_mean,
    })
}

/// Gather a per-cell null `[N, K] f32` from a dense `[N, D]` null tensor at
/// the encoder's `indices [N, K]`.
///
/// The earlier dense-scatter version walked every union slot for every
/// cell; the packed version pulls only the K positions that the encoder
/// will actually consume, so cost goes from O(N·S) to O(N·K).
pub(crate) fn gather_null_at_indices(
    x0_nd: &Tensor,
    indices: &Tensor,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let x0_rows: Vec<Vec<f32>> = x0_nd.to_vec2()?;
    let idx_vec: Vec<Vec<u32>> = indices.to_vec2()?;
    let n = idx_vec.len();
    let k = if idx_vec.is_empty() {
        0
    } else {
        idx_vec[0].len()
    };
    let mut buf = vec![0.0f32; n * k];
    for (i, idx_row) in idx_vec.iter().enumerate() {
        let null_row = &x0_rows[i];
        let off = i * k;
        for (kk, &feat) in idx_row.iter().enumerate() {
            buf[off + kk] = null_row[feat as usize];
        }
    }
    Ok(Tensor::from_vec(buf, (n, k), dev)?)
}

/// Config for [`evaluate_latent_masked`] — encoder-only inference for the
/// masked-imputation embedded topic model (no decoder / no refinement).
pub(crate) struct EvaluateLatentMaskedConfig<'a> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub enc_context_size: usize,
    pub shortlist_weights: &'a [f32],
    pub feature_mean: &'a [f32],
    /// Masked-VAE: use the Gaussian masked forward (posterior-mean `z`, no
    /// softmax) instead of the simplex `log θ` forward. The latent semantics
    /// differ but the eval path (all genes visible, no decoder) is identical.
    pub latent_gaussian: bool,
}

/// Encoder-only latent inference for the masked-imputation topic model.
///
/// Calls the deterministic masked-encoder forward with **all real genes
/// visible** (`visible = value>0`, no masking at inference) — matching the
/// training-time pooling (pads excluded) — and uses no decoder.
pub(crate) fn evaluate_latent_masked(
    data_vec: &SparseIoVec,
    encoder: &candle_util::encoder::IndexedEmbeddingEncoder,
    config: &EvaluateLatentMaskedConfig,
    delta: Option<&Tensor>,
    gene_remap: Option<&[Option<usize>]>,
) -> anyhow::Result<Mat> {
    let ntot = data_vec.num_columns();
    let kk = IndexedEncoderT::dim_latent(encoder);

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        let (lb, ub) = block;
        let x_dn = data_vec.read_columns_csc(lb..ub)?;
        let x0_nd = delta
            .map(|delta_bm| {
                expand_delta_for_block(data_vec, delta_bm, config.adj_method, lb, ub, config.dev)
            })
            .transpose()?;

        let ctx = PerGeneContext {
            feature_mean: Some(config.feature_mean),
        };
        let enc_pack = csc_to_indexed(
            &x_dn,
            config.enc_context_size,
            config.shortlist_weights,
            gene_remap,
            ctx,
            config.dev,
        )?;
        let enc_values_null = x0_nd
            .as_ref()
            .map(|x0| gather_null_at_indices(x0, &enc_pack.indices, config.dev))
            .transpose()?;
        let visible = enc_pack.values.gt(0.0)?.to_dtype(candle_core::DType::F32)?;
        let latent_nk = if config.latent_gaussian {
            // Posterior mean `z` (train=false → reparam returns the mean).
            let (z, _kl) = encoder.forward_indexed_masked_gaussian(
                &enc_pack.indices,
                &enc_pack.values,
                enc_values_null.as_ref(),
                enc_pack.values_mean.as_ref(),
                &visible,
                false,
            )?;
            z
        } else {
            encoder.forward_indexed_masked(
                &enc_pack.indices,
                &enc_pack.values,
                enc_values_null.as_ref(),
                enc_pack.values_mean.as_ref(),
                &visible,
                false,
            )?
        };
        let z_nk = latent_nk.to_device(&candle_core::Device::Cpu)?;
        Ok((lb, Mat::from_tensor(&z_nk)?))
    })
}
