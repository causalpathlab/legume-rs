use super::common::{expand_delta_for_block, process_blocks};
use crate::embed_common::*;

use candle_core::{Device, Tensor};
use candle_util::data::csc_columns_to_indexed_samples;
use candle_util::decoder::{EmbeddedNbTopicDecoder, MaskedNbTarget};
use candle_util::traits::*;
use candle_util::vae::masked_topic::{
    decoder_log_theta, masked_encode, LatentHead, MaskedEncoderInput, MaskedLikelihood,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

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
    /// Latent head to run at inference — must match the head the model was
    /// trained with. The eval path (all genes visible, no decoder) is identical
    /// across heads; only the final simplex/latent map differs. The Gaussian
    /// arm returns the posterior mean `z` (train=false → reparam returns mean).
    pub head: LatentHead,
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
        // Encoder-only inference: `train = false` (Gaussian returns its posterior
        // mean), and the KL is discarded — the latent alone is written out.
        let (latent_nk, _kl) = masked_encode(
            encoder,
            config.head,
            &MaskedEncoderInput {
                indices: &enc_pack.indices,
                values: &enc_pack.values,
                values_null: enc_values_null.as_ref(),
                values_mean: enc_pack.values_mean.as_ref(),
                visible_mask: &visible,
            },
            false,
        )?;
        let z_nk = latent_nk.to_device(&candle_core::Device::Cpu)?;
        Ok((lb, Mat::from_tensor(&z_nk)?))
    })
}

/// Config for [`evaluate_holdout_imputation`].
pub(crate) struct HoldoutEvalConfig<'a> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub enc_context_size: usize,
    pub shortlist_weights: &'a [f32],
    pub feature_mean: &'a [f32],
    /// Latent head the model was trained with (matches the persisted model).
    pub head: LatentHead,
    /// Per-gene likelihood — must match training for a comparable number.
    pub likelihood: MaskedLikelihood,
    /// Topic smoothing applied to the simplex heads, mirroring training.
    pub topic_smoothing: f64,
    /// Fraction of each cell's observed genes to hold out and score.
    pub mask_fraction: f64,
    /// Seed for the hold-out mask. A fixed seed masks the **same** per-cell
    /// positions across heads, so the comparison is apples-to-apples.
    pub seed: u64,
}

/// Held-out masked-imputation likelihood: for each cell, hide `mask_fraction`
/// of its observed genes, encode from the **visible** genes, impute the hidden
/// ones with the trained NB/multinomial ETM decoder, and return the mean
/// log-likelihood per held-out gene.
///
/// This is the honest generalization metric the per-epoch training likelihood
/// can't provide: the scored positions are never encoder inputs on their own
/// forward pass, and individual cells enter training only via their pseudobulk
/// aggregate — so a model that merely memorizes the training pseudobulks cannot
/// score well here. Runs one no-gradient pass; the decoder's `β = α·ρᵀ` (and
/// its log-partition) are fixed, so they are computed once.
pub(crate) fn evaluate_holdout_imputation(
    data_vec: &SparseIoVec,
    encoder: &candle_util::encoder::IndexedEmbeddingEncoder,
    decoder: &EmbeddedNbTopicDecoder,
    config: &HoldoutEvalConfig,
    delta: Option<&Tensor>,
) -> anyhow::Result<f32> {
    let ntot = data_vec.num_columns();
    let full_kd = decoder.full_logits_kd()?;
    let logz_11k = EmbeddedNbTopicDecoder::log_partition_from_logits(&full_kd)?;

    let mut llik_sum = 0f64;
    let mut mask_cnt = 0f64;
    for (lb, ub) in create_jobs(ntot, 0, Some(config.minibatch_size)) {
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
            None,
            ctx,
            config.dev,
        )?;
        let values_null = x0_nd
            .as_ref()
            .map(|x0| gather_null_at_indices(x0, &enc_pack.indices, config.dev))
            .transpose()?;

        // Seeded hold-out mask over the cell's real (value>0) top-K positions.
        // Host-side RNG keyed by `seed ^ lb` so the masked set is deterministic
        // and identical across heads for a fixed seed.
        let (n, k) = enc_pack.values.dims2()?;
        let values_host: Vec<f32> = enc_pack.values.flatten_all()?.to_vec1()?;
        let mut rng = StdRng::seed_from_u64(config.seed ^ (lb as u64));
        let mut mask_buf = vec![0f32; n * k];
        for (slot, &v) in values_host.iter().enumerate() {
            if v > 0.0 && rng.random::<f64>() < config.mask_fraction {
                mask_buf[slot] = 1.0;
            }
        }
        let masked = Tensor::from_vec(mask_buf, (n, k), config.dev)?;
        let real = enc_pack.values.gt(0.0)?.to_dtype(candle_core::DType::F32)?;
        let visible = (&real - &masked)?;

        // Encode from the visible genes only, mirroring the training split
        // (including the simplex-head topic smoothing).
        // KL is a training-time term only; the held-out metric is the
        // imputation likelihood alone.
        let (raw_z, _kl) = masked_encode(
            encoder,
            config.head,
            &MaskedEncoderInput {
                indices: &enc_pack.indices,
                values: &enc_pack.values,
                values_null: values_null.as_ref(),
                values_mean: enc_pack.values_mean.as_ref(),
                visible_mask: &visible,
            },
            false,
        )?;
        // Must match the training-time decoder coupling exactly, or the
        // held-out number is not comparable to the training trace.
        let log_z = decoder_log_theta(raw_z, config.head, config.topic_smoothing)?;

        let lib_n1 = (enc_pack.values.sum_keepdim(1)? + 1.0)?;
        let target = MaskedNbTarget {
            indices: &enc_pack.indices,
            residual: values_null.as_ref(),
            values: &enc_pack.values,
            lib: &lib_n1,
            mask: &masked,
        };
        let llik = match config.likelihood {
            MaskedLikelihood::Nb => decoder.impute_masked_nb(&log_z, &target, &logz_11k)?,
            MaskedLikelihood::Multinomial => {
                decoder.impute_masked_multinomial(&log_z, &target, &logz_11k)?
            }
        };
        llik_sum += f64::from(llik.sum_all()?.to_scalar::<f32>()?);
        mask_cnt += f64::from(masked.sum_all()?.to_scalar::<f32>()?);
    }

    Ok(if mask_cnt > 0.0 {
        (llik_sum / mask_cnt) as f32
    } else {
        f32::NAN
    })
}
