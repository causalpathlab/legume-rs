use super::common::{expand_delta_for_block, process_blocks};
use crate::embed_common::*;

use candle_core::{Device, Tensor, Var};
use candle_nn::ops;
use candle_util::candle_indexed_data_loader::{
    csc_columns_to_indexed_samples, top_k_indices_weighted,
};
use candle_util::candle_indexed_model_traits::*;
use candle_util::candle_topic_refinement::TopicRefinementConfig;
use std::collections::{BTreeSet, HashMap};

/// Packed top-K representation built from a dense `[N, D]` tensor.
///
/// Holds both the per-cell `(indices, values)` consumed by the encoder
/// and the `(union, scatter_pos)` consumed by the decoder, so a single
/// host pass over the dense rows is enough for both sides. Optional
/// `values_mean` / `values_weight` carry per-feature constants gathered
/// at the same per-cell positions, mirroring the loader's training-time
/// minibatch. Clone is cheap — Tensor is Arc-buffered.
#[derive(Clone)]
pub(crate) struct IndexedPack {
    /// [N, K] u32 — per-cell top-K feature ids
    pub indices: Tensor,
    /// [N, K] f32 — per-cell values in id order (= scatter_pos order)
    pub values: Tensor,
    /// [S] u32 — sorted-by-discovery union of feature ids
    pub union_indices: Tensor,
    /// [N, K] u32 in [0, S) — per-cell positions in the union
    pub scatter_pos: Tensor,
    /// [N, K] f32 — per-gene mean expression rate `μ_d` gathered at
    /// `indices` (encoder side; multiplicative count-rate divisor).
    pub values_mean: Option<Tensor>,
    /// [N, K] f32 — per-gene NB-Fisher weight gathered at `indices`
    /// (decoder side; multiplicative likelihood weight).
    pub values_weight: Option<Tensor>,
}

/// Optional per-gene context for `dense_to_indexed*`. Each slice is
/// length `D` (training-D order); the helpers gather them at the
/// per-cell top-K positions to produce the `[N, K]` tensors that flow
/// into the encoder / decoder.
#[derive(Clone, Copy, Default)]
pub(crate) struct PerGeneContext<'a> {
    pub feature_mean: Option<&'a [f32]>,
    pub feature_fisher_weights: Option<&'a [f32]>,
}

/// Convert a dense `[N, D]` tensor into packed top-K form.
///
/// `shortlist_weights` is the same per-gene weight vector used at training
/// time; required so inference picks shortlists from the same distribution
/// the encoder was trained on. `ctx` carries optional per-gene baseline
/// and Fisher weights to gather at the chosen top-K positions.
pub(crate) fn dense_to_indexed(
    x_nd: &Tensor,
    context_size: usize,
    shortlist_weights: &[f32],
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<IndexedPack> {
    let rows: Vec<Vec<f32>> = x_nd.to_vec2()?;
    dense_rows_to_indexed(&rows, context_size, shortlist_weights, ctx, dev)
}

/// Convert a dense `[N, D]` tensor into two packed top-K forms (encoder
/// and decoder windows) with a single host copy of the row data.
pub(crate) fn dense_to_indexed_pair(
    x_nd: &Tensor,
    enc_context_size: usize,
    dec_context_size: usize,
    shortlist_weights: &[f32],
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<(IndexedPack, IndexedPack)> {
    let rows: Vec<Vec<f32>> = x_nd.to_vec2()?;
    let enc = dense_rows_to_indexed(&rows, enc_context_size, shortlist_weights, ctx, dev)?;
    let dec = dense_rows_to_indexed(&rows, dec_context_size, shortlist_weights, ctx, dev)?;
    Ok((enc, dec))
}

fn dense_rows_to_indexed(
    rows: &[Vec<f32>],
    context_size: usize,
    shortlist_weights: &[f32],
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<IndexedPack> {
    let k = if rows.is_empty() {
        context_size
    } else {
        context_size.min(rows[0].len())
    };

    // Sequential per-row: `dense_rows_to_indexed`'s callers
    // (`evaluate_indexed_block`) already run inside `process_blocks`'
    // block-level rayon map, so a nested par_iter here would just add
    // task-splitting overhead.
    let all_top_k: Vec<(Vec<u32>, Vec<f32>)> = rows
        .iter()
        .map(|row| top_k_indices_weighted(row, shortlist_weights, context_size))
        .collect();

    pack_top_k_to_indexed(&all_top_k, k, ctx, dev)
}

/// Build an [`IndexedPack`] directly from a sparse `[D, N]` CSC matrix —
/// columns are cells. Skips the dense `[N, D]` materialization the
/// `dense_*` helpers need; only the stored nonzeros are visited.
pub(crate) fn csc_to_indexed(
    x_dn: &nalgebra_sparse::CscMatrix<f32>,
    context_size: usize,
    shortlist_weights: &[f32],
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<IndexedPack> {
    let k = context_size.min(x_dn.nrows());
    let samples = csc_columns_to_indexed_samples(x_dn, shortlist_weights, context_size);
    let all_top_k: Vec<(Vec<u32>, Vec<f32>)> = samples
        .into_iter()
        .map(|s| (s.indices, s.values))
        .collect();
    pack_top_k_to_indexed(&all_top_k, k, ctx, dev)
}

/// Like [`csc_to_indexed`] but produces both encoder and decoder windows
/// from a single sparse pass' worth of column scans.
pub(crate) fn csc_to_indexed_pair(
    x_dn: &nalgebra_sparse::CscMatrix<f32>,
    enc_context_size: usize,
    dec_context_size: usize,
    shortlist_weights: &[f32],
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<(IndexedPack, IndexedPack)> {
    let enc = csc_to_indexed(x_dn, enc_context_size, shortlist_weights, ctx, dev)?;
    let dec = csc_to_indexed(x_dn, dec_context_size, shortlist_weights, ctx, dev)?;
    Ok((enc, dec))
}

/// Pack per-cell top-K `(indices, values)` into the `[N, K]` / `[S]`
/// tensors of an [`IndexedPack`]. Shared by the dense and sparse
/// front-ends.
fn pack_top_k_to_indexed(
    all_top_k: &[(Vec<u32>, Vec<f32>)],
    k: usize,
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<IndexedPack> {
    let n_batch = all_top_k.len();

    let mut union_set = BTreeSet::new();
    for (indices, _) in all_top_k {
        for &idx in indices {
            union_set.insert(idx);
        }
    }

    let union_vec: Vec<u32> = union_set.into_iter().collect();
    let s = union_vec.len();
    let pos_map: HashMap<u32, usize> = union_vec
        .iter()
        .enumerate()
        .map(|(pos, &idx)| (idx, pos))
        .collect();

    // Pack per-cell (indices, values) and scatter positions into [N, K].
    // Short rows (when D < K) get padded with (idx=0, val=0.0, pos=0); the
    // matching zero value makes the gather + weighted-sum a no-op.
    let mut idx_buf = vec![0u32; n_batch * k];
    let mut val_buf = vec![0.0f32; n_batch * k];
    let mut scat_buf = vec![0u32; n_batch * k];
    let mut base_buf = ctx.feature_mean.map(|_| vec![0.0f32; n_batch * k]);
    let mut wt_buf = ctx
        .feature_fisher_weights
        .map(|_| vec![0.0f32; n_batch * k]);
    for (row, (indices, values)) in all_top_k.iter().enumerate() {
        let off = row * k;
        let take = indices.len().min(k);
        idx_buf[off..off + take].copy_from_slice(&indices[..take]);
        val_buf[off..off + take].copy_from_slice(&values[..take]);
        for (kk, &feat) in indices[..take].iter().enumerate() {
            scat_buf[off + kk] = pos_map[&feat] as u32;
            if let (Some(buf), Some(b)) = (base_buf.as_mut(), ctx.feature_mean) {
                buf[off + kk] = b[feat as usize];
            }
            if let (Some(buf), Some(w)) = (wt_buf.as_mut(), ctx.feature_fisher_weights) {
                buf[off + kk] = w[feat as usize];
            }
        }
    }

    let indices =
        Tensor::from_vec(idx_buf, (n_batch, k), dev)?.to_dtype(candle_core::DType::U32)?;
    let values = Tensor::from_vec(val_buf, (n_batch, k), dev)?;
    let union_indices =
        Tensor::from_vec(union_vec, (s,), dev)?.to_dtype(candle_core::DType::U32)?;
    let scatter_pos =
        Tensor::from_vec(scat_buf, (n_batch, k), dev)?.to_dtype(candle_core::DType::U32)?;
    let values_mean = base_buf
        .map(|buf| Tensor::from_vec(buf, (n_batch, k), dev))
        .transpose()?;
    let values_weight = wt_buf
        .map(|buf| Tensor::from_vec(buf, (n_batch, k), dev))
        .transpose()?;

    Ok(IndexedPack {
        indices,
        values,
        union_indices,
        scatter_pos,
        values_mean,
        values_weight,
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

pub(crate) struct EvaluateLatentConfig<'a, Dec> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub enc_context_size: usize,
    pub dec_context_size: usize,
    pub decoder: &'a Dec,
    pub refine_config: Option<&'a TopicRefinementConfig>,
    /// Same shortlist weights used during training.
    pub shortlist_weights: &'a [f32],
    /// Per-gene Anscombe baseline + NB-Fisher weights, in training-D
    /// order. The encoder subtracts the baseline at top-K positions
    /// before pooling; the decoder multiplies the Fisher weights into
    /// the per-position likelihood weight.
    pub feature_mean: &'a [f32],
    pub feature_fisher_weights: &'a [f32],
}

pub(crate) fn evaluate_latent_by_indexed_encoder<Enc, Dec>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    collapsed: &CollapsedOut,
    config: &EvaluateLatentConfig<Dec>,
) -> anyhow::Result<Mat>
where
    Enc: IndexedEncoderT + Send + Sync,
    Dec: IndexedDecoderT + Send + Sync,
{
    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();

    // Delta at full D — encoder operates at D_full
    let delta = match config.adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref(),
        AdjMethod::Residual => collapsed.mu_residual.as_ref(),
    }
    .map(|x| x.posterior_mean().clone())
    .map(|delta_db| {
        delta_db
            .to_tensor(config.dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
            .contiguous()
            .expect("contiguous")
    });

    let block_config = EvaluateBlockConfig {
        dev: config.dev,
        adj_method: config.adj_method,
        delta: delta.as_ref(),
        enc_context_size: config.enc_context_size,
        dec_context_size: config.dec_context_size,
        decoder: config.decoder,
        refine_config: config.refine_config,
        shortlist_weights: config.shortlist_weights,
        feature_mean: config.feature_mean,
        feature_fisher_weights: config.feature_fisher_weights,
    };

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        evaluate_indexed_block(block, data_vec, encoder, &block_config)
    })
}

struct EvaluateBlockConfig<'a, Dec> {
    dev: &'a Device,
    adj_method: &'a AdjMethod,
    delta: Option<&'a Tensor>,
    enc_context_size: usize,
    dec_context_size: usize,
    decoder: &'a Dec,
    refine_config: Option<&'a TopicRefinementConfig>,
    shortlist_weights: &'a [f32],
    feature_mean: &'a [f32],
    feature_fisher_weights: &'a [f32],
}

fn evaluate_indexed_block<Enc, Dec>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: &Enc,
    config: &EvaluateBlockConfig<Dec>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: IndexedEncoderT,
    Dec: IndexedDecoderT,
{
    let (lb, ub) = block;

    // Read sparse [D, N] at full D (no feature coarsening). Kept sparse:
    // the top-K packer only ever visits stored nonzeros, so the dense
    // `[N, D]` matrix is never materialized.
    let x_dn = data_vec.read_columns_csc(lb..ub)?;

    // Get batch/residual correction if available
    let x0_nd = config
        .delta
        .map(|delta_bm| {
            expand_delta_for_block(data_vec, delta_bm, config.adj_method, lb, ub, config.dev)
        })
        .transpose()?;

    // Convert sparse columns to packed top-K directly.
    let ctx = PerGeneContext {
        feature_mean: Some(config.feature_mean),
        feature_fisher_weights: Some(config.feature_fisher_weights),
    };
    let need_decoder = config.refine_config.is_some();
    let (enc_pack, dec_pack) = if need_decoder {
        let (enc, dec) = csc_to_indexed_pair(
            &x_dn,
            config.enc_context_size,
            config.dec_context_size,
            config.shortlist_weights,
            ctx,
            config.dev,
        )?;
        (enc, Some(dec))
    } else {
        let enc = csc_to_indexed(
            &x_dn,
            config.enc_context_size,
            config.shortlist_weights,
            ctx,
            config.dev,
        )?;
        (enc, None)
    };

    // Gather batch correction at the encoder's per-cell ids — O(N·K), no
    // host-side `[N, S]` scatter.
    let enc_values_null = match x0_nd.as_ref() {
        Some(x0) => Some(gather_null_at_indices(x0, &enc_pack.indices, config.dev)?),
        None => None,
    };

    let (log_z_nk, _) = encoder.forward_indexed_t(
        &enc_pack.indices,
        &enc_pack.values,
        enc_values_null.as_ref(),
        enc_pack.values_mean.as_ref(),
        false,
    )?;

    // Decoder refinement (inference — uniform log_q_s)
    let log_z_nk = if let Some(cfg) = config.refine_config {
        let dec = dec_pack.unwrap();
        let s = dec.union_indices.dim(0)?;
        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, config.dev)?;
        refine_indexed_topic_proportions(
            &log_z_nk,
            &dec.union_indices,
            &dec.scatter_pos,
            &dec.values,
            dec.values_weight.as_ref(),
            &log_q_s,
            config.decoder,
            cfg,
        )?
    } else {
        log_z_nk
    };

    let z_nk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}

/// Refine per-cell topic proportions by gradient descent against the frozen
/// indexed decoder. Consumes packed decoder inputs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_indexed_topic_proportions<Dec: IndexedDecoderT>(
    log_z_nk: &Tensor,
    union_indices: &Tensor,
    scatter_pos: &Tensor,
    values: &Tensor,
    values_weight: Option<&Tensor>,
    log_q_s: &Tensor,
    decoder: &Dec,
    config: &TopicRefinementConfig,
) -> candle_core::Result<Tensor> {
    let z_logits_init = log_z_nk.detach();
    let z_var = Var::from_tensor(&z_logits_init)?;

    for _step in 0..config.num_steps {
        let log_z = ops::log_softmax(z_var.as_tensor(), 1)?;
        let llik = decoder.forward_indexed(
            &log_z,
            union_indices,
            scatter_pos,
            values,
            values_weight,
            log_q_s,
        )?;

        let diff = (z_var.as_tensor() - &z_logits_init)?;
        let reg = (&diff * &diff)?.sum_all()?;

        let loss = ((reg * config.regularization)? - llik.mean_all()?)?;
        let grad = loss.backward()?;
        let z_grad = grad.get(z_var.as_tensor()).unwrap();

        let updated = (z_var.as_tensor() - (z_grad * config.learning_rate)?)?;
        z_var.set(&updated)?;
    }

    ops::log_softmax(z_var.as_tensor(), 1)
}
