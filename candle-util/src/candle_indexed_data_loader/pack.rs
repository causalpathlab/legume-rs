//! Per-cell pack/gather routines that materialise `[N, K]` host buffers
//! before the candle device upload. Every loop is parallel over
//! `sample_indices` via `par_chunks_mut`.

use super::types::IndexedSample;
use candle_core::{Device, Tensor};
use rayon::prelude::*;

/// Walk every cell's top-K positions and fill an `[N, K]` `f32` buffer.
///
/// `fill(row, kk, sample_index, feature_id) -> value` is invoked once
/// per observed slot; padded slots (when the sample has fewer than `k`
/// indices) keep the buffer's initial `0.0`. Pad indices are harmless
/// because every consumer either pairs them with a zero value side or
/// only multiplies them in (zero · anything = 0).
///
/// Parallel over rows. `fill` must be `Fn + Sync`.
pub(crate) fn pack_at_indices<F>(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    k: usize,
    target_device: &Device,
    fill: F,
) -> anyhow::Result<Tensor>
where
    F: Fn(usize, usize, usize, u32) -> f32 + Sync,
{
    let n = sample_indices.len();
    let mut buf = vec![0.0f32; n * k];
    buf.par_chunks_mut(k)
        .zip(sample_indices.par_iter())
        .enumerate()
        .for_each(|(row, (chunk, &si))| {
            let s = &samples[si];
            let take = s.indices.len().min(k);
            for (kk, &feat) in s.indices[..take].iter().enumerate() {
                chunk[kk] = fill(row, kk, si, feat);
            }
        });
    Ok(Tensor::from_vec(buf, (n, k), target_device)?)
}

/// Pack per-cell `(indices, values)` into `[N, K]` u32/f32 tensors in
/// parallel. Both buffers filled in lockstep via `par_chunks_mut`.
pub fn pack_indices_values(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    k: usize,
    target_device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let n = sample_indices.len();
    let mut idx_buf = vec![0u32; n * k];
    let mut val_buf = vec![0.0f32; n * k];
    idx_buf
        .par_chunks_mut(k)
        .zip(val_buf.par_chunks_mut(k))
        .zip(sample_indices.par_iter())
        .for_each(|((idx_chunk, val_chunk), &si)| {
            let s = &samples[si];
            let take = s.indices.len().min(k);
            idx_chunk[..take].copy_from_slice(&s.indices[..take]);
            val_chunk[..take].copy_from_slice(&s.values[..take]);
        });
    let indices =
        Tensor::from_vec(idx_buf, (n, k), target_device)?.to_dtype(candle_core::DType::U32)?;
    let values = Tensor::from_vec(val_buf, (n, k), target_device)?;
    Ok((indices, values))
}

/// Pack only the per-cell values into `[N, K]` f32 — for the decoder
/// path that already has its scatter positions and doesn't need the
/// indices tensor. Parallel over rows.
pub(crate) fn pack_values_only(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    k: usize,
    target_device: &Device,
) -> anyhow::Result<Tensor> {
    let n = sample_indices.len();
    let mut val_buf = vec![0.0f32; n * k];
    val_buf
        .par_chunks_mut(k)
        .zip(sample_indices.par_iter())
        .for_each(|(chunk, &si)| {
            let s = &samples[si];
            let take = s.values.len().min(k);
            chunk[..take].copy_from_slice(&s.values[..take]);
        });
    Ok(Tensor::from_vec(val_buf, (n, k), target_device)?)
}

/// Pack a per-cell row `(per_sample[si][feat])` at `samples[si].indices`
/// into `[N, K] f32`. Used for the encoder's μ_residual batch null.
pub(crate) fn pack_null_at_indices(
    samples: &[IndexedSample],
    null_rows: &[Vec<f32>],
    sample_indices: &[usize],
    k: usize,
    target_device: &Device,
) -> anyhow::Result<Tensor> {
    pack_at_indices(
        samples,
        sample_indices,
        k,
        target_device,
        |_, _, si, feat| null_rows[si][feat as usize],
    )
}

/// Gather a per-feature `[D]` slice at each cell's top-K positions into
/// `[N, K] f32`. Used for both encoder gene-mean (`μ_d`) and decoder
/// NB-Fisher weights — each cell sees the same constant per feature.
pub fn gather_per_feature_at_indices(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    per_feature: &[f32],
    k: usize,
    target_device: &Device,
) -> anyhow::Result<Tensor> {
    pack_at_indices(
        samples,
        sample_indices,
        k,
        target_device,
        |_, _, _, feat| per_feature[feat as usize],
    )
}
