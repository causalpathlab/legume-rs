//! Decoder-side union construction and per-cell scatter positions.
//!
//! Builds `(union_indices [S], scatter_pos [N, K])` from a minibatch's
//! per-cell top-K, where `S` is the de-duplicated count of distinct
//! feature ids across the cells. Uses a per-thread generation-tagged
//! lookup table to avoid `O(D)` zero-reset on every call.

use super::types::IndexedSample;
use candle_core::{Device, Tensor};
use std::cell::RefCell;

////////////////////////////////////////////////////////////////////////
// Per-thread generation-tagged lookup
////////////////////////////////////////////////////////////////////////

/// Per-feature lookup slot. `gen` carries the call generation that wrote
/// `pos`; a slot is "in the current union" iff its `gen` matches the
/// call's generation. Reusing entries via a generation tag instead of a
/// sentinel reset means a panic mid-call cannot leak stale state into
/// the next call on the same rayon worker — the next call simply bumps
/// the generation and every prior write is invalidated.
#[derive(Default, Clone, Copy)]
struct PosSlot {
    generation: u32,
    pos: u32,
}

#[derive(Default)]
struct PosLookup {
    entries: Vec<PosSlot>,
    current_generation: u32,
}

thread_local! {
    /// Per-thread `feature_id -> position` lookup. Grows monotonically to the
    /// largest `n_features` seen on this thread; never resets between calls.
    static POS_LOOKUP: RefCell<PosLookup> = const {
        RefCell::new(PosLookup { entries: Vec::new(), current_generation: 0 })
    };
}

////////////////////////////////////////////////////////////////////////
// Public builders
////////////////////////////////////////////////////////////////////////

/// Decoder-side union + per-cell scatter positions.
///
/// Returns:
/// - `union_indices [S] u32` — sorted-by-discovery union of feature ids
///   appearing in any selected cell's top-K.
/// - `scatter_pos   [N, K] u32 in [0, S)` — for each cell row and slot
///   k, the position of `samples[si].indices[k]` in `union_indices`.
///   Padded slots get position `0` (matched by zero values, harmless).
/// - `union_vec` — the same `[S]` ids as a host `Vec<u32>` for
///   downstream indexing into per-feature arrays (e.g. `output_log_q`).
pub fn build_union_and_scatter_pos(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    n_features: usize,
    k: usize,
    target_device: &Device,
) -> anyhow::Result<(Tensor, Tensor, Vec<u32>)> {
    let n_batch = sample_indices.len();

    POS_LOOKUP.with(|cell| {
        let mut lookup = cell.borrow_mut();
        let PosLookup {
            entries,
            current_generation,
        } = &mut *lookup;
        if entries.len() < n_features {
            entries.resize(n_features, PosSlot::default());
        }
        let mut new_generation = current_generation.wrapping_add(1);
        if new_generation == 0 {
            // Wrapped past u32::MAX — zero every gen so entries written
            // before the wrap are correctly invalidated.
            for slot in entries.iter_mut() {
                slot.generation = 0;
            }
            new_generation = 1;
        }
        *current_generation = new_generation;

        let mut union_vec: Vec<u32> = Vec::new();
        let mut scatter = vec![0u32; n_batch * k];
        for (row, &si) in sample_indices.iter().enumerate() {
            let s = &samples[si];
            let off = row * k;
            let take = s.indices.len().min(k);
            for (kk, &feat) in s.indices[..take].iter().enumerate() {
                let fi = feat as usize;
                if entries[fi].generation != new_generation {
                    entries[fi] = PosSlot {
                        generation: new_generation,
                        pos: union_vec.len() as u32,
                    };
                    union_vec.push(feat);
                }
                scatter[off + kk] = entries[fi].pos;
            }
            // Padded slots [take..k] keep scatter=0 (matches zero value).
        }
        let s = union_vec.len();

        let union_indices = Tensor::from_slice(&union_vec, (s,), target_device)?
            .to_dtype(candle_core::DType::U32)?;
        let scatter_pos = Tensor::from_vec(scatter, (n_batch, k), target_device)?
            .to_dtype(candle_core::DType::U32)?;

        Ok((union_indices, scatter_pos, union_vec))
    })
}

/// Slice the per-feature log selection frequency at the decoder union
/// positions into a `[1, S]` tensor.
pub fn slice_log_q_at_union(
    output_log_q: &[f32],
    union_vec: &[u32],
    target_device: &Device,
) -> anyhow::Result<Tensor> {
    let log_q_s: Vec<f32> = union_vec
        .iter()
        .map(|&idx| output_log_q[idx as usize])
        .collect();
    Ok(Tensor::from_vec(
        log_q_s,
        (1, union_vec.len()),
        target_device,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_union_eq_set(union: &[u32], expected: &[u32]) {
        let mut sorted = union.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, expected);
    }

    fn pos_of(union: &[u32], feat: u32) -> usize {
        union
            .iter()
            .position(|&x| x == feat)
            .unwrap_or_else(|| panic!("feature {feat} not in union {union:?}"))
    }

    /// A panic mid-`build_union_and_scatter_pos` must not poison the
    /// thread-local `POS_LOOKUP` for subsequent calls on the same thread.
    #[test]
    fn test_pos_lookup_panic_recovery() {
        use std::panic;

        // First call OOBs on idx=99, leaving slots 0..=2 with a dirty gen.
        let bad_samples = vec![
            IndexedSample {
                indices: vec![0, 1],
                values: vec![1.0, 1.0],
            },
            IndexedSample {
                indices: vec![2, 99],
                values: vec![1.0, 1.0],
            },
        ];
        let bad_indices = [0usize, 1];
        let result = panic::catch_unwind(|| {
            let _ = build_union_and_scatter_pos(&bad_samples, &bad_indices, 6, 2, &Device::Cpu);
        });
        assert!(result.is_err(), "expected the OOB sample to panic");

        // Clean call on the same thread: every feature in `good_samples`
        // must end up in the union — the gen bump invalidates the dirty
        // slots from the panicking call.
        let good_samples = vec![
            IndexedSample {
                indices: vec![0, 1],
                values: vec![10.0, 11.0],
            },
            IndexedSample {
                indices: vec![2, 3],
                values: vec![12.0, 13.0],
            },
        ];
        let (union_t, scatter_t, union_vec) =
            build_union_and_scatter_pos(&good_samples, &[0, 1], 6, 2, &Device::Cpu).unwrap();
        let union: Vec<u32> = union_t.to_vec1().unwrap();
        assert_union_eq_set(&union, &[0, 1, 2, 3]);
        assert_eq!(union_vec, union);

        let scat: Vec<Vec<u32>> = scatter_t.to_vec2().unwrap();
        assert_eq!(union[scat[0][0] as usize], 0);
        assert_eq!(union[scat[0][1] as usize], 1);
        assert_eq!(union[scat[1][0] as usize], 2);
        assert_eq!(union[scat[1][1] as usize], 3);

        let _ = pos_of(&union, 0);
    }
}
