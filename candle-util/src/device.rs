//! Device memory queries and memory-aware chunk sizing.
//!
//! A candle training step's footprint is dominated by the retained
//! autograd graph, which scales with the minibatch ("chunk") size, so
//! the safe chunk is a function of free device memory that users
//! otherwise have to guess (and guess again per GPU). This module
//! measures instead: [`gpu_mem_info`] asks the driver, and
//! [`auto_chunk_size`] probes the caller's own forward pass at growing
//! sizes until the next doubling would not fit.
//!
//! The one physical subtlety: the transient peak of a full step is NOT
//! observable after the step returns, because intermediates are freed
//! with the graph. But candle keeps the graph alive for as long as the
//! LOSS TENSOR lives, so a forward-only probe that returns its loss
//! lets us measure the resident graph before dropping it. `backward`
//! then roughly doubles that (one gradient per retained intermediate),
//! which is why the budget is halved rather than estimated exactly.

use crate::candle_core::{Device, Result as CandleResult, Tensor};
use log::info;

/// `(free, total)` device memory in bytes, when the device can say.
///
/// `None` on CPU and Metal, on builds without the `cuda` feature (the
/// dummy backend has no driver handle), and when the driver query
/// itself fails: `None` always means "keep your configured default",
/// never an error.
pub fn gpu_mem_info(dev: &Device) -> Option<(usize, usize)> {
    #[cfg(feature = "cuda")]
    {
        let cuda = dev.as_cuda_device().ok()?;
        cuda.cuda_stream().context().mem_get_info().ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = dev;
        None
    }
}

/// Pick a chunk size for the caller's training step from measured free
/// memory, by probing the caller's own forward pass.
///
/// `forward_at(n)` must run ONE representative forward at chunk size
/// `n` and return its (un-backwarded) loss tensor; this function
/// measures free memory while that tensor is alive, drops it, and
/// doubles `n` from `floor` while the projected footprint of the next
/// doubling stays inside `target_frac` of the initially free memory,
/// with half the budget reserved for `backward`'s gradient copies.
///
/// Returns `None` when the device cannot report memory (see
/// [`gpu_mem_info`]); the caller keeps its configured default. The
/// result is clamped to `[floor, cap]`.
pub fn auto_chunk_size(
    dev: &Device,
    cap: usize,
    floor: usize,
    target_frac: f32,
    mut forward_at: impl FnMut(usize) -> CandleResult<Tensor>,
) -> Option<usize> {
    let (free0, total) = gpu_mem_info(dev)?;
    let floor = floor.max(1).min(cap);
    // Half for the forward graph, half for backward's gradients.
    let budget = (free0 as f64 * f64::from(target_frac.clamp(0.05, 0.95)) * 0.5) as usize;

    let mut n = floor;
    let mut chosen = floor;
    let mut per_item = 0usize;
    loop {
        let loss = match forward_at(n) {
            Ok(t) => t,
            // A failed probe (often an allocation failure at this n) is
            // an answer, not an error: the previous size stood.
            Err(_) => break,
        };
        let (free_alive, _) = gpu_mem_info(dev)?;
        drop(loss);
        let used = free0.saturating_sub(free_alive);
        per_item = (used / n.max(1)).max(1);
        chosen = n;
        if n >= cap {
            break;
        }
        // Would the next doubling still fit the halved budget?
        let next = (n * 2).min(cap);
        if next * per_item > budget {
            break;
        }
        n = next;
    }

    info!(
        "auto chunk: {} (cap {}, floor {}) from {:.2} GiB free of {:.2} GiB, \
         ~{} KiB per item, {:.0}% target",
        chosen,
        cap,
        floor,
        free0 as f64 / 1024.0 / 1024.0 / 1024.0,
        total as f64 / 1024.0 / 1024.0 / 1024.0,
        per_item / 1024,
        f64::from(target_frac) * 100.0
    );
    Some(chosen)
}

/// The pure arithmetic behind [`auto_chunk_size`]'s stopping rule,
/// separated so it is testable without a device: given the measured
/// bytes per item and the initially free bytes, the largest power-of-two
/// multiple of `floor` within `[floor, cap]` whose footprint fits half
/// of `target_frac * free`.
pub fn chunk_from_measurements(
    free: usize,
    per_item: usize,
    cap: usize,
    floor: usize,
    target_frac: f32,
) -> usize {
    let floor = floor.max(1).min(cap.max(1));
    let budget = (free as f64 * f64::from(target_frac.clamp(0.05, 0.95)) * 0.5) as usize;
    let per_item = per_item.max(1);
    let mut n = floor;
    while n < cap && (n * 2).min(cap) * per_item <= budget {
        n = (n * 2).min(cap);
    }
    n
}
