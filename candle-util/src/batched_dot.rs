//! Batched dot products, as gemm rather than broadcast-multiply-then-sum.
//!
//! # Why this module exists
//!
//! `Σ_h a[b,k,h]·v[b,h]` is naturally written `a.broadcast_mul(&v.unsqueeze(1)?)?.sum(2)`,
//! and that form is slow twice over:
//!
//! - **The multiply.** Broadcasting `[B,1,H]` against `[B,K,H]` puts a stride-0
//!   dim BETWEEN two non-zero strides. `candle_core::Layout::offsets_b` strips
//!   only LEADING and TRAILING zero strides before requiring the rest to be
//!   contiguous, so that layout returns `None` and the op falls to a scalar
//!   `StridedIndex` loop — no SIMD, single-threaded.
//! - **The reduction.** `sum` takes candle's vectorized path only when it
//!   reduces over the LAST dims. Reducing over a middle axis walks every element
//!   doing an integer div/mod and a scattered accumulate.
//!
//! Either way an `[B,K,H]` product is materialized, and backward turns that one
//! temporary into several more. `matmul` keeps the product implicit and hits
//! gemm. Measured on this workspace: 10–26x on realistic shapes, and still 4–5x
//! at the smallest shapes tested — there is no size at which the broadcast form
//! wins, so these are unconditional replacements.
//!
//! Both helpers live here rather than in a model type because the callers span
//! `candle-util`, `graph-embedding-util` and `pinto`, and the dependency
//! direction (geu → candle-util → matrix-util) means only this crate is
//! reachable from all of them.

use candle_core::{Result, Tensor};

/// `out[b,k] = Σ_h a[b,k,h] · v[b,h]` — a per-batch mat-vec.
///
/// `a` is `[B, K, H]`, `v` is `[B, H]`, result is `[B, K]`.
pub fn batched_matvec(a: &Tensor, v: &Tensor) -> Result<Tensor> {
    a.matmul(&v.unsqueeze(2)?)?.squeeze(2)
}

/// `out[b,h] = Σ_k w[b,k] · a[b,k,h]` — the transposed case, i.e. a weighted sum
/// over `K` (attention pooling, mixture collapse).
///
/// `w` is `[B, K]`, `a` is `[B, K, H]`, result is `[B, H]`.
pub fn batched_weighted_sum(w: &Tensor, a: &Tensor) -> Result<Tensor> {
    w.unsqueeze(1)?.matmul(a)?.squeeze(1)
}

/// `out[b,k] = Σ_h a[b,k,h] · v[h]` — one shared vector against every batch.
///
/// `a` is `[B, K, H]`, `v` is `[H]`, result is `[B, K]`. Folds the batch into
/// the row axis so this is a single gemm rather than `B` of them.
pub fn batched_matvec_shared(a: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (b, k, h) = a.dims3()?;
    a.reshape((b * k, h))?
        .matmul(&v.reshape((h, 1))?)?
        .reshape((b, k))
}
