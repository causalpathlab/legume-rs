use super::VecPoint;
use instant_distance::Point;
use multiversion::multiversion;

///////////////////////////////////////
// SIMD Euclidean (L2) distance metric //
///////////////////////////////////////

/// Number of independent accumulator lanes. Chosen to fill a 512-bit register
/// (AVX-512) while decomposing cleanly into narrower ones (2×AVX2, 4×SSE); the
/// independent lanes also break the reduction's dependency chain.
const LANES: usize = 16;

/// Squared Euclidean distance kernel — a plain lane-accumulator loop with no
/// architecture intrinsics. `#[inline(always)]` so it is inlined into whichever
/// `#[multiversion]` entry point calls it and picks up that clone's target
/// features, letting LLVM autovectorise it to AVX-512 / AVX2+FMA / SSE per clone.
#[inline(always)]
pub(super) fn l2_sq_kernel(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "L2 distance on mismatched dimensions");

    let mut acc = [0.0f32; LANES];
    // `as_chunks` hands back `&[[f32; LANES]]` plus the remainder directly, so
    // LLVM sees fixed-size arrays -- no per-index bounds checks -- without the
    // `chunks_exact` + `try_into().unwrap()` dance this replaced.
    let (a_chunks, a_rest) = a.as_chunks::<LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<LANES>();
    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
        for l in 0..LANES {
            let d = ac[l] - bc[l];
            acc[l] += d * d;
        }
    }

    // Horizontal reduction of the lanes; folds left, so the sum order (and the
    // f32 result) matches the hand-rolled index loop this replaced.
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in a_rest.iter().zip(b_rest) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Squared Euclidean distance with runtime SIMD dispatch. Prefer this over
/// [`l2_simd`] when only *ranking* matters (the `sqrt` is monotone), e.g.
/// selecting nearest neighbours or a kernel bandwidth.
#[multiversion(targets = "simd")]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    l2_sq_kernel(a, b)
}

/// Euclidean (L2) distance with runtime SIMD dispatch.
///
/// Returns the *true* Euclidean distance (with `sqrt`) to preserve the semantics
/// of the previous `anndists::DistL2` backend: the kernels in `knn_graph`
/// (`exp_kernel_weights` median-σ, `fuzzy_kernel_weights`) consume distance
/// *values*, not just ranks, so they must match.
#[inline]
pub fn l2_simd(a: &[f32], b: &[f32]) -> f32 {
    l2_sq(a, b).sqrt()
}

impl Point for VecPoint {
    /// Called once per candidate pair inside instant-distance's HNSW traversal.
    ///
    /// Returns the **squared** distance: the traversal only ever *compares*
    /// distances, and squaring is monotone, so the per-pair `sqrt` is skipped
    /// here and applied only to the handful of neighbours actually returned (see
    /// `ColumnDict::search_indices`). Routes through the runtime-dispatched
    /// [`l2_sq`] — a cached indirect call, the portability trade-off.
    #[inline]
    fn distance(&self, other: &Self) -> f32 {
        l2_sq(&self.data, &other.data)
    }
}
