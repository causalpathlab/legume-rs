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
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (ac, bc) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        // chunks_exact yields exactly LANES elements; the array cast lets LLVM
        // drop per-index bounds checks and vectorise the lane loop.
        let ac: &[f32; LANES] = ac.try_into().unwrap();
        let bc: &[f32; LANES] = bc.try_into().unwrap();
        for l in 0..LANES {
            let d = ac[l] - bc[l];
            acc[l] += d * d;
        }
    }

    let mut sum = 0.0f32;
    for l in 0..LANES {
        sum += acc[l];
    }
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
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
    /// This routes through the runtime-dispatched [`l2_sq`]; the dispatch is a
    /// cached indirect call (portable across CPUs, the chosen trade-off) rather
    /// than a statically-inlined kernel.
    #[inline]
    fn distance(&self, other: &Self) -> f32 {
        l2_simd(&self.data, &other.data)
    }
}
