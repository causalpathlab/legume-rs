use super::metric::l2_sq_kernel;
use super::VecPoint;
use multiversion::multiversion;

////////////////////////////////////
// Exact brute-force k-NN fallback //
////////////////////////////////////

/// Fill `out` with `(index, squared-L2 distance)` for every point vs. `query`.
///
/// Multiversioned so the whole scan runs under one runtime dispatch (amortised
/// over all `points`), with `l2_sq_kernel` inlined and autovectorised for the
/// selected CPU. Squared distances are enough to rank; the `sqrt` is applied
/// only to the surviving top-`k` in [`topk`].
#[multiversion(targets = "simd")]
fn scan_sq(points: &[VecPoint], query: &[f32], out: &mut Vec<(usize, f32)>) {
    out.clear();
    out.extend(
        points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, l2_sq_kernel(&p.data, query))),
    );
}

/// Exact top-`k` nearest neighbours of `query` among `points`, returned as
/// `(indices, distances)` in nearest-first order (true Euclidean distances).
///
/// Serial by design: every call site fans out *across* queries with rayon, so
/// parallelising the inner scan too would nest pools and oversubscribe. Used
/// below `EXACT_THRESHOLD`, where an O(n) scan per query is cheaper than an HNSW
/// traversal and gives recall = 1.0 deterministically.
///
/// Self is *not* excluded here; when a self-search is wanted the caller keeps
/// the query itself in the top-`k` budget and drops it afterwards, matching the
/// historical `ColumnDict` semantics (see [`super::ColumnDict::search_indices`]).
pub(super) fn topk(points: &[VecPoint], query: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(points.len());
    scan_sq(points, query, &mut scored);

    let kk = k.min(scored.len());
    if kk == 0 {
        return (Vec::new(), Vec::new());
    }

    // Rank by squared distance (monotone in the true distance), then sort just
    // the kk smallest and take the sqrt only for those survivors.
    scored.select_nth_unstable_by(kk - 1, |a, b| a.1.total_cmp(&b.1));
    scored.truncate(kk);
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let indices = scored.iter().map(|&(i, _)| i).collect();
    let distances = scored.iter().map(|&(_, d2)| d2.sqrt()).collect();
    (indices, distances)
}
