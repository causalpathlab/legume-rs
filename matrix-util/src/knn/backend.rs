use super::{ColumnDict, MakeVecPoint, VecPoint};
use instant_distance::{Builder, HnswMap};
use rustc_hash::FxHashMap as HashMap;
use std::fmt::{Debug, Display};

////////////////////////////
// k-NN backend selection //
////////////////////////////

/// Fixed seed for HNSW construction. instant-distance's `Builder` seeds its
/// layer-assignment RNG from OS entropy by default; pinning it makes the index —
/// and therefore every downstream k-NN graph — bit-for-bit reproducible. This is
/// the property `hnsw_rs` could not provide and the root fix for the annotate
/// `--seed` non-reproducibility.
pub(super) const KNN_SEED: u64 = 20_240_517;

/// HNSW build-time beam width. Higher = better graph quality at more build cost.
pub(super) const EF_CONSTRUCTION: usize = 200;

/// HNSW search-time beam width, baked into the index at build. Recall is governed
/// by how far this exceeds `k`; the old `hnsw_rs` wrapper starved it at
/// `max(k, 24)`. 128 comfortably covers every call site's `k` (graph builds use
/// ~15–50, landmark/match queries use 1).
///
/// This is also the ceiling on results the approximate backend can return: a
/// query for `knn > EF_SEARCH` is silently truncated to `EF_SEARCH` neighbours
/// (guarded by a `debug_assert!` in `ColumnDict::search_indices`). Raise this if
/// a caller ever needs more than 128 approximate neighbours.
pub(super) const EF_SEARCH: usize = 128;

/// Below this many points, use the exact brute-force path: recall = 1.0,
/// deterministic, and cheaper than an HNSW traversal at small `n`.
pub(super) const EXACT_THRESHOLD: usize = 8_192;

/// Search backend behind a [`ColumnDict`].
pub(super) enum Backend {
    /// Exact O(n) scan per query (see [`super::exact`]).
    Exact,
    /// Approximate HNSW; the stored `u32` value is the original column index.
    Approx(HnswMap<VecPoint, u32>),
}

/// Build a [`ColumnDict`] from column views, choosing the exact backend at or
/// below `exact_threshold` points and the seeded HNSW above it.
///
/// The `_serial` / parallel constructor split upstream is now cosmetic:
/// instant-distance always builds via rayon's global pool internally, so there
/// is no serial insertion path to select here. The two-tier public API is kept
/// only for source compatibility with existing callers.
pub(super) fn build_column_dict<T, V>(
    data: Vec<V>,
    names: Vec<T>,
    exact_threshold: usize,
) -> ColumnDict<T>
where
    T: Clone + Eq + std::hash::Hash + Debug + Display,
    V: Sync + MakeVecPoint,
{
    let nn = data.len();
    debug_assert!(
        nn == names.len(),
        "Data and names must have the same length"
    );

    let data_vec: Vec<VecPoint> = data.iter().map(|v| v.to_vp()).collect();

    let mut name2index: HashMap<T, usize> = Default::default();
    names.iter().enumerate().for_each(|(j, x)| {
        name2index.insert(x.clone(), j);
    });

    let backend = if nn <= exact_threshold {
        Backend::Exact
    } else {
        // value = original column index, so search results map straight back to
        // `names` regardless of instant-distance's internal point reordering.
        let values: Vec<u32> = (0..nn as u32).collect();
        // `data_vec.clone()` duplicates only the `Vec<VecPoint>` spine — each
        // `VecPoint` holds an `Arc<[f32]>`, so the coordinate buffers are shared
        // with `ColumnDict.data_vec`, not copied.
        let map = Builder::default()
            .seed(KNN_SEED)
            .ef_construction(EF_CONSTRUCTION)
            .ef_search(EF_SEARCH)
            .build(data_vec.clone(), values);
        Backend::Approx(map)
    };

    let ret = ColumnDict {
        backend,
        data_vec,
        name2index,
        names,
    };

    #[cfg(debug_assertions)]
    {
        for (i, x) in ret.names.iter().enumerate() {
            if let Some(&j) = ret.name2index.get(x) {
                debug_assert_eq!(i, j);
            }
        }
    }
    ret
}
