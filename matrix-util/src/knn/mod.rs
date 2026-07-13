//! Named-column k-nearest-neighbour index ([`ColumnDict`]).
//!
//! # Querying
//!
//! There are two ways to query, and **neither requires you to build a
//! [`VecPoint`]** — that type is internal storage, not a query argument:
//!
//! * By the name of a column already in the dictionary — [`ColumnDict::search_others`]
//!   / [`ColumnDict::search_by_query_name`] / [`ColumnDict::match_by_query_name_against`].
//! * By an arbitrary query vector — [`ColumnDict::search_by_query_data`], which
//!   takes a plain `&[f32]`.
//!
//! ## Getting a `&[f32]` to query with
//!
//! The query slice must have the dictionary's dimension ([`ColumnDict::dim`]).
//! How you obtain it depends on how your data is laid out:
//!
//! * **A column of a column-major `DMatrix`** is contiguous — pass it directly:
//!   ```ignore
//!   let col = mat.column(j);            // DVectorView, contiguous
//!   dict.search_by_query_data(col.as_slice(), k)?;
//!   ```
//! * **A row of a `DMatrix`, or any non-contiguous view** — collect once into a
//!   `Vec<f32>` and pass a reference:
//!   ```ignore
//!   let q: Vec<f32> = mat.row(i).iter().copied().collect();
//!   dict.search_by_query_data(&q, k)?;
//!   ```
//! * **Querying many points in a loop** — reuse one `Vec<f32>` scratch buffer
//!   across iterations (`clear()` + `extend()`), passing `&buf` each time, rather
//!   than allocating per query. For repeated *approximate* queries also reuse a
//!   [`SearchScratch`] via [`ColumnDict::search_others_reuse`].
//!
//! The approximate backend internally wraps the query slice in a `VecPoint`
//! (an `Arc` over `query.len()` floats — not the point set), so callers never
//! construct one themselves.

use rustc_hash::FxHashMap as HashMap;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use instant_distance::Search;

mod backend;
mod exact;
pub mod metric;

#[cfg(test)]
mod tests;

use backend::{build_column_dict, Backend, EF_SEARCH, EXACT_THRESHOLD};
pub use metric::l2_simd;

/// A dictionary of named columns backed by a k-nearest-neighbour index.
///
/// The backend is chosen automatically at construction: an exact brute-force
/// scan below `EXACT_THRESHOLD` points, or a seeded instant-distance HNSW above
/// it (see the `backend` module). Both are deterministic. `data_vec` is the
/// canonical original-order store of the points (used for self-search queries,
/// `dim`, and the exact path); read it through [`points`](Self::points) /
/// [`num_points`](Self::num_points).
pub struct ColumnDict<K> {
    backend: Backend,
    data_vec: Vec<VecPoint>,
    name2index: HashMap<K, usize>,
    names: Vec<K>,
}

impl<K> ColumnDict<K>
where
    K: Clone + Eq + std::hash::Hash + Debug + Display + std::cmp::PartialEq,
{
    /// The point names in original (insertion) order: index `i` corresponds to
    /// `data_vec[i]` and to the indices the `search_*` methods map back to names.
    pub fn names(&self) -> &Vec<K> {
        &self.names
    }

    /// Build an index from `ndarray` **row** views — each view is one point,
    /// labelled by the matching entry of `names` (same length, same order).
    /// Coordinates are copied in, so the views need not outlive the dictionary.
    pub fn from_ndarray_views<'a>(data: Vec<ndarray::ArrayView1<'a, f32>>, names: Vec<K>) -> Self {
        <ColumnDict<K> as ColumnDictOps<K, ndarray::ArrayView1<'a, f32>>>::from_column_views(
            data, names,
        )
    }

    /// Build an index in which **each row** of `data` is a point (ndarray is
    /// row-major, so rows are the contiguous unit). `names.len()` must equal
    /// `data.nrows()`.
    pub fn from_ndarray(data: ndarray::Array2<f32>, names: Vec<K>) -> Self {
        let views: Vec<_> = data.outer_iter().collect();
        Self::from_ndarray_views(views, names)
    }

    /// Build an index from nalgebra column views — each view is one point,
    /// labelled by the matching entry of `names`.
    pub fn from_dvector_views(data: Vec<nalgebra::DVectorView<f32>>, names: Vec<K>) -> Self {
        <ColumnDict<K> as ColumnDictOps<K, nalgebra::DVectorView<f32>>>::from_column_views(
            data, names,
        )
    }

    /// Build an index in which **each column** of `data` is a point (nalgebra is
    /// column-major). `names.len()` must equal `data.ncols()`.
    ///
    /// Note the orientation flip versus [`from_ndarray`](Self::from_ndarray)
    /// (rows): each constructor takes the input's *contiguous* axis as the point.
    pub fn from_dmatrix(data: nalgebra::DMatrix<f32>, names: Vec<K>) -> Self {
        Self::from_dvector_views(data.column_iter().collect(), names)
    }

    /// An empty dictionary (no points; every search returns no neighbours). The
    /// `_ndarray_`/`_dvector_` suffix only fixes the `ColumnDictOps` view type and
    /// is otherwise identical.
    pub fn empty_ndarray_views() -> Self {
        <ColumnDict<K> as ColumnDictOps<K, ndarray::ArrayView1<f32>>>::empty()
    }

    /// See [`empty_ndarray_views`](Self::empty_ndarray_views).
    pub fn empty_dvector_views() -> Self {
        <ColumnDict<K> as ColumnDictOps<K, nalgebra::DVectorView<f32>>>::empty()
    }

    /// Dimension (length) of the stored points, or `None` when the dictionary is
    /// empty. A query passed to [`search_by_query_data`](Self::search_by_query_data)
    /// must have this length.
    pub fn dim(&self) -> Option<usize> {
        self.data_vec.first().map(|p| p.len())
    }

    /// Number of stored points.
    pub fn num_points(&self) -> usize {
        self.data_vec.len()
    }

    /// The stored point coordinates, in original (insertion) order — aligned
    /// with [`names`](Self::names).
    pub fn points(&self) -> impl Iterator<Item = &[f32]> {
        self.data_vec.iter().map(|p| p.as_slice())
    }

    /// The `knn` nearest neighbours of a stored column, **excluding the column
    /// itself** — a call with `knn` returns up to `knn` *other* columns (fewer
    /// only if the dictionary holds fewer than `knn + 1` points).
    ///
    /// * `query_name` - name of the column to search around
    /// * `knn` - number of other neighbours to return
    ///
    /// Returns `(names, distances)` in nearest-first order. Errors if `query_name`
    /// is not in the dictionary.
    pub fn search_others(&self, query_name: &K, knn: usize) -> anyhow::Result<(Vec<K>, Vec<f32>)> {
        self.search_by_query_name(query_name, knn, true)
    }

    /// k-nearest-neighbour match by name within the same dictionary.
    ///
    /// * `query_name` - the name of the column to match
    /// * `knn` - the number of nearest neighbours to return
    /// * `exclude_same` - exclude the query column from the results
    pub fn search_by_query_name(
        &self,
        query_name: &K,
        knn: usize,
        exclude_same: bool,
    ) -> anyhow::Result<(Vec<K>, Vec<f32>)> {
        let self_idx = self.index_of(query_name)?;
        let query = &self.data_vec[self_idx];
        let exclude = if exclude_same { Some(self_idx) } else { None };
        let mut search = Search::default();
        let (indices, distances) = self.search_indices(query, knn, exclude, &mut search);
        Ok((self.names_of(&indices), distances))
    }

    /// Like [`search_others`](Self::search_others) but reuses a caller-owned
    /// [`SearchScratch`] across many queries against this dictionary.
    ///
    /// Reusing one scratch avoids re-growing instant-distance's visited set
    /// (O(n)) on every query — worth it in tight loops such as building a k-NN
    /// graph. The exact backend ignores the scratch. Hold one scratch per rayon
    /// worker (queries against a shared `&self` are independent).
    pub fn search_others_reuse(
        &self,
        query_name: &K,
        knn: usize,
        scratch: &mut SearchScratch,
    ) -> anyhow::Result<(Vec<K>, Vec<f32>)> {
        let self_idx = self.index_of(query_name)?;
        let query = &self.data_vec[self_idx];
        let (indices, distances) = self.search_indices(query, knn, Some(self_idx), &mut scratch.0);
        Ok((self.names_of(&indices), distances))
    }

    /// k-nearest-neighbour match by an arbitrary query vector.
    ///
    /// Pass the query as a plain slice — do **not** wrap it in a [`VecPoint`].
    /// See the [module docs](self) for how to obtain a `&[f32]` from a matrix
    /// column/row without an extra copy.
    ///
    /// * `query` - query coordinates as a slice; its length must equal
    ///   [`dim`](Self::dim), otherwise an error is returned.
    /// * `knn` - the number of nearest neighbours to return (clamped to the
    ///   dictionary size).
    ///
    /// Returns `(names, distances)` in nearest-first order.
    pub fn search_by_query_data(
        &self,
        query: &[f32],
        knn: usize,
    ) -> anyhow::Result<(Vec<K>, Vec<f32>)> {
        if self.dim().unwrap_or(0) != query.len() {
            return Err(anyhow::anyhow!("query's dim does not match"));
        }
        // The approximate backend searches by `&VecPoint`; wrap the query once
        // (an Arc over `query.len()` floats, not the point set).
        let query = VecPoint {
            data: Arc::from(query),
        };
        let mut search = Search::default();
        let (indices, distances) = self.search_indices(&query, knn, None, &mut search);
        Ok((self.names_of(&indices), distances))
    }

    /// k-nearest-neighbour match by name against *another* dictionary, returning
    /// names from that dictionary.
    ///
    /// * `query_name` - the name of the column to match (in `self`)
    /// * `knn` - the number of nearest neighbours to return
    /// * `against` - the dictionary to search
    pub fn match_by_query_name_against(
        &self,
        query_name: &K,
        knn: usize,
        against: &Self,
    ) -> anyhow::Result<(Vec<K>, Vec<f32>)> {
        let self_idx = self.index_of(query_name)?;
        let query = &self.data_vec[self_idx];
        let mut search = Search::default();
        let (indices, distances) = against.search_indices(query, knn, None, &mut search);
        Ok((against.names_of(&indices), distances))
    }

    /// Core backend dispatch: return the `knn` nearest column *indices* and their
    /// distances to `query`.
    ///
    /// When `exclude` is set (self-search) it fetches one extra candidate and
    /// then drops that index, so exactly `knn` *others* are returned — i.e.
    /// `search_others(k)` yields `k` neighbours, not `k - 1`.
    fn search_indices(
        &self,
        query: &VecPoint,
        knn: usize,
        exclude: Option<usize>,
        search: &mut Search,
    ) -> (Vec<usize>, Vec<f32>) {
        // Fetch one extra when excluding self so dropping it still leaves `knn`.
        let fetch = if exclude.is_some() { knn + 1 } else { knn };
        let (mut indices, mut distances) = match &self.backend {
            Backend::Exact => exact::topk(&self.data_vec, &query.data, fetch),
            Backend::Approx(map) => {
                // `EF_SEARCH` is baked into the index at build, so the search
                // yields at most that many candidates; a larger request would be
                // silently truncated (see the const's doc).
                debug_assert!(
                    fetch <= EF_SEARCH,
                    "knn={knn} exceeds EF_SEARCH={EF_SEARCH}; approx results are truncated"
                );
                // `search` is reset internally by instant-distance and its `ef`
                // is overwritten from the index, so a reused (or fresh) buffer is
                // equally correct here.
                let mut indices = Vec::with_capacity(fetch);
                let mut distances = Vec::with_capacity(fetch);
                for item in map.search(query, search).take(fetch) {
                    indices.push(*item.value as usize);
                    // `Point::distance` returns squared L2; take the sqrt only
                    // for the neighbours we actually return (true Euclidean).
                    distances.push(item.distance.sqrt());
                }
                (indices, distances)
            }
        };

        if let Some(e) = exclude {
            if let Some(pos) = indices.iter().position(|&i| i == e) {
                indices.remove(pos);
                distances.remove(pos);
            }
            // Self may be absent (exact ties at distance 0 can crowd it out of
            // the top `knn + 1`); cap back to `knn`.
            indices.truncate(knn);
            distances.truncate(knn);
        }
        (indices, distances)
    }

    #[inline]
    fn names_of(&self, indices: &[usize]) -> Vec<K> {
        indices.iter().map(|&i| self.names[i].clone()).collect()
    }

    #[inline]
    fn index_of(&self, name: &K) -> anyhow::Result<usize> {
        self.name2index
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("name {} not found", name))
    }
}

/// Reusable scratch buffer for [`ColumnDict::search_others_reuse`].
///
/// Wraps instant-distance's per-search working state so it can be amortised
/// across many queries. Create one per rayon worker with `Default`. Not `Sync`
/// (it is mutated in place), so give each thread its own.
#[derive(Default)]
pub struct SearchScratch(Search);

////////////////////////////////////
// Construction trait (by view type) //
////////////////////////////////////

/// View-type-generic construction of a [`ColumnDict`].
///
/// Prefer the concrete `ColumnDict::from_*` constructors in application code;
/// this trait exists so the build logic is shared across the supported view
/// types `V` (`Vec<f32>`, nalgebra and ndarray views — anything [`MakeVecPoint`]).
pub trait ColumnDictOps<K, V> {
    /// An empty dictionary (no points).
    fn empty() -> Self;
    /// Build from a list of column views, one point per view, aligned to `names`.
    fn from_column_views(data: Vec<V>, names: Vec<K>) -> Self;
}

impl<T, V> ColumnDictOps<T, V> for ColumnDict<T>
where
    T: Clone + Eq + std::hash::Hash + Debug + Display,
    V: Sync + MakeVecPoint,
{
    fn empty() -> Self {
        Self {
            backend: Backend::Exact,
            data_vec: vec![],
            name2index: Default::default(),
            names: vec![],
        }
    }

    fn from_column_views(data: Vec<V>, names: Vec<T>) -> Self {
        build_column_dict(data, names, EXACT_THRESHOLD)
    }
}

//////////////////////////////
// Point wrapper + adapters //
//////////////////////////////

/// A stored point in the index.
///
/// The coordinates live behind an `Arc<[f32]>` so that cloning a `VecPoint` is a
/// refcount bump, not a data copy: the approximate backend hands instant-distance
/// its own `Vec<VecPoint>` (which it reorders) while `ColumnDict.data_vec` keeps
/// the originals — both share the same underlying buffers. Queries are passed as
/// plain `&[f32]`, so this type is only ever *constructed* internally (via
/// [`MakeVecPoint`]); read its coordinates with [`as_slice`](Self::as_slice).
#[derive(Clone, Debug)]
pub struct VecPoint {
    data: Arc<[f32]>,
}

impl VecPoint {
    /// Number of coordinates (the point dimension).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the point has zero coordinates.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow the coordinates as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Converts one column/vector view into the index's internal [`VecPoint`].
///
/// This is the extension point that lets the `from_*_views` constructors accept
/// `Vec<f32>`, nalgebra column views, and ndarray row views uniformly. It is a
/// *construction*-side adapter only — queries are `&[f32]`, not `VecPoint`.
pub trait MakeVecPoint {
    /// Copy this view's coordinates into a freshly-allocated `VecPoint`.
    fn to_vp(&self) -> VecPoint;
}

impl MakeVecPoint for Vec<f32> {
    fn to_vp(&self) -> VecPoint {
        VecPoint {
            data: self.iter().copied().collect(),
        }
    }
}

impl MakeVecPoint for nalgebra::DVectorView<'_, f32> {
    fn to_vp(&self) -> VecPoint {
        VecPoint {
            data: self.iter().copied().collect(),
        }
    }
}

impl MakeVecPoint for ndarray::ArrayView1<'_, f32> {
    fn to_vp(&self) -> VecPoint {
        VecPoint {
            data: self.iter().copied().collect(),
        }
    }
}
