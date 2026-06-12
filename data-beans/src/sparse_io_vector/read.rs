#![allow(dead_code)]

use super::*;

impl SparseIoVec {
    ////////////////////
    // access columns //
    ////////////////////

    /// Read a single column by global index and append offset triplets.
    /// Under [`ColumnAlignment::Union`] one global column can be backed
    /// by multiple `(didx, loc)` pairs — their triplets all land in the
    /// same output column and `col_offset` advances by 1.
    ///
    /// `pub(super)` so the matched/neighbour read paths in `matched.rs` (a
    /// sibling module) can reuse it.
    pub(super) fn read_column_offset(
        &self,
        glob: usize,
        col_offset: &mut usize,
        triplets: &mut Vec<(u64, u64, f32)>,
    ) -> anyhow::Result<()> {
        let off = *col_offset as u64;
        let g2c = self.global_to_compact_row.as_slice();
        for source in self.col_to_data[glob].iter() {
            let didx = source.backend as usize;
            let loc = source.local_col as usize;
            let (_, _, loc_triplets) = self.data_vec[didx].read_triplets_by_single_column(loc)?;
            let l2g = self.data_local_to_global_row[didx].as_slice();
            for (i, _j, v) in loc_triplets {
                if let Some(c) = g2c[l2g[i as usize]] {
                    triplets.push((c as u64, off, v));
                }
            }
        }
        *col_offset += 1;
        Ok(())
    }

    /// Stream every nonzero of the selected `cells` (global column
    /// indices) through `f(row, col, val)` **without** materializing the
    /// full triplet vector. `row` is a compact-row index and `col` runs
    /// over `0..ncol` in the iteration order of `cells` — exactly the
    /// indices [`Self::columns_triplets`] would emit.
    ///
    /// Columns are processed in slabs of at most `chunk_cols`, so the
    /// only transient allocation is one backend slab's worth of
    /// `(u64, u64, f32)` triplets: peak memory is bounded by `chunk_cols`,
    /// not by the total nnz. Callers can therefore build a compact edge
    /// list (e.g. 12-byte triplets) directly and never pay for the wide
    /// intermediate. Returns the `(nrow, ncol)` dimensions.
    pub fn for_each_triplet<I, F>(
        &self,
        cells: I,
        chunk_cols: usize,
        mut f: F,
    ) -> anyhow::Result<(usize, usize)>
    where
        I: Iterator<Item = usize>,
        F: FnMut(u64, u64, f32),
    {
        let nrow = self.num_rows();
        let cells: Vec<usize> = cells.collect();
        let ncol = cells.len();
        let chunk = chunk_cols.max(1);
        let g2c = self.global_to_compact_row.as_slice();

        for slab_start in (0..ncol).step_by(chunk) {
            let slab_end = (slab_start + chunk).min(ncol);

            // Group this slab's cells by backend, tracking (local_col,
            // out_col) where out_col is the index into the *full* `cells`
            // sequence. Under `ColumnAlignment::Union` one global cell can
            // contribute entries to multiple backend groups (one per
            // backend that observed it); all those reads target the same
            // out_col.
            let mut backend_groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::default();
            for (k, &glob) in cells[slab_start..slab_end].iter().enumerate() {
                let out_col = slab_start + k;
                for source in self.col_to_data[glob].iter() {
                    backend_groups
                        .entry(source.backend as usize)
                        .or_default()
                        .push((source.local_col as usize, out_col));
                }
            }

            for (&didx, group) in &backend_groups {
                let local_cols: Vec<usize> = group.iter().map(|&(loc, _)| loc).collect();
                let (_, _, group_triplets) =
                    self.data_vec[didx].read_triplets_by_columns(local_cols)?;

                let l2g = self.data_local_to_global_row[didx].as_slice();
                if self.data_has_intra_row_merges[didx] {
                    // Canonicalizer collapsed >=2 local rows in this dataset
                    // to the same global. Sum into a HashMap so downstream
                    // consumers don't see duplicate (row, col) entries. Every
                    // entry for a given out_col lives in this slab, so
                    // per-slab accumulation is exact.
                    let mut acc: HashMap<(u64, u64), f32> = HashMap::default();
                    for (i, j, v) in group_triplets {
                        if let Some(c) = g2c[l2g[i as usize]] {
                            let out_col = group[j as usize].1 as u64;
                            *acc.entry((c as u64, out_col)).or_insert(0.0) += v;
                        }
                    }
                    for ((r, c), v) in acc {
                        f(r, c, v);
                    }
                } else {
                    for (i, j, v) in group_triplets {
                        if let Some(c) = g2c[l2g[i as usize]] {
                            let out_col = group[j as usize].1 as u64;
                            f(c as u64, out_col, v);
                        }
                    }
                }
            }
        }

        Ok((nrow, ncol))
    }

    /// Collect all nonzeros of the selected `cells` into one triplet
    /// vector. Thin wrapper over [`Self::for_each_triplet`] with a single
    /// slab spanning every column (identical one-pass behavior); prefer
    /// `for_each_triplet` when the result is consumed once, to avoid the
    /// full-width intermediate.
    #[allow(clippy::type_complexity)]
    pub fn columns_triplets<I>(
        &self,
        cells: I,
    ) -> anyhow::Result<((usize, usize), Vec<(u64, u64, f32)>)>
    where
        I: Iterator<Item = usize>,
    {
        let cells: Vec<usize> = cells.collect();
        let one_slab = cells.len().max(1);
        let mut triplets = Vec::new();
        let dims = self.for_each_triplet(cells.into_iter(), one_slab, |r, c, v| {
            triplets.push((r, c, v));
        })?;
        Ok((dims, triplets))
    }

    pub fn read_columns_ndarray<I>(&self, cells: I) -> anyhow::Result<ndarray::Array2<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.columns_triplets(cells)?;
        ndarray::Array2::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    pub fn read_columns_dmatrix<I>(&self, cells: I) -> anyhow::Result<nalgebra::DMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.columns_triplets(cells)?;
        DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Direct-slice CSC read: bypasses the `triplet → COO → CSC` roundtrip
    /// when the underlying backends have preloaded column arrays.
    ///
    /// For each cell, we slice `(indices, values)` straight out of the
    /// backend's preloaded `by_column_indices` / `by_column_data`, remap
    /// row indices through `l2g` then `g2c` once, drop entries that fall
    /// outside the row intersection, and assemble final CSC arrays in one
    /// pass per column. Backends that aren't preloaded fall back to
    /// per-column triplet reads, which still avoids the global triplet
    /// vec and the column-major sort inside `CscMatrix::from(&coo)`.
    pub fn read_columns_csc<I>(&self, cells: I) -> anyhow::Result<CscMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let cells: Vec<usize> = cells.collect();
        let nrow = self.num_rows();
        let ncol = cells.len();

        // Group cells by backend, carrying the output column index so we
        // can scatter directly into the final per-column buckets. Under
        // `ColumnAlignment::Union` one global cell can be observed by
        // multiple backends — the loop visits every `(didx, loc)` for
        // the same `out_col`, so the bucket accumulates contributions
        // from all observing backends.
        let mut backend_groups: Vec<Vec<(usize, usize)>> =
            (0..self.data_vec.len()).map(|_| Vec::new()).collect();
        for (out_col, &glob) in cells.iter().enumerate() {
            for source in self.col_to_data[glob].iter() {
                backend_groups[source.backend as usize].push((source.local_col as usize, out_col));
            }
        }

        // Per-output-column buckets of (compact_row, value).
        let mut buckets: Vec<Vec<(u32, f32)>> = (0..ncol).map(|_| Vec::new()).collect();
        let g2c = self.global_to_compact_row.as_slice();

        for (didx, group) in backend_groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let l2g = self.data_local_to_global_row[didx].as_slice();

            if let Some((indptr, indices, values)) = self.data_vec[didx].csc_column_arrays() {
                // Fast path: zero-copy slicing into preloaded arrays.
                for &(loc, out_col) in group {
                    if loc + 1 >= indptr.len() {
                        continue;
                    }
                    let s = indptr[loc] as usize;
                    let e = indptr[loc + 1] as usize;
                    let bucket = &mut buckets[out_col];
                    bucket.reserve(e - s);
                    for k in s..e {
                        let local_row = indices[k] as usize;
                        if let Some(c) = g2c[l2g[local_row]] {
                            bucket.push((c as u32, values[k]));
                        }
                    }
                }
            } else {
                // Cold path: route through `read_triplets_by_columns` so the
                // backend can coalesce abutting indptr ranges into a single
                // zarr/hdf5 retrieval (zarr uses `coalesce_and_emit` + chunk
                // LRU cache). For a contiguous block of N cells this becomes
                // ONE retrieval instead of N — the dominant win on cold reads
                // from `.zarr.zip` over a slow disk.
                let cols: Vec<usize> = group.iter().map(|&(loc, _)| loc).collect();
                let (_, _, trip) = self.data_vec[didx].read_triplets_by_columns(cols)?;
                for (i, jj, v) in trip {
                    let (_, out_col) = group[jj as usize];
                    if let Some(c) = g2c[l2g[i as usize]] {
                        buckets[out_col].push((c as u32, v));
                    }
                }
            }
        }

        // Assemble final CSC arrays in one pass.
        let total_nnz: usize = buckets.iter().map(|b| b.len()).sum();
        let mut col_offsets: Vec<usize> = Vec::with_capacity(ncol + 1);
        let mut row_indices: Vec<usize> = Vec::with_capacity(total_nnz);
        let mut values: Vec<f32> = Vec::with_capacity(total_nnz);
        col_offsets.push(0);

        for bucket in &mut buckets {
            // Canonical CSC requires within-column row indices sorted
            // ascending AND unique. On-disk indices are sorted by local
            // row, but two sources can land on the same compact row:
            //   (a) a row canonicalizer that maps two local rows in the
            //       same backend to one global row;
            //   (b) `ColumnAlignment::Union` where multiple backends
            //       contribute to one output column.
            // The composition `l2g[g2c[..]]` is monotonic in the
            // single-source, no-canonicalizer case (the historical
            // fast path) — detect it cheaply and skip the rebuild;
            // otherwise sort and fold duplicates by summing.
            let strictly_sorted_unique = bucket.windows(2).all(|w| w[0].0 < w[1].0);
            if !strictly_sorted_unique {
                bucket.sort_by_key(|&(r, _)| r);
                // Compact duplicates in-place: sum values for equal rows.
                let mut write = 0usize;
                let mut read = 0usize;
                while read < bucket.len() {
                    let (r, mut v) = bucket[read];
                    read += 1;
                    while read < bucket.len() && bucket[read].0 == r {
                        v += bucket[read].1;
                        read += 1;
                    }
                    bucket[write] = (r, v);
                    write += 1;
                }
                bucket.truncate(write);
            }
            for &(r, v) in bucket.iter() {
                row_indices.push(r as usize);
                values.push(v);
            }
            col_offsets.push(row_indices.len());
        }

        CscMatrix::try_from_csc_data(nrow, ncol, col_offsets, row_indices, values)
            .map_err(|e| anyhow::anyhow!("CSC construction failed: {:?}", e))
    }

    pub fn read_columns_csr<I>(&self, cells: I) -> anyhow::Result<CsrMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.columns_triplets(cells)?;
        nalgebra_sparse::CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    pub fn read_columns_tensor<I>(&self, cells: I) -> anyhow::Result<Tensor>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.columns_triplets(cells)?;
        Tensor::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    /// Build (shape, triplets) for the requested compact rows across
    /// all backends. Output column index is the SparseIoVec-global
    /// column (concatenation of backends in push order); output row
    /// index is the position in `rows`.
    #[allow(clippy::type_complexity)]
    pub fn rows_triplets<I>(
        &self,
        rows: I,
    ) -> anyhow::Result<((usize, usize), Vec<(u64, u64, f32)>)>
    where
        I: Iterator<Item = usize>,
    {
        let rows_compact: Vec<usize> = rows.collect();
        let nrow_out = rows_compact.len();
        let ncol_out = self.num_columns();

        let n_compact = self.cached_num_rows;
        let compact_to_global = self.compact_to_global_row.as_slice();

        let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
        let mut local_to_out: Vec<usize> = Vec::with_capacity(rows_compact.len());
        for didx in 0..self.data_vec.len() {
            let g2l = &self.data_global_to_local_row[didx];

            local_to_out.clear();
            let mut local_rows: Vec<usize> = Vec::with_capacity(rows_compact.len());
            for (out_row, &c) in rows_compact.iter().enumerate() {
                if c >= n_compact {
                    continue;
                }
                let g = compact_to_global[c];
                if let Some(&l) = g2l.get(&g) {
                    local_rows.push(l);
                    local_to_out.push(out_row);
                }
            }
            if local_rows.is_empty() {
                continue;
            }

            let (_, _, group_triplets) = self.data_vec[didx].read_triplets_by_rows(local_rows)?;
            let cols_map = self
                .data_to_cols
                .get(&didx)
                .ok_or_else(|| anyhow::anyhow!("missing data_to_cols entry for didx {}", didx))?;
            triplets.reserve(group_triplets.len());
            for (i, j, v) in group_triplets {
                // `usize::MAX` marks a cell dropped by `mask_columns`.
                let mapped = cols_map[j as usize];
                if mapped == usize::MAX {
                    continue;
                }
                let out_row = local_to_out[i as usize] as u64;
                let out_col = mapped as u64;
                triplets.push((out_row, out_col, v));
            }
        }

        Ok(((nrow_out, ncol_out), triplets))
    }

    pub fn read_rows_ndarray<I>(&self, rows: I) -> anyhow::Result<ndarray::Array2<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.rows_triplets(rows)?;
        ndarray::Array2::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    pub fn read_rows_dmatrix<I>(&self, rows: I) -> anyhow::Result<nalgebra::DMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.rows_triplets(rows)?;
        DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    pub fn read_rows_csc<I>(&self, rows: I) -> anyhow::Result<CscMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.rows_triplets(rows)?;
        nalgebra_sparse::CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    pub fn read_rows_csr<I>(&self, rows: I) -> anyhow::Result<CsrMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.rows_triplets(rows)?;
        nalgebra_sparse::CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)
    }

    pub fn read_rows_tensor<I>(&self, rows: I) -> anyhow::Result<Tensor>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.rows_triplets(rows)?;
        Tensor::from_nonzero_triplets(nrow, ncol, &triplets)
    }
}
