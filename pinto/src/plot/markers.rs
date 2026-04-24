//! Marker gene ranking + chunked row extraction.
//!
//! Ranking: for topic `k`, rank genes by descending `gene_topic[g, k]`,
//! breaking ties by *specificity* (prefer genes with low activity in
//! other topics). Skips all-zero genes so we don't ship empty PDFs.
//!
//! Row extraction: data-beans is column-major (cells are columns), so
//! a "gene row for N cells" pass reads cells in blocks and harvests
//! the requested gene rows from each CSC block. This amortizes block
//! I/O across all marker genes in a single pass.

use crate::util::common::*;

/// Return up to `n` marker gene indices (into gt row-ids) + names for
/// topic `k`. Empty if topic has no signal.
pub fn top_n_markers(
    gt: &Mat,
    gene_names: &[Box<str>],
    k: usize,
    n: usize,
) -> Vec<(usize, Box<str>)> {
    if n == 0 || k >= gt.ncols() {
        return Vec::new();
    }
    let ng = gt.nrows();
    let mut scored: Vec<(f32, f32, usize)> = Vec::with_capacity(ng);
    for g in 0..ng {
        let v = gt[(g, k)];
        if !v.is_finite() || v <= 0.0 {
            continue;
        }
        // Tie-break by "specificity": penalize by mean rate in other topics.
        let other_sum: f32 = (0..gt.ncols())
            .filter(|&j| j != k)
            .map(|j| gt[(g, j)])
            .sum();
        let other_mean = if gt.ncols() > 1 {
            other_sum / (gt.ncols() - 1) as f32
        } else {
            0.0
        };
        scored.push((v, other_mean, g));
    }
    // Sort: primary high `v`, secondary low `other_mean` for specificity.
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    scored
        .into_iter()
        .take(n)
        .map(|(_, _, g)| {
            let name = gene_names
                .get(g)
                .cloned()
                .unwrap_or_else(|| format!("gene_{g}").into_boxed_str());
            (g, name)
        })
        .collect()
}

/// Fetch per-cell expression for `gene_names` and an **aligned**
/// per-cell slot list.
///
/// `data_col_ixs[i] = Some(c)` means "use data-beans column `c` for
/// output slot `i`"; `None` leaves slot `i` at `0.0`. The return value
/// has one `Vec<f32>` per gene, each of length `data_col_ixs.len()`,
/// positionally aligned with `core.cell_ixs`.
///
/// Internally we delegate row reads to each backend's `read_rows_csc`
/// (part of the `SparseIo` trait) so we touch only the rows we need
/// instead of loading every gene per cell chunk. That's the fast path
/// for typical top_markers × K ≈ 50 genes × hundreds-of-thousands of
/// cells.
pub fn fetch_gene_rows_aligned(
    data: &SparseIoVec,
    gene_names: &[Box<str>],
    data_col_ixs: &[Option<usize>],
) -> anyhow::Result<Vec<Vec<f32>>> {
    let n_cells = data_col_ixs.len();
    let n_genes = gene_names.len();
    if n_genes == 0 || n_cells == 0 {
        return Ok((0..n_genes).map(|_| vec![0.0; n_cells]).collect());
    }

    let mut out: Vec<Vec<f32>> = (0..n_genes).map(|_| vec![0.0; n_cells]).collect();

    // For each backend: translate gene *names* → backend-local row
    // indices, read just those rows (cheap), and scatter values into
    // slots whose `data_col_ixs` falls in this backend's column range.
    let mut col_cursor = 0usize;
    for didx in 0..data.len() {
        let backend = &data[didx];
        let backend_ncol = backend
            .num_columns()
            .ok_or_else(|| anyhow::anyhow!("backend {didx}: num_columns unavailable"))?;
        let col_start = col_cursor;
        let col_end = col_cursor + backend_ncol;
        col_cursor = col_end;

        // Cells in this backend, tagged with the output slot they feed.
        let backend_cells: Vec<(usize /*out_slot*/, usize /*local_col*/)> = data_col_ixs
            .iter()
            .enumerate()
            .filter_map(|(slot, o)| {
                o.and_then(|c| {
                    if c >= col_start && c < col_end {
                        Some((slot, c - col_start))
                    } else {
                        None
                    }
                })
            })
            .collect();

        if backend_cells.is_empty() {
            continue;
        }

        // Backend row_names → ix so we can translate marker names to
        // backend-local row ids. Intersection across backends is
        // handled by SparseIoVec at load time, but each backend still
        // has its own local row order.
        let row_names = backend.row_names()?;
        let row_ix: HashMap<Box<str>, usize> = row_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        let mut local_row_per_gene: Vec<Option<usize>> = Vec::with_capacity(n_genes);
        let mut present_rows: Vec<usize> = Vec::with_capacity(n_genes);
        for g in gene_names {
            match row_ix.get(g) {
                Some(&r) => {
                    local_row_per_gene.push(Some(present_rows.len()));
                    present_rows.push(r);
                }
                None => local_row_per_gene.push(None),
            }
        }
        if present_rows.is_empty() {
            continue;
        }

        // Read the thin (present_rows × backend_ncol) slab once. Dense
        // because rows are few; columns = entire backend cell count —
        // this is the cheap direction for row-oriented backends (hdf5
        // + zarr both stream rows by stripe).
        let slab = backend.read_rows_ndarray(present_rows.clone())?;

        for (g, maybe_local_row) in local_row_per_gene.iter().enumerate() {
            let Some(lr) = maybe_local_row else { continue };
            for &(out_slot, lc) in &backend_cells {
                out[g][out_slot] = slab[[*lr, lc]];
            }
        }
    }

    Ok(out)
}
