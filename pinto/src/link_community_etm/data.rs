//! Edge profile assembly for `pinto lc-etm`.
//!
//! Builds per-edge count vectors `y_e = x_i + x_j` as dense `[E, G]`
//! matrices. The V-cycle pipeline assembles one such matrix per coarsening
//! level, with `y_e` summing cells inside each endpoint super-cell at
//! coarser levels.

use data_beans::sparse_io_vector::SparseIoVec;
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;

type Mat = DMatrix<f32>;

/// Read the full cell × gene count matrix as a sparse CSC `[G, N]`.
///
/// Single read of every cell column from the backing `SparseIoVec`. The
/// result is held for the duration of training so per-edge `y_e` rows
/// can be assembled cheaply without rereading the backend.
pub fn read_cells_csc(data_vec: &SparseIoVec) -> anyhow::Result<CscMatrix<f32>> {
    let n_cells = data_vec.num_columns();
    data_vec.read_columns_csc(0..n_cells)
}

/// Build the dense per-edge count matrix `[E, G]` where row `e` is
/// `y_e = x_i + x_j` for edge `e = (i, j)`.
///
/// `cell_counts` is the cell × gene sparse matrix in `[G, N]` orientation
/// (genes as rows, cells as columns) — the layout produced by
/// [`read_cells_csc`].
pub fn build_edge_profiles(
    cell_counts: &CscMatrix<f32>,
    edges: &[(usize, usize)],
) -> Mat {
    let n_genes = cell_counts.nrows();
    let n_edges = edges.len();

    // Flat row-major buffer filled in parallel — one allocation total
    // before the final `from_row_slice` conversion to column-major DMatrix.
    let mut flat = vec![0f32; n_edges * n_genes];
    flat.par_chunks_mut(n_genes)
        .zip(edges.par_iter())
        .for_each(|(row, &(i, j))| {
            accumulate_column(row, cell_counts, i);
            accumulate_column(row, cell_counts, j);
        });
    Mat::from_row_slice(n_edges, n_genes, &flat)
}

fn accumulate_column(out: &mut [f32], csc: &CscMatrix<f32>, col: usize) {
    let column = csc.col(col);
    for (&r, &v) in column.row_indices().iter().zip(column.values().iter()) {
        out[r] += v;
    }
}

/// Sum cell columns within each super-cell label → `super_counts[label]` is
/// a dense `[G]` vector of summed counts. Returned as `Vec<Vec<f32>>` so
/// each super-cell's profile sits in a contiguous slice — convenient for
/// the per-super-edge double sum that follows.
pub fn build_super_cell_counts(
    cell_counts: &CscMatrix<f32>,
    cell_labels: &[usize],
) -> Vec<Vec<f32>> {
    let n_genes = cell_counts.nrows();
    let n_super = cell_labels
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let mut super_counts: Vec<Vec<f32>> = vec![vec![0f32; n_genes]; n_super];
    for (cell, &label) in cell_labels.iter().enumerate() {
        accumulate_column(&mut super_counts[label], cell_counts, cell);
    }
    super_counts
}

/// Per-super-edge profile matrix `[E_super, G]` where row `e` is
/// `super_counts[a] + super_counts[b]` for super-edge `(a, b)`.
pub fn build_super_edge_profiles(
    super_counts: &[Vec<f32>],
    super_edges: &[(usize, usize)],
) -> Mat {
    let n_genes = super_counts.first().map(|r| r.len()).unwrap_or(0);
    let n_edges = super_edges.len();
    let mut flat = vec![0f32; n_edges * n_genes];
    flat.par_chunks_mut(n_genes)
        .zip(super_edges.par_iter())
        .for_each(|(row, &(a, b))| {
            let ca = &super_counts[a];
            let cb = &super_counts[b];
            for g in 0..n_genes {
                row[g] = ca[g] + cb[g];
            }
        });
    Mat::from_row_slice(n_edges, n_genes, &flat)
}

/// Per-gene total counts across all cells — used as the per-feature
/// shortlist weight floor and for `--min-gene-count` filtering.
pub fn gene_total_counts(cell_counts: &CscMatrix<f32>) -> Vec<f32> {
    let mut totals = vec![0f32; cell_counts.nrows()];
    for col_idx in 0..cell_counts.ncols() {
        let col = cell_counts.col(col_idx);
        for (&r, &v) in col.row_indices().iter().zip(col.values().iter()) {
            totals[r] += v;
        }
    }
    totals
}

/// Shortlist weights: per-gene non-negative scalars used to *rank*
/// candidates inside the top-K selector. Stored values stay as raw counts.
///
/// `min_count` floors genes with total < threshold to zero, effectively
/// removing them from being selected into the encoder/decoder window.
pub fn shortlist_weights(gene_totals: &[f32], min_count: f32) -> Vec<f32> {
    gene_totals
        .iter()
        .map(|&t| if t >= min_count { 1.0 } else { 0.0 })
        .collect()
}
