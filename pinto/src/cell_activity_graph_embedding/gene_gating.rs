//! Per-gene cell activity: which edges each gene is active on, and how
//! strongly.
//!
//! - `GeneActiveEdges.gene_active_edges[g]` is the precomputed list of
//!   global edge ids where gene `g` is active at *both* endpoints.
//!   Built via an edge-driven merge of per-cell sorted gene lists, so
//!   the cost is O(n_edges × avg_genes_per_cell) instead of the
//!   per-gene scan over all edges (O(n_genes × n_edges)).
//! - `GeneActiveEdges.gene_active_edge_weights[g][i]` = `a_g[u] * a_g[v]`
//!   for the corresponding edge. The gene-gated sampler rebuilds a
//!   per-call `WeightedIndex` only over this small list.
//!
//! Purely the data-derived sampling weight: this module owns no
//! learnable parameters, and nothing here is fit.

use crate::util::common::*;
use crate::util::gene_axis::GeneAxis;
use clap::ValueEnum;
use matrix_util::utils::generate_minibatch_intervals;
use nalgebra_sparse::{CooMatrix, CsrMatrix};

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ActivityNorm {
    L1,
    L2,
    Log1p,
}

////////////////////
// GeneActiveEdges //
////////////////////

pub struct GeneActiveEdges {
    /// `gene_active_edges[g]`: global edge ids where both endpoints have
    /// non-zero activity for gene `g`. Sorted ascending. The struct is
    /// used at TWO levels: fine cell-cell edge ids as built here, then
    /// PB super-edge ids after `fold_active_edges_to_super`.
    pub gene_active_edges: Vec<Vec<u32>>,
    /// `gene_active_edge_weights[g][i]`: weight = `a_g[u] * a_g[v]` for
    /// edge `gene_active_edges[g][i]` (same length).
    pub gene_active_edge_weights: Vec<Vec<f32>>,
}

/// Build per-cell activity (log1p + per-gene L1/L2/identity normalization)
/// and derive per-gene active-edge lists + endpoint-product weights via
/// an edge-driven merge of per-cell sorted gene lists.
///
/// Activity is per GENE, not per matrix row: on a splice-channelized input a
/// gene's two rows are summed BEFORE the `log1p`, so `a_g[c]` is one gene's
/// evidence rather than two half-evidence copies competing for the same
/// positives. `log1p(s) + log1p(u) != log1p(s + u)`, so the order matters and
/// the summing cannot be moved after the transform.
pub fn build_gene_active_fine_edges(
    data: &SparseIoVec,
    edges: &[(u32, u32)],
    block_size: Option<usize>,
    norm: ActivityNorm,
    axis: &GeneAxis,
) -> anyhow::Result<GeneActiveEdges> {
    let n_cells = data.num_columns();
    let n_rows = data.num_rows();
    let n_genes = axis.n_genes();
    anyhow::ensure!(
        axis.n_rows() == n_rows,
        "gene axis has {} rows, data has {n_rows}",
        axis.n_rows()
    );

    // Step 1: read counts in column blocks; rayon-fold per-thread
    // (gene, cell, count) entries, then concat once on the main thread.
    // Avoids the per-row mutex on a shared CooMatrix. The COO→CSR conversion
    // SUMS duplicate entries, which is what folds a gene's channel rows
    // together; `log1p` is applied to the summed value afterwards.
    let block = block_size.unwrap_or(n_rows.max(1));
    let intervals = generate_minibatch_intervals(n_cells, block, None);

    let per_block: Vec<Vec<(usize, usize, f32)>> = intervals
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<Vec<(usize, usize, f32)>> {
            let csc = data.read_columns_csc(lb..ub)?;
            let mut local = Vec::new();
            for col in 0..(ub - lb) {
                let global_col = lb + col;
                let column = csc.col(col);
                for (&r, &v) in column.row_indices().iter().zip(column.values().iter()) {
                    if v > 0.0 {
                        local.push((axis.gene_of_row(r), global_col, v));
                    }
                }
            }
            Ok(local)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut coo = CooMatrix::<f32>::new(n_genes, n_cells);
    for block in per_block {
        for (g, c, a) in block {
            coo.push(g, c, a);
        }
    }
    let mut gene_by_cell_csr = CsrMatrix::from(&coo);
    // Over every nonzero of the matrix, so it is worth the rayon hop: this used to
    // ride along inside the parallel block read, and moving it after the sum (which
    // it has to be) would otherwise have made it the one serial pass.
    gene_by_cell_csr.values_mut().par_iter_mut().for_each(|x| {
        *x = x.ln_1p();
    });
    normalize_rows_inplace(&mut gene_by_cell_csr, norm);

    // Step 2: invert to per-cell sorted (gene, activity) lists.
    // Iterating CSR rows in gene order gives each cell's gene-list
    // already sorted ascending, which is what the edge-merge needs.
    let mut cell_to_genes: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_cells];
    for g in 0..n_genes {
        let row = gene_by_cell_csr.row(g);
        for (&c, &v) in row.col_indices().iter().zip(row.values().iter()) {
            cell_to_genes[c].push((g as u32, v));
        }
    }

    // Step 3: edge-driven merge — for each edge (u, v), walk the
    // two sorted lists in tandem, emitting `(edge_idx, a_u·a_v)` for
    // every common gene. Genes are appended in edge-iteration order,
    // so per-gene edge lists are sorted ascending by construction.
    let mut gene_active_edges: Vec<Vec<u32>> = vec![Vec::new(); n_genes];
    let mut gene_active_edge_weights: Vec<Vec<f32>> = vec![Vec::new(); n_genes];
    for (e_idx, &(u, v)) in edges.iter().enumerate() {
        let gu = &cell_to_genes[u as usize];
        let gv = &cell_to_genes[v as usize];
        let (mut i, mut j) = (0usize, 0usize);
        while i < gu.len() && j < gv.len() {
            match gu[i].0.cmp(&gv[j].0) {
                std::cmp::Ordering::Equal => {
                    let g = gu[i].0 as usize;
                    gene_active_edges[g].push(e_idx as u32);
                    gene_active_edge_weights[g].push(gu[i].1 * gv[j].1);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
    }

    Ok(GeneActiveEdges {
        gene_active_edges,
        gene_active_edge_weights,
    })
}

/// In-place row-wise renormalization of a CSR. Skips the COO round-trip
/// that an out-of-place build would require.
fn normalize_rows_inplace(csr: &mut CsrMatrix<f32>, norm: ActivityNorm) {
    if matches!(norm, ActivityNorm::Log1p) {
        return;
    }
    let n_rows = csr.nrows();
    let offsets: Vec<usize> = csr.row_offsets().to_vec();
    let values = csr.values_mut();
    for g in 0..n_rows {
        let (start, end) = (offsets[g], offsets[g + 1]);
        if start == end {
            continue;
        }
        let scale = match norm {
            ActivityNorm::Log1p => 1.0_f32,
            ActivityNorm::L1 => {
                let s: f32 = values[start..end].iter().sum();
                if s > 0.0 {
                    1.0 / s
                } else {
                    0.0
                }
            }
            ActivityNorm::L2 => {
                let s: f32 = values[start..end].iter().map(|v| v * v).sum::<f32>().sqrt();
                if s > 0.0 {
                    1.0 / s
                } else {
                    0.0
                }
            }
        };
        if scale != 1.0 {
            for v in &mut values[start..end] {
                *v *= scale;
            }
        }
    }
}

/// Fold per-gene ACTIVE FINE EDGES onto PB-PB super edges: each fine
/// edge's endpoint-product weight is summed into the super edge it maps
/// to, and intra-PB fine edges (`fine_to_super[e] == None`) are dropped
/// — they fold into the node, matching `build_super_graph` semantics.
///
/// Returns the same `GeneActiveEdges` shape one level up, so the
/// existing `(gene, batch)` cache builder consumes it unchanged: edge
/// ids are super-edge ids, sorted ascending per gene, weights parallel.
pub fn fold_active_edges_to_super(
    activities: GeneActiveEdges,
    fine_to_super: &[Option<usize>],
) -> GeneActiveEdges {
    use rayon::prelude::*;
    let folded: Vec<(Vec<u32>, Vec<f32>)> = activities
        .gene_active_edges
        .into_par_iter()
        .zip(activities.gene_active_edge_weights.into_par_iter())
        .map(|(edges, weights)| {
            let mut acc: std::collections::BTreeMap<u32, f32> = std::collections::BTreeMap::new();
            for (&e, &w) in edges.iter().zip(weights.iter()) {
                if let Some(s) = fine_to_super[e as usize] {
                    *acc.entry(s as u32).or_insert(0.0) += w;
                }
            }
            acc.into_iter().unzip()
        })
        .collect();
    let (gene_active_edges, gene_active_edge_weights): (Vec<_>, Vec<_>) =
        folded.into_iter().unzip();
    GeneActiveEdges {
        gene_active_edges,
        gene_active_edge_weights,
    }
}
