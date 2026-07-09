//! Per-gene cell activity + per-level per-dim gate in the score function.
//!
//! - `CellActivities.gene_active_edges[g]` is the precomputed list of
//!   global edge ids where gene `g` is active at *both* endpoints.
//!   Built via an edge-driven merge of per-cell sorted gene lists, so
//!   the cost is O(n_edges × avg_genes_per_cell) instead of the
//!   per-gene scan over all edges (O(n_genes × n_edges)).
//! - `CellActivities.gene_active_edge_weights[g][i]` = `a_g[u] * a_g[v]`
//!   for the corresponding edge. The gene-gated sampler rebuilds a
//!   per-call `WeightedIndex` only over this small list.
//! - `LevelDimGate.gamma` is `[L × D]` pre-softplus. The per-level
//!   gated gene direction enters the score function:
//!   `e_gene[g] ⊙ softplus_floored(γ[ℓ, :])`. Different chain levels
//!   can emphasize different embedding directions; γ gets gradient
//!   from every positive and negative pair at level ℓ.

use crate::util::common::*;
use candle_util::candle_core::{Device, Result as CResult, Tensor};
use candle_util::candle_nn::VarMap;
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
// CellActivities //
////////////////////

pub struct CellActivities {
    /// `gene_active_edges[g]`: global edge ids where both endpoints have
    /// non-zero activity for gene `g`. Sorted ascending.
    pub gene_active_edges: Vec<Vec<u32>>,
    /// `gene_active_edge_weights[g][i]`: weight = `a_g[u] * a_g[v]` for
    /// edge `gene_active_edges[g][i]` (same length).
    pub gene_active_edge_weights: Vec<Vec<f32>>,
}

/// Build per-cell activity (log1p + per-gene L1/L2/identity normalization)
/// and derive per-gene active-edge lists + endpoint-product weights via
/// an edge-driven merge of per-cell sorted gene lists.
pub fn build_cell_activities(
    data: &SparseIoVec,
    edges: &[(u32, u32)],
    block_size: Option<usize>,
    norm: ActivityNorm,
) -> anyhow::Result<CellActivities> {
    let n_cells = data.num_columns();
    let n_genes = data.num_rows();

    // Phase 1: read counts in column blocks; rayon-fold per-thread
    // (gene, cell, log1p) entries, then concat once on the main thread.
    // Avoids the per-row mutex on a shared CooMatrix.
    let block = block_size.unwrap_or(n_genes.max(1));
    let intervals = generate_minibatch_intervals(n_cells, block, None);

    let per_block: Vec<Vec<(usize, usize, f32)>> = intervals
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<Vec<(usize, usize, f32)>> {
            let csc = data.read_columns_csc(lb..ub)?;
            let mut local = Vec::new();
            for col in 0..(ub - lb) {
                let global_col = lb + col;
                let column = csc.col(col);
                for (&g, &v) in column.row_indices().iter().zip(column.values().iter()) {
                    let a = v.ln_1p();
                    if a > 0.0 {
                        local.push((g, global_col, a));
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
    let mut cell_csr = CsrMatrix::from(&coo);
    normalize_rows_inplace(&mut cell_csr, norm);

    // Phase 2: invert to per-cell sorted (gene, activity) lists.
    // Iterating CSR rows in gene order gives each cell's gene-list
    // already sorted ascending, which is what the edge-merge needs.
    let mut cell_to_genes: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_cells];
    for g in 0..n_genes {
        let row = cell_csr.row(g);
        for (&c, &v) in row.col_indices().iter().zip(row.values().iter()) {
            cell_to_genes[c].push((g as u32, v));
        }
    }

    // Phase 3: edge-driven merge — for each edge (u, v), walk the
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

    Ok(CellActivities {
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

//////////////////
// LevelDimGate //
//////////////////

/// Linear-floor coefficient on the positive side of softplus. The
/// floored softplus is `softplus(x) + GAMMA_EPS · relu(x)`:
///
/// - x > 0: `~x + ε·x = (1+ε)·x`, gradient `sigmoid(x) + ε ≥ 0.5 + ε`.
///   Permanent linear-from-0 gradient so γ keeps moving even when the
///   sigmoid has saturated.
/// - x < 0: `~0 + 0 = 0`, gradient `sigmoid(x)` (intrinsic softplus
///   vanishing on the negative side; consistent with "this direction
///   is off at this level").
const GAMMA_EPS: f32 = 1e-2;

/// Numerically stable softplus: `max(x, 0) + log(1 + exp(-|x|))`. The
/// naive `log(1 + exp(x))` overflows for large positive `x`.
fn softplus_stable(x: &Tensor) -> CResult<Tensor> {
    let abs_x = x.abs()?;
    let one = Tensor::ones_like(&abs_x)?;
    let log_term = (one + abs_x.neg()?.exp()?)?.log()?;
    let relu_x = x.relu()?;
    relu_x + log_term
}

/// Stable softplus with a small linear floor on the positive side so
/// the gradient never falls below `GAMMA_EPS` once γ is active.
pub fn softplus_floored(x: &Tensor) -> CResult<Tensor> {
    let sp = softplus_stable(x)?;
    let floor = x.relu()?.affine(GAMMA_EPS as f64, 0.0)?;
    sp + floor
}

/// Per-level per-dim learnable gate `γ[L × D]`. Pre-softplus storage
/// initialized at `0.0` so `softplus_floored(γ₀) = ln(2) ≈ 0.693`
/// uniformly. Owned by the shared `VarMap`; AdamW over
/// `varmap.all_vars()` picks it up.
pub struct LevelDimGate {
    pub gamma: Tensor, // [L, D]
}

impl LevelDimGate {
    pub fn new(
        n_levels: usize,
        embedding_dim: usize,
        varmap: &VarMap,
        dev: &Device,
    ) -> CResult<Self> {
        let init = Tensor::zeros(
            (n_levels, embedding_dim),
            candle_util::candle_core::DType::F32,
            dev,
        )?;
        let var = candle_util::candle_core::Var::from_tensor(&init)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("cage_gamma".to_string(), var.clone());
        Ok(Self {
            gamma: var.as_tensor().clone(),
        })
    }

    /// Snapshot the post-softplus_floored γ matrix `[L × D]` for output.
    pub fn snapshot_gates(&self) -> CResult<Tensor> {
        softplus_floored(&self.gamma)
    }
}
