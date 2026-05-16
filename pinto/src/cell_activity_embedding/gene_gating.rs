//! Per-gene cell activity + per-gene per-level learnable gates.
//!
//! v1 design:
//! - `CellActivities.cell_csr` holds per-gene per-cell activity
//!   (`log1p` of raw counts, optionally L1/L2 normalized per gene).
//! - `CellActivities.gene_active_edges[g]` is the precomputed list of
//!   global edge ids where gene `g` is active at *both* endpoints.
//!   The gene-gated sampler rebuilds a per-call `WeightedIndex` only
//!   over this small list.
//! - `GeneGating.alpha` is `[G × L]` pre-softplus. `softplus(α[g,:])`
//!   multiplies the per-level chain-NCE losses in `fit.rs`, giving each
//!   gene a learnable spatial-scale gating without changing the chain
//!   sampler's negative-side logic.

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

////////////////////////////////////////////////////////////////
//                                                            //
// CellActivities                                             //
//                                                            //
////////////////////////////////////////////////////////////////

pub struct CellActivities {
    /// `gene_active_edges[g]`: global edge ids where both endpoints have
    /// non-zero activity for gene `g`. Sorted ascending.
    pub gene_active_edges: Vec<Vec<u32>>,
    /// `gene_active_edge_weights[g][i]`: weight = `a_g[u] * a_g[v]` for
    /// edge `gene_active_edges[g][i]` (same length).
    pub gene_active_edge_weights: Vec<Vec<f32>>,
}

/// Build per-cell activity in column blocks (rayon over blocks), then
/// derive per-gene active-edge lists + endpoint-product weights.
pub fn build_cell_activities(
    data: &SparseIoVec,
    edges: &[(u32, u32)],
    block_size: Option<usize>,
    norm: ActivityNorm,
) -> anyhow::Result<CellActivities> {
    let n_cells = data.num_columns();
    let n_genes = data.num_rows();

    // ---- Phase 1: read counts in column blocks, log1p each entry.
    let block = block_size.unwrap_or(n_genes.max(1));
    let intervals = generate_minibatch_intervals(n_cells, block, None);

    let coo = Mutex::new(CooMatrix::<f32>::new(n_genes, n_cells));
    intervals
        .par_iter()
        .try_for_each(|&(lb, ub)| -> anyhow::Result<()> {
            let csc = data.read_columns_csc(lb..ub)?;
            let mut local = Vec::<(usize, usize, f32)>::new();
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
            let mut coo = coo.lock().expect("coo mutex");
            for (g, c, a) in local {
                coo.push(g, c, a);
            }
            Ok(())
        })?;

    let raw_csr = CsrMatrix::from(&coo.into_inner().expect("coo final"));
    let cell_csr = normalize_rows(raw_csr, norm);

    // ---- Phase 2: precompute per-gene active edges + endpoint weights.
    // For each gene, build a HashSet<cell_id> of cells with nonzero
    // activity, then scan edges once. Rayon over genes.
    let gene_active: Vec<(Vec<u32>, Vec<f32>)> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let row = cell_csr.row(g);
            if row.nnz() == 0 {
                return (Vec::new(), Vec::new());
            }
            // cell -> activity lookup
            let mut a: HashMap<u32, f32> =
                HashMap::with_capacity_and_hasher(row.nnz(), Default::default());
            for (&c, &v) in row.col_indices().iter().zip(row.values().iter()) {
                a.insert(c as u32, v);
            }
            let mut active = Vec::<u32>::new();
            let mut weights = Vec::<f32>::new();
            for (e_idx, &(u, v)) in edges.iter().enumerate() {
                if let (Some(&au), Some(&av)) = (a.get(&u), a.get(&v)) {
                    active.push(e_idx as u32);
                    weights.push(au * av);
                }
            }
            (active, weights)
        })
        .collect();

    let mut gene_active_edges = Vec::with_capacity(n_genes);
    let mut gene_active_edge_weights = Vec::with_capacity(n_genes);
    for (e, w) in gene_active {
        gene_active_edges.push(e);
        gene_active_edge_weights.push(w);
    }

    Ok(CellActivities {
        gene_active_edges,
        gene_active_edge_weights,
    })
}

fn normalize_rows(csr: CsrMatrix<f32>, norm: ActivityNorm) -> CsrMatrix<f32> {
    let n_rows = csr.nrows();
    let n_cols = csr.ncols();
    let mut coo = CooMatrix::<f32>::new(n_rows, n_cols);
    for g in 0..n_rows {
        let row = csr.row(g);
        if row.nnz() == 0 {
            continue;
        }
        let scale = match norm {
            ActivityNorm::Log1p => 1.0_f32,
            ActivityNorm::L1 => {
                let s: f32 = row.values().iter().sum();
                if s > 0.0 {
                    1.0 / s
                } else {
                    0.0
                }
            }
            ActivityNorm::L2 => {
                let s: f32 = row.values().iter().map(|v| v * v).sum::<f32>().sqrt();
                if s > 0.0 {
                    1.0 / s
                } else {
                    0.0
                }
            }
        };
        for (&c, &v) in row.col_indices().iter().zip(row.values().iter()) {
            coo.push(g, c, v * scale);
        }
    }
    CsrMatrix::from(&coo)
}

////////////////////////////////////////////////////////////////
//                                                            //
// GeneGating                                                 //
//                                                            //
////////////////////////////////////////////////////////////////

/// Per-gene per-level learnable gate `α[G × L]`. Pre-softplus storage
/// initialized at `ln(e − 1) ≈ 0.5413` so `softplus(α₀) = 1.0`. Owned
/// by the shared `VarMap`; AdamW over `varmap.all_vars()` picks it up.
pub struct GeneGating {
    pub alpha: Tensor, // [G, L]
}

impl GeneGating {
    pub fn new(n_genes: usize, n_levels: usize, varmap: &VarMap, dev: &Device) -> CResult<Self> {
        let init_val: f32 = (std::f32::consts::E - 1.0).ln(); // softplus -> 1.0
        let init = Tensor::full(init_val, (n_genes, n_levels), dev)?;
        let var = candle_util::candle_core::Var::from_tensor(&init)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert("cage_alpha".to_string(), var.clone());
        Ok(Self {
            alpha: var.as_tensor().clone(),
        })
    }

    /// Return `softplus(α[genes, :])` as a `[G, L]` tensor with an ε
    /// floor. One `index_select + softplus + clamp` chain regardless of
    /// `G`, so cage's batched fwd/bwd makes a single CUDA call here.
    pub fn gates_batch(&self, genes: &[usize], dev: &Device) -> CResult<Tensor> {
        let idx_vec: Vec<u32> = genes.iter().map(|&g| g as u32).collect();
        let g = idx_vec.len();
        let idx = Tensor::from_vec(idx_vec, g, dev)?;
        let rows = self.alpha.index_select(&idx, 0)?; // [G, L]
        let one = Tensor::ones_like(&rows)?;
        let exp_x = rows.exp()?;
        let softplus = (one + exp_x)?.log()?;
        softplus.clamp(1e-4_f32, f32::INFINITY)
    }

    /// Snapshot the post-softplus gate matrix `[G × L]` for output.
    pub fn snapshot_gates(&self) -> CResult<Tensor> {
        let one = Tensor::ones_like(&self.alpha)?;
        let exp_x = self.alpha.exp()?;
        (one + exp_x)?.log()
    }
}
