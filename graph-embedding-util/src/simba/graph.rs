//! `si.tl.gen_graph`: the cell–gene edge list with one relation per level.
//!
//! Entities are the cells (every column of the backend) and the HVGs (the
//! caller's `hvg_rows`, in that order). Every nonzero HVG entry is one edge
//! `(cell, gene, level)`. The level comes from [`super::discretize`], which
//! sees the log-normalized nonzero values of ALL genes — SIMBA discretizes
//! before it subsets to the HVGs — so the build streams the backend twice:
//! pass 1 takes library sizes, the global value range and the HVG entries;
//! pass 2 fills the histogram over every gene. Peak memory is one column slab
//! plus the HVG stash (`cell u32 + gene u32 + value f32` per edge, the value
//! replaced by a `u8` level once the bins are known).

use super::discretize::{log_norm, Discretization, Histogram};
use super::HIST_BINS;
use crate::progress::new_progress_bar;
use data_beans::sparse_io_vector::SparseIoVec;
use rand::{Rng, RngExt};
use std::ops::Range;

/// PBG's `weight` per relation (bin level): `round(linspace(1, 5, n), 2)`
/// over the levels PRESENT in the graph, ascending. Rust's `round` is
/// half-away-from-zero where Python's is half-to-even; no `linspace(1, 5, n ≤ 5)`
/// value sits on a half, so the two agree here.
#[derive(Clone, Debug)]
pub struct RelationTable {
    /// Present levels, ascending.
    pub levels: Vec<u8>,
    /// Loss weight of each relation, aligned with `levels`.
    pub weights: Vec<f32>,
    level_to_rel: [u8; 256],
}

impl RelationTable {
    #[must_use]
    pub fn from_levels(present: &[u8]) -> Self {
        let mut levels = present.to_vec();
        levels.sort_unstable();
        levels.dedup();
        let n = levels.len();
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                let w = if n > 1 {
                    1.0 + 4.0 * i as f64 / (n - 1) as f64
                } else {
                    1.0
                };
                ((w * 100.0).round() / 100.0) as f32
            })
            .collect();
        let mut level_to_rel = [u8::MAX; 256];
        for (r, &l) in levels.iter().enumerate() {
            level_to_rel[l as usize] = r as u8;
        }
        Self {
            levels,
            weights,
            level_to_rel,
        }
    }

    /// Relation index of a level (panics on a level absent from the graph).
    #[must_use]
    pub fn rel(&self, level: u8) -> usize {
        let r = self.level_to_rel[level as usize];
        assert!(r != u8::MAX, "level {level} has no relation");
        r as usize
    }

    #[must_use]
    pub fn weight(&self, level: u8) -> f32 {
        self.weights[self.rel(level)]
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
}

/// `pbg_train(auto_wd=True)`: the weight decay SIMBA fits to the edge count,
/// scaled off two reference graphs (`0.013` at 2,725,781 edges below 5e7
/// edges, `0.0004` at 59,103,481 edges above), rounded to 6 decimals
/// (half-away-from-zero here vs numpy's half-to-even: a tie needs the 7th
/// decimal to be exactly 5, which no edge count of interest produces).
#[must_use]
pub fn auto_wd(n_edges: usize) -> f64 {
    let n = n_edges.max(1) as f64;
    let wd = if n < 5e7 {
        0.013 * 2_725_781.0 / n
    } else {
        0.0004 * 59_103_481.0 / n
    };
    (wd * 1e6).round() / 1e6
}

/// The graph as three parallel arrays (structure of arrays), so an epoch's
/// shuffle is an in-place permutation with no index vector.
#[derive(Clone, Debug)]
pub struct EdgeList {
    pub n_cells: usize,
    pub n_genes: usize,
    pub cell: Vec<u32>,
    pub gene: Vec<u32>,
    /// Expression level, `1..=n_levels`.
    pub level: Vec<u8>,
}

impl EdgeList {
    #[must_use]
    pub fn len(&self) -> usize {
        self.cell.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cell.is_empty()
    }

    pub(crate) fn swap(&mut self, i: usize, j: usize) {
        self.cell.swap(i, j);
        self.gene.swap(i, j);
        self.level.swap(i, j);
    }

    /// Fisher–Yates over `range` only; edges outside it stay where they are.
    pub(crate) fn shuffle_range<R: Rng>(&mut self, range: Range<usize>, rng: &mut R) {
        let start = range.start;
        let n = range.end - range.start;
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            self.swap(start + i, start + j);
        }
    }

    /// Distinct levels in the graph, ascending.
    #[must_use]
    pub fn levels_present(&self) -> Vec<u8> {
        let mut seen = [false; 256];
        for &l in &self.level {
            seen[l as usize] = true;
        }
        (0..=u8::MAX).filter(|&l| seen[l as usize]).collect()
    }
}

/// Column-slab width targeting ~8M nonzeros per slab (as the composite
/// trainer's sampler does), so the streaming memory bound holds.
fn slab_width(data: &SparseIoVec) -> usize {
    let n_cells = data.num_columns().max(1);
    match data.num_non_zeros() {
        Ok(nnz) if nnz > 0 => {
            let avg_per_col = (nnz / n_cells).max(1);
            (8_000_000 / avg_per_col).clamp(1, n_cells)
        }
        _ => (1usize << 14).min(n_cells),
    }
}

/// Build the edge list over `hvg_rows` (backend row indices, which become
/// gene entities `0..hvg_rows.len()` in that order) with `n_bins` levels.
pub fn build_edge_list(
    data: &SparseIoVec,
    hvg_rows: &[usize],
    n_bins: usize,
) -> anyhow::Result<(EdgeList, Discretization)> {
    let n_cells = data.num_columns();
    let n_rows = data.num_rows();
    anyhow::ensure!(n_cells > 0, "simba: no cells");
    anyhow::ensure!(!hvg_rows.is_empty(), "simba: no genes selected");
    let mut hvg_pos = vec![u32::MAX; n_rows];
    for (g, &row) in hvg_rows.iter().enumerate() {
        anyhow::ensure!(
            row < n_rows,
            "simba: gene row {row} outside the {n_rows} backend rows"
        );
        anyhow::ensure!(
            hvg_pos[row] == u32::MAX,
            "simba: gene row {row} listed twice"
        );
        hvg_pos[row] = g as u32;
    }
    let chunk = slab_width(data);

    // Pass 1: library sizes, the global value range, and the HVG entries.
    let mut lib_size = vec![0f64; n_cells];
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut cell: Vec<u32> = Vec::new();
    let mut gene: Vec<u32> = Vec::new();
    let mut val: Vec<f32> = Vec::new();
    let bar = new_progress_bar(n_cells as u64);
    bar.set_message("simba graph: library sizes + HVG entries");
    let mut start = 0usize;
    while start < n_cells {
        let end = (start + chunk).min(n_cells);
        let csc = data.read_columns_csc(start..end)?;
        for (lc, col) in csc.col_iter().enumerate() {
            let c = start + lc;
            let s: f64 = col.values().iter().map(|&x| f64::from(x)).sum();
            lib_size[c] = s;
            if s <= 0.0 {
                continue;
            }
            for (&row, &x) in col.row_indices().iter().zip(col.values()) {
                if x <= 0.0 {
                    continue;
                }
                let v = log_norm(x, s);
                let vf = f64::from(v);
                lo = lo.min(vf);
                hi = hi.max(vf);
                let g = hvg_pos[row];
                if g != u32::MAX {
                    cell.push(c as u32);
                    gene.push(g);
                    val.push(v);
                }
            }
        }
        bar.inc((end - start) as u64);
        start = end;
    }
    bar.finish_and_clear();
    anyhow::ensure!(
        lo.is_finite(),
        "simba: the count matrix has no nonzero entries"
    );
    anyhow::ensure!(
        !cell.is_empty(),
        "simba: the selected genes have no nonzero entries"
    );

    // Pass 2: the histogram over every nonzero value of every gene.
    let mut hist = Histogram::new(lo, hi, HIST_BINS)?;
    let bar = new_progress_bar(n_cells as u64);
    bar.set_message("simba graph: value histogram");
    let mut start = 0usize;
    while start < n_cells {
        let end = (start + chunk).min(n_cells);
        data.for_each_triplet(start..end, end - start, |_brow, lc, x| {
            if x > 0.0 {
                let s = lib_size[start + lc as usize];
                if s > 0.0 {
                    hist.add(f64::from(log_norm(x, s)));
                }
            }
        })?;
        bar.inc((end - start) as u64);
        start = end;
    }
    bar.finish_and_clear();

    let disc = Discretization::fit(&hist, n_bins)?;
    let level: Vec<u8> = val.iter().map(|&v| disc.level(v)).collect();
    drop(val);
    log::info!(
        "simba graph: {} cells × {} genes, {} edges, {} levels (edges {:?})",
        n_cells,
        hvg_rows.len(),
        cell.len(),
        disc.n_levels(),
        disc.bin_edges
            .iter()
            .map(|e| (e * 1e3).round() / 1e3)
            .collect::<Vec<_>>()
    );
    Ok((
        EdgeList {
            n_cells,
            n_genes: hvg_rows.len(),
            cell,
            gene,
            level,
        },
        disc,
    ))
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod graph_tests;
