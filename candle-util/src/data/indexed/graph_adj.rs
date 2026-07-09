//! Feature-feature graph CSR + per-cell sub-adjacency cache + sparse
//! edge-batch construction for the γ-gated GCN encoder block.
//!
//! `GraphCsr` is the symmetric CSR at the encoder's feature axis.
//! `SubAdjCache` holds per-cell pre-normalised triples in slot-space so
//! the per-minibatch sparse edge tensors are just a `par_iter` flatten.

use super::types::IndexedSample;
use candle_core::{Device, Tensor};
use rayon::prelude::*;

///////////////////////////////////////////////////////////////////
// GraphCsr — symmetric undirected adjacency at the feature axis //
///////////////////////////////////////////////////////////////////

/// Compressed-sparse-row representation of a (symmetrized) feature-
/// feature graph over the same feature axis as the encoder input.
///
/// Built once when a `--feature-network` edge list is supplied; held on
/// the `IndexedInMemoryData` so per-minibatch `[N, K_in, K_in]`
/// sub-adjacency tensors can be assembled with `O(K · avg_degree)`
/// lookups per cell.
#[derive(Debug, Clone)]
pub struct GraphCsr {
    pub n_features: usize,
    /// Length `n_features + 1`; `row_ptr[u..u+1]` slices into `col_idx` /
    /// `values` to enumerate neighbours of feature `u`.
    pub row_ptr: Vec<u32>,
    /// Column indices, sorted ascending within each row.
    pub col_idx: Vec<u32>,
    /// Edge weights aligned with `col_idx`.
    pub values: Vec<f32>,
}

impl GraphCsr {
    /// Build a symmetric CSR from a list of `(u, v)` edges with optional
    /// per-edge weights. Self-loops are dropped; the result is
    /// symmetrized (both `(u,v)` and `(v,u)` recorded) and each row
    /// sorted by column.
    pub fn from_edges(
        n_features: usize,
        edges: &[(usize, usize)],
        weights: Option<&[f32]>,
    ) -> Self {
        debug_assert!(weights.is_none_or(|w| w.len() == edges.len()));
        let mut per_row: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_features];
        for (i, &(u, v)) in edges.iter().enumerate() {
            if u == v || u >= n_features || v >= n_features {
                continue;
            }
            let w = weights.map_or(1.0, |ws| ws[i]);
            per_row[u].push((v as u32, w));
            per_row[v].push((u as u32, w));
        }
        // Sort + dedup each row in parallel — for PPI-scale graphs (1M+
        // edges) the per-row sort dominates the build.
        per_row.par_iter_mut().for_each(|row| {
            row.sort_unstable_by_key(|&(c, _)| c);
            row.dedup_by_key(|&mut (c, _)| c);
        });
        let total: usize = per_row.iter().map(Vec::len).sum();
        let mut row_ptr: Vec<u32> = Vec::with_capacity(n_features + 1);
        row_ptr.push(0);
        let mut col_idx: Vec<u32> = Vec::with_capacity(total);
        let mut values: Vec<f32> = Vec::with_capacity(total);
        for row in per_row.iter() {
            for &(c, w) in row.iter() {
                col_idx.push(c);
                values.push(w);
            }
            row_ptr.push(col_idx.len() as u32);
        }
        Self {
            n_features,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Borrow the neighbours of feature `u` as `(col_idx, weights)` slices.
    pub fn row(&self, u: u32) -> (&[u32], &[f32]) {
        let lo = self.row_ptr[u as usize] as usize;
        let hi = self.row_ptr[u as usize + 1] as usize;
        (&self.col_idx[lo..hi], &self.values[lo..hi])
    }

    /// Number of stored undirected entries (each contributes two CSR rows).
    pub fn n_directed_entries(&self) -> usize {
        self.col_idx.len()
    }
}

///////////////////////////////////////////////////////////////////////
// SubAdjCache — per-cell pre-normalised sub-adjacency in slot-space //
///////////////////////////////////////////////////////////////////////

/// Per-cell, pre-normalised sub-adjacency triples flattened across
/// cells. Each triple `(kk_u, kk_v, w)` is an edge in slot-space
/// `[0, K)` on the cell's measured top-K, with `w` already
/// symmetric-normalised à la Kipf-Welling:
/// `Ã[u, v] = (A+I)[u, v] / sqrt(d[u] · d[v])` where `d[u]` is the
/// row-sum of `(A+I)` restricted to the cell's measured top-K. Self-
/// loops are included with weight `1 / d[u]`. The forward pass needs
/// no further row-norm — pure gather + scatter-add.
pub(super) struct SubAdjCache {
    /// Flattened triples for every cell, in cell order.
    pub(super) triples: Vec<(u16, u16, f32)>,
    /// `[n_cells + 1]` offsets into `triples` (CSR-like over cells).
    offsets: Vec<u32>,
}

/// Sym-normalised (Kipf-Welling) sub-adjacency triples for one cell.
///
/// Two passes over the cell's measured top-K:
/// 1. Per-slot degree `d[kk] = 1 (self) + Σ_v∈top-K (A[u, v])`.
/// 2. Emit self-loop `(kk_u, kk_u, 1/d[kk_u])` and each in-top-K
///    neighbour edge `(kk_u, kk_v, w / sqrt(d[kk_u]·d[kk_v]))`.
///
/// `cell_idx` must be sorted ascending (we `binary_search` into it).
fn cell_sym_norm_triples(cell_idx: &[u32], graph: &GraphCsr) -> Vec<(u16, u16, f32)> {
    let take = cell_idx.len();
    if take == 0 {
        return Vec::new();
    }
    let mut d: Vec<f32> = vec![1.0; take];
    for (kk_u, &u) in cell_idx.iter().enumerate() {
        let (cols, weights) = graph.row(u);
        for (&c, &w) in cols.iter().zip(weights.iter()) {
            if cell_idx.binary_search(&c).is_ok() {
                d[kk_u] += w;
            }
        }
    }
    let inv_sqrt_d: Vec<f32> = d.iter().map(|&x| 1.0 / x.max(1e-6).sqrt()).collect();

    let mut out: Vec<(u16, u16, f32)> = Vec::new();
    for (kk_u, &u) in cell_idx.iter().enumerate() {
        let inv_u = inv_sqrt_d[kk_u];
        out.push((kk_u as u16, kk_u as u16, inv_u * inv_u));
        let (cols, weights) = graph.row(u);
        for (&c, &w) in cols.iter().zip(weights.iter()) {
            if let Ok(kk_v) = cell_idx.binary_search(&c) {
                out.push((kk_u as u16, kk_v as u16, w * inv_u * inv_sqrt_d[kk_v]));
            }
        }
    }
    out
}

impl SubAdjCache {
    pub(super) fn build(samples: &[IndexedSample], graph: &GraphCsr, k: usize) -> Self {
        // Top-K size fits in u16 in all realistic uses (k ≤ a few
        // thousand); assert once so the cast is verifiable.
        assert!(
            k <= u16::MAX as usize,
            "encoder top-K {k} exceeds u16; cached adjacency triples would overflow"
        );
        let per_cell: Vec<Vec<(u16, u16, f32)>> = samples
            .par_iter()
            .map(|s| {
                let take = s.indices.len().min(k);
                cell_sym_norm_triples(&s.indices[..take], graph)
            })
            .collect();
        let total: usize = per_cell.iter().map(Vec::len).sum();
        let mut triples: Vec<(u16, u16, f32)> = Vec::with_capacity(total);
        let mut offsets: Vec<u32> = Vec::with_capacity(per_cell.len() + 1);
        offsets.push(0);
        for cell in per_cell.iter() {
            triples.extend_from_slice(cell);
            offsets.push(triples.len() as u32);
        }
        Self { triples, offsets }
    }

    fn cell_triples(&self, cell_idx: usize) -> &[(u16, u16, f32)] {
        let lo = self.offsets[cell_idx] as usize;
        let hi = self.offsets[cell_idx + 1] as usize;
        &self.triples[lo..hi]
    }
}

//////////////////////////////////////////////////////////////////
// SparseEdgeBatch — three [E] flat tensors for the GCN forward //
//////////////////////////////////////////////////////////////////

/// Sparse encoding of a minibatch's stacked sub-adjacencies. Replaces
/// the dense `[B, K, K]` adjacency with three small `[E]` tensors —
/// `E` is the total number of cell-local edges in the minibatch
/// (typically a few × 10⁵ for `K ≈ 500`). Consumed by
/// [`crate::nn::gcn::GcnBlock::forward`] via
/// `index_select` + `index_add` instead of a dense matmul.
pub struct SparseEdgeBatch {
    /// `[E]` u32 — destination flat index `(b * K + kk_u)`.
    pub dst_flat: Tensor,
    /// `[E]` u32 — source flat index `(b * K + kk_v)`.
    pub src_flat: Tensor,
    /// `[E]` f32 — row-normalised edge weight (incl. self-loops).
    pub weight: Tensor,
}

////////////////////////////////////////////////////////////
// Per-minibatch edge scatter (cached + on-the-fly paths) //
////////////////////////////////////////////////////////////

/// Scatter cached per-cell triples (pre-normalised) into the three
/// `[E]` flat tensors of a [`SparseEdgeBatch`]. Parallel per-cell
/// transform; sequential flatten to host buffers. No dense
/// `[B, K, K]` is ever materialised.
pub(super) fn scatter_sparse_edges(
    cache: &SubAdjCache,
    sample_indices: &[usize],
    k: usize,
    target_device: &Device,
) -> anyhow::Result<SparseEdgeBatch> {
    let per_cell: Vec<Vec<(u32, u32, f32)>> = sample_indices
        .par_iter()
        .enumerate()
        .map(|(row, &si)| {
            let row_off = (row * k) as u32;
            cache
                .cell_triples(si)
                .iter()
                .map(|&(kk_u, kk_v, weight)| {
                    (row_off + u32::from(kk_u), row_off + u32::from(kk_v), weight)
                })
                .collect()
        })
        .collect();
    let total: usize = per_cell.iter().map(Vec::len).sum();
    let mut dst: Vec<u32> = Vec::with_capacity(total);
    let mut src: Vec<u32> = Vec::with_capacity(total);
    let mut w: Vec<f32> = Vec::with_capacity(total);
    for triples in &per_cell {
        for &(d, s, weight) in triples {
            dst.push(d);
            src.push(s);
            w.push(weight);
        }
    }
    let dst_flat = Tensor::from_vec(dst, (total,), target_device)?;
    let src_flat = Tensor::from_vec(src, (total,), target_device)?;
    let weight = Tensor::from_vec(w, (total,), target_device)?;
    Ok(SparseEdgeBatch {
        dst_flat,
        src_flat,
        weight,
    })
}

/// Build a [`SparseEdgeBatch`] directly from a packed `[N, K]` indices
/// tensor (the predict-time entry point — at inference we don't keep
/// the `IndexedSample` vectors around). The CSR walk happens inline,
/// parallel over the batch rows.
pub fn build_sparse_edges_from_tensor(
    indices: &Tensor,
    graph: &GraphCsr,
    target_device: &Device,
) -> anyhow::Result<SparseEdgeBatch> {
    let (_n, k) = indices.dims2()?;
    let idx_host = indices
        .to_device(&Device::Cpu)?
        .to_dtype(candle_core::DType::U32)?
        .to_vec2::<u32>()?;

    let per_cell: Vec<Vec<(u32, u32, f32)>> = idx_host
        .par_iter()
        .enumerate()
        .map(|(b, idx_row)| {
            let mut t = k;
            for kk in (1..k).rev() {
                if idx_row[kk] == 0 {
                    t = kk;
                } else {
                    break;
                }
            }
            if t == 0 {
                return Vec::new();
            }
            let row_off = (b * k) as u32;
            cell_sym_norm_triples(&idx_row[..t], graph)
                .into_iter()
                .map(|(kk_u, kk_v, w)| (row_off + u32::from(kk_u), row_off + u32::from(kk_v), w))
                .collect()
        })
        .collect();

    let total: usize = per_cell.iter().map(Vec::len).sum();
    let mut dst: Vec<u32> = Vec::with_capacity(total);
    let mut src: Vec<u32> = Vec::with_capacity(total);
    let mut w: Vec<f32> = Vec::with_capacity(total);
    for triples in &per_cell {
        for &(d, s, weight) in triples {
            dst.push(d);
            src.push(s);
            w.push(weight);
        }
    }
    let dst_flat = Tensor::from_vec(dst, (total,), target_device)?;
    let src_flat = Tensor::from_vec(src, (total,), target_device)?;
    let weight = Tensor::from_vec(w, (total,), target_device)?;
    Ok(SparseEdgeBatch {
        dst_flat,
        src_flat,
        weight,
    })
}
