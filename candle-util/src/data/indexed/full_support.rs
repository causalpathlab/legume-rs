//! Packing a cell's WHOLE observed support, with no context window.
//!
//! The top-K packing ([`super::top_k`]) exists because the per-gene encoder
//! gathers an `H`-wide row per slot: an `[N, K, H]` block whose size forces a
//! cap on `K`, and the cap in turn forces a choice about which genes a cell is
//! read through. Genes outside the window are invisible to the encoder however
//! much they were expressed.
//!
//! A coarse read has no such block. Every observed gene is added into its
//! group, giving an `[N, C]` profile whose width is the group count and not the
//! support size, so the slot dimension never meets `H` and there is nothing to
//! rank. This module is what feeds it: the support, whole, unweighted.
//!
//! No shortlist weights appear here on purpose. Weighting existed to decide
//! what survived the cap; with nothing discarded there is nothing to decide,
//! and applying a weight would silently rescale counts the decoder still
//! scores in raw units.

use super::types::IndexedSample;
use nalgebra_sparse::CscMatrix;

/// Every stored nonzero of each column, as one [`IndexedSample`] per cell.
///
/// Columns are cells and rows are features, matching
/// [`super::top_k::csc_columns_to_indexed_samples`]. Indices come out in the
/// matrix's own row order, ascending, and values are the stored counts
/// untouched.
///
/// `gene_remap = Some(new_to_train)` renumbers each stored row from a held-out
/// gene axis onto the training axis and drops rows that do not map, for
/// scoring a cohort whose gene set differs. `None` when the matrix is already
/// on the training axis.
pub fn csc_columns_to_full_samples(
    x_dn: &CscMatrix<f32>,
    gene_remap: Option<&[Option<usize>]>,
) -> Vec<IndexedSample> {
    debug_assert!(gene_remap.is_none_or(|rm| rm.len() == x_dn.nrows()));
    (0..x_dn.ncols())
        .map(|j| {
            let col = x_dn.col(j);
            let pairs = col.row_indices().iter().zip(col.values().iter());
            let (indices, values) = match gene_remap {
                // A row the training axis does not have is dropped, not
                // folded into a neighbour: the profile is keyed by group and a
                // misattributed row would land in the wrong one.
                Some(rm) => pairs
                    .filter_map(|(&r, &v)| rm[r].map(|rt| (rt as u32, v)))
                    .unzip(),
                None => pairs.map(|(&r, &v)| (r as u32, v)).unzip(),
            };
            IndexedSample { indices, values }
        })
        .collect()
}

/// The widest support in a batch of samples, i.e. the `K` a rectangular pack
/// needs. Returned rather than assumed so a caller pads to the batch it has
/// instead of to a configured constant.
pub fn widest_support(samples: &[IndexedSample]) -> usize {
    samples.iter().map(|s| s.indices.len()).max().unwrap_or(0)
}

#[cfg(test)]
#[path = "full_support_tests.rs"]
mod full_support_tests;
