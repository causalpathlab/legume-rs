//! What "no context window" has to mean, pinned before the loader changes.
//!
//! The anchor is the last test: full-support packing must equal top-K packing
//! whenever K is at least the widest support. If it does, removing the window
//! is provably a relaxation of the existing loader rather than a new one, and
//! any measured difference comes from genes the window used to drop.

use super::{csc_columns_to_full_samples, widest_support};
use crate::data::indexed::top_k::csc_columns_to_indexed_samples;
use nalgebra_sparse::{CooMatrix, CscMatrix};

const D: usize = 8;
const N: usize = 3;

/// Three cells over eight genes, with deliberately uneven supports:
/// cell 0 has four genes, cell 1 has one, cell 2 has none.
fn planted() -> CscMatrix<f32> {
    let mut coo = CooMatrix::<f32>::new(D, N);
    for (g, v) in [(0usize, 3.0f32), (2, 1.0), (5, 7.0), (7, 2.0)] {
        coo.push(g, 0, v);
    }
    coo.push(4, 1, 5.0);
    CscMatrix::from(&coo)
}

#[test]
fn every_stored_nonzero_survives_in_its_own_cell() {
    let x = planted();
    let s = csc_columns_to_full_samples(&x, None);
    assert_eq!(s.len(), N);
    assert_eq!(s[0].indices, vec![0, 2, 5, 7], "ascending row order");
    assert_eq!(s[0].values, vec![3.0, 1.0, 7.0, 2.0], "raw stored counts");
    assert_eq!(s[1].indices, vec![4]);
    assert_eq!(s[1].values, vec![5.0]);
    assert!(
        s[2].indices.is_empty() && s[2].values.is_empty(),
        "a cell with no observation packs to an empty support, not to padding"
    );
}

/// Nothing here may rescale a count: the decoder scores raw units, and the
/// weighting that top-K used to rank candidates has no job once nothing is
/// discarded.
#[test]
fn values_are_not_reweighted() {
    let x = planted();
    let s = csc_columns_to_full_samples(&x, None);
    let total: f32 = s.iter().flat_map(|c| c.values.iter()).sum();
    assert!((total - 18.0).abs() < 1e-6, "3+1+7+2+5 = 18, got {total}");
}

/// Scoring a cohort on a different gene axis: mapped rows are renumbered onto
/// the training axis, unmapped rows are dropped rather than misattributed.
#[test]
fn a_gene_remap_renumbers_and_drops() {
    let x = planted();
    // Held-out axis -> training axis. Genes 2 and 7 do not exist in training.
    let remap: Vec<Option<usize>> =
        vec![Some(10), None, None, None, Some(11), Some(12), None, None];
    let s = csc_columns_to_full_samples(&x, Some(&remap));
    assert_eq!(s[0].indices, vec![10, 12], "gene 0 -> 10, gene 5 -> 12");
    assert_eq!(s[0].values, vec![3.0, 7.0], "their counts follow them");
    assert_eq!(s[1].indices, vec![11]);
    assert!(s[2].indices.is_empty());
}

/// A rectangular pack needs the batch's widest support, and a batch of empty
/// cells must not report a width that would index nothing.
#[test]
fn the_widest_support_is_the_pack_width() {
    let x = planted();
    let s = csc_columns_to_full_samples(&x, None);
    assert_eq!(widest_support(&s), 4, "cell 0 has the widest support");
    assert_eq!(widest_support(&[]), 0, "an empty batch has no width");
    assert_eq!(
        widest_support(&s[2..]),
        0,
        "a batch of unobserved cells has no width"
    );
}

/// THE anchor. With `context_size` at or above the widest support, top-K keeps
/// everything, so the two packings must agree cell by cell. This is what makes
/// removing the window a relaxation of the existing loader rather than a
/// rewrite of it.
#[test]
fn full_support_equals_top_k_once_k_covers_the_support() {
    let x = planted();
    let full = csc_columns_to_full_samples(&x, None);
    let weights = vec![1.0f32; D];

    for k in [4usize, 8, 64] {
        let capped = csc_columns_to_indexed_samples(&x, &weights, k, None);
        assert_eq!(capped.len(), full.len());
        for (n, (a, b)) in capped.iter().zip(&full).enumerate() {
            let mut got: Vec<(u32, f32)> = a
                .indices
                .iter()
                .copied()
                .zip(a.values.iter().copied())
                .collect();
            let mut want: Vec<(u32, f32)> = b
                .indices
                .iter()
                .copied()
                .zip(b.values.iter().copied())
                .collect();
            got.sort_by_key(|e| e.0);
            want.sort_by_key(|e| e.0);
            assert_eq!(got, want, "cell {n} differs at context_size {k}");
        }
    }
}

/// And the converse, so the test above cannot pass for a trivial reason: below
/// the widest support the window really does drop genes, which is the loss the
/// coarse read exists to avoid.
#[test]
fn a_window_narrower_than_the_support_drops_genes() {
    let x = planted();
    let full = csc_columns_to_full_samples(&x, None);
    let weights = vec![1.0f32; D];
    let capped = csc_columns_to_indexed_samples(&x, &weights, 2, None);
    assert_eq!(capped[0].indices.len(), 2);
    assert_eq!(full[0].indices.len(), 4);
}
