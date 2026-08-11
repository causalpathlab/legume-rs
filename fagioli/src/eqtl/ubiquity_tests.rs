//! Tests for the leave-one-out ubiquitous context and the ubiquity index.

use super::*;
use crate::eqtl::evidence::classify_states;
use crate::eqtl::select::{Pair, PairObs};

fn obs(rows: &[(u32, f32, f32)]) -> Vec<PairObs> {
    rows.iter()
        .map(|&(celltype, beta, se)| PairObs { celltype, beta, se })
        .collect()
}

fn selection(rows: &[(u32, f32, f32)]) -> crate::eqtl::select::Selection {
    let pair = Pair {
        gene: 0,
        variant: 0,
        obs: obs(rows),
    };
    crate::eqtl::select::Selection {
        n_rows: pair.obs.len(),
        pairs: vec![pair],
        n_selected_variants: 1,
        n_pairs_dropped: 0,
    }
}

fn celltypes(n: usize) -> Vec<Box<str>> {
    (0..n).map(|k| Box::from(format!("C{k}"))).collect()
}

/// The point of leaving one out: an effect carried by a single cell type
/// leaves NO ubiquitous edge, while an effect shared across cell types
/// survives. A plain meta-analysis would leave a diluted-but-nonzero
/// effect in both cases, turning edge/non-edge into strong/weak.
#[test]
fn leave_one_out_nulls_a_single_celltype_effect_and_keeps_a_shared_one() {
    let specific = [
        (0, 1.0, 0.1),
        (1, 0.0, 0.1),
        (2, 0.0, 0.1),
        (3, 0.0, 0.1),
        (4, 0.0, 0.1),
    ];
    let (beta, se) = loo_meta(&obs(&specific)).unwrap();
    assert!(beta.abs() < 1e-6, "the lone effect must not leak through");
    assert!((beta / se).abs() < 4.0);

    let shared = [
        (0, 0.5, 0.1),
        (1, 0.5, 0.1),
        (2, 0.5, 0.1),
        (3, 0.5, 0.1),
        (4, 0.5, 0.1),
    ];
    let (beta, se) = loo_meta(&obs(&shared)).unwrap();
    assert!((beta - 0.5).abs() < 1e-5);
    assert!((beta / se).abs() > 4.0);

    // The same contrast, read off the classified states.
    let ubi_state = |rows: &[(u32, f32, f32)]| {
        let t = classify_states(&selection(rows), &celltypes(5), 4.0).unwrap();
        t.rows
            .iter()
            .find(|r| r.context == t.ubiquitous)
            .unwrap()
            .state
    };
    assert_ne!(ubi_state(&specific), State::Edge);
    assert_eq!(ubi_state(&shared), State::Edge);
}

/// Precision weighting, not a plain average: the imprecise cell type barely
/// moves the combination.
#[test]
fn leave_one_out_combines_by_precision() {
    let rows = obs(&[(0, 2.0, 0.1), (1, 1.0, 0.1), (2, 0.0, 1.0)]);
    let (beta, _) = loo_meta(&rows).unwrap();
    // Weights 100 and 1: (100*1.0 + 1*0.0) / 101.
    assert!((beta - 100.0 / 101.0).abs() < 1e-4);
}

#[test]
fn leave_one_out_needs_two_observations() {
    assert!(loo_meta(&obs(&[(0, 0.5, 0.1)])).is_none());
    assert!(loo_meta(&[]).is_none());
}

/// Only powered contexts enter the denominator, so an unpowered cell type
/// cannot make a pair look narrow.
#[test]
fn the_ubiquity_index_counts_only_powered_contexts() {
    // Two edges, one certified absence, one unpowered context.
    let rows = vec![
        (0, 1.0, 0.1),  // edge
        (1, 0.9, 0.1),  // edge
        (2, 0.0, 0.1),  // certified absent (mde 0.28 < b_ref 0.9)
        (3, 0.0, 10.0), // unknown: mde 28 dwarfs the reference
    ];
    let table = classify_states(&selection(&rows), &celltypes(4), 4.0).unwrap();
    let index = ubiquity_index(&table);

    assert_eq!(index.len(), 1);
    let row = index[0];
    assert_eq!(row.n_powered, 3);
    assert_eq!(row.n_edge, 2);
    assert_eq!(row.n_unknown, 1);
    assert!((row.ubiquity.unwrap() - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn a_pair_with_no_powered_context_has_no_ubiquity() {
    // Everything is imprecise, so nothing is an edge and nothing is
    // certified; the index must abstain rather than report zero breadth.
    let rows = vec![(0, 0.1, 10.0), (1, 0.1, 10.0), (2, 0.1, 10.0)];
    let table = classify_states(&selection(&rows), &celltypes(3), 4.0).unwrap();
    let index = ubiquity_index(&table);
    assert_eq!(index[0].n_powered, 0);
    assert!(index[0].ubiquity.is_none());
}
