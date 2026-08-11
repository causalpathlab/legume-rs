//! Tests for the three-state rule: what counts as an edge, what may be
//! certified absent, and what must stay unknown.

use super::*;
use crate::eqtl::select::{Pair, PairObs};

fn celltypes(n: usize) -> Vec<Box<str>> {
    (0..n).map(|k| Box::from(format!("C{k}"))).collect()
}

/// One pair from `(celltype, beta, se)` triples.
fn selection(obs: &[(u32, f32, f32)]) -> Selection {
    let pair = Pair {
        gene: 0,
        variant: 0,
        obs: obs
            .iter()
            .map(|&(celltype, beta, se)| PairObs { celltype, beta, se })
            .collect(),
    };
    Selection {
        n_rows: pair.obs.len(),
        pairs: vec![pair],
        n_selected_variants: 1,
        n_pairs_dropped: 0,
    }
}

fn state_at(table: &EvidenceTable, context: u32) -> State {
    table
        .rows
        .iter()
        .find(|r| r.context == context)
        .expect("context present")
        .state
}

#[test]
fn an_edge_needs_the_detection_threshold() {
    let sel = selection(&[(0, 0.40, 0.1), (1, 0.39, 0.1), (2, 0.30, 0.1)]);
    let table = classify_states(&sel, &celltypes(3), 4.0).unwrap();

    assert_eq!(state_at(&table, 0), State::Edge); // z = 4.0, at the threshold
    assert_ne!(state_at(&table, 1), State::Edge); // z = 3.9
}

/// The certificate compares the minimum detectable effect against the
/// RUNNER-UP effect. Using the maximum would certify absence in a context
/// that only the inflated maximum makes look under-powered.
#[test]
fn the_reference_effect_is_the_runner_up_not_the_maximum() {
    // Cell type 0 is the (inflated) maximum, cell type 1 the runner-up,
    // cell type 2 the quiet context under judgement.
    let sel = selection(&[(0, 1.0, 0.1), (1, 0.4, 0.1), (2, 0.0, 0.3)]);
    let table = classify_states(&sel, &celltypes(3), 4.0).unwrap();

    assert!((table.pairs[0].b_ref - 0.4).abs() < 1e-6);

    // mde = 2.8 * 0.3 = 0.84 sits below the maximum effect and above the
    // runner-up, so the two rules disagree on exactly this cell.
    let quiet = table.rows.iter().find(|r| r.context == 2).unwrap();
    let mde = MDE_MULT * quiet.se;
    let b_max = table.rows.iter().map(|r| r.beta.abs()).fold(0.0, f32::max);
    assert!(mde < b_max, "the maximum would have certified it");
    assert!(mde > table.pairs[0].b_ref, "the runner-up must not");
    assert_eq!(state_at(&table, 2), State::Unknown);
    assert_eq!(table.n_unknown, 1);
}

#[test]
fn a_powered_quiet_context_is_certified_absent() {
    // Same shape, but the quiet context is precise enough to have seen it.
    let sel = selection(&[(0, 1.0, 0.1), (1, 0.9, 0.1), (2, 0.0, 0.1)]);
    let table = classify_states(&sel, &celltypes(3), 4.0).unwrap();

    assert!((table.pairs[0].b_ref - 0.9).abs() < 1e-6);
    assert_eq!(state_at(&table, 2), State::CertifiedAbsent);
}

#[test]
fn an_unpowered_context_stays_unknown() {
    // The quiet context's mde (2.8 * 1.0) dwarfs the reference effect.
    let sel = selection(&[(0, 1.0, 0.1), (1, 0.9, 0.1), (2, 0.0, 1.0)]);
    let table = classify_states(&sel, &celltypes(3), 4.0).unwrap();
    assert_eq!(state_at(&table, 2), State::Unknown);
}

#[test]
fn the_ubiquitous_row_is_appended_with_the_pair_reference() {
    let sel = selection(&[(0, 0.5, 0.1), (1, 0.5, 0.1), (2, 0.5, 0.1)]);
    let table = classify_states(&sel, &celltypes(3), 4.0).unwrap();

    assert_eq!(table.contexts.len(), 4);
    assert_eq!(table.contexts[3].as_ref(), UBIQUITOUS);
    assert_eq!(table.ubiquitous, 3);
    assert_eq!(table.contexts.len() - 1, 3);
    assert_eq!(table.pairs[0].len, 4);
    // A shared effect survives the leave-one-out meta as an edge.
    assert_eq!(state_at(&table, 3), State::Edge);
}

#[test]
fn a_pair_seen_in_one_context_gets_no_ubiquitous_row() {
    let sel = selection(&[(0, 0.5, 0.1)]);
    let table = classify_states(&sel, &celltypes(1), 4.0).unwrap();
    assert_eq!(table.pairs[0].len, 1);
    assert!(table.rows.iter().all(|r| r.context != table.ubiquitous));
}

#[test]
fn a_celltype_named_ubiquitous_is_rejected() {
    let sel = selection(&[(0, 0.5, 0.1), (1, 0.5, 0.1)]);
    let names: Vec<Box<str>> = vec![Box::from("CT1"), Box::from(UBIQUITOUS)];
    let err = classify_states(&sel, &names, 4.0).unwrap_err();
    assert!(err.to_string().contains("collides"));
}
