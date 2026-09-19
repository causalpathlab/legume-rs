//! Tests for root resolution and its precedence order.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

// resolve_root_hint signature: (root_node, root_cell, cell_names, labels, k, type_root)
// -> Result<Option<usize>>. Priority: node > cell > type > None.

#[test]
fn resolve_root_node_override() {
    // Explicit --root-node beats type.
    let (nm, lab) = (names(&["a", "b", "c"]), vec![0usize, 1, 1]);
    assert_eq!(
        resolve_root_hint(Some(2), None, &nm, &lab, 3, Some(0)).unwrap(),
        Some(2)
    );
    assert!(resolve_root_hint(Some(5), None, &nm, &lab, 3, None).is_err());
    // out of range
}

#[test]
fn resolve_root_cell_maps_to_its_node() {
    let (nm, lab) = (names(&["a", "b", "c"]), vec![0usize, 1, 1]);
    assert_eq!(
        resolve_root_hint(None, Some("b"), &nm, &lab, 3, None).unwrap(),
        Some(1)
    );
    assert!(resolve_root_hint(None, Some("zzz"), &nm, &lab, 3, None).is_err());
}

#[test]
fn resolve_root_type_is_used_when_no_node_or_cell_is_given() {
    let (nm, lab) = (names(&["a", "b"]), vec![0usize, 1]);
    assert_eq!(
        resolve_root_hint(None, None, &nm, &lab, 2, Some(1)).unwrap(),
        Some(1)
    );
}

#[test]
fn resolve_root_falls_back_to_none() {
    // With no override at all: None (the branching picks the roots).
    let (nm, lab) = (names(&["a", "b"]), vec![0usize, 1]);
    assert_eq!(
        resolve_root_hint(None, None, &nm, &lab, 2, None).unwrap(),
        None
    );
}
