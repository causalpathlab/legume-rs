//! Tests for root resolution and its precedence order.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

// resolve_root_hint signature: (root_node, root_cell, cell_names, labels, k, type_root,
// gem_root) -> Result<Option<usize>>. Priority: node > cell > type > gem > None.

#[test]
fn resolve_root_node_override() {
    // Explicit --root-node beats type/gem.
    let (nm, lab) = (names(&["a", "b", "c"]), vec![0usize, 1, 1]);
    assert_eq!(
        resolve_root_hint(Some(2), None, &nm, &lab, 3, Some(0), Some(0)).unwrap(),
        Some(2)
    );
    assert!(resolve_root_hint(Some(5), None, &nm, &lab, 3, None, None).is_err());
    // out of range
}

#[test]
fn resolve_root_cell_maps_to_its_node() {
    let (nm, lab) = (names(&["a", "b", "c"]), vec![0usize, 1, 1]);
    assert_eq!(
        resolve_root_hint(None, Some("b"), &nm, &lab, 3, None, None).unwrap(),
        Some(1)
    );
    assert!(resolve_root_hint(None, Some("zzz"), &nm, &lab, 3, None, None).is_err());
}

#[test]
fn resolve_root_type_beats_gem() {
    // --root-type (type_root) is preferred over gem_root.
    let (nm, lab) = (names(&["a", "b"]), vec![0usize, 1]);
    assert_eq!(
        resolve_root_hint(None, None, &nm, &lab, 2, Some(1), Some(0)).unwrap(),
        Some(1)
    );
}

#[test]
fn resolve_root_falls_back_to_gem_then_none() {
    // With no override/type: gem_root, else None (the branching picks the roots).
    let (nm, lab) = (names(&["a", "b"]), vec![0usize, 1]);
    assert_eq!(
        resolve_root_hint(None, None, &nm, &lab, 2, None, Some(0)).unwrap(),
        Some(0)
    );
    assert_eq!(
        resolve_root_hint(None, None, &nm, &lab, 2, None, None).unwrap(),
        None
    );
}

#[test]
fn gem_dag_n_terminals_parses_field_and_zero_vetoes() {
    let dir = std::env::temp_dir();
    let prefix = dir
        .join(format!("faba_qc_{}", std::process::id()))
        .to_string_lossy()
        .into_owned();
    let write = |body: &str| std::fs::write(format!("{prefix}.lineage_qc.json"), body).unwrap();

    // Well-formed descriptive JSON (no `flag` field): the terminal count is read out.
    write("{\n  \"n_roots\": 1,\n  \"n_terminals\": 42,\n  \"top_source_reach\": 0.31\n}\n");
    assert_eq!(gem_dag_n_terminals(&prefix), Some(42));

    // A structureless DAG → Some(0), which is the only case that vetoes --root-from-gem.
    write("{\n  \"n_roots\": 0,\n  \"n_terminals\": 0,\n  \"top_source_reach\": 0.0\n}\n");
    assert_eq!(gem_dag_n_terminals(&prefix), Some(0));

    // Missing file → None (no signal → do NOT veto the gem root).
    std::fs::remove_file(format!("{prefix}.lineage_qc.json")).unwrap();
    assert_eq!(gem_dag_n_terminals(&prefix), None);
}
