//! Parent maps of the pseudobulk levels from per-level cell memberships.

use chickpea::p2g::pb_levels::parent_maps;

#[test]
fn parent_maps_follow_strict_nesting() {
    // 6 cells; level 0 (finest) has 4 pbs, level 1 has 2, level 2 has 1.
    let cell_to_pb = vec![
        vec![0, 0, 1, 2, 3, 3],
        vec![0, 0, 0, 1, 1, 1],
        vec![0, 0, 0, 0, 0, 0],
    ];
    let parent = parent_maps(&cell_to_pb, &[4, 2, 1]).unwrap();
    assert_eq!(parent, vec![vec![0, 0, 1, 1], vec![0, 0]]);
}

#[test]
fn a_straddling_pb_is_an_error() {
    // pb 0 at level 0 holds cells {0,1,2}; they sit in two coarse pbs.
    let cell_to_pb = vec![vec![0, 0, 0, 1], vec![0, 1, 1, 1]];
    assert!(parent_maps(&cell_to_pb, &[2, 2]).is_err());
}

#[test]
fn an_empty_pb_is_an_error() {
    let cell_to_pb = vec![vec![0, 0, 0], vec![0, 0, 0]];
    assert!(parent_maps(&cell_to_pb, &[2, 1]).is_err());
}

#[test]
fn parent_maps_reject_ragged_levels() {
    let cell_to_pb = vec![vec![0, 1, 1], vec![0, 0]];
    assert!(parent_maps(&cell_to_pb, &[2, 1]).is_err());
}

#[test]
fn single_level_has_no_parents() {
    let parent = parent_maps(&[vec![0, 1, 1]], &[2]).unwrap();
    assert!(parent.is_empty());
}
