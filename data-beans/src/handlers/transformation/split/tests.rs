//! Every cell must land on exactly one side of every fold. A split that quietly
//! drops or duplicates groups produces numbers that look fine and mean nothing,
//! so the partition is checked for coverage and disjointness directly.

use super::*;
use std::collections::HashSet;

fn covers(n: usize, train: &[usize], test: &[usize]) {
    let t: HashSet<usize> = train.iter().copied().collect();
    let e: HashSet<usize> = test.iter().copied().collect();
    assert_eq!(t.len(), train.len(), "train repeats a group");
    assert_eq!(e.len(), test.len(), "test repeats a group");
    assert!(t.is_disjoint(&e), "a group is in both halves");
    assert_eq!(t.len() + e.len(), n, "the halves do not cover the groups");
}

#[test]
fn a_fraction_split_covers_every_group_exactly_once() {
    let folds = partition_groups(100, Some(0.2), None, 42).expect("split");
    assert_eq!(folds.len(), 1);
    let (train, test) = &folds[0];
    assert_eq!(test.len(), 20);
    covers(100, train, test);
}

#[test]
fn k_folds_each_cover_everything_and_the_test_halves_tile_the_input() {
    let n = 37;
    let k = 5;
    let folds = partition_groups(n, None, Some(k), 7).expect("split");
    assert_eq!(folds.len(), k);

    let mut seen: Vec<usize> = Vec::new();
    for (train, test) in &folds {
        covers(n, train, test);
        seen.extend(test);
    }
    // Every group is tested exactly once across the K folds — the property that
    // makes a per-fold score an out-of-sample score for the whole input.
    seen.sort_unstable();
    assert_eq!(seen, (0..n).collect::<Vec<_>>());
}

#[test]
fn neither_half_is_ever_empty() {
    // A fraction that rounds to nothing, and one that rounds to everything.
    for (n, frac) in [(10usize, 0.01f64), (10, 0.99), (3, 0.001), (3, 0.999)] {
        let folds = partition_groups(n, Some(frac), None, 1).expect("split");
        let (train, test) = &folds[0];
        assert!(!train.is_empty() && !test.is_empty(), "n={n} frac={frac}");
        covers(n, train, test);
    }
}

#[test]
fn the_split_is_reproducible_from_the_seed_and_moves_with_it() {
    let a = partition_groups(50, Some(0.3), None, 11).expect("split");
    let b = partition_groups(50, Some(0.3), None, 11).expect("split");
    let c = partition_groups(50, Some(0.3), None, 12).expect("split");
    assert_eq!(a[0].1, b[0].1);
    assert_ne!(
        a[0].1, c[0].1,
        "a different seed must give a different half"
    );
}

#[test]
fn impossible_requests_are_refused() {
    assert!(partition_groups(10, Some(0.0), None, 1).is_err(), "frac 0");
    assert!(partition_groups(10, Some(1.0), None, 1).is_err(), "frac 1");
    assert!(
        partition_groups(10, Some(-0.5), None, 1).is_err(),
        "negative"
    );
    assert!(partition_groups(10, None, Some(1), 1).is_err(), "one fold");
    assert!(partition_groups(3, None, Some(9), 1).is_err(), "k > groups");
    assert!(partition_groups(10, None, None, 1).is_err(), "no mode");
}

/// The flag says TABLE and the help says "two-column table"; reading only
/// parquet meant the TSV a user would naturally write failed with a parquet
/// decode error.
#[test]
fn a_group_table_can_be_delimited_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, body) in [
        ("g.tsv", "c1\tdonorA\nc2\tdonorB\nc3\tdonorA\n"),
        ("g.csv", "c1,donorA\nc2,donorB\nc3,donorA\n"),
    ] {
        let p = dir.path().join(name);
        std::fs::write(&p, body).expect("write");
        let (names, labels) = read_group_table(p.to_str().expect("utf8")).expect(name);
        assert_eq!(names.len(), 3, "{name}");
        assert_eq!(labels[0].as_ref(), "donorA", "{name}");
        assert_eq!(labels[1].as_ref(), "donorB", "{name}");
    }
}

#[test]
fn a_one_column_group_table_is_refused() {
    // Silently treating a bare cell list as a grouping would put every cell in
    // its own group, i.e. a plain random split wearing the flag's name.
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("bad.tsv");
    std::fs::write(&p, "c1\nc2\n").expect("write");
    assert!(read_group_table(p.to_str().expect("utf8")).is_err());
}

#[test]
fn cells_sharing_a_label_land_in_one_group() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("g.tsv");
    std::fs::write(&p, "c1\tA\nc2\tB\nc3\tA\n").expect("write");
    let cols: Vec<Box<str>> = ["c1", "c2", "c3"].iter().map(|s| Box::from(*s)).collect();
    let groups = column_groups_from_table(p.to_str().expect("utf8"), &cols).expect("groups");
    assert_eq!(groups.len(), 2, "two labels -> two groups");
    assert!(groups.iter().any(|g| g == &vec![0, 2]), "A holds c1 and c3");
}

#[test]
fn a_cell_with_no_label_is_an_error_not_a_singleton() {
    // A partial grouping would split part of the data by group and the rest by
    // cell, and the resulting fold is neither.
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("g.tsv");
    std::fs::write(&p, "c1\tA\n").expect("write");
    let cols: Vec<Box<str>> = ["c1", "c2"].iter().map(|s| Box::from(*s)).collect();
    assert!(column_groups_from_table(p.to_str().expect("utf8"), &cols).is_err());
}
