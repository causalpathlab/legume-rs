//! The propensity parquet schema is shared by three writers (lc, cage,
//! prop): `cell, C0..C{K-1}, cluster, entropy`. Plot's reader keys on the
//! names, and every consumer of `cluster` assumes the same argmax; these
//! tests pin the schema at lc's writer and the tie rule at the shared
//! helper both writers derive their `cluster` column from.

use crate::link_community::outputs::write_propensity_parquet;
use crate::link_community::profiles::dominant_cluster_rows;
use crate::util::common::*;
use matrix_util::traits::MatWithNames;

/// Argmax per row; ties go to the lowest index (matching
/// `CommunityStrata`), and a row with no mass maps to 0.
#[test]
fn dominant_cluster_rows_picks_argmax_with_lowest_index_ties() {
    let mut p = Mat::zeros(4, 3);
    p[(0, 0)] = 0.1;
    p[(0, 2)] = 0.9; // clear argmax
    p[(1, 0)] = 0.5; // exact tie with column 1 → lowest index wins
    p[(1, 1)] = 0.5;
    p[(2, 1)] = 1.0;
    // row 3 stays all-zero → 0

    assert_eq!(dominant_cluster_rows(&p), vec![2.0, 0.0, 1.0, 0.0]);
}

/// The columns written are exactly `C0..C{K-1}, cluster, entropy` (the
/// `cell` axis rides as row names), and `cluster` equals the argmax of the
/// propensity columns on the same row.
#[test]
fn propensity_parquet_carries_the_shared_schema() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_str().unwrap().to_string();

    // 4 cells, 2 communities: cells 0,1 wired in community 0, cells 2,3 in
    // community 1, one bridging edge to give cell 1 mixed propensity.
    let edges = vec![(0usize, 1usize), (2, 3), (1, 2), (0, 1)];
    let fine_labels = vec![0usize, 1, 1, 0];
    let cell_names: Vec<Box<str>> = (0..4).map(|i| format!("c{i}").into()).collect();

    let propensity =
        write_propensity_parquet(&prefix, &edges, &fine_labels, 4, 2, &cell_names).unwrap();

    let MatWithNames { rows, cols, mat } =
        Mat::from_parquet(&format!("{prefix}.propensity.parquet")).unwrap();

    let col_names: Vec<&str> = cols.iter().map(|c| c.as_ref()).collect();
    assert_eq!(
        col_names,
        vec!["C0", "C1", "cluster", "entropy"],
        "one schema for lc, cage and prop"
    );
    assert_eq!(rows.len(), 4, "one row per cell");

    // `cluster` is the argmax of the propensity columns, row by row.
    for i in 0..4 {
        let expect = if propensity[(i, 0)] >= propensity[(i, 1)] {
            0.0
        } else {
            1.0
        };
        assert_eq!(mat[(i, 2)], expect, "cell {i} cluster is the row argmax");
    }
}
