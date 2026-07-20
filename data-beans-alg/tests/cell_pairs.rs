//! Tests for the shared cell-pair structure.
//!
//! Pinto's spatial wrapper (`SrtCellPairs`) layers coordinates on top of this;
//! everything exercised here is coordinate-free.

use data_beans::sparse_io::*;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::cell_pairs::{collapse_pairs, CellPairs};
use matrix_util::knn_graph::KnnGraph;
use matrix_util::parquet::{peek_parquet_field_names, Column, ParquetReader};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use ndarray::Array2;
use std::sync::{Arc, Mutex};

fn str_vec(n: usize, prefix: &str) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{prefix}{i}").into_boxed_str())
        .collect()
}

/// A `SparseIoVec` of `n_genes × n_cells`. Only the shape and the column
/// names matter here — no test reads the counts back.
fn make_data(n_genes: usize, n_cells: usize) -> anyhow::Result<SparseIoVec> {
    let raw = Array2::<f32>::zeros((n_genes, n_cells));
    let mut sp = create_sparse_from_ndarray(&raw, None, None)?;
    sp.register_row_names_vec(&str_vec(n_genes, "gene"));
    sp.register_column_names_vec(&str_vec(n_cells, "cell"));
    sp.preload_columns()?;

    let mut data = SparseIoVec::new();
    data.push(Arc::from(sp), Some("batch0".into()))?;
    Ok(data)
}

/// Undirected graph from an edge list, distances `1, 2, 3, ...` so each edge
/// is distinguishable in the written table.
fn make_graph(n_nodes: usize, edges: Vec<(usize, usize)>) -> KnnGraph {
    let distances = (0..edges.len()).map(|i| (i + 1) as f32).collect();

    let mut coo = CooMatrix::new(n_nodes, n_nodes);
    for &(i, j) in &edges {
        coo.push(i, j, 1.0f32);
        coo.push(j, i, 1.0f32);
    }

    KnnGraph {
        adjacency: CscMatrix::from(&coo),
        edges,
        distances,
        n_nodes,
    }
}

#[test]
fn pairs_alias_graph_edges() -> anyhow::Result<()> {
    let data = make_data(7, 6)?;
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let graph = make_graph(6, edges.clone());
    let pairs = CellPairs::from_graph(&data, &graph);

    assert_eq!(pairs.pairs(), edges.as_slice());
    assert_eq!(pairs.num_pairs(), 5);
    assert_eq!(pairs.num_features(), 7);
    assert_eq!(pairs.weights(), Some(&[1.0, 2.0, 3.0, 4.0, 5.0][..]));

    // `pairs()` must borrow the graph's own edges, not a copy of them.
    assert!(std::ptr::eq(pairs.pairs(), graph.edges.as_slice()));
    Ok(())
}

#[test]
fn visit_pairs_by_block_covers_every_pair_once() -> anyhow::Result<()> {
    let data = make_data(4, 6)?;
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let graph = make_graph(6, edges);
    let pairs = CellPairs::from_graph(&data, &graph);

    let mut visits = vec![0usize; pairs.num_pairs()];
    pairs.visit_pairs_by_block(
        &|(lb, ub), _data: &CellPairs, _shared: &(), out: Arc<Mutex<&mut Vec<usize>>>| {
            let mut out = out.lock().expect("lock visits");
            for v in out.iter_mut().take(ub).skip(lb) {
                *v += 1;
            }
            Ok(())
        },
        &(),
        &mut visits,
        Some(2),
    )?;

    assert_eq!(visits, vec![1; 5]);
    Ok(())
}

#[test]
fn to_parquet_writes_extra_columns_between_names_and_distance() -> anyhow::Result<()> {
    let data = make_data(3, 4)?;
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let graph = make_graph(4, edges);
    let pairs = CellPairs::from_graph(&data, &graph);

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("pairs.parquet");
    let path = path.to_str().expect("utf8 path");

    let left_x = vec![10.0f32, 11.0, 12.0];
    let right_x = vec![20.0f32, 21.0, 22.0];
    pairs.to_parquet(
        path,
        &[
            ("left_x".into(), Column::F32(&left_x)),
            ("right_x".into(), Column::F32(&right_x)),
        ],
    )?;

    // Caller columns land between `right_cell` and `distance`.
    let fields = peek_parquet_field_names(path)?;
    let names: Vec<&str> = fields.iter().map(|f| f.as_ref()).collect();
    assert_eq!(
        &names[1..],
        &["left_cell", "right_cell", "left_x", "right_x", "distance"]
    );

    // Numeric columns read back in that same order, distance last.
    let read = ParquetReader::new(path, Some(0), None, None)?;
    assert_eq!(
        read.column_names
            .iter()
            .map(|c| c.as_ref())
            .collect::<Vec<_>>(),
        vec!["left_x", "right_x", "distance"]
    );

    let ncol = read.column_names.len();
    let row = |i: usize| &read.row_major_data[i * ncol..(i + 1) * ncol];
    for i in 0..3 {
        assert_eq!(row(i)[0] as f32, left_x[i]);
        assert_eq!(row(i)[1] as f32, right_x[i]);
        assert_eq!(row(i)[2] as f32, (i + 1) as f32); // distance
    }
    Ok(())
}

#[test]
fn to_parquet_without_extra_columns() -> anyhow::Result<()> {
    let data = make_data(3, 4)?;
    let edges = vec![(0, 1), (2, 3)];
    let graph = make_graph(4, edges);
    let pairs = CellPairs::from_graph(&data, &graph);

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("bare.parquet");
    let path = path.to_str().expect("utf8 path");
    pairs.to_parquet(path, &[])?;

    let fields = peek_parquet_field_names(path)?;
    let names: Vec<&str> = fields.iter().map(|f| f.as_ref()).collect();
    assert_eq!(&names[1..], &["left_cell", "right_cell", "distance"]);
    Ok(())
}

#[test]
fn to_parquet_rejects_wrong_length_column() -> anyhow::Result<()> {
    let data = make_data(3, 4)?;
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let graph = make_graph(4, edges);
    let pairs = CellPairs::from_graph(&data, &graph);

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("bad.parquet");
    let path = path.to_str().expect("utf8 path");

    let short = vec![1.0f32, 2.0];
    let err = pairs
        .to_parquet(path, &[("left_x".into(), Column::F32(&short))])
        .expect_err("mismatched column length must be rejected");
    assert!(err.to_string().contains("left_x"));
    Ok(())
}

// ── collapse_pairs ────────────────────────────────────────────────────────
// The pair-level analogue of column collapsing. Pinto's
// `cell_labels_to_pair_samples` and `build_super_edges` are both views on this.

#[test]
fn collapse_pairs_merges_by_unordered_label_pair() {
    // labels: 0,1 -> group 0 ; 2,3 -> group 1
    let pairs = vec![(0, 1), (2, 3), (0, 3), (1, 2), (1, 0)];
    let labels = vec![0, 0, 1, 1];
    let (coarse, fine_to_coarse) = collapse_pairs(&pairs, &labels);

    // (0,1) and (1,0) are both within group 0 -> the same self-loop (0,0).
    assert_eq!(fine_to_coarse[0], fine_to_coarse[4]);
    // (0,3) and (1,2) both cross 0<->1 -> same coarse pair, regardless of order.
    assert_eq!(fine_to_coarse[2], fine_to_coarse[3]);
    // (2,3) is within group 1 -> distinct from both.
    assert_ne!(fine_to_coarse[0], fine_to_coarse[1]);
    assert_ne!(fine_to_coarse[1], fine_to_coarse[2]);

    assert_eq!(coarse.len(), 3);
    assert_eq!(coarse.pairs, vec![(0, 0), (1, 1), (0, 1)]); // first-seen order
    assert!(coarse.weights.is_none());
}

#[test]
fn collapse_pairs_is_deterministic_and_composes() {
    let pairs = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
    let labels = vec![0, 0, 1, 1];

    let a = collapse_pairs(&pairs, &labels);
    let b = collapse_pairs(&pairs, &labels);
    assert_eq!(a.0.pairs, b.0.pairs, "ids must not depend on hasher order");
    assert_eq!(a.1, b.1);

    // A collapsed set collapses again — this is what multi-level needs.
    let (level1, _) = a;
    let regroup = vec![0, 0]; // fold both groups into one
    let (level2, _) = level1.collapse_by(&regroup);
    assert_eq!(level2.pairs, vec![(0, 0)]);
}

#[test]
fn collapse_pairs_identity_labels_is_a_no_op() {
    let pairs = vec![(0, 1), (1, 2), (2, 3)];
    let identity: Vec<usize> = (0..4).collect();
    let (coarse, fine_to_coarse) = collapse_pairs(&pairs, &identity);
    assert_eq!(coarse.pairs, pairs);
    assert_eq!(fine_to_coarse, vec![0, 1, 2]);
}

#[test]
fn cell_pairs_over_a_collapsed_pair_set() -> anyhow::Result<()> {
    // The point of borrowing the pair list rather than owning a graph: the
    // output of `collapse_pairs` can back a `CellPairs` directly, with no
    // KnnGraph in sight and no copy of the edge list.
    let data = make_data(3, 4)?;
    let fine = vec![(0, 1), (1, 2), (2, 3)];
    let labels = vec![0, 0, 1, 1];
    let (coarse, _) = collapse_pairs(&fine, &labels);

    let pairs = CellPairs::new(&data, &coarse.pairs);
    assert!(std::ptr::eq(pairs.pairs(), coarse.pairs.as_slice()));
    assert_eq!(pairs.num_pairs(), coarse.len());
    assert!(pairs.weights().is_none());

    // With no weights there is no `distance` column to write.
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("coarse.parquet");
    let path = path.to_str().expect("utf8 path");
    pairs.to_parquet(path, &[])?;

    let fields = peek_parquet_field_names(path)?;
    let names: Vec<&str> = fields.iter().map(|f| f.as_ref()).collect();
    assert_eq!(&names[1..], &["left_cell", "right_cell"]);
    Ok(())
}
