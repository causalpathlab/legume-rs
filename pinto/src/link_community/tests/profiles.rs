use crate::gene_network::graph::test_graph_from_edges;
use crate::link_community::profiles::*;
use crate::util::common::*;

#[test]
fn test_transfer_labels() {
    let f2s = vec![0, 1, 1, 2, 0];
    let super_mem = vec![2, 0, 1];
    let fine_mem = transfer_labels(&f2s, &super_mem);
    assert_eq!(fine_mem, vec![2, 0, 0, 1, 2]);
}

#[test]
fn test_shannon_entropy_rows() {
    let mut p = Mat::zeros(4, 4);
    // row 0: one-hot → H = 0
    p[(0, 0)] = 1.0;
    // row 1: uniform over K=4 → H = ln 4
    for j in 0..4 {
        p[(1, j)] = 0.25;
    }
    // row 2: [0.5, 0.5, 0, 0] → H = ln 2
    p[(2, 0)] = 0.5;
    p[(2, 1)] = 0.5;
    // row 3: zero row → NaN

    let h = shannon_entropy_rows(&p);
    assert!(h[0].abs() < 1e-6, "got {}", h[0]);
    assert!((h[1] - 4.0f32.ln()).abs() < 1e-6, "got {}", h[1]);
    assert!((h[2] - 2.0f32.ln()).abs() < 1e-6, "got {}", h[2]);
    assert!(h[3].is_nan(), "got {}", h[3]);
}

#[test]
fn test_compute_node_membership() {
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3)];
    let membership = vec![0, 0, 1, 1];
    let nm = compute_node_membership(&edges, &membership, 4, 2);

    assert_eq!(nm.nrows(), 4);
    assert_eq!(nm.ncols(), 2);

    // Cell 0: edges 0,1 → communities 0,0 → [1.0, 0.0]
    assert!((nm[(0, 0)] - 1.0).abs() < 1e-6);
    assert!((nm[(0, 1)] - 0.0).abs() < 1e-6);

    // Cell 2: edges 1,2,3 → communities 0,1,1 → [1/3, 2/3]
    assert!((nm[(2, 0)] - 1.0 / 3.0).abs() < 1e-6);
    assert!((nm[(2, 1)] - 2.0 / 3.0).abs() < 1e-6);
}

use test_graph_from_edges as make_graph;

#[test]
fn test_module_pair_basis_counts_and_degrees() {
    // 4 genes, 2 modules: {0,1} -> 0, {2,3} -> 1.
    // Edges: (0,1) internal to mod 0, (2,3) internal to mod 1, (1,2) cross.
    let graph = make_graph(&[(0, 1), (2, 3), (1, 2)], 4);
    let mods = vec![Some(0), Some(0), Some(1), Some(1)];
    let basis = ModulePairBasis::build(&graph, mods);

    assert_eq!(basis.n_modules, 2);
    assert_eq!(basis.n_pairs, 3);
    let adj0: Vec<u32> = basis.pair_adj[0].iter().map(|e| e.b).collect();
    assert!(adj0.contains(&0));
    assert!(adj0.contains(&1));
    let adj1: Vec<u32> = basis.pair_adj[1].iter().map(|e| e.b).collect();
    assert!(adj1.contains(&0));
    assert!(adj1.contains(&1));
}

#[test]
fn test_module_pair_basis_drops_unmodule_genes() {
    // Gene 3 has no module (e.g., k-core-trimmed). Edges touching it are ignored.
    let graph = make_graph(&[(0, 1), (0, 3), (1, 3)], 4);
    let mods = vec![Some(0), Some(0), None, None];
    let basis = ModulePairBasis::build(&graph, mods);
    // Only (0,0) from (0,1) survives.
    assert_eq!(basis.n_pairs, 1);
}

#[test]
fn test_coarsen_module_expression() {
    // 3 cells × 2 modules, cell 0+1 -> super 0, cell 2 -> super 1.
    let mut expr = Mat::zeros(2, 3);
    expr[(0, 0)] = 1.0;
    expr[(1, 0)] = 2.0;
    expr[(0, 1)] = 3.0;
    expr[(1, 2)] = 4.0;
    let (super_expr, super_totals) = coarsen_module_expression(&expr, &[0, 0, 1], 2);
    assert_eq!(super_expr[(0, 0)], 4.0);
    assert_eq!(super_expr[(1, 0)], 2.0);
    assert_eq!(super_expr[(0, 1)], 0.0);
    assert_eq!(super_expr[(1, 1)], 4.0);
    assert_eq!(super_totals, vec![6.0, 4.0]);
}

#[test]
fn test_module_pair_profiles_residual() {
    // 2 modules, 2 cells. Pair (0,1) has null_ab = deg(0)*deg(1)/(2W)^2.
    // Construct so the residual is a known positive value.
    //
    // One gene-gene edge crossing modules 0 and 1: (0,1) with genes in each.
    // deg(0) = 1, deg(1) = 1, 2W = 2, null_{0,1} = 1*1 / 4 = 0.25.
    //
    // Cell 0: x_{0,0}=2, x_{0,1}=0 → X_0 = 2
    // Cell 1: x_{1,0}=0, x_{1,1}=3 → X_1 = 3
    //
    // For edge (cell 0, cell 1), pair (0,1):
    //   y_obs = x_{0,0}*x_{1,1} + x_{0,1}*x_{1,0} = 6
    //   y_exp = 2 * 3 * 0.25 = 1.5
    //   y = max(0, 6 - 1.5) = 4.5
    let graph = make_graph(&[(0, 1)], 2);
    let mods = vec![Some(0), Some(1)];
    let basis = ModulePairBasis::build(&graph, mods);

    let mut module_expr = Mat::zeros(2, 2);
    module_expr[(0, 0)] = 2.0;
    module_expr[(1, 1)] = 3.0;
    let cell_totals = vec![2.0, 3.0];
    let all_edges = vec![(0usize, 1usize)];
    let edge_indices = vec![0usize];

    let store = build_module_pair_profiles_for_edges(
        &module_expr,
        &cell_totals,
        &all_edges,
        &edge_indices,
        &basis,
    );

    assert_eq!(store.n_edges, 1);
    let (cols, vals) = store.row(0);
    assert_eq!(cols.len(), 1);
    assert!((vals[0] - 4.5).abs() < 1e-5, "got y = {}", vals[0]);
}

use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;

/// Build a tiny on-disk backend from triplets, deduplicated by `(row, col)`.
fn backend_from(
    dir: &tempfile::TempDir,
    triplets: Vec<(u64, u64, f32)>,
    n_rows: usize,
    n_cols: usize,
) -> SparseIoVec {
    let mut merged: HashMap<(u64, u64), f32> = HashMap::default();
    for (r, c, v) in triplets {
        *merged.entry((r, c)).or_insert(0.0) += v;
    }
    let triplets: Vec<(u64, u64, f32)> = merged.into_iter().map(|((r, c), v)| (r, c, v)).collect();

    let path = dir.path().join("coarsen_alignment.zarr");
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (n_rows, n_cols, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    let rows: Vec<Box<str>> = (0..n_rows).map(|g| format!("GENE{g}").into()).collect();
    backend.register_row_names_vec(&rows);
    let cols: Vec<Box<str>> = (0..n_cols).map(|i| format!("c{i}").into()).collect();
    backend.register_column_names_vec(&cols);
    let mut data = SparseIoVec::new();
    data.push(std::sync::Arc::from(backend), None).unwrap();
    data
}

/// Jobs are keyed by a chunk of pb-samples while `par_chunks_mut` splits the
/// output column-major, so the two partitions have to agree exactly. They only
/// do because `n_genes` divides the chunk width. Break that correspondence —
/// write to the global pb column rather than the job-local one — and this test
/// fails.
///
/// The label map is deliberately scrambled and the counts deliberately not a
/// multiple of the chunking, so a cell's pb-sample is unrelated to its position.
#[test]
fn coarsening_by_pb_chunks_equals_the_brute_force_pooling() {
    let (n_genes, n_cells, n_pb) = (30usize, 517usize, 53usize);

    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..n_cells {
        for k in 0..3usize {
            let g = (c * 7 + k * 11 + 3) % n_genes;
            // Integer-valued, so summation order cannot introduce rounding.
            let v = (((c * 13 + g * 3 + k) % 9) + 1) as f32;
            triplets.push((g as u64, c as u64, v));
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let data = backend_from(&dir, triplets.clone(), n_genes, n_cells);
    let labels: Vec<usize> = (0..n_cells).map(|c| (c * 17 + 5) % n_pb).collect();

    // The generator never repeats a `(gene, cell)` key, so the triplets can be
    // accumulated straight into the reference.
    let mut expected = Mat::zeros(n_genes, n_pb);
    for &(r, c, v) in &triplets {
        expected[(r as usize, labels[c as usize])] += v;
    }

    let actual = coarsen_cell_expression_dense(&data, &labels, n_pb).unwrap();
    assert_eq!((actual.nrows(), actual.ncols()), (n_genes, n_pb));
    for g in 0..n_genes {
        for p in 0..n_pb {
            assert!(
                (actual[(g, p)] - expected[(g, p)]).abs() < 1e-3,
                "gene {g}, pb {p}: {} != {}",
                actual[(g, p)],
                expected[(g, p)]
            );
        }
    }
}

/// A label the coarsener has no column for means the caller and the label
/// vector disagree, and pooling anyway would drop those counts silently.
#[test]
fn coarsening_rejects_a_label_with_no_pb_sample_and_a_short_label_vector() {
    let dir = tempfile::tempdir().unwrap();
    let data = backend_from(&dir, vec![(0, 0, 1.0), (0, 1, 2.0)], 2, 2);

    // Label 5 with only 2 samples declared.
    let err = coarsen_cell_expression_dense(&data, &[0, 5], 2).unwrap_err();
    assert!(err.to_string().contains("label 5"), "{err}");

    // One label for two cells: the tail would otherwise vanish in release.
    let err = coarsen_cell_expression_dense(&data, &[0], 1).unwrap_err();
    assert!(err.to_string().contains("one pb-sample label"), "{err}");
}

/// `par_chunks_mut` panics on a zero-width chunk rather than yielding nothing,
/// so both degenerate widths return early instead.
#[test]
fn degenerate_widths_return_empty_rather_than_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let data = backend_from(&dir, vec![(0, 0, 1.0)], 2, 2);
    let out = coarsen_cell_expression_dense(&data, &[0, 0], 0).unwrap();
    assert_eq!((out.nrows(), out.ncols()), (2, 0));

    // Zero-width basis: every edge gets an empty profile, no panic.
    let basis = Mat::zeros(2, 0);
    let edges = vec![(0usize, 1usize)];
    let store = build_projection_profiles_for_edges(&data, &[0], &edges, &basis, None).unwrap();
    assert_eq!(store.n_edges, 1);
    assert_eq!(store.m, 0);
}

/// The output is chunked by a width taken from the FIRST job, so a short final
/// job must still line up. Take the width from the last job instead and the
/// zip misaligns; this catches that.
#[test]
fn edge_profiles_survive_a_short_final_job() {
    let (n_genes, n_cells, m) = (5usize, 20usize, 4usize);

    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..n_cells {
        for g in 0..n_genes {
            if (c + g) % 3 != 0 {
                triplets.push((g as u64, c as u64, (((c * 5 + g * 2) % 7) + 1) as f32));
            }
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let data = backend_from(&dir, triplets.clone(), n_genes, n_cells);

    let mut dense = Mat::zeros(n_genes, n_cells);
    for &(r, c, v) in &triplets {
        dense[(r as usize, c as usize)] = v;
    }

    // 17 edges at a block of 4 gives jobs of 4,4,4,4,1 — a short final job.
    let edges: Vec<(usize, usize)> = (0..17).map(|i| (i % n_cells, (i * 3 + 1) % n_cells)).collect();
    let edge_indices: Vec<usize> = (0..edges.len()).collect();

    let mut basis = Mat::zeros(n_genes, m);
    for g in 0..n_genes {
        for d in 0..m {
            basis[(g, d)] = (((g * 3 + d * 2) % 5) as f32) - 2.0;
        }
    }

    let store =
        build_projection_profiles_for_edges(&data, &edge_indices, &edges, &basis, Some(4)).unwrap();
    assert_eq!(store.n_edges, edges.len());

    let basis_t = basis.transpose();
    for (e, &(u, v)) in edges.iter().enumerate() {
        let expected = &basis_t * (dense.column(u) + dense.column(v));
        let (cols, vals) = store.row(e);
        let mut got = vec![0.0f32; m];
        for (&c, &x) in cols.iter().zip(vals.iter()) {
            got[c as usize] = x;
        }
        for d in 0..m {
            assert!(
                (got[d] - expected[d].max(0.0)).abs() < 1e-4,
                "edge {e}, dim {d}: {} != {}",
                got[d],
                expected[d].max(0.0)
            );
        }
    }
}
