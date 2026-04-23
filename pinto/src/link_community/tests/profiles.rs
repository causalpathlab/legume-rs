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
