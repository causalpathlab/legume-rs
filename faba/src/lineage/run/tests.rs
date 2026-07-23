//! End-to-end tests for the `faba lineage` run.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

// resolve_root_hint signature: (root_node, root_cell, cell_names, labels, k, type_root,
// gem_root) -> Result<Option<usize>>. Priority: node > cell > type > gem > None.

/// End-to-end recovery on a planted two-lineage sim: two x-chains far apart in y, joined
/// by the single geometric MST bridge. Velocity flows +x in both, so the bridge (a +y edge)
/// carries no net flow → abstains → the max-weight branching cuts it into TWO trees, each
/// rooted at its x=0 source, with monotone within-tree pseudotime.
#[test]
fn forest_recovers_two_bridged_lineages() {
    use crate::lineage::forest::fit_forest_curves;
    use crate::lineage::orient::{
        candidate_edges, edge_directionality, EdgeCall, EdgeDirectionConfig,
    };
    use matrix_util::branching::max_branching;
    use matrix_util::principal_curve::PrincipalCurveArgs;
    use matrix_util::principal_graph::{mst_from_sqdist, pairwise_sqdist_rows_to_rows};
    use nalgebra::DMatrix;
    use std::collections::HashMap;

    // 8 centroids: chain A at y=0, chain B at y=100, each x = 0,1,2,3.
    let k = 8;
    let mut centroids = DMatrix::<f32>::zeros(k, 2);
    for c in 0..4 {
        centroids[(c, 0)] = c as f32; // A: (0,0)..(3,0)
        centroids[(c, 1)] = 0.0;
        centroids[(4 + c, 0)] = c as f32; // B: (0,100)..(3,100)
        centroids[(4 + c, 1)] = 100.0;
    }
    // 12 cells per node, sitting on the node; velocity = +x everywhere.
    let per = 12;
    let n = k * per;
    let mut theta = DMatrix::<f32>::zeros(n, 2);
    let mut velocity = DMatrix::<f32>::zeros(n, 2);
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let node = i / per;
        labels[i] = node;
        theta[(i, 0)] = centroids[(node, 0)];
        theta[(i, 1)] = centroids[(node, 1)];
        velocity[(i, 0)] = 1.0; // +x flow
    }

    let (mst, _) = mst_from_sqdist(&pairwise_sqdist_rows_to_rows(&centroids, &centroids));
    let cand = candidate_edges(&centroids, &mst, 2);
    let cfg = EdgeDirectionConfig {
        n_boot: 50,
        n_perm: 100,
        alpha: 0.05,
        min_cells: 2,
        seed: 7,
    };
    let dirs = edge_directionality(&centroids, &velocity, &labels, &cand, &mst, &cfg);

    // Within-chain +x edges are called; the +y bridge abstains.
    let get = |a: usize, b: usize| {
        let key = if a < b { (a, b) } else { (b, a) };
        dirs.iter().find(|d| d.edge == key).cloned()
    };
    assert_eq!(
        get(0, 1).unwrap().call,
        EdgeCall::Forward,
        "A: 0→1 flows +x"
    );
    assert_eq!(
        get(4, 5).unwrap().call,
        EdgeCall::Forward,
        "B: 4→5 flows +x"
    );
    // The y-bridge is whichever candidate edge crosses the two groups (nodes <4 vs ≥4).
    let bridge = dirs
        .iter()
        .find(|d| (d.edge.0 < 4) != (d.edge.1 < 4))
        .expect("a bridge edge joins the two chains");
    assert_eq!(
        bridge.call,
        EdgeCall::Abstain,
        "the y-bridge carries no net flow → abstain"
    );

    // Branching cuts the abstained bridge → two trees.
    let (arcs, root_aff) = assemble_arcs(&dirs, k, None, None);
    let br = max_branching(k, &arcs, &root_aff);
    assert!(br.roots.len() >= 2, "the bridge is cut into ≥2 trees");
    assert_ne!(
        br.tree[0], br.tree[4],
        "the two chains land in different trees"
    );
    for c in 1..4 {
        assert_eq!(br.tree[c], br.tree[0], "chain A stays one tree");
        assert_eq!(br.tree[4 + c], br.tree[4], "chain B stays one tree");
    }
    // Each tree roots at its x=0 source (no incoming edge).
    assert!(br.parent[0].is_none() && br.parent[4].is_none());

    // Per-tree curves + pseudotime: cells split by tree, order increases along +x.
    let dirs_map: HashMap<(usize, usize), &_> = dirs.iter().map(|d| (d.edge, d)).collect();
    let forest = fit_forest_curves(
        &theta,
        &centroids,
        &labels,
        &br,
        &dirs_map,
        &PrincipalCurveArgs::default(),
    )
    .unwrap();
    // A cell at x=0 (node 0) and a cell at x=3 (node 3) are in the same tree, ordered.
    let cell_at = |node: usize| node * per; // first cell of that node
    assert_eq!(forest.cell_tree[cell_at(0)], forest.cell_tree[cell_at(3)]);
    assert_ne!(forest.cell_tree[cell_at(0)], forest.cell_tree[cell_at(4)]);
    let pt0 = forest.curves.pseudotime[cell_at(0)];
    let pt3 = forest.curves.pseudotime[cell_at(3)];
    assert!(
        pt0.is_finite() && pt3.is_finite() && pt3 > pt0,
        "pseudotime rises along +x"
    );
    // Downstream cells carry positive order confidence (their edge was called).
    assert!(forest.order_conf[cell_at(3)] > 0.0);
}

#[test]
fn end_to_end_run_writes_forest_outputs() {
    use clap::Parser;
    use matrix_util::dmatrix_io::DMatrix;
    use matrix_util::traits::IoOps;

    // Two well-separated chains (y=0 and y=100), four x-positions each as tight blobs,
    // with a uniform +x velocity — the y-bridge should be cut into two trees.
    let per = 15usize;
    let groups = 8usize;
    let n = groups * per;
    let mut theta = DMatrix::<f32>::zeros(n, 2);
    let mut velocity = DMatrix::<f32>::zeros(n, 2);
    let mut rows: Vec<Box<str>> = Vec::with_capacity(n);
    for i in 0..n {
        let g = i / per;
        let jit = ((i % per) as f32 - per as f32 / 2.0) * 0.01;
        theta[(i, 0)] = (g % 4) as f32 + jit;
        theta[(i, 1)] = if g < 4 { 0.0 } else { 100.0 } + jit;
        velocity[(i, 0)] = 1.0; // +x flow along each chain
        rows.push(format!("cell_{i}").into());
    }
    let cols = names(&["d0", "d1"]);

    let prefix = std::env::temp_dir()
        .join(format!("faba_lin_e2e_{}", std::process::id()))
        .to_string_lossy()
        .into_owned();
    let hdr = (Some(rows.as_slice()), Some("cell"));
    theta
        .to_parquet_with_names(
            &format!("{prefix}.cell_embedding.parquet"),
            hdr,
            Some(cols.as_slice()),
        )
        .unwrap();
    velocity
        .to_parquet_with_names(
            &format!("{prefix}.velocity.parquet"),
            hdr,
            Some(cols.as_slice()),
        )
        .unwrap();

    // Drive the real command through clap parsing + run_lineage (skip PHATE for speed).
    #[derive(Parser)]
    struct W {
        #[command(flatten)]
        a: LineageArgs,
    }
    let w = W::parse_from([
        "t",
        "--from",
        &prefix,
        "--layout",
        "none",
        // This fixture is a CARTESIAN grid (two chains at y=0 and y=100), not a cell
        // embedding: under the cosine default every y=100 row normalizes to ~(0, 1),
        // collapsing the four x-positions into one point and leaving nothing to cut.
        // Pin the Euclidean geometry the fixture is drawn in, so this stays a test of
        // the forest logic rather than of the default metric.
        "--latent-geometry",
        "euclidean",
        "--n-centroids",
        "8",
        "--edge-direction-n-boot",
        "50",
        "--edge-direction-n-perm",
        "100",
        "--seed",
        "1",
    ]);
    run_lineage(&w.a).unwrap();

    // The forest outputs exist and describe a ≥2-tree split with finite pseudotime.
    let trees = DMatrix::<f32>::from_parquet(&format!("{prefix}.trees.parquet")).unwrap();
    assert!(
        trees.mat.nrows() >= 2,
        "the y-bridge is cut → ≥2 trees, got {}",
        trees.mat.nrows()
    );
    let pt = DMatrix::<f32>::from_parquet(&format!("{prefix}.pseudotime.parquet")).unwrap();
    assert_eq!(pt.mat.nrows(), n, "one pseudotime row per cell");
    assert!(
        (0..n).any(|i| pt.mat[(i, 0)].is_finite()),
        "some cells get finite pseudotime"
    );
    let trees_seen: std::collections::HashSet<i64> =
        (0..n).map(|i| pt.mat[(i, 2)] as i64).collect();
    assert!(trees_seen.len() >= 2, "cells span ≥2 trees");
    assert!(std::path::Path::new(&format!("{prefix}.edges.parquet")).exists());

    for suf in [
        "cell_embedding",
        "velocity",
        "trees",
        "pseudotime",
        "edges",
        "nodes",
        "node_velocity",
        "lineages",
        "cell_lineage_weights",
        "lineage_pseudotime",
        "curves",
    ] {
        let _ = std::fs::remove_file(format!("{prefix}.{suf}.parquet"));
    }
}
