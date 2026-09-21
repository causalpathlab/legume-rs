//! Typed graph joining the peak→gene link relation with the pb-tree count
//! relations (SIMBA-binned) and the tree's parent edges.

mod common;

use chickpea::p2g::context_graph::*;
use chickpea::p2g::link_map::PeakGeneEdge;
use chickpea::p2g::pb_levels::PbLevels;
use common::mat;

/// 2 genes × 4 finest pbs and 3 peaks × 4 pbs, coarser level of 2 pbs.
fn two_level_tree(with_rna: bool) -> PbLevels {
    let rna0 = mat(2, 4, |g, s| ((g + 1) * (s + 1)) as f32);
    let rna1 = mat(2, 2, |g, s| ((g + 1) * (s + 1) * 3) as f32);
    let atac0 = mat(3, 4, |p, s| if (p + s) % 2 == 0 { 2.0 } else { 0.0 });
    let atac1 = mat(3, 2, |p, s| ((p + 1) * (s + 2)) as f32);
    PbLevels {
        rna: with_rna.then(|| vec![rna0, rna1]),
        atac: vec![atac0, atac1],
        parent: vec![vec![0, 0, 1, 1]],
    }
}

fn link() -> Vec<PeakGeneEdge> {
    vec![
        PeakGeneEdge {
            peak: 0,
            gene: 0,
            weight: 0.9,
        },
        PeakGeneEdge {
            peak: 2,
            gene: 1,
            weight: 0.4,
        },
    ]
}

#[test]
fn node_types_are_region_gene_and_one_per_level() {
    let g = build_context_graph(&link(), &two_level_tree(true), 3, 2, 5).unwrap();
    let names: Vec<&str> = g.types.names().iter().map(|s| &**s).collect();
    assert_eq!(names, vec!["region", "gene", "pb@0", "pb@1"]);
    assert_eq!(g.types.n_nodes(0), 3);
    assert_eq!(g.types.n_nodes(1), 2);
    assert_eq!(g.types.n_nodes(2), 4);
    assert_eq!(g.types.n_nodes(3), 2);
}

#[test]
fn every_nonzero_count_is_one_edge_and_parents_are_edges() {
    let tree = two_level_tree(true);
    let g = build_context_graph(&link(), &tree, 3, 2, 5).unwrap();
    let names: Vec<&str> = g.rels.iter().map(|r| &*r.name).collect();
    assert_eq!(names[0], "region:gene/link");
    assert!(names.iter().any(|n| n.starts_with("pb@0:gene/rna@")));
    assert!(names.iter().any(|n| n.starts_with("pb@1:region/atac@")));
    assert!(names.contains(&"pb@0:pb@1/parent"));
    // 2 link + RNA nonzeros (8 + 4) + ATAC nonzeros (6 + 6) + 4 parent edges.
    assert_eq!(g.edges.len(), 2 + 12 + 12 + 4);
    let counts = g.edges.counts_per_relation(g.rels.len());
    assert_eq!(counts[0], 2);
    let parent_rel = names.iter().position(|n| *n == "pb@0:pb@1/parent").unwrap();
    assert_eq!(counts[parent_rel], 4);
    g.edges.validate(&g.types, &g.rels).unwrap();
}

#[test]
fn atac_only_has_no_rna_relation() {
    let g = build_context_graph(&link(), &two_level_tree(false), 3, 2, 5).unwrap();
    assert!(g.rels.iter().all(|r| !r.name.contains("rna")));
    assert_eq!(g.edges.len(), 2 + 12 + 4);
}

#[test]
fn binned_relations_carry_simba_weights_and_link_is_repeated() {
    let g = build_context_graph(&link(), &two_level_tree(true), 3, 2, 2).unwrap();
    let rna0: Vec<f32> = g
        .rels
        .iter()
        .filter(|r| r.name.starts_with("pb@0:gene/rna@"))
        .map(|r| r.weight)
        .collect();
    assert_eq!(rna0, vec![1.0, 5.0]);
    // The link relation is lifted to about one count level's share.
    assert!(g.repeats[0] > 1, "link repeats = {}", g.repeats[0]);
}

#[test]
fn zeros_are_not_edges() {
    let mut tree = two_level_tree(false);
    tree.atac[0][(0, 0)] = 0.0;
    tree.atac[0][(1, 1)] = 0.0;
    tree.atac[0][(2, 2)] = 0.0;
    let g = build_context_graph(&link(), &tree, 3, 2, 5).unwrap();
    assert_eq!(g.edges.len(), 2 + (12 - 3) + 4);
}
