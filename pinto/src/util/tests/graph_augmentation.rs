//! What the union graph may and may not be used for.
//!
//! Augmenting the pair graph with expression neighbours changes what the graph
//! MEANS. It is still the right object to model, because its edges are the
//! pairs we score. It is the wrong object to navigate, because its edges are
//! no longer statements about physical adjacency.
//!
//! The failure that motivates these tests is silent. Nothing errors; the run
//! simply reports one batch where there were two, skips batch-effect
//! estimation, and collapses every tissue core into one frame downstream.

use crate::util::input::auto_batch_from_components;
use crate::util::srt_pipeline::topology_graph;
use matrix_util::knn_graph::{DistanceMerge, KnnGraph, KnnGraphArgs};
use nalgebra::DMatrix;

/// Two tissue sections, far apart in space, so the spatial graph has two
/// connected components: cells 0-3 and cells 4-7.
fn two_sections() -> KnnGraph {
    let coords = DMatrix::from_row_slice(
        8,
        2,
        &[
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // section one
            500.0, 0.0, 501.0, 0.0, 500.0, 1.0, 501.0, 1.0, // section two
        ],
    );
    KnnGraph::from_rows(
        &coords,
        KnnGraphArgs {
            knn: 2,
            block_size: 8,
            reciprocal: false,
        },
    )
    .unwrap()
}

/// Expression neighbours ignore geometry, so the same cell type in either
/// section pairs up and the two components become one.
fn expression_pairs() -> KnnGraph {
    let embedding = DMatrix::from_row_slice(8, 1, &[0.0, 10.0, 20.0, 30.0, 0.1, 10.1, 20.1, 30.1]);
    KnnGraph::from_rows(
        &embedding,
        KnnGraphArgs {
            knn: 1,
            block_size: 8,
            reciprocal: false,
        },
    )
    .unwrap()
}

/// The premise. If the union did NOT bridge the sections there would be
/// nothing here to get wrong, and the tests below would pass vacuously.
#[test]
fn the_union_really_does_merge_two_separate_sections() {
    let spatial = two_sections();
    let (merged, _) = spatial
        .union_with(&expression_pairs(), DistanceMerge::SourceRank)
        .unwrap();

    let mut labels = vec!["b".into(); 8];
    assert_eq!(
        auto_batch_from_components(&spatial, &mut labels),
        2,
        "the two sections are separate before augmentation"
    );
    let mut labels = vec!["b".into(); 8];
    assert_eq!(
        auto_batch_from_components(&merged, &mut labels),
        1,
        "and the union joins them, which is exactly the hazard"
    );
}

/// So batch detection must be handed the spatial graph. Run it on the union
/// and a two-section slide silently becomes a one-batch run.
#[test]
fn batches_are_detected_on_the_spatial_graph_not_the_union() {
    let spatial = two_sections();
    let (merged, _) = spatial
        .union_with(&expression_pairs(), DistanceMerge::SourceRank)
        .unwrap();
    let spatial_graph = Some(spatial);

    let mut labels = vec!["b".into(); 8];
    let n = auto_batch_from_components(topology_graph(&merged, &spatial_graph), &mut labels);

    assert_eq!(n, 2, "topology_graph must hand back the spatial graph");
    assert_eq!(
        labels
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        2,
        "and the cells must carry two distinct batch labels"
    );
    assert_eq!(labels[0], labels[3], "one section shares a label");
    assert_eq!(labels[4], labels[7], "so does the other");
    assert_ne!(labels[0], labels[4], "and the two sections differ");
}

/// With no augmentation there is no separate spatial graph to fall back to,
/// and the modelled graph IS the spatial one.
#[test]
fn without_augmentation_the_topology_is_the_graph_itself() {
    let spatial = two_sections();
    let none: Option<KnnGraph> = None;
    let picked = topology_graph(&spatial, &none);
    assert_eq!(picked.edges, spatial.edges);
    assert_eq!(picked.n_nodes, spatial.n_nodes);
}

/// Expression neighbours must not leave the spatial component they came from.
///
/// A tissue microarray is the case that matters: each core is a separate
/// sample, and expression similarity ignores geometry, so an unrestricted
/// search pairs cells across cores and therefore across samples. The rest of
/// the pipeline treats a spatial component as a batch, so those pairs
/// contradict its own definition of one.
#[test]
fn expression_neighbours_stay_inside_their_spatial_component() {
    use crate::util::cell_pairs::{
        build_expression_knn, build_expression_knn_within, SrtCellPairsArgs,
    };
    use crate::util::common::Mat;

    // Two spatial components. Cells alternate between two expression profiles,
    // so a cell's nearest neighbours by expression exist in BOTH components
    // and an unrestricted search would happily cross.
    let n = 12usize;
    let proj = Mat::from_fn(2, n, |r, c| {
        let kind = (c % 2) as f32;
        if r == 0 {
            kind
        } else {
            1.0 - kind
        }
    });
    let component: Vec<usize> = (0..n).map(|c| usize::from(c >= n / 2)).collect();
    let args = || SrtCellPairsArgs {
        knn: 3,
        block_size: Some(64),
        reciprocal: false,
    };

    let within = build_expression_knn_within(&proj, &component, 2, args()).unwrap();
    let global = build_expression_knn(&proj, args()).unwrap();

    assert!(!within.edges.is_empty(), "the fixture must produce edges");
    for &(i, j) in &within.edges {
        assert_eq!(
            component[i], component[j],
            "edge {i}-{j} left its component"
        );
    }
    // The fixture only means something if an unrestricted search DOES cross,
    // otherwise the assertion above passes for the wrong reason.
    assert!(
        global
            .edges
            .iter()
            .any(|&(i, j)| component[i] != component[j]),
        "a global search must cross here, or this proves nothing"
    );
    assert_eq!(within.n_nodes, n, "nodes are all cells, not one component");
}

/// A component smaller than the neighbour count cannot supply that many, and
/// taking whatever it has would give its cells a denser neighbourhood than
/// everyone else's. They get none instead.
#[test]
fn a_component_too_small_for_k_contributes_no_expression_edges() {
    use crate::util::cell_pairs::{build_expression_knn_within, SrtCellPairsArgs};
    use crate::util::common::Mat;

    let n = 10usize;
    let proj = Mat::from_fn(2, n, |r, c| if r == 0 { c as f32 } else { 0.0 });
    let component: Vec<usize> = (0..n).map(|c| usize::from(c >= 8)).collect();
    let g = build_expression_knn_within(
        &proj,
        &component,
        2,
        SrtCellPairsArgs {
            knn: 3,
            block_size: Some(64),
            reciprocal: false,
        },
    )
    .unwrap();
    assert!(
        g.edges
            .iter()
            .all(|&(i, j)| component[i] == 0 && component[j] == 0),
        "the two-cell component must contribute nothing"
    );
}
