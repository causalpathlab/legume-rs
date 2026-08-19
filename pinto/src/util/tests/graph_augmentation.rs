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

use crate::util::common::Mat;
use crate::util::input::auto_batch_from_components;
use crate::util::srt_pipeline::topology_graph;
use matrix_util::knn_graph::{DistanceMerge, KnnGraph, KnnGraphArgs};
use nalgebra::DMatrix;

/// The same eight points `two_sections` places, as a coordinate matrix.
fn two_section_coords() -> Mat {
    Mat::from_row_slice(
        8,
        2,
        &[
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // section one
            500.0, 0.0, 501.0, 0.0, 500.0, 1.0, 501.0, 1.0, // section two
        ],
    )
}

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
        auto_batch_from_components(&spatial, &two_section_coords(), &mut labels),
        2,
        "the two sections are separate before augmentation"
    );
    let mut labels = vec!["b".into(); 8];
    assert_eq!(
        auto_batch_from_components(&merged, &two_section_coords(), &mut labels),
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
    let n = auto_batch_from_components(
        topology_graph(&merged, &spatial_graph),
        &two_section_coords(),
        &mut labels,
    );

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

/// A connected component stands in for a sample, but the spatial graph breaks
/// wherever the tissue does, so one piece arrives as a large component plus
/// stray fragments. Batching per fragment estimates an effect from a handful
/// of cells. A fragment is told apart from a real piece by WHERE it lies, not
/// by how small it is: it sits inside the piece it broke off.
#[test]
fn a_fragment_inside_a_piece_joins_it_rather_than_becoming_its_own_batch() {
    use crate::util::input::auto_batch_from_components;

    // Two pieces, each internally well connected at unit spacing, plus a
    // two-cell fragment two units above the first piece. Two units clears the
    // neighbour link (their own partner is one away) but sits inside the three
    // units of slack the median edge length implies, which is exactly the
    // situation a torn piece of tissue produces.
    let coords = Mat::from_row_slice(
        14,
        2,
        &[
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, // piece one
            1.0, 3.0, 2.0, 3.0, // its fragment
            500.0, 0.0, 501.0, 0.0, 502.0, 0.0, 500.0, 1.0, 501.0, 1.0, 502.0,
            1.0, // piece two
        ],
    );
    let graph = KnnGraph::from_rows(
        &coords,
        KnnGraphArgs {
            knn: 2,
            block_size: 16,
            reciprocal: true,
        },
    )
    .unwrap();

    let mut labels = vec!["b".into(); 14];
    let n = auto_batch_from_components(&graph, &coords, &mut labels);

    assert_eq!(n, 2, "two pieces, not two pieces plus a fragment");
    assert_eq!(labels[6], labels[0], "the fragment takes its piece's label");
    assert_eq!(labels[7], labels[0]);
    assert_ne!(labels[8], labels[0], "the far piece stays its own batch");
}

/// Nothing is merged when the pieces are simply separate. The sizes here are
/// deliberately UNEQUAL: a rule that folded the smaller piece into the larger
/// on size alone would pass a fixture of two equal pieces while being wrong,
/// so the test has to be able to tell size from position.
#[test]
fn separate_pieces_are_left_alone_however_unequal() {
    use crate::util::input::auto_batch_from_components;

    // Four cells here, ten far away. Nothing lies inside anything else.
    let mut v: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    for i in 0..10 {
        v.push(500.0 + (i % 5) as f32);
        v.push((i / 5) as f32);
    }
    let coords = Mat::from_row_slice(14, 2, &v);
    let graph = KnnGraph::from_rows(
        &coords,
        KnnGraphArgs {
            knn: 2,
            block_size: 16,
            reciprocal: true,
        },
    )
    .unwrap();

    let mut labels = vec!["b".into(); 14];
    let n = auto_batch_from_components(&graph, &coords, &mut labels);
    assert_eq!(
        n, 2,
        "two separated pieces stay two batches whatever their sizes"
    );
    assert_ne!(
        labels[0], labels[8],
        "the small piece is not absorbed by the big one"
    );
}
