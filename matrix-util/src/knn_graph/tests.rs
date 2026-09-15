use super::*;

/// Two tight clusters of 5 points each in 2D, well separated
fn two_cluster_matrix() -> DMatrix<f32> {
    DMatrix::from_row_slice(
        10,
        2,
        &[
            // Cluster A near origin
            0.0, 0.0, //
            0.1, 0.0, //
            0.0, 0.1, //
            0.1, 0.1, //
            0.05, 0.05, //
            // Cluster B far away
            10.0, 10.0, //
            10.1, 10.0, //
            10.0, 10.1, //
            10.1, 10.1, //
            10.05, 10.05, //
        ],
    )
}

#[test]
fn test_knn_graph_construction() {
    let data = two_cluster_matrix();
    let graph = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 4,
            block_size: 100,
            reciprocal: true,
        },
    )
    .unwrap();

    // Basic properties
    assert_eq!(graph.num_nodes(), 10);
    assert!(graph.num_edges() > 0);
    assert_eq!(graph.edges.len(), graph.distances.len());

    // All edges should be (i < j)
    for &(i, j) in &graph.edges {
        assert!(i < j, "Edge ({}, {}) not canonical", i, j);
    }

    // All distances should be non-negative
    for &d in &graph.distances {
        assert!(d >= 0.0);
    }

    // Adjacency matrix dimensions
    assert_eq!(graph.adjacency.nrows(), 10);
    assert_eq!(graph.adjacency.ncols(), 10);

    // With k=4 and well-separated clusters, no edges should cross clusters
    for &(i, j) in &graph.edges {
        let same_cluster = (i < 5 && j < 5) || (i >= 5 && j >= 5);
        assert!(same_cluster, "Cross-cluster edge ({}, {}) found", i, j);
    }

    // Adjacency should be symmetric: if i is neighbor of j, j is neighbor of i
    for node in 0..graph.num_nodes() {
        for &neighbor in graph.neighbors(node) {
            let reverse_neighbors = graph.neighbors(neighbor);
            assert!(
                reverse_neighbors.contains(&node),
                "Node {} has neighbor {} but not vice versa",
                node,
                neighbor
            );
        }
    }
}

#[test]
fn test_from_columns_equivalent_to_from_rows() {
    let data = two_cluster_matrix();
    let transposed = data.transpose();

    let g_rows = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 3,
            block_size: 100,
            reciprocal: true,
        },
    )
    .unwrap();

    let g_cols = KnnGraph::from_columns(
        &transposed,
        KnnGraphArgs {
            knn: 3,
            block_size: 100,
            reciprocal: true,
        },
    )
    .unwrap();

    assert_eq!(g_rows.num_nodes(), g_cols.num_nodes());
    let diff = (g_rows.num_edges() as i64 - g_cols.num_edges() as i64).unsigned_abs();
    assert!(
        diff <= 2,
        "Edge counts differ: {} vs {}",
        g_rows.num_edges(),
        g_cols.num_edges()
    );
}

#[test]
fn test_exp_kernel_weights() {
    let data = two_cluster_matrix();
    let graph = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 4,
            block_size: 100,
            reciprocal: true,
        },
    )
    .unwrap();

    let weights = graph.exp_kernel_weights();
    assert_eq!(weights.len(), graph.num_edges());

    // All weights should be in (0, 1]
    for &w in &weights {
        assert!(w > 0.0, "Weight {} should be > 0", w);
        assert!(w <= 1.0, "Weight {} should be <= 1", w);
    }

    // Median edge gets exp(-1) ≈ 0.37; closer edges get higher weights
    let mean_w: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
    assert!(
        mean_w > 0.2 && mean_w < 0.9,
        "Mean weight {} should be in a reasonable range",
        mean_w
    );
}

#[test]
fn test_fuzzy_kernel_weights() {
    let data = two_cluster_matrix();
    let graph = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 4,
            block_size: 100,
            reciprocal: false, // union, like scanpy default
        },
    )
    .unwrap();

    let weights = graph.fuzzy_kernel_weights();
    assert_eq!(weights.len(), graph.num_edges());

    // All weights should be in (0, 1]
    for &w in &weights {
        assert!(w > 0.0, "Weight {} should be > 0", w);
        assert!(w <= 1.0, "Weight {} should be <= 1", w);
    }

    // With UMAP weights, no edge should be near zero (local sigma adapts)
    let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        min_w > 0.01,
        "Min fuzzy weight {} is too small; local sigma should prevent near-zero weights",
        min_w
    );
}

#[test]
fn test_smooth_knn_sigma() {
    // 5 distances, rho = 0.1 (nearest neighbor)
    let dists = [0.1, 0.2, 0.3, 0.5, 1.0];
    let rho = 0.1;
    let target = (5.0f32).log2(); // log2(k)

    let sigma = super::smooth_knn_sigma(&dists, rho, target);
    assert!(sigma > 0.0, "sigma should be positive");

    // Verify the sigma achieves the target
    let psum: f32 = dists
        .iter()
        .map(|&d| {
            let gap = d - rho;
            if gap > 0.0 {
                (-gap / sigma).exp()
            } else {
                1.0
            }
        })
        .sum();

    assert!(
        (psum - target).abs() < 0.1,
        "psum {:.3} should be close to target {:.3}",
        psum,
        target
    );
}

#[test]
fn test_median() {
    assert_eq!(crate::utils::median(&[1.0, 3.0, 2.0]), 2.0);
    assert_eq!(crate::utils::median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    assert_eq!(crate::utils::median(&[5.0]), 5.0);
}

#[test]
fn test_create_jobs_helper() {
    let jobs = create_jobs(10, 3);
    assert_eq!(jobs, vec![(0, 3), (3, 6), (6, 9), (9, 10)]);

    let jobs = create_jobs(6, 3);
    assert_eq!(jobs, vec![(0, 3), (3, 6)]);

    let jobs = create_jobs(1, 100);
    assert_eq!(jobs, vec![(0, 1)]);

    // block_size=0 should fall back to DEFAULT_BLOCK_SIZE
    let jobs = create_jobs(5, 0);
    assert_eq!(jobs, vec![(0, 5)]);
}

/// A line of 4 points, so the spatial graph is a path.
fn line_matrix() -> DMatrix<f32> {
    DMatrix::from_row_slice(4, 1, &[0.0, 1.0, 2.0, 3.0])
}

fn path_graph(points: &DMatrix<f32>) -> KnnGraph {
    KnnGraph::from_rows(
        points,
        KnnGraphArgs {
            knn: 1,
            block_size: 8,
            reciprocal: false,
        },
    )
    .unwrap()
}

/// Two graphs over the same nodes. The union must carry every edge once, tag
/// each with where it came from, and keep the sorted `i < j` invariant the
/// rest of the type depends on.
#[test]
fn union_merges_edges_and_records_which_graph_each_came_from() {
    let a = path_graph(&line_matrix());
    // A second embedding: 0 and 3 are close here, and 1 and 2 again.
    let b = path_graph(&DMatrix::from_row_slice(4, 1, &[0.0, 10.0, 10.5, 0.5]));

    let a_edges = a.edges.clone();
    let b_edges = b.edges.clone();
    let (merged, source) = a.union_with(&b, DistanceMerge::SourceRank).unwrap();

    assert_eq!(merged.edges.len(), source.len(), "one source tag per edge");
    assert_eq!(
        merged.distances.len(),
        merged.edges.len(),
        "distances parallel"
    );
    assert_eq!(merged.n_nodes, 4);
    assert!(
        merged.edges.windows(2).all(|w| w[0] < w[1]),
        "sorted and deduplicated"
    );
    for &(i, j) in &merged.edges {
        assert!(i < j, "canonical orientation");
    }

    for e in &a_edges {
        let k = merged.edges.iter().position(|x| x == e).expect("kept");
        let want = if b_edges.contains(e) {
            EdgeSource::Both
        } else {
            EdgeSource::Primary
        };
        assert_eq!(source[k], want, "edge {e:?}");
    }
    for e in &b_edges {
        let k = merged.edges.iter().position(|x| x == e).expect("kept");
        let want = if a_edges.contains(e) {
            EdgeSource::Both
        } else {
            EdgeSource::Secondary
        };
        assert_eq!(source[k], want, "edge {e:?}");
    }

    // The adjacency is derived state and must be rebuilt. A union that merges
    // the edge list but keeps an input's adjacency passes everything above.
    for &(i, j) in &merged.edges {
        assert!(
            merged.neighbors(i).contains(&j),
            "adjacency missing {i}->{j}"
        );
        assert!(
            merged.neighbors(j).contains(&i),
            "adjacency missing {j}->{i}"
        );
    }
}

/// The two inputs measure different things, so each side is replaced by its
/// own within-source quantile rank before merging. Without this the
/// user-facing `distance` column interleaves two incomparable units and a
/// median-sigma kernel over the result is meaningless.
#[test]
fn union_ranks_each_sources_distances_within_that_source() {
    let a = path_graph(&line_matrix());
    // Same spacing scaled by 1000, but PERMUTED so `b` contributes edges of
    // its own. With identical topology every edge would be shared and the
    // `min` rule would always return the near graph's value, so the fixture
    // rather than the policy would decide the result.
    let b = path_graph(&DMatrix::from_row_slice(
        4,
        1,
        &[0.0, 3000.0, 1000.0, 2000.0],
    ));
    assert!(
        b.edges.iter().any(|e| !a.edges.contains(e)),
        "fixture must give `b` at least one edge of its own"
    );

    let (ranked, _) = a.union_with(&b, DistanceMerge::SourceRank).unwrap();
    for &d in &ranked.distances {
        assert!((0.0..=1.0).contains(&d), "rank out of range: {d}");
    }

    // Raw must keep the spread, or the assertion above proves nothing about
    // the policy as opposed to the fixture.
    let (raw, _) = a.union_with(&b, DistanceMerge::Raw).unwrap();
    let hi = raw.distances.iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        hi > 100.0,
        "Raw should preserve the original scale, got {hi}"
    );
}

/// A pair must be counted once however an input happened to store it. Edge
/// order is a constructor invariant, not a type invariant: `build_super_graph`
/// in pinto builds a `KnnGraph` by hand from hash-map order.
#[test]
fn union_counts_a_pair_once_even_when_an_input_stores_it_reversed() {
    let a = path_graph(&line_matrix());
    let mut flipped = path_graph(&line_matrix());
    for e in flipped.edges.iter_mut() {
        *e = (e.1, e.0);
    }

    let (merged, source) = a.union_with(&flipped, DistanceMerge::SourceRank).unwrap();

    assert_eq!(
        merged.edges.len(),
        a.edges.len(),
        "reversed duplicates must collapse, not double"
    );
    assert!(
        source.iter().all(|s| *s == EdgeSource::Both),
        "every edge is present in both inputs"
    );
    for &(i, j) in &merged.edges {
        assert!(i < j, "canonical orientation restored");
        assert_eq!(
            merged.neighbors(i).iter().filter(|&&n| n == j).count(),
            1,
            "edge {i}-{j} recorded once in the adjacency"
        );
    }
}

#[test]
fn union_rejects_graphs_over_different_node_counts() {
    let a = path_graph(&line_matrix());
    let b = path_graph(&DMatrix::from_row_slice(3, 1, &[0.0, 1.0, 2.0]));
    let err = match a.union_with(&b, DistanceMerge::SourceRank) {
        Ok(_) => panic!("a union over mismatched node counts must not succeed"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("node"), "{err}");
}

/// Ranks are written into the merged graph's `distances`, and that lands in a
/// file. The sort is parallel and unstable, so ties must be broken on
/// something stable or the file stops being reproducible between runs. Equal
/// distances are the common case, not a corner: a grid of coordinates ties
/// everywhere.
///
/// Stated as an exact expectation rather than a repeat-and-compare, because
/// a repeat inside one process can agree by luck.
#[test]
fn tied_distances_rank_in_index_order_so_the_result_is_reproducible() {
    // The fixture has to do two things at once: be large enough to engage the
    // parallel sort, and mix ties WITH distinct values. All-equal input does
    // not work, because the sort recognises that pattern and moves nothing, so
    // ties survive even with no tie-break and the test cannot fail.
    const N: usize = 100_000;
    let tied_and_distinct: Vec<f32> = (0..N).map(|i| (i % 4) as f32).collect();
    let r = within_source_rank(&tied_and_distinct);

    for i in 0..N {
        for j in [i + 4, i + 400] {
            if j < N {
                assert!(
                    r[i] < r[j],
                    "equal values must rank in index order: {i} vs {j}"
                );
            }
        }
    }
    // And the ranks are still exactly the ladder, just permuted.
    let mut sorted = r.clone();
    sorted.sort_by(f32::total_cmp);
    let ladder: Vec<f32> = (0..N).map(|i| i as f32 / (N - 1) as f32).collect();
    assert_eq!(sorted, ladder);

    // A partial tie: the two 1.0s must keep their relative index order, and
    // ranks must still be a permutation of the same ladder.
    let mixed = vec![9.0f32, 1.0, 5.0, 1.0];
    let r = within_source_rank(&mixed);
    assert!(r[1] < r[3], "the earlier of two equal values ranks first");
    assert!(
        r[1] < r[2] && r[2] < r[0],
        "distinct values keep their order"
    );
    let mut sorted = r.clone();
    sorted.sort_by(f32::total_cmp);
    assert_eq!(sorted, vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
}

#[test]
fn ranking_handles_degenerate_lengths() {
    assert!(within_source_rank(&[]).is_empty());
    assert_eq!(
        within_source_rank(&[7.0]),
        vec![0.0],
        "no divide by n-1 == 0"
    );
}

//////////////////////////////////////////////////
// Neighbour lists → edges, and the search arms //
//////////////////////////////////////////////////

/// The edge filter as it was written over a keyed map of directed triplets:
/// the oracle the flat sort-and-fold must reproduce.
fn keyed_edges(
    neighbours: &[(Vec<usize>, Vec<f32>)],
    reciprocal: bool,
) -> Vec<((usize, usize), f32)> {
    let mut triplets = std::collections::HashMap::new();
    for (i, (nb, ds)) in neighbours.iter().enumerate() {
        for (&j, &d) in nb.iter().zip(ds) {
            triplets.insert((i, j), d);
        }
    }
    let mut edges: Vec<((usize, usize), f32)> = triplets
        .iter()
        .filter_map(|(&(i, j), &d)| {
            if reciprocal {
                (i < j && triplets.contains_key(&(j, i))).then_some(((i, j), d))
            } else if i < j {
                let d_ji = triplets.get(&(j, i)).copied().unwrap_or(d);
                Some(((i, j), d.min(d_ji)))
            } else if !triplets.contains_key(&(j, i)) {
                Some(((j, i), d))
            } else {
                None
            }
        })
        .collect();
    edges.sort_by_key(|&(ij, _)| ij);
    edges
}

/// Hand-built directed lists: reversed duplicates with different distances,
/// one-sided pairs, and a pair present in both directions with equal distance.
fn directed_lists() -> Vec<(Vec<usize>, Vec<f32>)> {
    vec![
        (vec![1, 2, 4], vec![0.5, 1.5, 3.0]), // 0→1, 0→2, 0→4
        (vec![0, 3], vec![0.6, 2.0]),         // 1→0 (asymmetric), 1→3
        (vec![0, 3], vec![1.5, 0.2]),         // 2→0 (symmetric), 2→3
        (vec![2], vec![0.2]),                 // 3→2 only
        (vec![3], vec![9.0]),                 // 4→3 one-sided, 4→0 missing
    ]
}

#[test]
fn flat_fold_reproduces_the_keyed_filter_in_both_modes() {
    let lists = directed_lists();
    for reciprocal in [false, true] {
        let got = edges_from_neighbours(&lists, reciprocal);
        let want = keyed_edges(&lists, reciprocal);
        assert_eq!(got, want, "reciprocal={reciprocal}");
    }
    // Spot checks of what the oracle encodes.
    let union = edges_from_neighbours(&lists, false);
    assert!(union.contains(&((0, 1), 0.5)), "min of the two directions");
    assert!(
        union.contains(&((3, 4), 9.0)),
        "one-sided pair kept, canonical order"
    );
    let recip = edges_from_neighbours(&lists, true);
    assert_eq!(recip, vec![((0, 1), 0.5), ((0, 2), 1.5), ((2, 3), 0.2)]);
}

#[test]
fn direct_csc_matches_the_coo_route() {
    let edges = vec![(0, 1), (0, 3), (1, 2), (1, 5), (2, 5), (3, 4)];
    let distances = vec![0.1, 0.3, 0.2, 0.6, 0.5, 0.4];
    let got = symmetric_adjacency(6, &edges, &distances);
    let mut coo = nalgebra_sparse::CooMatrix::new(6, 6);
    for (&(i, j), &v) in edges.iter().zip(&distances) {
        coo.push(i, j, v);
        coo.push(j, i, v);
    }
    let want = CscMatrix::from(&coo);
    assert_eq!(got.col_offsets(), want.col_offsets());
    assert_eq!(got.row_indices(), want.row_indices());
    assert_eq!(got.values(), want.values());
    // A node with no edges keeps an empty column.
    let got = symmetric_adjacency(3, &[(0, 2)], &[1.0]);
    assert_eq!(got.col_offsets(), &[0, 1, 1, 2]);
}

#[test]
fn the_all_pairs_arm_equals_brute_force_above_the_exact_threshold() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let n = EXACT_THRESHOLD + 100;
    let mut rng = StdRng::seed_from_u64(9);
    let data = DMatrix::from_fn(n, 4, |_, _| rng.random_range(-1.0f32..1.0));
    let graph = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 5,
            block_size: 1000,
            reciprocal: false,
        },
    )
    .unwrap();
    // The same graph from a brute-force scan.
    let points: Vec<Vec<f32>> = (0..n)
        .map(|i| data.row(i).iter().copied().collect())
        .collect();
    let lists: Vec<(Vec<usize>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|q| crate::knn::tests::brute_others(&points, q, 5))
        .collect();
    let want = edges_from_neighbours(&lists, false);
    assert_eq!(graph.edges.len(), want.len());
    for ((e, d), w) in graph.edges.iter().zip(&graph.distances).zip(&want) {
        assert_eq!(*e, w.0);
        assert!((d - w.1).abs() < 1e-5);
    }
}
