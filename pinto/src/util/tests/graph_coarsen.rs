use crate::test_support::make_test_graph;
use crate::util::common::*;
use crate::util::graph_coarsen::*;

#[test]
fn test_graph_coarsen_small() {
    // 6 nodes in a path: 0-1-2-3-4-5
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let graph = make_test_graph(6, edges);

    let mut features = Mat::zeros(6, 6);
    for i in 0..6 {
        features[(i, i)] = 1.0;
    }

    let result = graph_coarsen(&graph, &mut features, None);
    assert_eq!(result.n_nodes, 6);
    assert_eq!(result.merges.len(), 5);

    // Extract via multilevel with 3 cluster target
    let pairs: Vec<Pair> = (0..5)
        .map(|i| Pair {
            left: i,
            right: i + 1,
        })
        .collect();
    let ml = graph_coarsen_multilevel(
        &make_test_graph(6, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        &mut {
            let mut f = Mat::zeros(6, 6);
            for i in 0..6 {
                f[(i, i)] = 1.0;
            }
            f
        },
        &pairs,
        CoarsenConfig {
            n_clusters: 3,
            num_levels: 1,
            refine_iterations: 0,
            seeding: None,
            modularity_veto: None,
            dc_poisson: None,
        },
    );
    let unique: HashSet<usize> = ml.all_pair_to_sample[0].iter().cloned().collect();
    assert!(unique.len() <= ml.all_num_samples[0]);
}

#[test]
fn test_graph_coarsen_two_cliques() {
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)];
    let graph = make_test_graph(6, edges.clone());

    let dim = 4;
    let mut features = Mat::zeros(dim, 6);
    for i in 0..3 {
        features[(0, i)] = 1.0;
        features[(1, i)] = 0.1 * (i as f32);
    }
    for i in 3..6 {
        features[(3, i)] = 1.0;
        features[(2, i)] = 0.1 * ((i - 3) as f32);
    }

    let pairs: Vec<Pair> = edges
        .iter()
        .map(|&(i, j)| Pair { left: i, right: j })
        .collect();

    let ml = graph_coarsen_multilevel(
        &graph,
        &mut features,
        &pairs,
        CoarsenConfig {
            n_clusters: 2,
            num_levels: 1,
            refine_iterations: 0,
            seeding: None,
            modularity_veto: None,
            dc_poisson: None,
        },
    );
    // With 2 clusters the two cliques should separate
    let p2s = &ml.all_pair_to_sample[0];
    // Intra-clique pairs should share a sample pattern
    // edges: (0,1),(0,2),(1,2) are clique A, (3,4),(3,5),(4,5) are clique B
    // (2,3) is the bridge
    assert_eq!(ml.all_num_samples[0], 3); // (A,A), (B,B), (A,B)
    assert_eq!(p2s[0], p2s[1]); // (0,1) and (0,2) both in clique A
    assert_eq!(p2s[0], p2s[2]); // (1,2) also in clique A
    assert_eq!(p2s[4], p2s[5]); // (3,5) and (4,5) both in clique B
    assert_ne!(p2s[0], p2s[4]); // clique A ≠ clique B
}

#[test]
fn test_cell_labels_to_pair_samples() {
    let pairs = vec![
        Pair { left: 0, right: 1 },
        Pair { left: 2, right: 3 },
        Pair { left: 0, right: 3 },
        Pair { left: 1, right: 0 },
    ];
    let labels = vec![0, 0, 1, 1];

    let (p2s, n_samples) = cell_labels_to_pair_samples(&labels, &pairs);

    assert_eq!(p2s[0], p2s[3]);
    assert_ne!(p2s[0], p2s[1]);
    assert_ne!(p2s[0], p2s[2]);
    assert_ne!(p2s[1], p2s[2]);
    assert_eq!(n_samples, 3);
}

#[test]
fn test_compute_level_n_clusters() {
    assert_eq!(compute_level_n_clusters(1024, 1), vec![1024]);
    assert_eq!(compute_level_n_clusters(1024, 2), vec![16, 1024]);

    let levels = compute_level_n_clusters(1024, 3);
    assert_eq!(levels.len(), 3);
    assert_eq!(levels[0], 16);
    assert_eq!(levels[2], 1024);
    assert!(levels[1] > 16 && levels[1] < 1024);

    assert_eq!(compute_level_n_clusters(8, 2), vec![8, 8]);
}

#[test]
fn test_graph_coarsen_multilevel() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let graph = make_test_graph(6, edges.clone());

    let pairs: Vec<Pair> = edges
        .iter()
        .map(|&(i, j)| Pair { left: i, right: j })
        .collect();

    let mut features = Mat::zeros(4, 6);
    for i in 0..6 {
        features[(i % 4, i)] = 1.0;
    }

    // 2 levels, finest = 3 clusters
    let ml = graph_coarsen_multilevel(
        &graph,
        &mut features,
        &pairs,
        CoarsenConfig {
            n_clusters: 3,
            num_levels: 2,
            refine_iterations: 0,
            seeding: None,
            modularity_veto: None,
            dc_poisson: None,
        },
    );

    assert_eq!(ml.all_pair_to_sample.len(), 2);
    assert_eq!(ml.all_num_samples.len(), 2);

    // Coarsest level has fewer or equal samples
    assert!(ml.all_num_samples[0] <= ml.all_num_samples[1]);

    // Each pair_to_sample has one entry per pair
    for (level, p2s) in ml.all_pair_to_sample.iter().enumerate() {
        assert_eq!(p2s.len(), pairs.len());
        for &s in p2s {
            assert!(s < ml.all_num_samples[level]);
        }
    }

    // Single level
    let ml1 = graph_coarsen_multilevel(
        &make_test_graph(6, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        &mut Mat::from_fn(4, 6, |r, c| if r == c % 4 { 1.0 } else { 0.0 }),
        &pairs,
        CoarsenConfig {
            n_clusters: 3,
            num_levels: 1,
            refine_iterations: 0,
            seeding: None,
            modularity_veto: None,
            dc_poisson: None,
        },
    );
    assert_eq!(ml1.all_pair_to_sample.len(), 1);
    assert_eq!(ml1.all_num_samples.len(), 1);
}

#[test]
fn test_spatial_seeding() {
    // 100 nodes on a 10×10 grid, KNN edges to 4-neighbors
    let n = 100;
    let mut coords = Mat::zeros(n, 2);
    let mut edges = Vec::new();
    for row in 0..10 {
        for col in 0..10 {
            let i = row * 10 + col;
            coords[(i, 0)] = col as f32;
            coords[(i, 1)] = row as f32;
            if col + 1 < 10 {
                edges.push((i, i + 1));
            }
            if row + 1 < 10 {
                edges.push((i, i + 10));
            }
        }
    }
    let graph = make_test_graph(n, edges.clone());

    // Seed with target ~25 super-nodes (5×5 grid)
    let (labels, num_super) = spatial_seed_labels(&coords, &graph, 25, None);
    assert_eq!(labels.len(), n);
    assert!(num_super > 0 && num_super <= 30);
    // Every cell assigned
    assert!(labels.iter().all(|&l| l < num_super));

    // Build super-graph
    let dim = 4;
    let mut features = Mat::from_fn(dim, n, |r, c| if r == c % dim { 1.0 } else { 0.0 });
    let (super_graph, super_features) = build_super_graph(&labels, num_super, &graph, &features);
    assert_eq!(super_features.nrows(), dim);
    assert_eq!(super_features.ncols(), num_super);
    assert!(!super_graph.edges.is_empty());
    assert_eq!(super_graph.n_nodes, num_super);

    // Full multilevel with seeding
    let pairs: Vec<Pair> = edges
        .iter()
        .map(|&(i, j)| Pair { left: i, right: j })
        .collect();
    let sp = SeedingParams {
        coordinates: &coords,
        batch_membership: None,
    };
    let ml = graph_coarsen_multilevel(
        &graph,
        &mut features,
        &pairs,
        CoarsenConfig {
            n_clusters: 10,
            num_levels: 2,
            refine_iterations: 0,
            seeding: Some(sp),
            modularity_veto: None,
            dc_poisson: None,
        },
    );
    assert_eq!(ml.all_cell_labels.len(), 2);
    for level_labels in &ml.all_cell_labels {
        assert_eq!(level_labels.len(), n);
    }
}

#[test]
fn test_spatial_seeding_capacity_limit() {
    // Dense cluster of 80 cells at (0,0) plus 20 cells spread across (1..20, 0).
    // Without capacity limits, the dense cluster would all join one super-node.
    let n = 100;
    let mut coords = Mat::zeros(n, 2);
    let mut edges = Vec::new();

    // Dense cluster: cells 0..80 all at (0, 0) with chain edges
    for i in 0..80 {
        coords[(i, 0)] = 0.0;
        coords[(i, 1)] = (i as f32) * 0.01; // tiny offsets so they're nearby
        if i + 1 < 80 {
            edges.push((i, i + 1));
        }
    }
    // Connect dense cluster to spread region
    edges.push((79, 80));

    // Spread region: cells 80..100 at increasing x
    for i in 80..100 {
        coords[(i, 0)] = (i - 79) as f32;
        coords[(i, 1)] = 0.0;
        if i + 1 < 100 {
            edges.push((i, i + 1));
        }
    }

    let graph = make_test_graph(n, edges);

    // Target 10 super-nodes → avg 10 cells each, max_cells = 20
    let (labels, num_super) = spatial_seed_labels(&coords, &graph, 10, None);
    assert_eq!(labels.len(), n);
    assert!(labels.iter().all(|&l| l < num_super));

    // Count cells per super-node
    let mut counts = vec![0usize; num_super];
    for &l in &labels {
        counts[l] += 1;
    }

    // No super-node should have more than 2× the average
    let max_cells = (n / num_super) * 2;
    let largest = *counts.iter().max().unwrap();
    assert!(
        largest <= max_cells + 1, // +1 for rounding
        "largest super-node has {} cells, limit is {} (avg={})",
        largest,
        max_cells,
        n / num_super
    );
}

#[test]
fn test_refine_labels_in_multilevel() {
    // Two-clique setup — coarsening + refinement integration.
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)];
    let graph = make_test_graph(6, edges.clone());

    let dim = 4;
    let mut features = Mat::zeros(dim, 6);
    for i in 0..3 {
        features[(0, i)] = 1.0;
    }
    for i in 3..6 {
        features[(3, i)] = 1.0;
    }

    let pairs: Vec<Pair> = edges
        .iter()
        .map(|&(i, j)| Pair { left: i, right: j })
        .collect();

    let ml = graph_coarsen_multilevel(
        &graph,
        &mut features,
        &pairs,
        CoarsenConfig {
            n_clusters: 2,
            num_levels: 1,
            refine_iterations: 5,
            seeding: None,
            modularity_veto: None,
            dc_poisson: None,
        },
    );
    // Must still produce the expected (A,A), (B,B), (A,B) sample structure.
    assert_eq!(ml.all_num_samples[0], 3);
    let p2s = &ml.all_pair_to_sample[0];
    assert_eq!(p2s[0], p2s[1]);
    assert_eq!(p2s[0], p2s[2]);
    assert_eq!(p2s[4], p2s[5]);
    assert_ne!(p2s[0], p2s[4]);
}

#[test]
fn test_modularity_veto_rejects_bridge() {
    // Two triangles {0,1,2} and {3,4,5} joined by one bridge (2,3). Intra-clique
    // features are nearly identical (high sim) while the bridge endpoints sit
    // on orthogonal axes (zero sim). The modularity veto should reject the
    // bridge merge regardless of how greedy we get.
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)];
    let graph = make_test_graph(6, edges.clone());

    let dim = 2;
    // Clique A lives on axis 0, clique B on axis 1; both L2-normalised.
    let mut features = Mat::zeros(dim, 6);
    for i in 0..3 {
        features[(0, i)] = 1.0;
    }
    for i in 3..6 {
        features[(1, i)] = 1.0;
    }

    // Without the veto, the greedy coarsener will eventually cross the
    // bridge — confirm the baseline produces only a single cluster when
    // asked for one.
    let result_no_veto = graph_coarsen(&graph, &mut features.clone(), None);
    assert_eq!(
        result_no_veto.merges.len(),
        5,
        "without veto, coarsening should merge everything"
    );

    // With γ = 1.0, the bridge edge (orthogonal endpoints, sim = 0) should
    // fail the modularity-gain test and be vetoed. The two triangles merge
    // internally (5 intra-clique merges → 2 clusters) but the bridge itself
    // is never taken.
    let veto = ModularityVeto { gamma: 1.0 };
    let result_with_veto = graph_coarsen(&graph, &mut features, Some(&veto));
    assert_eq!(
        result_with_veto.merges.len(),
        4,
        "with veto, each triangle produces 2 merges (6 nodes → 2 clusters)"
    );
}
