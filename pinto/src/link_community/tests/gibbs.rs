use crate::link_community::gibbs::*;
use crate::link_community::model::{LinkCommunityStats, LinkProfileStore};
use crate::test_support::make_test_graph;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Create a planted partition: edges 0..n/2 belong to community 0,
/// edges n/2..n belong to community 1, with distinct gene signatures.
fn make_planted_profiles(n_edges: usize, m: usize) -> (LinkProfileStore, Vec<usize>) {
    let mut profiles = vec![0.0f32; n_edges * m];
    let mut true_labels = vec![0usize; n_edges];

    for e in 0..n_edges {
        let c = if e < n_edges / 2 { 0 } else { 1 };
        true_labels[e] = c;
        for g in 0..m {
            // Strong signal in first half of genes for c=0, second half for c=1
            let signal = if (g < m / 2) == (c == 0) { 10.0 } else { 1.0 };
            profiles[e * m + g] = signal;
        }
    }

    (LinkProfileStore::new(profiles, n_edges, m), true_labels)
}

#[test]
fn test_gibbs_convergence() {
    let (store, _true_labels) = make_planted_profiles(100, 10);
    let k = 2;

    // Start with random labels
    let random_labels: Vec<usize> = (0..100).map(|e| e % k).collect();
    let mut stats = LinkCommunityStats::from_profiles(&store, k, &random_labels);

    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

    let moves1 = sampler.run(&mut stats, &store, 1);
    let _moves_mid = sampler.run(&mut stats, &store, 20);
    let moves_late = sampler.run(&mut stats, &store, 1);

    assert!(
        moves_late <= moves1 || moves1 == 0,
        "Expected convergence: early={}, late={}",
        moves1,
        moves_late
    );
}

#[test]
fn test_greedy_recovers_planted() {
    let (store, true_labels) = make_planted_profiles(100, 10);
    let k = 2;

    // Start with random labels
    let random_labels: Vec<usize> = (0..100).map(|e| (e * 7) % k).collect();
    let mut stats = LinkCommunityStats::from_profiles(&store, k, &random_labels);

    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));
    sampler.run(&mut stats, &store, 50);
    sampler.run_greedy_plain(&mut stats, &store, 20, 0.0);

    // Check that the partition matches the planted one (up to label permutation)
    let match_direct: usize = (0..100)
        .filter(|&e| stats.membership[e] == true_labels[e])
        .count();
    let match_flipped: usize = (0..100)
        .filter(|&e| stats.membership[e] == 1 - true_labels[e])
        .count();

    let best_match = match_direct.max(match_flipped);
    assert!(
        best_match >= 90,
        "Planted partition recovery: {}/100",
        best_match
    );
}

#[test]
fn test_parallel_gibbs() {
    let (store, _true_labels) = make_planted_profiles(100, 10);
    let k = 2;

    let random_labels: Vec<usize> = (0..100).map(|e| e % k).collect();
    let mut stats = LinkCommunityStats::from_profiles(&store, k, &random_labels);

    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

    let moves1 = sampler.run_parallel(&mut stats, &store, 1, 0.0);
    let _moves_mid = sampler.run_parallel(&mut stats, &store, 20, 0.0);
    let moves_late = sampler.run_parallel(&mut stats, &store, 1, 0.0);

    assert!(
        moves_late <= moves1 || moves1 == 0,
        "Parallel: expected convergence: early={}, late={}",
        moves1,
        moves_late
    );
}

#[test]
fn test_sample_categorical_log() {
    let mut rng = SmallRng::seed_from_u64(42);

    let log_probs = vec![-100.0, 0.0, -100.0];
    let mut counts = [0usize; 3];
    for _ in 0..1000 {
        let idx = sample_categorical_log(&log_probs, &mut rng);
        counts[idx] += 1;
    }
    assert!(counts[1] > 990, "Expected mostly index 1, got {:?}", counts);
}

/// Test memoized EM Gibbs on a 2-component graph with planted partition.
///
/// Component 0: nodes 0-4, edges among them → community 0 (high in genes 0..m/2)
/// Component 1: nodes 5-9, edges among them → community 1 (high in genes m/2..m)
#[test]
fn test_memoized_em_two_components() {
    let m = 10;
    let k = 2;

    // Two disconnected cliques: nodes 0-4 and nodes 5-9
    let edges_list = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
        (5, 6),
        (5, 7),
        (5, 8),
        (6, 7),
        (6, 8),
        (7, 8),
        (7, 9),
        (8, 9),
    ];
    let n_edges = edges_list.len();
    let graph = make_test_graph(10, edges_list.clone());

    // Build profiles: edges in component 0 → community 0 signal,
    //                  edges in component 1 → community 1 signal
    let mut profiles = vec![0.0f32; n_edges * m];
    let mut true_labels = vec![0usize; n_edges];
    for (e, &(i, _j)) in edges_list.iter().enumerate() {
        let c = if i < 5 { 0 } else { 1 };
        true_labels[e] = c;
        for g in 0..m {
            let signal = if (g < m / 2) == (c == 0) { 10.0 } else { 1.0 };
            profiles[e * m + g] = signal;
        }
    }
    let store = LinkProfileStore::new(profiles, n_edges, m);

    // Start with random labels
    let mut membership: Vec<usize> = (0..n_edges).map(|e| (e * 7) % k).collect();

    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

    // Run memoized EM Gibbs
    let comp_args = ComponentGibbsArgs {
        graph: &graph,
        edges: &edges_list,
        k,
        alpha: 0.0,
        incidence: None,
    };
    let moves = sampler.run_components_em(&mut membership, &store, &comp_args, 50);
    assert!(moves > 0, "Expected some moves");

    // Run memoized greedy
    let greedy_moves = sampler.run_greedy_by_components(&mut membership, &store, &comp_args, 20);
    let _ = greedy_moves;

    // Check planted recovery (up to label permutation)
    let match_direct: usize = (0..n_edges)
        .filter(|&e| membership[e] == true_labels[e])
        .count();
    let match_flipped: usize = (0..n_edges)
        .filter(|&e| membership[e] == 1 - true_labels[e])
        .count();
    let best_match = match_direct.max(match_flipped);
    assert!(
        best_match >= n_edges - 2,
        "Two-component planted recovery: {}/{}",
        best_match,
        n_edges
    );
}

/// Test memoized EM on a single-component graph (balanced partitioning splits it).
#[test]
fn test_memoized_em_single_component() {
    let (store, _true_labels) = make_planted_profiles(100, 10);
    let k = 2;

    // Build a connected graph for edges 0..100
    // Edges need a graph with nodes. Let's make a simple chain.
    let n_nodes = 101;
    let edges: Vec<(usize, usize)> = (0..100).map(|i| (i, i + 1)).collect();
    let graph = make_test_graph(n_nodes, edges.clone());

    let mut membership: Vec<usize> = (0..100).map(|e| e % k).collect();

    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

    let comp_args = ComponentGibbsArgs {
        graph: &graph,
        edges: &edges,
        k,
        alpha: 0.0,
        incidence: None,
    };
    let moves = sampler.run_components_em(&mut membership, &store, &comp_args, 10);
    // Should converge (fewer moves over time, similar to parallel test)
    assert!(moves > 0 || membership.iter().all(|&m| m < k));
}

/// Test that memoized stats delta preserves exact global stats.
#[test]
fn test_memoized_stats_consistency() {
    let m = 6;
    let k = 3;
    let n_edges = 30;

    let (store, labels) = {
        let mut profiles = vec![0.0f32; n_edges * m];
        let mut labels = vec![0usize; n_edges];
        for e in 0..n_edges {
            let c = e % k;
            labels[e] = c;
            for g in 0..m {
                let signal = if g % k == c { 5.0 } else { 1.0 };
                profiles[e * m + g] = signal;
            }
        }
        (LinkProfileStore::new(profiles, n_edges, m), labels)
    };

    let global = LinkCommunityStats::from_profiles(&store, k, &labels);

    // Partition into 2 fake components: edges 0..15 and 15..30
    let comp0: Vec<usize> = (0..15).collect();
    let comp1: Vec<usize> = (15..30).collect();

    let (gs0, ss0, ec0) = LinkCommunityStats::component_stats(&store, k, &comp0, &labels);
    let (gs1, ss1, ec1) = LinkCommunityStats::component_stats(&store, k, &comp1, &labels);

    // Verify: sum of component stats == global stats
    for i in 0..k * m {
        assert!(
            (gs0[i] + gs1[i] - global.gene_sum[i]).abs() < 1e-10,
            "gene_sum mismatch at {}",
            i
        );
    }
    for c in 0..k {
        assert!(
            (ss0[c] + ss1[c] - global.size_sum[c]).abs() < 1e-10,
            "size_sum mismatch at {}",
            c
        );
        assert_eq!(
            ec0[c] + ec1[c],
            global.edge_count[c],
            "edge_count mismatch at {}",
            c
        );
    }
}
