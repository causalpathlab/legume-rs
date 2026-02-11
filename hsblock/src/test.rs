//! Integration tests for the hsblock crate.

use crate::btree::BTree;
use crate::inference::{Hsblock, HsbmOptions};
use crate::model::poisson_score_cpu;
use crate::sufficient_stats::SufficientStats;
use leiden::clustering::SimpleClustering;
use leiden::leiden::Leiden;
use leiden::network::{Graph, Network};
use leiden::Clustering;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Build a planted partition graph as a leiden Network.
///
/// If `fractional_weights` is true, edge weights are drawn from U(0.1, 1.0)
/// to simulate fuzzy kernel weights. Otherwise, weights are 1.0.
fn planted_partition_network(
    n_per_cluster: usize,
    n_clusters: usize,
    p_in: f32,
    p_out: f32,
    fractional_weights: bool,
    seed: u64,
) -> (Network, Vec<usize>) {
    let n = n_per_cluster * n_clusters;
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut true_labels = Vec::with_capacity(n);
    for c in 0..n_clusters {
        for _ in 0..n_per_cluster {
            true_labels.push(c);
        }
    }

    let mut edge_list = Vec::new();
    let mut degree = vec![0.0f32; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let p = if true_labels[i] == true_labels[j] {
                p_in
            } else {
                p_out
            };
            if rng.random::<f32>() < p {
                let w = if fractional_weights {
                    // Simulate fuzzy kernel weights in (0, 1]
                    // Within-cluster edges get higher weights
                    if true_labels[i] == true_labels[j] {
                        0.5 + 0.5 * rng.random::<f32>() // U(0.5, 1.0)
                    } else {
                        0.1 + 0.3 * rng.random::<f32>() // U(0.1, 0.4)
                    }
                } else {
                    1.0f32
                };
                degree[i] += w;
                degree[j] += w;
                edge_list.push((i, j, w));
            }
        }
    }

    let mut graph = Graph::with_capacity(n, edge_list.len());
    for i in 0..n {
        graph.add_node(degree[i]);
    }
    for &(i, j, w) in &edge_list {
        graph.add_edge((i as u32).into(), (j as u32).into(), w);
    }

    (Network::new_from_graph(graph), true_labels)
}

/// Compute Adjusted Rand Index between two label vectors.
fn adjusted_rand_index(labels_a: &[usize], labels_b: &[usize]) -> f64 {
    assert_eq!(labels_a.len(), labels_b.len());
    let n = labels_a.len();
    if n < 2 {
        return 1.0;
    }

    // Build contingency table
    let max_a = *labels_a.iter().max().unwrap_or(&0) + 1;
    let max_b = *labels_b.iter().max().unwrap_or(&0) + 1;
    let mut nij = vec![0i64; max_a * max_b];
    let mut ni = vec![0i64; max_a];
    let mut nj = vec![0i64; max_b];

    for i in 0..n {
        nij[labels_a[i] * max_b + labels_b[i]] += 1;
        ni[labels_a[i]] += 1;
        nj[labels_b[i]] += 1;
    }

    let choose2 = |x: i64| -> f64 { (x * (x - 1)) as f64 / 2.0 };

    let sum_nij_c2: f64 = nij.iter().map(|&x| choose2(x)).sum();
    let sum_ni_c2: f64 = ni.iter().map(|&x| choose2(x)).sum();
    let sum_nj_c2: f64 = nj.iter().map(|&x| choose2(x)).sum();
    let n_c2 = choose2(n as i64);

    let expected = sum_ni_c2 * sum_nj_c2 / n_c2;
    let max_index = (sum_ni_c2 + sum_nj_c2) / 2.0;

    if (max_index - expected).abs() < 1e-10 {
        return 1.0;
    }

    (sum_nij_c2 - expected) / (max_index - expected)
}

/// Run Leiden on a network and return labels.
fn run_leiden(network: &Network, n: usize, resolution: f64) -> Vec<usize> {
    // Scale resolution to CPM form: gamma / (2m)
    let total_edge_weight: f64 = (0..n)
        .flat_map(|i| network.neighbors(i).map(|(_, w)| w as f64))
        .sum::<f64>()
        / 2.0;
    let resolution_scaled = resolution / (2.0 * total_edge_weight);

    let mut leiden = Leiden::new(resolution_scaled, 0.01, Some(42));
    let mut clustering = SimpleClustering::init_different_clusters(n);

    for _iter in 0..10 {
        let updated = leiden.iterate(network, &mut clustering);
        if !updated {
            break;
        }
    }

    (0..n).map(|i| clustering.get(i)).collect()
}

/// Run HSBM on a network and return labels.
fn run_hsblock(
    network: &Network,
    n: usize,
    tree_depth: usize,
    degree_corrected: bool,
    edge_scale: f64,
) -> Vec<usize> {
    let options = HsbmOptions {
        tree_depth,
        num_sweeps: 100,
        degree_corrected,
        edge_scale,
        seed: 42,
        ..Default::default()
    };

    let mut hsblock = Hsblock::new(options);
    let mut clustering = SimpleClustering::init_different_clusters(n);
    hsblock.iterate(&network, &mut clustering);

    (0..n).map(|i| clustering.get(i)).collect()
}

/// Print cluster size distribution
fn cluster_sizes(labels: &[usize]) -> Vec<usize> {
    let k = *labels.iter().max().unwrap_or(&0) + 1;
    let mut counts = vec![0usize; k];
    for &l in labels {
        counts[l] += 1;
    }
    counts.into_iter().filter(|&s| s > 0).collect()
}

// ─── Tests ───

#[test]
fn test_end_to_end_planted_partition() {
    let (network, true_labels) = planted_partition_network(40, 2, 0.5, 0.02, false, 42);
    let n = network.nodes();

    let options = HsbmOptions {
        tree_depth: 2,
        num_sweeps: 100,
        degree_corrected: false,
        seed: 42,
        ..Default::default()
    };

    let mut hsblock = Hsblock::new(options);
    let mut clustering = SimpleClustering::init_different_clusters(n);
    hsblock.iterate(&network, &mut clustering);

    let inferred: Vec<usize> = (0..n).map(|i| clustering.get(i)).collect();

    let ari = adjusted_rand_index(&true_labels, &inferred);
    println!(
        "ARI = {:.4}, n_clusters = {}, true = 2",
        ari,
        clustering.num_clusters()
    );

    // With a well-separated planted partition, ARI should be high
    assert!(
        ari > 0.5,
        "ARI ({:.4}) should be > 0.5 for well-separated clusters",
        ari
    );
}

#[test]
fn test_tree_score_improves() {
    let (network, _) = planted_partition_network(30, 2, 0.4, 0.05, false, 99);
    let n = network.nodes();

    // Extract edges
    let mut edges = Vec::new();
    for i in 0..n {
        for (j, w) in network.neighbors(i) {
            if j > i {
                edges.push((i, j, w as f32));
            }
        }
    }

    let k = 2;
    let tree = BTree::new(2, 1.0, 1.0);

    // Random initial labels
    let mut rng = SmallRng::seed_from_u64(42);
    let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..k)).collect();
    let stats_initial = SufficientStats::from_edges(&edges, n, k, &labels);

    let (node_edge, node_total) = stats_initial.aggregate_to_tree(&tree, false);
    let initial_score: f64 = (1..=tree.num_nodes())
        .map(|node| {
            let (a0, b0) = tree.node_params(node);
            poisson_score_cpu(a0, b0, node_edge[node], node_total[node])
        })
        .sum();

    // Run HSBM — the score should improve
    let options = HsbmOptions {
        tree_depth: 2,
        num_sweeps: 100,
        degree_corrected: false,
        seed: 42,
        ..Default::default()
    };

    let mut hsblock = Hsblock::new(options);
    let mut clustering = SimpleClustering::init_different_clusters(n);
    hsblock.iterate(&network, &mut clustering);

    // Recompute final score
    let final_labels: Vec<usize> = (0..n).map(|i| clustering.get(i)).collect();
    // Number of effective clusters may differ from k
    let k_final = clustering.num_clusters();
    let tree_final = BTree::new(2, 1.0, 1.0);
    if k_final <= tree_final.num_leaves() {
        let stats_final =
            SufficientStats::from_edges(&edges, n, tree_final.num_leaves(), &final_labels);
        let (ne, nt) = stats_final.aggregate_to_tree(&tree_final, false);
        let final_score: f64 = (1..=tree_final.num_nodes())
            .map(|node| {
                let (a0, b0) = tree_final.node_params(node);
                poisson_score_cpu(a0, b0, ne[node], nt[node])
            })
            .sum();

        println!(
            "Initial score: {:.4}, Final score: {:.4}",
            initial_score, final_score
        );
        // The final score should be better (higher) than the random initial
        assert!(
            final_score >= initial_score,
            "Final score ({:.4}) should be >= initial score ({:.4})",
            final_score,
            initial_score
        );
    }
}

/// Diagnostic: compare hsblock vs leiden on planted partition with integer weights.
/// Both should recover the planted structure well.
#[test]
fn test_diagnostic_integer_weights() {
    println!("\n=== DIAGNOSTIC: Integer weights (w=1.0) ===");

    let (network, true_labels) = planted_partition_network(50, 3, 0.4, 0.02, false, 42);
    let n = network.nodes();

    println!("Graph: {} nodes, 3 true clusters of 50", n);

    // Leiden
    let leiden_labels = run_leiden(&network, n, 1.0);
    let leiden_ari = adjusted_rand_index(&true_labels, &leiden_labels);
    let leiden_sizes = cluster_sizes(&leiden_labels);
    println!(
        "Leiden:  ARI={:.4}, K={}, sizes={:?}",
        leiden_ari,
        leiden_sizes.len(),
        leiden_sizes
    );

    // HSBM with edge_scale=1.0 (no scaling)
    let hsb_labels_1 = run_hsblock(&network, n, 3, false, 1.0);
    let hsb_ari_1 = adjusted_rand_index(&true_labels, &hsb_labels_1);
    let hsb_sizes_1 = cluster_sizes(&hsb_labels_1);
    println!(
        "HSBM (scale=1):   ARI={:.4}, K={}, sizes={:?}",
        hsb_ari_1,
        hsb_sizes_1.len(),
        hsb_sizes_1
    );

    // HSBM with edge_scale=100
    let hsb_labels_100 = run_hsblock(&network, n, 3, false, 100.0);
    let hsb_ari_100 = adjusted_rand_index(&true_labels, &hsb_labels_100);
    let hsb_sizes_100 = cluster_sizes(&hsb_labels_100);
    println!(
        "HSBM (scale=100): ARI={:.4}, K={}, sizes={:?}",
        hsb_ari_100,
        hsb_sizes_100.len(),
        hsb_sizes_100
    );

    // At least one hsblock run should have decent ARI
    let best_ari = hsb_ari_1.max(hsb_ari_100);
    assert!(
        best_ari > 0.3,
        "Best HSBM ARI ({:.4}) should be > 0.3",
        best_ari
    );
}

/// Diagnostic: compare hsblock vs leiden on planted partition with fractional weights.
/// This simulates real KNN fuzzy kernel weights in (0, 1].
#[test]
fn test_diagnostic_fractional_weights() {
    println!("\n=== DIAGNOSTIC: Fractional weights (fuzzy kernel simulation) ===");

    let (network, true_labels) = planted_partition_network(50, 3, 0.4, 0.02, true, 42);
    let n = network.nodes();

    // Print weight statistics
    let mut all_weights = Vec::new();
    for i in 0..n {
        for (j, w) in network.neighbors(i) {
            if j > i {
                all_weights.push(w);
            }
        }
    }
    all_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_w = all_weights.first().copied().unwrap_or(0.0);
    let max_w = all_weights.last().copied().unwrap_or(0.0);
    let mean_w = all_weights.iter().sum::<f64>() / all_weights.len().max(1) as f64;
    println!(
        "Graph: {} nodes, {} edges, weights: min={:.3}, max={:.3}, mean={:.3}",
        n,
        all_weights.len(),
        min_w,
        max_w,
        mean_w,
    );

    // Leiden
    let leiden_labels = run_leiden(&network, n, 1.0);
    let leiden_ari = adjusted_rand_index(&true_labels, &leiden_labels);
    let leiden_sizes = cluster_sizes(&leiden_labels);
    println!(
        "Leiden:             ARI={:.4}, K={}, sizes={:?}",
        leiden_ari,
        leiden_sizes.len(),
        leiden_sizes
    );

    // HSBM: sweep edge_scale values
    for &scale in &[1.0, 10.0, 100.0, 1000.0] {
        let hsb_labels = run_hsblock(&network, n, 3, false, scale);
        let hsb_ari = adjusted_rand_index(&true_labels, &hsb_labels);
        let hsb_sizes = cluster_sizes(&hsb_labels);
        println!(
            "HSBM (scale={:>5}): ARI={:.4}, K={}, sizes={:?}",
            scale as u32,
            hsb_ari,
            hsb_sizes.len(),
            hsb_sizes
        );
    }

    // Also try degree-corrected
    for &scale in &[1.0, 100.0] {
        let hsb_labels = run_hsblock(&network, n, 3, true, scale);
        let hsb_ari = adjusted_rand_index(&true_labels, &hsb_labels);
        let hsb_sizes = cluster_sizes(&hsb_labels);
        println!(
            "HSBM DC (scale={:>3}): ARI={:.4}, K={}, sizes={:?}",
            scale as u32,
            hsb_ari,
            hsb_sizes.len(),
            hsb_sizes
        );
    }
}

/// Diagnostic: compare hsblock vs leiden on 2-cluster problem (depth=2, K=2).
/// This is the simplest possible case — should always work.
#[test]
fn test_diagnostic_two_clusters() {
    println!("\n=== DIAGNOSTIC: 2 clusters, depth=2 (K=2) ===");

    for &frac in &[false, true] {
        let label = if frac { "fractional" } else { "integer" };
        let (network, true_labels) = planted_partition_network(50, 2, 0.5, 0.02, frac, 42);
        let n = network.nodes();

        let leiden_labels = run_leiden(&network, n, 1.0);
        let leiden_ari = adjusted_rand_index(&true_labels, &leiden_labels);

        let hsb_labels_1 = run_hsblock(&network, n, 2, false, 1.0);
        let hsb_ari_1 = adjusted_rand_index(&true_labels, &hsb_labels_1);

        let hsb_labels_100 = run_hsblock(&network, n, 2, false, 100.0);
        let hsb_ari_100 = adjusted_rand_index(&true_labels, &hsb_labels_100);

        let hsb_dc_1 = run_hsblock(&network, n, 2, true, 1.0);
        let hsb_dc_ari_1 = adjusted_rand_index(&true_labels, &hsb_dc_1);
        let hsb_dc_sizes_1 = cluster_sizes(&hsb_dc_1);

        let hsb_dc_100 = run_hsblock(&network, n, 2, true, 100.0);
        let hsb_dc_ari_100 = adjusted_rand_index(&true_labels, &hsb_dc_100);
        let hsb_dc_sizes_100 = cluster_sizes(&hsb_dc_100);

        println!(
            "[{}]  Leiden={:.4}  HSBM(s=1)={:.4}  HSBM(s=100)={:.4}",
            label, leiden_ari, hsb_ari_1, hsb_ari_100,
        );
        println!(
            "       DC(s=1)={:.4} {:?}  DC(s=100)={:.4} {:?}",
            hsb_dc_ari_1, hsb_dc_sizes_1, hsb_dc_ari_100, hsb_dc_sizes_100,
        );
    }
}
