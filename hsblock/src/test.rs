//! Integration tests for the hsblock crate.

use crate::btree::BTree;
use crate::model::poisson_score_cpu;
use crate::sufficient_stats::SufficientStats;
use crate::variational::{Hsblock, HsbmOptions};
use candle_core::Device;
use leiden::clustering::SimpleClustering;
use leiden::network::{Graph, Network};
use leiden::Clustering;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Build a planted partition graph as a leiden Network.
fn planted_partition_network(
    n_per_cluster: usize,
    n_clusters: usize,
    p_in: f32,
    p_out: f32,
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
                let w = 1.0f32;
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

#[test]
fn test_end_to_end_planted_partition() {
    let (network, true_labels) = planted_partition_network(40, 2, 0.5, 0.02, 42);
    let n = network.nodes();

    let options = HsbmOptions {
        tree_depth: 2,
        vb_iter: 30,
        inner_iter: 10,
        final_inner_iter: 50,
        burnin_iter: 5,
        degree_corrected: false,
        seed: 42,
        ..Default::default()
    };

    let mut hsblock = Hsblock::new(options, Device::Cpu);
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
    let (network, _) = planted_partition_network(30, 2, 0.4, 0.05, 99);
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
        vb_iter: 20,
        inner_iter: 5,
        final_inner_iter: 20,
        burnin_iter: 2,
        degree_corrected: false,
        seed: 42,
        ..Default::default()
    };

    let mut hsblock = Hsblock::new(options, Device::Cpu);
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
