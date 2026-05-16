use crate::leiden::Leiden;
use crate::objective::cpm;
use crate::{Clustering, Network, SimpleClustering};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

/// Generate a random test graph using the method described in section 4B of the Leiden paper.
fn gen_sample_network(
    rng: &mut impl Rng,
    num_clusters: usize,
    nodes_per_cluster: usize,
    mean_degree: f64,
    mu: f64,
) -> (Network, SimpleClustering) {
    assert!(num_clusters > 1);
    assert!(nodes_per_cluster > 1);

    let total_nodes = num_clusters * nodes_per_cluster;
    let total_edges = (total_nodes as f64 * mean_degree).ceil() as usize;

    // which cluster is each node assigned to?
    let mut cluster = Vec::with_capacity(total_nodes);

    for c in 0..num_clusters {
        for _ in 0..nodes_per_cluster {
            cluster.push(c);
        }
    }
    let true_clusters = SimpleClustering::new_from_labels(&cluster);

    let mut network = Network::with_capacity(total_nodes);

    for _ in 0..total_nodes {
        network.add_node(1.0);
    }

    for _ in 0..total_edges {
        let in_cluster = rng.random_bool(1.0 - mu);

        // Choose Node 1
        let n1 = rng.random_range(0..total_nodes);
        let c1 = true_clusters.get(n1);

        // Choose Node 2 -- optimize the case when the node2 should be in the same cluster as node1
        let mut n2 = if in_cluster {
            rng.random_range(c1 * nodes_per_cluster..(c1 + 1) * nodes_per_cluster)
        } else {
            rng.random_range(0..total_nodes)
        };

        let mut c2 = true_clusters.get(n2);

        // Iterate until node2 is of the kind we want
        loop {
            if n1 != n2 && in_cluster && c1 == c2 {
                break;
            }

            if n1 != n2 && !in_cluster && c1 != c2 {
                break;
            }

            n2 = rng.random_range(0..total_nodes);
            c2 = cluster[n2];
        }

        network.add_edge(n1, n2, 1.0);
        network.add_edge(n2, n1, 1.0);
    }

    (network, true_clusters)
}

const DEFAULT_RESOLUTION: f64 = 1.0;
const DEFAULT_RANDOMNESS: f64 = 1e-2;

#[test]
fn run_leiden() {
    let mut rng = SmallRng::seed_from_u64(0);

    let num_clusters = 100_000 / 50;
    let nodes_per_cluster = 50;

    let (n, true_clusters) =
        gen_sample_network(&mut rng, num_clusters, nodes_per_cluster, 10.0, 0.4);
    check_edge_weight_par(&n);

    println!("best cpm: {}", cpm(DEFAULT_RESOLUTION, &n, &true_clusters));

    let mut c = SimpleClustering::init_different_clusters(n.nodes());
    let mut l = Leiden::new(DEFAULT_RESOLUTION, DEFAULT_RANDOMNESS, None);

    let score = cpm(DEFAULT_RESOLUTION, &n, &c);
    println!("initial cpm: {score}");

    for i in 0..10 {
        let update = l.iterate(&n, &mut c);

        let score = cpm(DEFAULT_RESOLUTION, &n, &c);
        check_edge_weight_par(&n);
        println!("iter: {i}, cpm: {score}");

        if !update {
            break;
        }
    }
}

fn check_edge_weight_par(n: &Network) {
    let total_edge_weight_par = n.get_total_edge_weight_par();

    // Ensure parallelized edge weight matches normal codepath
    let total_edge_weight = n.get_total_edge_weight();
    let e = (total_edge_weight_par - total_edge_weight) / total_edge_weight;

    if e.abs() > 1e-7 {
        println!("{total_edge_weight} {total_edge_weight_par} {e}");
    }
    assert!(e.abs() < 1e-6);

    // Ensure parallelized version is deterministic
    let total_edge_weight_par2 = n.get_total_edge_weight_par();
    assert_eq!(total_edge_weight_par, total_edge_weight_par2);
}

#[test]
fn edge_weight_par() {
    // (seed, num_clusters, nodes_per_cluster)
    let settings = [
        (0, 100, 50),
        (1, 100, 50),
        (2, 100, 250),
        (3, 100, 250),
        (4, 150, 1000),
        (5, 150, 1000),
    ];

    for (seed, num_clusters, nodes_per_cluster) in settings {
        let mut rng = SmallRng::seed_from_u64(seed);

        let (n, _) = gen_sample_network(&mut rng, num_clusters, nodes_per_cluster, 10.0, 0.4);
        check_edge_weight_par(&n);
    }
}
