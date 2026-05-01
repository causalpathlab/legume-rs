//! Edge-case tests for `cpm` and `par_cpm`. The objective is undefined
//! on edgeless graphs (denominator vanishes) — these tests pin down the
//! defensive `return 0.0` behaviour rather than letting NaN escape.

use crate::objective::{cpm, par_cpm};
use crate::{Clustering, Network, SimpleClustering};

fn empty_network(n_nodes: usize) -> Network {
    let mut net = Network::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        net.add_node(1.0);
    }
    net
}

#[test]
fn cpm_returns_zero_on_edgeless_graph() {
    let net = empty_network(10);
    let clustering: SimpleClustering = Clustering::init_different_clusters(10);
    let q = cpm(1.0, &net, &clustering);
    assert_eq!(q, 0.0, "cpm on edgeless graph should be 0, got {q}");
    assert!(q.is_finite(), "cpm result must be finite");
}

#[test]
fn par_cpm_returns_zero_on_edgeless_graph() {
    let net = empty_network(10);
    let clustering: SimpleClustering = Clustering::init_different_clusters(10);
    let q = par_cpm(1.0, &net, &clustering);
    assert_eq!(q, 0.0, "par_cpm on edgeless graph should be 0, got {q}");
    assert!(q.is_finite(), "par_cpm result must be finite");
}

#[test]
fn cpm_returns_zero_on_zero_node_graph() {
    let net = empty_network(0);
    let clustering: SimpleClustering = Clustering::init_different_clusters(0);
    let q = cpm(1.0, &net, &clustering);
    assert_eq!(q, 0.0);
    let q_par = par_cpm(1.0, &net, &clustering);
    assert_eq!(q_par, 0.0);
}

#[test]
fn cpm_and_par_cpm_agree_on_small_graph() {
    let mut net = Network::with_capacity(4);
    let nodes: Vec<_> = (0..4).map(|_| net.add_node(1.0)).collect();
    net.add_edge(nodes[0], nodes[1], 1.0);
    net.add_edge(nodes[1], nodes[0], 1.0);
    net.add_edge(nodes[2], nodes[3], 1.0);
    net.add_edge(nodes[3], nodes[2], 1.0);

    let clustering: SimpleClustering = SimpleClustering::new_from_labels(&[0, 0, 1, 1]);
    let serial = cpm(1.0, &net, &clustering);
    let parallel = par_cpm(1.0, &net, &clustering);
    assert!(
        (serial - parallel).abs() < 1e-12,
        "serial={serial}, parallel={parallel}"
    );
}
