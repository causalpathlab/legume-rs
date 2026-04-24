//! Shared graph APIs.
//!
//! A lightweight `WeightedGraph` trait + a handful of algorithms that
//! operate generically over anything that implements it. Individual
//! graph types (kNN graph, gene-pair graph, Leiden network) live in
//! their own modules/crates and opt in via a single `impl` block.

use std::collections::VecDeque;

/// Undirected weighted graph over contiguous node ids `0..num_nodes()`.
///
/// `neighbors_with_weight` yields each edge incident to `node` as
/// `(other_node, weight)`. The trait is deliberately boxed-iterator to
/// keep object safety and let implementations pick whatever internal
/// storage they already have (CSC, adjacency list, petgraph, etc.).
pub trait WeightedGraph {
    fn num_nodes(&self) -> usize;
    fn num_edges(&self) -> usize;
    fn neighbors_with_weight<'a>(
        &'a self,
        node: usize,
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a>;

    fn degree(&self, node: usize) -> usize {
        self.neighbors_with_weight(node).count()
    }

    fn weighted_degree(&self, node: usize) -> f32 {
        self.neighbors_with_weight(node).map(|(_, w)| w).sum()
    }
}

/// Generic symmetric adjacency-list graph. Useful as a target when adapting
/// an edge-list representation (e.g. `GenePairGraph`) to algorithms that
/// consume `WeightedGraph`.
pub struct AdjListGraph {
    adj: Vec<Vec<(usize, f32)>>,
    num_edges: usize,
}

impl AdjListGraph {
    /// Build from a canonical (u ≤ v) edge list. Each edge is expanded to
    /// both directions internally; self-loops are dropped.
    pub fn from_edges(n_nodes: usize, edges: &[(usize, usize, f32)]) -> Self {
        let mut adj = vec![Vec::new(); n_nodes];
        let mut num_edges = 0usize;
        for &(u, v, w) in edges {
            if u == v {
                continue;
            }
            adj[u].push((v, w));
            adj[v].push((u, w));
            num_edges += 1;
        }
        Self { adj, num_edges }
    }

    /// Build from unweighted canonical edges; each edge gets weight 1.0.
    pub fn from_unweighted_edges(n_nodes: usize, edges: &[(usize, usize)]) -> Self {
        let weighted: Vec<(usize, usize, f32)> = edges.iter().map(|&(u, v)| (u, v, 1.0)).collect();
        Self::from_edges(n_nodes, &weighted)
    }
}

impl WeightedGraph for AdjListGraph {
    fn num_nodes(&self) -> usize {
        self.adj.len()
    }

    fn num_edges(&self) -> usize {
        self.num_edges
    }

    fn neighbors_with_weight<'a>(
        &'a self,
        node: usize,
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        Box::new(self.adj[node].iter().copied())
    }
}

/// Per-node component id via BFS. Unreached nodes (isolated) get their own id.
pub fn connected_components<G: WeightedGraph + ?Sized>(g: &G) -> Vec<usize> {
    let n = g.num_nodes();
    let mut label = vec![usize::MAX; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut next_id = 0usize;

    for root in 0..n {
        if label[root] != usize::MAX {
            continue;
        }
        label[root] = next_id;
        queue.push_back(root);
        while let Some(u) = queue.pop_front() {
            for (v, _) in g.neighbors_with_weight(u) {
                if label[v] == usize::MAX {
                    label[v] = next_id;
                    queue.push_back(v);
                }
            }
        }
        next_id += 1;
    }
    label
}

/// Count of connected components.
pub fn num_connected_components<G: WeightedGraph + ?Sized>(g: &G) -> usize {
    connected_components(g)
        .into_iter()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Weighted degree per node.
pub fn weighted_degrees<G: WeightedGraph + ?Sized>(g: &G) -> Vec<f32> {
    (0..g.num_nodes()).map(|i| g.weighted_degree(i)).collect()
}

/// Sum of all edge weights (each undirected edge counted once). Assumes
/// `neighbors_with_weight(u)` yields `v > u` edges too (symmetric listing).
pub fn total_edge_weight<G: WeightedGraph + ?Sized>(g: &G) -> f32 {
    let mut s = 0f32;
    for u in 0..g.num_nodes() {
        for (v, w) in g.neighbors_with_weight(u) {
            if u < v {
                s += w;
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TinyGraph {
        n: usize,
        adj: Vec<Vec<(usize, f32)>>,
    }

    impl WeightedGraph for TinyGraph {
        fn num_nodes(&self) -> usize {
            self.n
        }
        fn num_edges(&self) -> usize {
            self.adj.iter().map(|v| v.len()).sum::<usize>() / 2
        }
        fn neighbors_with_weight<'a>(
            &'a self,
            node: usize,
        ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
            Box::new(self.adj[node].iter().copied())
        }
    }

    fn two_components() -> TinyGraph {
        // 0 -- 1,   2 -- 3 -- 4,   5 isolated
        let mut adj = vec![vec![]; 6];
        adj[0].push((1, 1.0));
        adj[1].push((0, 1.0));
        adj[2].push((3, 2.0));
        adj[3].push((2, 2.0));
        adj[3].push((4, 3.0));
        adj[4].push((3, 3.0));
        TinyGraph { n: 6, adj }
    }

    #[test]
    fn connected_components_identifies_three_groups() {
        let g = two_components();
        let label = connected_components(&g);
        assert_eq!(label[0], label[1]);
        assert_eq!(label[2], label[3]);
        assert_eq!(label[3], label[4]);
        assert_ne!(label[0], label[2]);
        assert_ne!(label[0], label[5]);
        assert_ne!(label[2], label[5]);
        assert_eq!(num_connected_components(&g), 3);
    }

    #[test]
    fn weighted_degree_and_total_weight() {
        let g = two_components();
        assert_eq!(g.degree(3), 2);
        assert_eq!(g.weighted_degree(3), 5.0);
        assert_eq!(weighted_degrees(&g), vec![1.0, 1.0, 2.0, 5.0, 3.0, 0.0]);
        assert_eq!(total_edge_weight(&g), 6.0);
    }
}
