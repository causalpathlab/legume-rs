//! Undirected weighted graph used by the Leiden / Louvain clustering algorithms.

use crate::Clustering;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use rustc_hash::FxHashMap as HashMap;

/// Undirected weighted graph backing the Leiden / Louvain algorithms.
///
/// Node ids are dense `usize` in `0..nodes()`. Each undirected edge is stored
/// twice (once per endpoint) so adjacency iteration is O(deg). Internal
/// adjacency uses `u32` neighbour ids for compactness; callers see only `usize`.
pub struct Network {
    /// adj\[u\] holds (neighbour_id, edge_weight) for each undirected edge {u, v}.
    /// Each undirected edge appears in both adj\[u\] and adj\[v\].
    adj: Vec<Vec<(u32, f32)>>,
    /// Node weights, parallel to `adj`.
    node_weights: Vec<f32>,
    /// Number of undirected edges (each pair counted once).
    edge_count: usize,
}

/// Iterator over `(neighbour_id, edge_weight)` for all neighbours of a chosen node.
pub struct NeighborAndWeightIter<'a> {
    iter: std::slice::Iter<'a, (u32, f32)>,
}

impl Iterator for NeighborAndWeightIter<'_> {
    type Item = (usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|&(n, w)| (n as usize, f64::from(w)))
    }
}

/// Iterator over each undirected edge once, yielding `(source, target, weight)`
/// with `target >= source`. Iteration order is by `source` ascending; within a
/// source, in insertion order. Stable across calls.
pub struct EdgeReferences<'a> {
    adj: &'a [Vec<(u32, f32)>],
    src: usize,
    pos: usize,
}

impl Iterator for EdgeReferences<'_> {
    type Item = (usize, usize, f32);

    fn next(&mut self) -> Option<Self::Item> {
        while self.src < self.adj.len() {
            let edges = &self.adj[self.src];
            while self.pos < edges.len() {
                let (tgt32, w) = edges[self.pos];
                self.pos += 1;
                let tgt = tgt32 as usize;
                if tgt >= self.src {
                    return Some((self.src, tgt, w));
                }
            }
            self.src += 1;
            self.pos = 0;
        }
        None
    }
}

impl Network {
    /// Create a new empty network.
    #[must_use]
    pub fn new() -> Network {
        Network::with_capacity(0)
    }

    /// Create a new empty network with capacity for `n_nodes` nodes.
    #[must_use]
    pub fn with_capacity(n_nodes: usize) -> Network {
        Network {
            adj: Vec::with_capacity(n_nodes),
            node_weights: Vec::with_capacity(n_nodes),
            edge_count: 0,
        }
    }

    /// Append a node with `weight`. Returns its node id.
    pub fn add_node(&mut self, weight: f32) -> usize {
        let id = self.node_weights.len();
        self.node_weights.push(weight);
        self.adj.push(Vec::new());
        id
    }

    /// Add an undirected edge between `source` and `target` with `weight`.
    ///
    /// # Panics
    /// If `source` or `target` exceeds `u32::MAX`, or is out of range.
    pub fn add_edge(&mut self, source: usize, target: usize, weight: f32) {
        let s32 = u32::try_from(source).expect("node index exceeds u32::MAX");
        let t32 = u32::try_from(target).expect("node index exceeds u32::MAX");
        self.adj[source].push((t32, weight));
        self.adj[target].push((s32, weight));
        self.edge_count += 1;
    }

    /// Number of nodes.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.node_weights.len()
    }

    /// Number of undirected edges (each pair counted once).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Get the weight of `node` as `f64`.
    ///
    /// # Panics
    /// If `node >= nodes()`.
    #[must_use]
    pub fn weight(&self, node: usize) -> f64 {
        f64::from(self.node_weights[node])
    }

    /// Mutable access to a node's weight.
    ///
    /// # Panics
    /// If `node >= nodes()`.
    pub fn node_weight_mut(&mut self, node: usize) -> &mut f32 {
        &mut self.node_weights[node]
    }

    /// Iterator over `(neighbour_id, edge_weight)` for all neighbours of `node`.
    ///
    /// # Panics
    /// If `node >= nodes()`.
    #[must_use]
    pub fn neighbors(&self, node: usize) -> NeighborAndWeightIter<'_> {
        NeighborAndWeightIter {
            iter: self.adj[node].iter(),
        }
    }

    /// Iterator over each undirected edge once.
    pub fn edge_references(&self) -> EdgeReferences<'_> {
        EdgeReferences {
            adj: &self.adj,
            src: 0,
            pos: 0,
        }
    }

    /// Sum of all node weights.
    #[must_use]
    pub fn get_total_node_weight(&self) -> f64 {
        let mut w = 0.0;
        for i in 0..self.nodes() {
            w += self.weight(i);
        }
        w
    }

    /// Sum of all edge weights (each undirected edge counted once).
    #[must_use]
    pub fn get_total_edge_weight(&self) -> f64 {
        let mut s = 0.0;
        for (src, edges) in self.adj.iter().enumerate() {
            for &(t, w) in edges {
                if t as usize >= src {
                    s += f64::from(w);
                }
            }
        }
        s
    }

    /// Parallel total edge weight. Sums the double-counted weights then divides by 2.
    /// Chunked + serially reduced for determinism.
    #[must_use]
    pub fn get_total_edge_weight_par(&self) -> f64 {
        let mut partial_sums = vec![];

        self.adj
            .par_chunks(256)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|edges| edges.iter().fold(0.0, |a, &(_, w)| a + f64::from(w)))
                    .sum::<f64>()
            })
            .collect_into_vec(&mut partial_sums);

        partial_sums.iter().sum::<f64>() / 2.0
    }

    /// Tabulate the total edge weight of each node into `result`.
    pub fn get_total_edge_weight_per_node(&self, result: &mut Vec<f64>) {
        result.clear();
        for edges in &self.adj {
            let w = edges.iter().fold(0.0, |a, &(_, w)| a + f64::from(w));
            result.push(w);
        }
    }

    /// Aggregate network where each cluster becomes a single node.
    ///
    /// Node weights are summed within each cluster. Edge weights between
    /// distinct clusters are summed. Within-cluster edges are dropped — only
    /// the inter-cluster summary is preserved.
    ///
    /// # Panics
    /// If a cluster id exceeds `u32::MAX`.
    #[must_use]
    pub fn create_reduced_network(&self, clustering: &impl Clustering) -> Network {
        let mut cluster_g = Network::with_capacity(clustering.num_clusters());

        for _ in 0..clustering.num_clusters() {
            cluster_g.add_node(0.0);
        }

        for n in 0..self.nodes() {
            let cluster = clustering.get(n);
            cluster_g.node_weights[cluster] += self.node_weights[n];
        }

        let mut edge_memo: HashMap<(u32, u32), f32> = HashMap::default();

        for (src, tgt, w) in self.edge_references() {
            let c1 = u32::try_from(clustering.get(src)).expect("cluster id exceeds u32::MAX");
            let c2 = u32::try_from(clustering.get(tgt)).expect("cluster id exceeds u32::MAX");

            if c1 == c2 {
                continue;
            }

            let (mn, mx) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            *edge_memo.entry((mn, mx)).or_insert(0.0) += w;
        }

        for (&(c1, c2), &weight) in &edge_memo {
            cluster_g.add_edge(c1 as usize, c2 as usize, weight);
        }

        cluster_g
    }

    /// One subnetwork per cluster, containing only intra-cluster edges.
    ///
    /// # Panics
    /// If a node index exceeds `u32::MAX`.
    pub fn create_subnetworks(&self, c: &impl Clustering) -> Vec<Network> {
        let mut graphs: Vec<Network> = (0..c.num_clusters())
            .map(|_| Network::with_capacity(0))
            .collect();
        let mut new_id_map = Vec::with_capacity(c.nodes());
        let mut counts = vec![0usize; c.num_clusters()];

        for i in 0..self.nodes() {
            let cluster = c.get(i);
            let new_id = counts[cluster];
            new_id_map.push(new_id);
            counts[cluster] += 1;
            graphs[cluster].add_node(self.node_weights[i]);
        }

        for (n1, n2, w) in self.edge_references() {
            let c1 = c.get(n1);
            let c2 = c.get(n2);
            if c1 == c2 {
                graphs[c1].add_edge(new_id_map[n1], new_id_map[n2], w);
            }
        }

        graphs
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}
