use crate::srt_common::*;
use crate::srt_knn_graph::KnnGraph;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone)]
struct CoarsenLevelResult {
    pair_to_sample: Vec<usize>,
    num_samples: usize,
    cell_labels: Vec<usize>,
}

/// Union-Find (disjoint set) with path halving and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    #[inline]
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    #[inline]
    fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (big, small) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
        if self.rank[big] == self.rank[small] {
            self.rank[big] += 1;
        }
        big
    }
}

/// A candidate merge for the priority queue (max-heap by similarity).
struct MergeCandidate {
    similarity: f32,
    node_a: usize,
    node_b: usize,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.similarity
            .partial_cmp(&other.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

/// Result of graph coarsening: a sequence of merges forming a dendrogram.
pub struct CoarsenResult {
    /// Each entry (a, b) records that groups rooted at a and b were merged.
    merges: Vec<(usize, usize)>,
    /// Total number of original nodes.
    n_nodes: usize,
}

/// L2-normalize a column in-place (zero allocation).
#[inline]
fn normalize_column_inplace(features: &mut Mat, col: usize) {
    let dim = features.nrows();
    let mut sq = 0.0f32;
    for r in 0..dim {
        let v = features[(r, col)];
        sq += v * v;
    }
    let norm = sq.sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for r in 0..dim {
            features[(r, col)] *= inv;
        }
    }
}

/// Dot product of two columns (zero allocation).
#[inline]
fn dot_columns(features: &Mat, a: usize, b: usize) -> f32 {
    let dim = features.nrows();
    let mut s = 0.0f32;
    for r in 0..dim {
        s += features[(r, a)] * features[(r, b)];
    }
    s
}

/// Graph-constrained agglomerative coarsening.
///
/// Merges cells along KNN graph edges, prioritized by cosine similarity
/// of their projected feature vectors. Produces a full dendrogram that
/// can be cut at any level.
///
/// * `graph` - spatial KNN graph
/// * `cell_features` - `[proj_dim × n_cells]` matrix, mutated for centroid updates
pub fn graph_coarsen(graph: &KnnGraph, cell_features: &mut Mat) -> CoarsenResult {
    let n = graph.n_nodes;
    let dim = cell_features.nrows();

    // L2-normalize all feature columns in-place
    for i in 0..n {
        normalize_column_inplace(cell_features, i);
    }

    let mut uf = UnionFind::new(n);

    // Build adjacency lists and initial heap
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut heap = BinaryHeap::new();

    for &(i, j) in &graph.edges {
        adj[i].insert(j);
        adj[j].insert(i);
        let sim = dot_columns(cell_features, i, j);
        if sim.is_finite() {
            heap.push(MergeCandidate {
                similarity: sim,
                node_a: i,
                node_b: j,
            });
        }
    }

    let max_merges = n.saturating_sub(1);
    let mut merges = Vec::with_capacity(max_merges);

    let pb = new_progress_bar(
        max_merges as u64,
        "Coarsening {bar:40} {pos}/{len} merges ({eta})",
    );

    while let Some(candidate) = heap.pop() {
        let ra = uf.find(candidate.node_a);
        let rb = uf.find(candidate.node_b);
        if ra == rb {
            continue;
        }

        merges.push((ra, rb));
        pb.inc(1);

        // Weighted average of feature vectors — in-place, zero allocation
        let size_a = uf.size[ra] as f32;
        let size_b = uf.size[rb] as f32;
        let total = size_a + size_b;

        let new_rep = uf.union(ra, rb);
        let absorbed = if new_rep == ra { rb } else { ra };

        // Weighted average: feat[ra] * size_a + feat[rb] * size_b
        // Use ra/rb directly since union doesn't move column data.
        for r in 0..dim {
            cell_features[(r, new_rep)] =
                (cell_features[(r, ra)] * size_a + cell_features[(r, rb)] * size_b) / total;
        }

        normalize_column_inplace(cell_features, new_rep);

        // Merge adjacency: take absorbed's set (frees its allocation)
        let absorbed_neighbors = std::mem::take(&mut adj[absorbed]);
        for &c in &absorbed_neighbors {
            adj[c].remove(&absorbed);
            if c != new_rep {
                adj[c].insert(new_rep);
                adj[new_rep].insert(c);
            }
        }
        drop(absorbed_neighbors);
        adj[new_rep].remove(&absorbed);
        adj[new_rep].remove(&new_rep);

        // Push new edges for merged node — stale check on pop handles duplicates
        for &c in &adj[new_rep] {
            let rc = uf.find(c);
            if rc != new_rep {
                let sim = dot_columns(cell_features, new_rep, rc);
                if sim.is_finite() {
                    heap.push(MergeCandidate {
                        similarity: sim,
                        node_a: new_rep,
                        node_b: rc,
                    });
                }
            }
        }
    }

    pb.finish_and_clear();
    info!(
        "Graph coarsening: {} nodes, {} merges recorded",
        n,
        merges.len()
    );

    CoarsenResult { merges, n_nodes: n }
}

/// Map cell labels to pair sample indices.
///
/// Each pair (i,j) maps to a canonical sample key `(min(label[i], label[j]), max(...))`.
/// Returns `(pair_to_sample, n_samples)`.
pub fn cell_labels_to_pair_samples(cell_labels: &[usize], pairs: &[Pair]) -> (Vec<usize>, usize) {
    let mut pair_key_to_sample: HashMap<(usize, usize), usize> = HashMap::new();
    let mut next_sample = 0usize;

    let pair_to_sample: Vec<usize> = pairs
        .iter()
        .map(|p| {
            let la = cell_labels[p.left];
            let lb = cell_labels[p.right];
            let key = (la.min(lb), la.max(lb));
            *pair_key_to_sample.entry(key).or_insert_with(|| {
                let s = next_sample;
                next_sample += 1;
                s
            })
        })
        .collect();

    (pair_to_sample, next_sample)
}

/// Compute target cluster counts for multi-level extraction.
///
/// Linearly interpolates from coarsest (16 or fewer) to finest (`n_clusters`).
pub fn compute_level_n_clusters(n_clusters: usize, num_levels: usize) -> Vec<usize> {
    if num_levels <= 1 {
        return vec![n_clusters];
    }
    let coarsest = 16usize.min(n_clusters);
    let mut result = Vec::with_capacity(num_levels);
    for level in 0..num_levels {
        let t = level as f32 / (num_levels - 1) as f32;
        let nc = coarsest as f32 + t * (n_clusters - coarsest) as f32;
        result.push(nc.round() as usize);
    }
    result
}

/// Multi-level graph coarsening result.
pub struct MultiLevelCoarsenResult {
    /// Per-level pair-to-sample mapping (coarse → fine).
    pub all_pair_to_sample: Vec<Vec<usize>>,
    /// Per-level number of distinct samples.
    pub all_num_samples: Vec<usize>,
    /// Per-level cell cluster labels (coarse → fine).
    pub all_cell_labels: Vec<Vec<usize>>,
}

/// Graph-constrained agglomerative coarsening with multi-level extraction.
///
/// Coarsens cells along KNN graph edges prioritized by cosine similarity,
/// then cuts the dendrogram at `num_levels` linearly-spaced cluster
/// counts from coarsest to `n_clusters`.
///
/// Returns per-level pair-to-sample mappings ready for the fused or
/// per-sample visitors.
pub fn graph_coarsen_multilevel(
    graph: &KnnGraph,
    cell_features: &mut Mat,
    pairs: &[Pair],
    n_clusters: usize,
    num_levels: usize,
) -> MultiLevelCoarsenResult {
    let result = graph_coarsen(graph, cell_features);
    let level_n_clusters = compute_level_n_clusters(n_clusters, num_levels);

    // Extract all levels with a single incremental UF pass.
    // Process from finest (most clusters = fewest merges) to coarsest,
    // then reverse to restore coarse→fine order.
    let n = result.n_nodes;
    let mut sorted: Vec<(usize, usize)> = level_n_clusters
        .iter()
        .enumerate()
        .map(|(i, &nc)| (nc, i))
        .collect();
    sorted.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // descending n_clusters

    let mut uf = UnionFind::new(n);
    let mut merge_idx = 0usize;
    let mut results: Vec<Option<CoarsenLevelResult>> = vec![None; level_n_clusters.len()];

    for (nc, orig_idx) in sorted {
        let target = n.saturating_sub(nc).min(result.merges.len());
        while merge_idx < target {
            let (a, b) = result.merges[merge_idx];
            uf.union(a, b);
            merge_idx += 1;
        }
        // Compact labels
        let mut rep_to_label: HashMap<usize, usize> = HashMap::new();
        let mut next_label = 0usize;
        let cell_labels: Vec<usize> = (0..n)
            .map(|i| {
                let r = uf.find(i);
                *rep_to_label.entry(r).or_insert_with(|| {
                    let l = next_label;
                    next_label += 1;
                    l
                })
            })
            .collect();
        rep_to_label.clear();

        let (p2s, ns) = cell_labels_to_pair_samples(&cell_labels, pairs);
        results[orig_idx] = Some(CoarsenLevelResult {
            pair_to_sample: p2s,
            num_samples: ns,
            cell_labels,
        });
    }

    let mut all_pair_to_sample = Vec::with_capacity(level_n_clusters.len());
    let mut all_num_samples = Vec::with_capacity(level_n_clusters.len());
    let mut all_cell_labels = Vec::with_capacity(level_n_clusters.len());
    for slot in results {
        let r = slot.unwrap();
        all_pair_to_sample.push(r.pair_to_sample);
        all_num_samples.push(r.num_samples);
        all_cell_labels.push(r.cell_labels);
    }

    info!(
        "Multi-level coarsening: {} levels, samples per level: {:?}",
        level_n_clusters.len(),
        all_num_samples
    );

    MultiLevelCoarsenResult {
        all_pair_to_sample,
        all_num_samples,
        all_cell_labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    fn make_test_graph(n_nodes: usize, edges: Vec<(usize, usize)>) -> KnnGraph {
        let distances = vec![1.0; edges.len()];

        let mut coo = CooMatrix::new(n_nodes, n_nodes);
        for &(i, j) in &edges {
            coo.push(i, j, 1.0f32);
            coo.push(j, i, 1.0f32);
        }
        let adjacency = CscMatrix::from(&coo);

        KnnGraph {
            adjacency,
            edges,
            distances,
            n_nodes,
        }
    }

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(4), 4);
        assert_eq!(uf.size[0], 1);

        let rep = uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.size[rep], 2);

        let rep2 = uf.union(2, 3);
        assert_eq!(uf.size[rep2], 2);

        let rep3 = uf.union(0, 3);
        assert_eq!(uf.find(0), uf.find(3));
        assert_eq!(uf.find(1), uf.find(2));
        assert_eq!(uf.size[rep3], 4);
    }

    #[test]
    fn test_graph_coarsen_small() {
        // 6 nodes in a path: 0-1-2-3-4-5
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
        let graph = make_test_graph(6, edges);

        let mut features = Mat::zeros(6, 6);
        for i in 0..6 {
            features[(i, i)] = 1.0;
        }

        let result = graph_coarsen(&graph, &mut features);
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
            3,
            1,
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

        let ml = graph_coarsen_multilevel(&graph, &mut features, &pairs, 2, 1);
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
        let ml = graph_coarsen_multilevel(&graph, &mut features, &pairs, 3, 2);

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
            3,
            1,
        );
        assert_eq!(ml1.all_pair_to_sample.len(), 1);
        assert_eq!(ml1.all_num_samples.len(), 1);
    }
}
