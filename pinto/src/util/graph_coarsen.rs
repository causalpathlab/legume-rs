use crate::util::common::*;
use crate::util::knn_graph::KnnGraph;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use std::cmp::Ordering;
use std::collections::VecDeque;

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

#[inline]
fn dot_columns(features: &Mat, a: usize, b: usize) -> f32 {
    features.column(a).dot(&features.column(b))
}

/// Build symmetric adjacency lists from edge pairs.
fn build_adjacency(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j) in edges {
        adj[i].push(j);
        adj[j].push(i);
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    adj
}

/// Parallel graph-constrained agglomerative coarsening.
///
/// Uses a Leiden-style parallel sweep strategy:
/// 1. Each active root finds its best neighbor by cosine similarity (parallel)
/// 2. Greedy matching accepts non-conflicting merges sorted by similarity
/// 3. Batch-apply merges: update Union-Find, features, adjacency
/// 4. Repeat until no more merges possible
///
/// * `graph` - spatial KNN graph
/// * `cell_features` - `[proj_dim × n_cells]` matrix, mutated for centroid updates
pub fn graph_coarsen(graph: &KnnGraph, cell_features: &mut Mat) -> CoarsenResult {
    let n = graph.n_nodes;
    let dim = cell_features.nrows();

    // L2-normalize all feature columns
    for i in 0..n {
        normalize_column_inplace(cell_features, i);
    }

    let mut uf = UnionFind::new(n);
    let mut adj = build_adjacency(n, &graph.edges);

    let mut alive = vec![true; n]; // false once absorbed
    let max_merges = n.saturating_sub(1);
    let mut merges = Vec::with_capacity(max_merges);

    let pb = new_progress_bar(
        max_merges as u64,
        "Coarsening {bar:40} {pos}/{len} merges ({eta})",
    );

    // Working edge list — shrinks each round as intra-cluster edges vanish.
    let mut edges: Vec<(usize, usize)> = graph.edges.clone();
    // Reusable HashSet for edge dedup (FxHashSet via common)
    let mut edge_set: HashSet<(usize, usize)> = HashSet::default();

    let mut active: Vec<usize> = (0..n).filter(|&i| !adj[i].is_empty()).collect();
    let mut taken = vec![false; n];
    let mut n_rounds = 0u32;

    loop {
        if active.is_empty() {
            break;
        }

        // Parallel: each active node finds its best neighbor by cosine similarity.
        let features_ref = &*cell_features;
        let adj_ref = &adj;

        let proposals: Vec<(usize, usize, f32)> = active
            .par_iter()
            .filter_map(|&i| {
                let mut best_j = 0usize;
                let mut best_sim = f32::NEG_INFINITY;
                for &j in &adj_ref[i] {
                    let sim = dot_columns(features_ref, i, j);
                    if sim > best_sim || (sim == best_sim && j < best_j) {
                        best_sim = sim;
                        best_j = j;
                    }
                }
                best_sim.is_finite().then_some((i, best_j, best_sim))
            })
            .collect();

        // Sort by similarity descending for greedy matching
        let mut sorted = proposals;
        sorted.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

        // Greedy matching: accept merges where neither endpoint is already taken
        let mut round_merges: Vec<(usize, usize)> = Vec::new();
        for (a, b, _) in sorted {
            if !taken[a] && !taken[b] {
                taken[a] = true;
                taken[b] = true;
                round_merges.push((a, b));
            }
        }
        for &(a, b) in &round_merges {
            taken[a] = false;
            taken[b] = false;
        }

        if round_merges.is_empty() {
            break;
        }

        // UF merges + feature updates (sequential)
        for &(a, b) in &round_merges {
            merges.push((a, b));
            pb.inc(1);

            let size_a = uf.size[a] as f32;
            let size_b = uf.size[b] as f32;
            let total = size_a + size_b;

            let new_rep = uf.union(a, b);
            let absorbed = if new_rep == a { b } else { a };

            for r in 0..dim {
                cell_features[(r, new_rep)] =
                    (cell_features[(r, a)] * size_a + cell_features[(r, b)] * size_b) / total;
            }
            normalize_column_inplace(cell_features, new_rep);

            alive[absorbed] = false;
        }

        // Flatten UF for O(1) lookups
        for i in 0..n {
            uf.parent[i] = uf.find(i);
        }

        // Rebuild edge list (Leiden-style): resolve endpoints, drop
        // intra-cluster edges, deduplicate via FxHashSet.
        edge_set.clear();
        let mut write = 0;
        for read in 0..edges.len() {
            let (i, j) = edges[read];
            let ri = uf.parent[i];
            let rj = uf.parent[j];
            if ri == rj {
                continue;
            }
            let key = (ri.min(rj), ri.max(rj));
            if edge_set.insert(key) {
                edges[write] = key;
                write += 1;
            }
        }
        edges.truncate(write);

        // Rebuild adjacency from clean edge list
        for &i in &active {
            adj[i].clear();
        }
        for &(i, j) in &edges {
            adj[i].push(j);
            adj[j].push(i);
        }

        // Update active set
        active.retain(|&i| alive[i] && !adj[i].is_empty());

        n_rounds += 1;
    }

    pb.finish_and_clear();
    info!(
        "Graph coarsening: {} nodes, {} merges in {} rounds",
        n,
        merges.len(),
        n_rounds
    );

    CoarsenResult { merges, n_nodes: n }
}

/// Map cell labels to pair sample indices.
///
/// Each pair (i,j) maps to a canonical sample key `(min(label[i], label[j]), max(...))`.
/// Returns `(pair_to_sample, n_samples)`.
pub fn cell_labels_to_pair_samples(cell_labels: &[usize], pairs: &[Pair]) -> (Vec<usize>, usize) {
    let mut pair_key_to_sample: HashMap<(usize, usize), usize> = Default::default();
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

/// Threshold: if n_nodes > this, use spatial seeding pre-pass.
const SEEDING_THRESHOLD: usize = 20_000;

/// Assign cells to spatially-seeded super-nodes via BFS flood-fill.
///
/// Places seeds on a regular grid over each batch's bounding box
/// (different batches have independent coordinate systems), then
/// expands all seeds simultaneously along graph edges.
///
/// Returns `(labels, num_super_nodes)` where `labels[cell] = super_node_id`.
fn spatial_seed_labels(
    coordinates: &Mat,
    graph: &KnnGraph,
    target_super: usize,
    batch_membership: Option<&[Box<str>]>,
) -> (Vec<usize>, usize) {
    let n = graph.n_nodes;
    let n_dims = coordinates.ncols();

    // Partition cells by batch (single pass over cells)
    let n_batches: usize;
    let mut batch_cells: Vec<Vec<usize>>;
    if let Some(batches) = batch_membership {
        let mut batch_map: HashMap<&str, usize> = Default::default();
        let mut next_id = 0usize;
        batch_cells = Vec::new();
        for (i, b) in batches.iter().enumerate() {
            let bid = *batch_map.entry(b.as_ref()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                batch_cells.push(Vec::new());
                id
            });
            batch_cells[bid].push(i);
        }
        n_batches = next_id;
    } else {
        batch_cells = vec![(0..n).collect()];
        n_batches = 1;
    }

    let mut seeds: Vec<usize> = Vec::new();
    let mut labels = vec![usize::MAX; n];
    let mut num_super = 0usize;

    for cells in &batch_cells {
        let batch_target = (target_super as f64 * cells.len() as f64 / n as f64).ceil() as usize;
        let batch_target = batch_target.max(1);

        // Bounding box for this batch
        let mut mins = vec![f32::INFINITY; n_dims];
        let mut maxs = vec![f32::NEG_INFINITY; n_dims];
        for &i in cells {
            for d in 0..n_dims {
                let v = coordinates[(i, d)];
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }

        // Grid dimensions
        let side = (batch_target as f64).powf(1.0 / n_dims as f64).ceil() as usize;
        let side = side.max(1);
        let mut tile_strides = vec![1usize; n_dims];
        for d in (0..n_dims - 1).rev() {
            tile_strides[d] = tile_strides[d + 1] * side;
        }
        let total_tiles = tile_strides[0] * side;

        // Assign cells in this batch to tiles, pick one seed per tile
        let mut tile_seed: Vec<Option<usize>> = vec![None; total_tiles];
        for &i in cells {
            let mut tile_id = 0usize;
            for d in 0..n_dims {
                let range = (maxs[d] - mins[d]).max(1e-12);
                let frac = ((coordinates[(i, d)] - mins[d]) / range).clamp(0.0, 0.999999);
                let bin = (frac * side as f32) as usize;
                tile_id += bin * tile_strides[d];
            }
            if tile_seed[tile_id].is_none() {
                tile_seed[tile_id] = Some(i);
            }
        }

        for seed in tile_seed.into_iter().flatten() {
            labels[seed] = num_super;
            seeds.push(seed);
            num_super += 1;
        }
    }

    // Capacity-limited multi-source BFS: each super-node absorbs at most
    // 2× the average cell count to keep sizes uniform across density variation.
    let max_cells = (n / num_super.max(1)) * 2;
    let mut seed_counts = vec![1usize; num_super]; // each seed is one cell

    let mut queue = VecDeque::with_capacity(n);
    let mut spillover: VecDeque<usize> = VecDeque::new();
    for &cell in &seeds {
        queue.push_back(cell);
    }
    let csc = &graph.adjacency;

    while let Some(cell) = queue.pop_front() {
        let label = labels[cell];
        if seed_counts[label] >= max_cells {
            continue;
        }
        let col = csc.col(cell);
        for &nb in col.row_indices() {
            if labels[nb] == usize::MAX {
                labels[nb] = label;
                seed_counts[label] += 1;
                if seed_counts[label] < max_cells {
                    queue.push_back(nb);
                } else {
                    // Seed is now full — collect overflow for uncapped spillover
                    spillover.push_back(nb);
                }
            }
        }
    }

    // Propagate remaining unassigned cells without capacity limit.
    while let Some(cell) = spillover.pop_front() {
        let label = labels[cell];
        let col = csc.col(cell);
        for &nb in col.row_indices() {
            if labels[nb] == usize::MAX {
                labels[nb] = label;
                spillover.push_back(nb);
            }
        }
    }

    // Truly disconnected cells get their own singleton super-node
    for label in labels.iter_mut() {
        if *label == usize::MAX {
            *label = num_super;
            num_super += 1;
        }
    }

    info!(
        "Spatial seeding: {} cells, {} batches → {} super-nodes (target {})",
        n, n_batches, num_super, target_super
    );

    (labels, num_super)
}

/// Build a super-graph from seed labels.
///
/// Averages cell features per super-node and creates edges between
/// adjacent super-nodes (from original graph edges crossing boundaries).
fn build_super_graph(
    seed_labels: &[usize],
    num_super: usize,
    graph: &KnnGraph,
    cell_features: &Mat,
) -> (KnnGraph, Mat) {
    let dim = cell_features.nrows();

    // Aggregate features: sum columns per super-node via column ops
    let mut super_features = Mat::zeros(dim, num_super);
    let mut counts = vec![0usize; num_super];
    for (cell, &label) in seed_labels.iter().enumerate() {
        counts[label] += 1;
        let mut col = super_features.column_mut(label);
        col += cell_features.column(cell);
    }
    for (s, &c) in counts.iter().enumerate() {
        if c > 0 {
            super_features.column_mut(s).scale_mut(1.0 / c as f32);
        }
    }

    let est_edges = graph.edges.len();
    let mut edge_set: HashSet<(usize, usize)> =
        HashSet::with_capacity_and_hasher(est_edges, Default::default());
    let mut super_edges: Vec<(usize, usize)> = Vec::with_capacity(est_edges);
    for &(i, j) in &graph.edges {
        let si = seed_labels[i];
        let sj = seed_labels[j];
        if si != sj {
            let key = (si.min(sj), si.max(sj));
            if edge_set.insert(key) {
                super_edges.push(key);
            }
        }
    }

    let distances = vec![1.0f32; super_edges.len()];

    let mut coo = CooMatrix::new(num_super, num_super);
    for &(i, j) in &super_edges {
        coo.push(i, j, 1.0f32);
        coo.push(j, i, 1.0f32);
    }
    let adjacency = CscMatrix::from(&coo);

    let super_graph = KnnGraph {
        adjacency,
        edges: super_edges,
        distances,
        n_nodes: num_super,
    };

    info!(
        "Super-graph: {} nodes, {} edges",
        num_super,
        super_graph.edges.len()
    );

    (super_graph, super_features)
}

/// Optional spatial seeding parameters for large datasets.
pub struct SeedingParams<'a> {
    pub coordinates: &'a Mat,
    pub batch_membership: Option<&'a [Box<str>]>,
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
/// When `coordinates` is provided and n > 20k, a spatial grid seeding
/// pre-pass reduces the problem to ~8× n_clusters super-nodes before
/// running the expression-guided agglomerative coarsening.
///
/// Returns per-level pair-to-sample mappings ready for the fused or
/// per-sample visitors.
pub fn graph_coarsen_multilevel(
    graph: &KnnGraph,
    cell_features: &mut Mat,
    pairs: &[Pair],
    n_clusters: usize,
    num_levels: usize,
    seeding: Option<SeedingParams<'_>>,
) -> MultiLevelCoarsenResult {
    let n_original = graph.n_nodes;

    // Optional seeding pre-pass for large datasets
    let seed_labels: Option<Vec<usize>>;
    let coarsen_result;

    if let Some(sp) = seeding.filter(|_| n_original > SEEDING_THRESHOLD) {
        let target_super = (n_clusters * 8).min(n_original / 4).max(n_clusters + 1);
        let (labels, num_super) =
            spatial_seed_labels(sp.coordinates, graph, target_super, sp.batch_membership);
        let (super_graph, mut super_features) =
            build_super_graph(&labels, num_super, graph, cell_features);
        coarsen_result = graph_coarsen(&super_graph, &mut super_features);
        seed_labels = Some(labels);
    } else {
        coarsen_result = graph_coarsen(graph, cell_features);
        seed_labels = None;
    }

    let level_n_clusters = compute_level_n_clusters(n_clusters, num_levels);

    // Extract all levels with a single incremental UF pass.
    let n_coarsen = coarsen_result.n_nodes; // super-nodes if seeded, cells otherwise
    let mut sorted: Vec<(usize, usize)> = level_n_clusters
        .iter()
        .enumerate()
        .map(|(i, &nc)| (nc, i))
        .collect();
    sorted.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // descending n_clusters

    let mut uf = UnionFind::new(n_coarsen);
    let mut merge_idx = 0usize;
    let mut results: Vec<Option<CoarsenLevelResult>> = vec![None; level_n_clusters.len()];

    for (nc, orig_idx) in sorted {
        let target = n_coarsen
            .saturating_sub(nc)
            .min(coarsen_result.merges.len());
        while merge_idx < target {
            let (a, b) = coarsen_result.merges[merge_idx];
            uf.union(a, b);
            merge_idx += 1;
        }
        // Compact labels at the coarsened level
        let mut rep_to_label: HashMap<usize, usize> = Default::default();
        let mut next_label = 0usize;
        let coarse_labels: Vec<usize> = (0..n_coarsen)
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

        // Compose with seed labels to get original-cell-level labels
        let cell_labels: Vec<usize> = if let Some(ref sl) = seed_labels {
            (0..n_original).map(|i| coarse_labels[sl[i]]).collect()
        } else {
            coarse_labels
        };

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
            None,
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

        let ml = graph_coarsen_multilevel(&graph, &mut features, &pairs, 2, 1, None);
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
        let ml = graph_coarsen_multilevel(&graph, &mut features, &pairs, 3, 2, None);

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
            None,
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
        let (super_graph, super_features) =
            build_super_graph(&labels, num_super, &graph, &features);
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
        let ml = graph_coarsen_multilevel(&graph, &mut features, &pairs, 10, 2, Some(sp));
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
}
