use crate::util::common::*;
use crate::util::graph_refine::refine_labels;
use crate::util::knn_graph::KnnGraph;
use data_beans_alg::union_find::UnionFind;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use std::cmp::Ordering;
use std::collections::VecDeque;

#[derive(Clone)]
struct CoarsenLevelResult {
    pair_to_sample: Vec<usize>,
    num_samples: usize,
    cell_labels: Vec<usize>,
}

/// Result of graph coarsening: a sequence of merges forming a dendrogram.
pub struct CoarsenResult {
    /// Each entry (a, b) records that groups rooted at a and b were merged.
    pub(super) merges: Vec<(usize, usize)>,
    /// Total number of original nodes.
    pub(super) n_nodes: usize,
}

/// Degree-corrected modularity-gain merge veto.
///
/// Reject a proposed merge `(i, j)` with similarity `sim(i,j)` when
/// `sim(i,j) < gamma · deg(i) · deg(j) / (2W)`, the standard Louvain/Leiden
/// merge-gain criterion adapted to a cosine-weighted adjacency. Weights are
/// clamped at zero so anti-correlated cell pairs never contribute to the
/// null.
///
/// `gamma = 1.0` is a natural default (modularity resolution). Set to `0.0`
/// to accept any proposal with non-negative similarity, or pass `None` to
/// the coarsener to disable the veto entirely.
#[derive(Clone, Copy, Debug)]
pub struct ModularityVeto {
    pub gamma: f32,
}

impl Default for ModularityVeto {
    fn default() -> Self {
        Self { gamma: 1.0 }
    }
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
/// * `cell_features` - `[proj_dim × n_cells]` matrix with L2-normalized columns;
///   mutated in place as merged centroids replace absorbed columns.
/// * `veto` - optional [`ModularityVeto`] — when supplied, proposals are
///   rejected if their similarity is no better than expected given each
///   endpoint's weighted degree in the current super-graph.
pub fn graph_coarsen(
    graph: &KnnGraph,
    cell_features: &mut Mat,
    veto: Option<&ModularityVeto>,
) -> CoarsenResult {
    let n = graph.n_nodes;
    let dim = cell_features.nrows();

    let mut uf = UnionFind::new(n);
    let mut adj = build_adjacency(n, &graph.edges);

    let mut alive = vec![true; n];
    let max_merges = n.saturating_sub(1);
    let mut merges = Vec::with_capacity(max_merges);

    let pb = new_progress_bar(
        max_merges as u64,
        "Coarsening {bar:40} {pos}/{len} merges ({eta})",
    );

    let mut edges: Vec<(usize, usize)> = graph.edges.clone();
    let mut edge_set: HashSet<(usize, usize)> = HashSet::default();

    let mut active: Vec<usize> = (0..n).filter(|&i| !adj[i].is_empty()).collect();
    let mut taken = vec![false; n];
    let mut deg_cos = vec![0.0f32; n];
    let mut total_w = 0.0f32;
    let mut total_vetoed = 0usize;
    let mut n_rounds = 0u32;

    loop {
        if active.is_empty() {
            break;
        }

        // Recompute per-node weighted degree and total_w from the current
        // super-graph edge list. Negative cosines clamp to zero so the null
        // model doesn't credit anti-correlated neighborhoods.
        if veto.is_some() {
            for d in deg_cos.iter_mut().take(n) {
                *d = 0.0;
            }
            total_w = 0.0;
            for &(i, j) in &edges {
                let s = cell_features
                    .column(i)
                    .dot(&cell_features.column(j))
                    .max(0.0);
                deg_cos[i] += s;
                deg_cos[j] += s;
                total_w += s;
            }
        }

        let features_ref = &*cell_features;
        let adj_ref = &adj;
        let veto_ref = veto;
        let deg_ref = &deg_cos;
        let total_w_ref = total_w;

        let proposals: Vec<(usize, usize, f32)> = active
            .par_iter()
            .filter_map(|&i| {
                let mut best_j = 0usize;
                let mut best_sim = f32::NEG_INFINITY;
                for &j in &adj_ref[i] {
                    let sim = features_ref.column(i).dot(&features_ref.column(j));
                    if sim > best_sim || (sim == best_sim && j < best_j) {
                        best_sim = sim;
                        best_j = j;
                    }
                }
                if !best_sim.is_finite() {
                    return None;
                }
                // Degree-corrected modularity-gain veto. Require strictly
                // positive similarity (no merging anti-correlated or
                // orthogonal pairs) AND strict gain over the null rate.
                if let Some(v) = veto_ref {
                    if best_sim <= 0.0 {
                        return None;
                    }
                    let expected = if total_w_ref > 0.0 {
                        v.gamma * deg_ref[i] * deg_ref[best_j] / (2.0 * total_w_ref)
                    } else {
                        0.0
                    };
                    if best_sim <= expected {
                        return None;
                    }
                }
                Some((i, best_j, best_sim))
            })
            .collect();

        let n_accepted_proposals = proposals.len();
        if veto.is_some() {
            total_vetoed += active.len() - n_accepted_proposals;
        }

        let mut sorted = proposals;
        sorted.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

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

            let size_a = uf.size(a) as f32;
            let size_b = uf.size(b) as f32;
            let total = size_a + size_b;

            let new_rep = uf.union(a, b);
            let absorbed = if new_rep == a { b } else { a };

            for r in 0..dim {
                cell_features[(r, new_rep)] =
                    (cell_features[(r, a)] * size_a + cell_features[(r, b)] * size_b) / total;
            }
            cell_features.column_mut(new_rep).normalize_mut();

            alive[absorbed] = false;
        }

        uf.flatten();

        // Rebuild edge list Leiden-style: resolve endpoints, drop intra-cluster
        // edges, dedupe — the HashSet is retained across rounds to amortize its
        // capacity, since edge counts fall geometrically.
        edge_set.clear();
        let mut write = 0;
        for read in 0..edges.len() {
            let (i, j) = edges[read];
            let ri = uf.parent(i);
            let rj = uf.parent(j);
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

        for &i in &active {
            adj[i].clear();
        }
        for &(i, j) in &edges {
            adj[i].push(j);
            adj[j].push(i);
        }

        active.retain(|&i| alive[i] && !adj[i].is_empty());

        n_rounds += 1;
    }

    pb.finish_and_clear();
    if veto.is_some() {
        info!(
            "Graph coarsening: {} nodes, {} merges in {} rounds ({} proposals vetoed)",
            n,
            merges.len(),
            n_rounds,
            total_vetoed
        );
    } else {
        info!(
            "Graph coarsening: {} nodes, {} merges in {} rounds",
            n,
            merges.len(),
            n_rounds
        );
    }

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
pub(super) fn spatial_seed_labels(
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
pub(super) fn build_super_graph(
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

/// Inputs required to run DC-Poisson refinement per level.
///
/// Bundled as a single struct so the three fields — scoring params, raw
/// counts, and feature count — are either all present or all absent. No
/// runtime check needed: `CoarsenConfig::dc_poisson: Option<DcPoissonConfig>`
/// makes the joint nature explicit in the type.
pub struct DcPoissonConfig<'a> {
    pub params: data_beans_alg::dc_poisson::RefineParams,
    pub data: &'a SparseIoVec,
    pub num_genes: usize,
}

/// Configuration for multi-level graph coarsening.
pub struct CoarsenConfig<'a> {
    /// Finest-level target cluster count.
    pub n_clusters: usize,
    /// Number of dendrogram cuts to extract, linearly spaced coarsest → finest.
    pub num_levels: usize,
    /// Leiden-style refinement sweeps run at each extracted level (0 disables).
    pub refine_iterations: usize,
    /// Spatial grid seeding pre-pass for large datasets (see [`SeedingParams`]).
    pub seeding: Option<SeedingParams<'a>>,
    /// Optional degree-corrected modularity-gain veto on merge proposals.
    /// `None` disables the veto (legacy behaviour — accept any sim).
    pub modularity_veto: Option<ModularityVeto>,
    /// Optional gene-level DC-Poisson refinement per level.
    pub dc_poisson: Option<DcPoissonConfig<'a>>,
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
/// then cuts the dendrogram at `config.num_levels` linearly-spaced cluster
/// counts from coarsest to `config.n_clusters`.
///
/// When `config.seeding` is provided and n > 20k, a spatial grid seeding
/// pre-pass reduces the problem to ~8× n_clusters super-nodes before
/// running the expression-guided agglomerative coarsening.
///
/// Returns per-level pair-to-sample mappings ready for the fused or
/// per-sample visitors.
pub fn graph_coarsen_multilevel(
    graph: &KnnGraph,
    cell_features: &mut Mat,
    pairs: &[Pair],
    config: CoarsenConfig<'_>,
) -> MultiLevelCoarsenResult {
    let CoarsenConfig {
        n_clusters,
        num_levels,
        refine_iterations,
        seeding,
        modularity_veto,
        dc_poisson,
    } = config;
    let veto_ref = modularity_veto.as_ref();
    let n_original = graph.n_nodes;

    // Optional seeding pre-pass for large datasets.
    // We snapshot the L2-normalised pre-agglomeration features and (if seeded)
    // the super-graph so refinement can run on the same level the agglomeration
    // ran on.
    let seed_labels: Option<Vec<usize>>;
    let coarsen_result;
    let coarsen_features: Mat;
    let coarsen_graph_owned: Option<KnnGraph>;

    if let Some(sp) = seeding.filter(|_| n_original > SEEDING_THRESHOLD) {
        let target_super = (n_clusters * 8).min(n_original / 4).max(n_clusters + 1);
        let (labels, num_super) =
            spatial_seed_labels(sp.coordinates, graph, target_super, sp.batch_membership);
        let (super_graph, mut super_features) =
            build_super_graph(&labels, num_super, graph, cell_features);

        super_features.normalize_columns_inplace();
        let snapshot = super_features.clone();

        coarsen_result = graph_coarsen(&super_graph, &mut super_features, veto_ref);
        coarsen_features = snapshot;
        seed_labels = Some(labels);
        coarsen_graph_owned = Some(super_graph);
    } else {
        cell_features.normalize_columns_inplace();
        let snapshot = cell_features.clone();

        coarsen_result = graph_coarsen(graph, cell_features, veto_ref);
        coarsen_features = snapshot;
        seed_labels = None;
        coarsen_graph_owned = None;
    }

    let coarsen_graph: &KnnGraph = coarsen_graph_owned.as_ref().unwrap_or(graph);

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

    // Build DC-Poisson context once if configured. The entity axis (super-
    // nodes when seeded, cells otherwise) and the raw-count-derived profiles
    // + guard are identical across all levels, so pay the I/O cost once.
    let dc_poisson_ctx: Option<crate::util::graph_dc_poisson_refine::DcPoissonContext<'_>> =
        if let Some(cfg) = dc_poisson.as_ref() {
            let cell_to_entity: Vec<usize> = if let Some(ref sl) = seed_labels {
                sl.clone()
            } else {
                (0..n_original).collect()
            };
            Some(
                crate::util::graph_dc_poisson_refine::DcPoissonContext::build(
                    cfg.data,
                    graph,
                    cell_to_entity,
                    n_coarsen,
                    cfg.num_genes,
                    cfg.params.gene_weighting,
                )
                .expect("DC-Poisson context build failed"),
            )
        } else {
            None
        };

    // Previous (next-coarser) level's refined entity labels drive the sibling
    // constraint. `None` at the coarsest level.
    let mut prev_entity_labels: Option<Vec<usize>> = None;

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
        let mut coarse_labels: Vec<usize> = (0..n_coarsen)
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

        // Refine this level's clustering. Per-level seed keeps levels reproducible
        // and independent.
        if refine_iterations > 0 {
            let seed = (orig_idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
            refine_labels(
                &coarsen_features,
                coarsen_graph,
                &mut coarse_labels,
                refine_iterations,
                seed,
            );
        }

        // Gene-level DC-Poisson refinement (second opinion on raw counts).
        if let (Some(ctx), Some(cfg)) = (dc_poisson_ctx.as_ref(), dc_poisson.as_ref()) {
            use rand::SeedableRng;
            let level_seed = cfg
                .params
                .seed
                .wrapping_add((orig_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let mut rng = rand::rngs::SmallRng::seed_from_u64(level_seed);
            let level_label = format!(
                "DC-Poisson L{}/{}",
                level_n_clusters.len() - orig_idx,
                level_n_clusters.len()
            );
            let moves = crate::util::graph_dc_poisson_refine::refine_level_dc_poisson(
                ctx,
                &mut coarse_labels,
                prev_entity_labels.as_deref(),
                &cfg.params,
                &mut rng,
                &level_label,
            );
            info!("  level {} DC-Poisson refined: {} moves", orig_idx, moves);
            prev_entity_labels = Some(coarse_labels.clone());
        }

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
