use crate::graph::WeightedGraph;
use crate::knn_match::{ColumnDict, SearchScratch};

use dashmap::DashMap;
use indicatif::ParallelProgressIterator;
use log::info;
use nalgebra::DMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use rayon::prelude::*;

const DEFAULT_BLOCK_SIZE: usize = 1000;

pub struct KnnGraph {
    /// Symmetric CSC adjacency matrix (n_nodes x n_nodes)
    pub adjacency: CscMatrix<f32>,
    /// Sorted edge list (i < j), deduplicated
    pub edges: Vec<(usize, usize)>,
    /// Edge distances/weights, parallel to `edges`
    pub distances: Vec<f32>,
    /// Number of nodes
    pub n_nodes: usize,
}

/// Which input graph an edge of a [`KnnGraph::union_with`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeSource {
    Primary,
    Secondary,
    /// Present in both inputs. Still a primary-graph edge for any consumer
    /// filtering on the primary relation, since the primary relation holds.
    Both,
}

/// How to reconcile two graphs' `distances` when merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMerge {
    /// Keep raw values. Correct only when both inputs measure the same thing.
    Raw,
    /// Replace each side's distances with its own within-source quantile rank
    /// in `[0, 1]` before merging. Use this whenever the inputs measure
    /// different things, so the merged column stays comparable across sources
    /// and monotone within each.
    SourceRank,
}

pub struct KnnGraphArgs {
    pub knn: usize,
    pub block_size: usize,
    /// If true, keep only reciprocal edges (i→j AND j→i).
    /// If false, keep union edges (i→j OR j→i), using min distance.
    pub reciprocal: bool,
}

impl KnnGraph {
    /// Build a KNN graph from column vectors.
    ///
    /// * `points` - transposed coordinate matrix (d x n), where each column is a point
    /// * `args` - KNN graph construction parameters
    pub fn from_columns(points: &DMatrix<f32>, args: KnnGraphArgs) -> anyhow::Result<KnnGraph> {
        let nn = points.ncols();
        let points_vec = points.column_iter().collect::<Vec<_>>();
        let names = (0..nn).collect::<Vec<_>>();

        let dict = ColumnDict::from_dvector_views(points_vec, names);
        Self::build_from_dict(dict, nn, &args)
    }

    /// Build a KNN graph from row vectors (cells × features).
    ///
    /// * `data` - matrix (n x d), where each row is a point
    /// * `args` - KNN graph construction parameters
    pub fn from_rows(data: &DMatrix<f32>, args: KnnGraphArgs) -> anyhow::Result<KnnGraph> {
        let transposed = data.transpose();
        Self::from_columns(&transposed, args)
    }

    /// Merge two graphs over the same nodes, keeping every pair exactly once
    /// and reporting which input each came from.
    ///
    /// Borrows both inputs: a caller that unions a spatial graph with an
    /// expression one generally still needs the spatial graph afterwards, as
    /// the topology for anything that reasons about physical adjacency.
    ///
    /// Neither input is assumed sorted, nor assumed to store `i < j`. Edge
    /// order is a constructor invariant here, not a type invariant, and a
    /// hand-built `KnnGraph` can violate both.
    ///
    /// `distances` after a union are NOT a metric. Under
    /// [`DistanceMerge::SourceRank`] they are within-source quantile ranks,
    /// which keeps them comparable across sources without pretending the two
    /// measurements are the same quantity. When an edge is in both inputs the
    /// smaller value wins, matching the `reciprocal: false` convention in
    /// `build_from_dict`.
    pub fn union_with(
        &self,
        other: &KnnGraph,
        policy: DistanceMerge,
    ) -> anyhow::Result<(KnnGraph, Vec<EdgeSource>)> {
        anyhow::ensure!(
            self.n_nodes == other.n_nodes,
            "cannot union graphs over different node counts: {} vs {}",
            self.n_nodes,
            other.n_nodes
        );
        let n_nodes = self.n_nodes;

        let (a_dist, b_dist) = match policy {
            DistanceMerge::Raw => (self.distances.clone(), other.distances.clone()),
            DistanceMerge::SourceRank => (
                within_source_rank(&self.distances),
                within_source_rank(&other.distances),
            ),
        };

        // Sort then fold, NOT a keyed map. At a few million edges an ordered
        // map costs a pointer-chasing O(log n) descent and a node allocation
        // per insert, all of it serial. One flat buffer, one parallel sort and
        // one linear scan replaces that, and measured an order of magnitude
        // faster. The allocation count drops too, though only the time was
        // measured.
        //
        // The canonical key is what makes this a set operation: a pair stored
        // one way round in one input and the other way round in the other must
        // land on the same key. The dedup has to finish before the COO below,
        // which SUMS duplicate entries rather than rejecting them.
        let canonical = |&(i, j): &(usize, usize)| if i <= j { (i, j) } else { (j, i) };
        // Source as a bitmask, so folding a run is an OR rather than a case
        // analysis: 1 = primary, 2 = secondary, 3 = both.
        let mut tagged: Vec<((usize, usize), f32, u8)> =
            Vec::with_capacity(self.edges.len() + other.edges.len());
        tagged.par_extend(
            self.edges
                .par_iter()
                .zip(a_dist.par_iter())
                .map(|(e, &d)| (canonical(e), d, 1u8)),
        );
        tagged.par_extend(
            other
                .edges
                .par_iter()
                .zip(b_dist.par_iter())
                .map(|(e, &d)| (canonical(e), d, 2u8)),
        );
        tagged.par_sort_unstable_by_key(|&(key, _, _)| key);

        let mut edges = Vec::with_capacity(tagged.len());
        let mut distances = Vec::with_capacity(tagged.len());
        let mut source = Vec::with_capacity(tagged.len());
        for &(key, dist, tag) in tagged.iter() {
            if edges.last() == Some(&key) {
                let last = distances.len() - 1;
                distances[last] = f32::min(distances[last], dist);
                source[last] |= tag;
            } else {
                edges.push(key);
                distances.push(dist);
                source.push(tag);
            }
        }
        let source: Vec<EdgeSource> = source
            .into_iter()
            .map(|mask| match mask {
                1 => EdgeSource::Primary,
                2 => EdgeSource::Secondary,
                _ => EdgeSource::Both,
            })
            .collect();

        // Derived state, so rebuild rather than merge.
        let adjacency = symmetric_adjacency(n_nodes, &edges, &distances);

        Ok((
            KnnGraph {
                adjacency,
                edges,
                distances,
                n_nodes,
            },
            source,
        ))
    }

    fn build_from_dict(
        dict: ColumnDict<usize>,
        nn: usize,
        args: &KnnGraphArgs,
    ) -> anyhow::Result<KnnGraph> {
        // `search_others` now returns exactly this many *other* neighbours
        // (self excluded). Clamp to the available others and floor at 1.
        let n_neighbours = args.knn.min(nn.saturating_sub(1)).max(1);

        let jobs = create_jobs(nn, args.block_size);
        let njobs = jobs.len() as u64;

        //////////////////////////////////////////
        // step 1: searching nearest neighbours //
        //////////////////////////////////////////

        let triplets: DashMap<(usize, usize), f32> = DashMap::new();

        // Every bar draws through `crate::progress` so it shares the one
        // `MultiProgress` the log bridge writes above; a bar built straight
        // from indicatif registers with neither and corrupts the log.
        let search_bar = crate::progress::new_progress_bar(njobs).with_message("kNN blocks");
        jobs.into_par_iter()
            .progress_with(search_bar.clone())
            .try_for_each(|(lb, ub)| -> anyhow::Result<()> {
                // One scratch per block, reused across the block's queries to
                // avoid re-growing the approximate index's visited set each call.
                let mut scratch = SearchScratch::default();
                for i in lb..ub {
                    let (_indices, _distances) =
                        dict.search_others_reuse(&i, n_neighbours, &mut scratch)?;
                    for (j, d_ij) in _indices.into_iter().zip(_distances) {
                        triplets.insert((i, j), d_ij);
                    }
                }
                Ok(())
            })?;
        search_bar.finish_and_clear();

        info!("{} triplets by kNN matching", triplets.len());

        if triplets.is_empty() {
            return Err(anyhow::anyhow!("empty triplets"));
        }

        //////////////////////////////////////////////////
        // step 2: edge filtering (reciprocal or union) //
        //////////////////////////////////////////////////

        // Filtering and the sort below ran silent, which on a large pair
        // graph is a half-minute of nothing between the search bar and the
        // next log line, and reads as a hang.
        let filter_bar =
            crate::progress::new_progress_bar(triplets.len() as u64).with_message("edge filtering");
        let mut edges: Vec<((usize, usize), f32)> = if args.reciprocal {
            // Intersection: keep (i,j) only if both i→j and j→i exist
            triplets
                .par_iter()
                .progress_with(filter_bar.clone())
                .filter_map(|entry| {
                    let &(i, j) = entry.key();
                    if i < j && triplets.contains_key(&(j, i)) {
                        Some(((i, j), *entry.value()))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            // Union: keep (i,j) if either i→j or j→i exists, min distance
            triplets
                .par_iter()
                .progress_with(filter_bar.clone())
                .filter_map(|entry| {
                    let &(i, j) = entry.key();
                    if i < j {
                        let d_ij = *entry.value();
                        let d_ji = triplets.get(&(j, i)).map(|e| *e).unwrap_or(d_ij);
                        Some(((i, j), d_ij.min(d_ji)))
                    } else if i > j && !triplets.contains_key(&(j, i)) {
                        // Only (i→j) exists with i > j; emit as canonical (j, i)
                        Some(((j, i), *entry.value()))
                    } else {
                        None
                    }
                })
                .collect()
        };

        filter_bar.finish_and_clear();

        // A parallel sort cannot report position, so this is a spinner:
        // unbounded work, but visibly alive.
        let sort_spin = crate::progress::new_spinner("{spinner} [{elapsed_precise}] {msg}")
            .with_message("sorting and deduplicating edges");
        edges.par_sort_by_key(|&(ij, _)| ij);
        edges.dedup();
        sort_spin.finish_and_clear();

        info!(
            "{} edges after {} matching",
            edges.len(),
            if args.reciprocal {
                "reciprocal"
            } else {
                "union"
            }
        );

        ///////////////////////////////////////////////
        // step 3: construct sparse network backbone //
        ///////////////////////////////////////////////

        let (edge_pairs, distances): (Vec<_>, Vec<_>) = edges.into_iter().unzip();
        let adjacency = symmetric_adjacency(nn, &edge_pairs, &distances);

        Ok(KnnGraph {
            adjacency,
            edges: edge_pairs,
            distances,
            n_nodes: nn,
        })
    }

    /// Get neighbors of a node from the CSC adjacency matrix
    pub fn neighbors(&self, node: usize) -> &[usize] {
        let offsets = self.adjacency.col_offsets();
        let start = offsets[node];
        let end = offsets[node + 1];
        &self.adjacency.row_indices()[start..end]
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Convert distances to similarity weights using an exponential kernel:
    /// `w = exp(-d / σ)` where σ = median distance.
    ///
    /// Returns weights parallel to `self.edges`, all in (0, 1].
    /// Consistent with the softmax(-d) pattern used in counterfactual
    /// inference (data-beans-alg) but with a global bandwidth.
    pub fn exp_kernel_weights(&self) -> Vec<f32> {
        if self.distances.is_empty() {
            return Vec::new();
        }
        let sigma = crate::utils::median(&self.distances);
        let sigma = if sigma <= 0.0 { 1.0 } else { sigma };
        info!("exp_kernel_weights: σ (median distance) = {:.4}", sigma);
        self.distances.iter().map(|&d| (-d / sigma).exp()).collect()
    }

    /// Adaptive-bandwidth kernel weights with local connectivity.
    ///
    /// Per-point sigma calibration (originated in t-SNE, van der Maaten
    /// & Hinton 2008) ensures every node has the same effective number
    /// of neighbors, preventing isolated singletons in sparse regions.
    /// The rho subtraction and fuzzy-union symmetrization follow UMAP
    /// (McInnes et al. 2018), matching the scanpy default for Leiden.
    ///
    /// Algorithm:
    /// 1. rho_i = distance to nearest neighbor (local connectivity)
    /// 2. sigma_i via binary search: sum_j exp(-(d_ij - rho_i)/sigma_i) = log2(k)
    /// 3. Directed weight: w(i→j) = exp(-(d_ij - rho_i) / sigma_i)
    /// 4. Symmetrize: w_sym = w(i→j) + w(j→i) - w(i→j) * w(j→i)
    ///
    /// Returns weights parallel to `self.edges`, all in (0, 1].
    pub fn fuzzy_kernel_weights(&self) -> Vec<f32> {
        if self.distances.is_empty() {
            return Vec::new();
        }

        let offsets = self.adjacency.col_offsets();
        let row_indices = self.adjacency.row_indices();
        let values = self.adjacency.values();

        // Step 1-2: compute rho and sigma per node — independent per node.
        let (rho, sigma): (Vec<f32>, Vec<f32>) = (0..self.n_nodes)
            .into_par_iter()
            .map(|i| {
                let start = offsets[i];
                let end = offsets[i + 1];
                let dists: Vec<f32> = (start..end).map(|idx| values[idx]).collect();
                if dists.is_empty() {
                    return (0.0_f32, 1.0_f32);
                }
                let rho_i = dists.iter().cloned().fold(f32::INFINITY, f32::min);
                let target = (dists.len() as f32).log2();
                let sigma_i = smooth_knn_sigma(&dists, rho_i, target);
                (rho_i, sigma_i)
            })
            .unzip();

        // Step 3-4: compute directed weights and symmetrize per edge —
        // independent per edge, only reads rho/sigma.
        self.edges
            .par_iter()
            .map(|&(i, j)| {
                let d_ij = self.edge_distance_directed(offsets, row_indices, values, i, j);
                let w_ij = directed_umap_weight(d_ij, rho[i], sigma[i]);
                let d_ji = self.edge_distance_directed(offsets, row_indices, values, j, i);
                let w_ji = directed_umap_weight(d_ji, rho[j], sigma[j]);
                // fuzzy union: P(at least one edge) = P(A) + P(B) - P(A)*P(B)
                w_ij + w_ji - w_ij * w_ji
            })
            .collect()
    }

    /// Look up the distance from node `from` to node `to` in the CSC adjacency.
    fn edge_distance_directed(
        &self,
        offsets: &[usize],
        row_indices: &[usize],
        values: &[f32],
        from: usize,
        to: usize,
    ) -> f32 {
        let start = offsets[from];
        let end = offsets[from + 1];
        for idx in start..end {
            if row_indices[idx] == to {
                return values[idx];
            }
        }
        f32::INFINITY
    }
}

impl WeightedGraph for KnnGraph {
    fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    fn num_edges(&self) -> usize {
        self.edges.len()
    }

    fn neighbors_with_weight<'a>(
        &'a self,
        node: usize,
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        let offsets = self.adjacency.col_offsets();
        let start = offsets[node];
        let end = offsets[node + 1];
        let rows = &self.adjacency.row_indices()[start..end];
        let vals = &self.adjacency.values()[start..end];
        Box::new(rows.iter().zip(vals.iter()).map(|(&i, &w)| (i, w)))
    }
}

/// Binary search for per-point sigma (UMAP's smooth_knn_dist).
///
/// Finds sigma such that: sum_j exp(-max(0, d_j - rho) / sigma) = target
fn smooth_knn_sigma(dists: &[f32], rho: f32, target: f32) -> f32 {
    const TOLERANCE: f32 = 1e-5;
    const MAX_ITER: usize = 64;

    let mean_dist: f32 = dists.iter().sum::<f32>() / dists.len().max(1) as f32;
    let min_sigma = 1e-3 * mean_dist;

    let mut lo = 0.0f32;
    let mut hi = f32::INFINITY;
    let mut mid = 1.0f32;

    for _ in 0..MAX_ITER {
        let mut psum = 0.0f32;
        for &d in dists {
            let gap = d - rho;
            if gap > 0.0 {
                psum += (-gap / mid).exp();
            } else {
                psum += 1.0;
            }
        }

        if (psum - target).abs() < TOLERANCE {
            break;
        }

        if psum > target {
            hi = mid;
            mid = (lo + hi) / 2.0;
        } else {
            lo = mid;
            if hi.is_infinite() {
                mid *= 2.0;
            } else {
                mid = (lo + hi) / 2.0;
            }
        }
    }

    mid.max(min_sigma)
}

/// Compute a single directed UMAP membership weight.
fn directed_umap_weight(d: f32, rho: f32, sigma: f32) -> f32 {
    if d.is_infinite() || sigma <= 0.0 {
        return 0.0;
    }
    let gap = d - rho;
    if gap <= 0.0 {
        1.0
    } else {
        (-gap / sigma).exp()
    }
}

////////////////////////
// Leiden integration //
////////////////////////

impl WeightedGraph for leiden::Network {
    fn num_nodes(&self) -> usize {
        self.nodes()
    }

    fn num_edges(&self) -> usize {
        leiden::Network::edge_count(self)
    }

    fn neighbors_with_weight<'a>(
        &'a self,
        node: usize,
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        Box::new(self.neighbors(node).map(|(n, w)| (n, w as f32)))
    }
}

/// Convert a modularity resolution `gamma` to the CPM scale expected by the
/// Leiden crate, given the total undirected edge weight of the graph.
///
/// CPM resolution = `gamma / (2 * total_edge_weight)`. Guards against division
/// by zero for degenerate graphs by clamping the denominator to at least 1.
#[must_use]
pub fn modularity_to_cpm_resolution(modularity_gamma: f64, total_edge_weight: f64) -> f64 {
    modularity_gamma / (2.0 * total_edge_weight).max(1.0)
}

impl KnnGraph {
    /// Convert this KNN graph to a Leiden `Network` with modularity objective.
    ///
    /// Node weights = weighted degree, edge weights = fuzzy kernel weights.
    /// Returns `(network, total_edge_weight)`. Pass `total_edge_weight` to
    /// [`modularity_to_cpm_resolution`] to get a CPM-scale resolution.
    pub fn to_leiden_network(&self) -> (leiden::Network, f64) {
        let n = self.n_nodes;
        let weights = self.fuzzy_kernel_weights();

        let mut node_degree = vec![0.0f32; n];
        let mut total_edge_weight = 0.0f64;
        for (&(i, j), &w) in self.edges.iter().zip(weights.iter()) {
            node_degree[i] += w;
            node_degree[j] += w;
            total_edge_weight += w as f64;
        }

        let mut network = leiden::Network::with_capacity(n);
        for &nd in &node_degree {
            network.add_node(nd);
        }
        for (&(i, j), &w) in self.edges.iter().zip(weights.iter()) {
            network.add_edge(i, j, w);
        }

        (network, total_edge_weight)
    }
}

/// Run Leiden clustering at a fixed (already-scaled) resolution.
///
/// Returns cluster labels as `Vec<usize>` (not necessarily contiguous).
pub fn run_leiden(
    network: &leiden::Network,
    n: usize,
    resolution: f64,
    seed: Option<usize>,
) -> Vec<usize> {
    use leiden::clustering::SimpleClustering;
    use leiden::leiden::Leiden;
    use leiden::Clustering;

    let mut leiden = Leiden::new(resolution, 0.01, seed);
    let mut clustering = SimpleClustering::init_different_clusters(n);

    for iter in 0..10 {
        let updated = leiden.iterate(network, &mut clustering);
        info!(
            "  Leiden iter {}: {} clusters{}",
            iter + 1,
            clustering.num_clusters(),
            if !updated { " (converged)" } else { "" }
        );
        if !updated {
            break;
        }
    }

    (0..n).map(|i| clustering.get(i)).collect()
}

/// Binary search on Leiden resolution to approximate `target_k` clusters.
///
/// `initial_resolution` should already be on the CPM scale
/// (i.e., `modularity_gamma / (2 * total_edge_weight)`).
/// Returns cluster labels (not necessarily contiguous).
pub fn tune_leiden_resolution(
    network: &leiden::Network,
    n: usize,
    target_k: usize,
    initial_resolution: f64,
    seed: Option<usize>,
) -> Vec<usize> {
    let mut lo = 1e-6_f64;
    let mut hi = 10.0_f64;
    let mut best = run_leiden(network, n, initial_resolution, seed);
    let best_k = count_distinct(&best);

    info!(
        "  resolution={:.6e} → {} clusters (target {})",
        initial_resolution, best_k, target_k
    );

    if best_k == target_k {
        return best;
    }
    if best_k > target_k {
        hi = initial_resolution;
    } else {
        lo = initial_resolution;
    }

    let mut best_diff = best_k.abs_diff(target_k);

    for _ in 0..20 {
        let mid = (lo + hi) / 2.0;
        let result = run_leiden(network, n, mid, seed);
        let k = count_distinct(&result);
        info!("  resolution={:.6e} → {} clusters", mid, k);

        if k > target_k {
            hi = mid;
        } else {
            lo = mid;
        }

        let diff = k.abs_diff(target_k);
        if diff < best_diff {
            best = result;
            best_diff = diff;
        }

        if k == target_k || (hi - lo) / hi.max(1e-10) < 1e-4 {
            break;
        }
    }

    best
}

/// Count distinct values in a label vector.
fn count_distinct(labels: &[usize]) -> usize {
    let max = labels.iter().copied().max().unwrap_or(0);
    let mut seen = vec![false; max + 1];
    for &l in labels {
        seen[l] = true;
    }
    seen.iter().filter(|&&s| s).count()
}

/// Remap labels to contiguous 0..k.
pub fn compact_labels(labels: &mut [usize]) {
    let max = labels.iter().copied().max().unwrap_or(0);
    let mut mapping = vec![usize::MAX; max + 1];
    let mut next = 0usize;
    for l in labels.iter_mut() {
        if mapping[*l] == usize::MAX {
            mapping[*l] = next;
            next += 1;
        }
        *l = mapping[*l];
    }
}

fn create_jobs(ntot: usize, block_size: usize) -> Vec<(usize, usize)> {
    let block_size = if block_size == 0 {
        DEFAULT_BLOCK_SIZE
    } else {
        block_size
    };
    let nblock = ntot.div_ceil(block_size);
    (0..nblock)
        .map(|block| {
            let lb = block * block_size;
            let ub = ((block + 1) * block_size).min(ntot);
            (lb, ub)
        })
        .collect()
}

#[cfg(test)]
mod tests;

/// The `n x n` symmetric adjacency implied by an undirected edge list.
///
/// Both constructors need this and the invariant is easy to get subtly wrong:
/// each edge must be pushed in BOTH directions, and exactly once per
/// direction, because building a `CscMatrix` from a `CooMatrix` SUMS entries
/// that share a coordinate rather than rejecting them. A duplicate would
/// silently double that edge's weight.
pub fn symmetric_adjacency(
    n_nodes: usize,
    edges: &[(usize, usize)],
    distances: &[f32],
) -> CscMatrix<f32> {
    let mut coo = CooMatrix::new(n_nodes, n_nodes);
    for (&(i, j), &v) in edges.iter().zip(distances.iter()) {
        coo.push(i, j, v);
        coo.push(j, i, v);
    }
    CscMatrix::from(&coo)
}

/// Each value replaced by its rank among the others, scaled to `[0, 1]`.
///
/// Ties take distinct adjacent ranks, which is harmless: the point is to put
/// two incomparable distance scales on one axis, not to be a faithful
/// empirical CDF.
fn within_source_rank(d: &[f32]) -> Vec<f32> {
    if d.len() <= 1 {
        return vec![0.0; d.len()];
    }
    let mut order: Vec<usize> = (0..d.len()).collect();
    // Ties break on the index, which is what makes this safe to sort in
    // parallel: an unstable parallel sort would otherwise put equal distances
    // in a run-dependent order, and these ranks are written out, so the file
    // would stop being reproducible. Grid-spaced coordinates tie constantly,
    // so this is the common case rather than a corner.
    //
    // `total_cmp` rather than `partial_cmp().unwrap_or(Equal)`: it is a total
    // order, so it needs no per-comparison branch and it gives NaN a definite
    // position instead of making it a wildcard that breaks transitivity.
    order.par_sort_unstable_by(|&a, &b| d[a].total_cmp(&d[b]).then(a.cmp(&b)));
    let mut out = vec![0.0f32; d.len()];
    let denom = (d.len() - 1) as f32;
    for (rank, &idx) in order.iter().enumerate() {
        out[idx] = rank as f32 / denom;
    }
    out
}
