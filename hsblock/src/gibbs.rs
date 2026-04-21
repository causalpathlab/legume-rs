//! Collapsed Gibbs sampler for the HSBM (E-step).
//!
//! For each vertex, computes the conditional posterior over all K clusters
//! using delta-score evaluation on the binary tree, then samples a new
//! cluster assignment from the categorical distribution.
//!
//! Supports both sequential and parallel (rayon) sweeps.

use crate::btree::{BTree, GammaPoissonParam};
use crate::model::poisson_score_cpu;
use crate::sufficient_stats::{SufficientStats, WeightedEdge};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Collapsed Gibbs sampler for HSBM cluster assignments.
pub struct GibbsSampler {
    rng: SmallRng,
    /// Scratch space for per-cluster log-probabilities
    log_probs: Vec<f64>,
    /// Base seed for parallel RNG (derived from main rng on first use)
    parallel_seed: u64,
    /// Stagnation threshold (fraction of vertices moving per sweep). When
    /// three consecutive sweeps land below this threshold, Gibbs returns
    /// early. `0.0` disables.
    stagnation_frac: f64,
}

impl GibbsSampler {
    /// Create a new Gibbs sampler with the given RNG.
    pub fn new(rng: SmallRng) -> Self {
        GibbsSampler {
            rng,
            log_probs: Vec::new(),
            parallel_seed: 0,
            stagnation_frac: 0.0,
        }
    }

    /// Enable Gibbs early-exit when three consecutive sweeps move fewer
    /// than `frac * n` vertices. `frac = 0.0` keeps the fixed-sweep
    /// behavior (the default).
    pub fn with_stagnation(mut self, frac: f64) -> Self {
        self.stagnation_frac = frac;
        self
    }

    /// Run `num_sweeps` full Gibbs sweeps over all vertices (sequential).
    ///
    /// Returns the total number of vertex moves across all sweeps.
    ///
    /// Each vertex sees the most recent state including moves from earlier
    /// vertices in the same sweep (standard sequential Gibbs).
    ///
    /// * `tree` - Binary tree with current variational parameters
    /// * `stats` - Sufficient statistics (modified in place)
    /// * `adj_list` - Pre-built adjacency list: `adj_list[v]` = vec of (neighbor, weight)
    /// * `num_sweeps` - Number of full sweeps over all vertices
    /// * `degree_corrected` - Whether to use degree-corrected model
    pub fn run(
        &mut self,
        tree: &BTree<GammaPoissonParam>,
        stats: &mut SufficientStats,
        adj_list: &[Vec<(usize, f64)>],
        num_sweeps: usize,
        degree_corrected: bool,
    ) -> usize {
        let k = stats.k;
        let n = stats.n;
        self.log_probs.resize(k, 0.0);

        let mut total_moves = 0;
        let early_exit = self.stagnation_frac > 0.0;
        let threshold = (self.stagnation_frac * n as f64).ceil() as usize;
        let mut low_streak = 0usize;

        for _sweep in 0..num_sweeps {
            let mut sweep_moves = 0;
            for v in 0..n {
                let old_c = stats.membership[v];

                // Compute log-probability for each candidate cluster
                compute_log_probs_for_vertex(
                    v,
                    tree,
                    stats,
                    adj_list,
                    degree_corrected,
                    &mut self.log_probs,
                );

                // Sample new cluster from categorical(softmax(log_probs))
                let new_c = sample_categorical_log(&self.log_probs, &mut self.rng);

                if new_c != old_c {
                    stats.delta_move(v, old_c, new_c, &adj_list[v]);
                    sweep_moves += 1;
                }
            }
            total_moves += sweep_moves;
            if early_exit {
                if sweep_moves < threshold {
                    low_streak += 1;
                    if low_streak >= 3 {
                        break;
                    }
                } else {
                    low_streak = 0;
                }
            }
        }

        total_moves
    }

    /// Run `num_sweeps` full Gibbs sweeps with parallel proposal computation.
    ///
    /// Uses a "synchronous Gibbs" (Jacobi-style) strategy: within each sweep,
    /// all vertices compute their conditional proposals against a frozen snapshot
    /// of the sufficient statistics using rayon `par_chunks`, then moves are
    /// applied sequentially.
    ///
    /// This matches the pattern used by `leiden::parallel_local_moving`.
    ///
    /// Returns the total number of vertex moves across all sweeps.
    pub fn run_parallel(
        &mut self,
        tree: &BTree<GammaPoissonParam>,
        stats: &mut SufficientStats,
        adj_list: &[Vec<(usize, f64)>],
        num_sweeps: usize,
        degree_corrected: bool,
    ) -> usize {
        let k = stats.k;
        let n = stats.n;

        // Generate a base seed for deterministic parallel RNG
        if self.parallel_seed == 0 {
            self.parallel_seed = self.rng.random::<u64>() | 1; // ensure nonzero
        }
        let base_seed = self.parallel_seed;

        let mut total_moves = 0;
        let early_exit = self.stagnation_frac > 0.0;
        let threshold = (self.stagnation_frac * n as f64).ceil() as usize;
        let mut low_streak = 0usize;

        // Pre-allocate node order for par_chunks
        let node_order: Vec<usize> = (0..n).collect();
        let chunk_size = std::cmp::max(256, n / rayon::current_num_threads().max(1));

        for sweep in 0..num_sweeps {
            // Phase 1: Parallel proposal computation (read-only on stats)
            // Each chunk gets its own log_probs scratch buffer.
            let sweep_seed = base_seed.wrapping_mul(sweep as u64 + 1);

            let proposals: Vec<usize> = node_order
                .par_chunks(chunk_size)
                .flat_map(|chunk| {
                    let mut log_probs = vec![0.0f64; k];
                    chunk
                        .iter()
                        .map(|&v| {
                            compute_log_probs_for_vertex(
                                v,
                                tree,
                                stats,
                                adj_list,
                                degree_corrected,
                                &mut log_probs,
                            );
                            // Deterministic per-vertex RNG
                            let vertex_seed = sweep_seed ^ (v as u64).wrapping_mul(2654435761);
                            let mut rng = SmallRng::seed_from_u64(vertex_seed);
                            sample_categorical_log(&log_probs, &mut rng)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            // Phase 2: Sequential apply (mutates stats)
            let mut sweep_moves = 0;
            for v in 0..n {
                let old_c = stats.membership[v];
                let new_c = proposals[v];
                if new_c != old_c {
                    stats.delta_move(v, old_c, new_c, &adj_list[v]);
                    sweep_moves += 1;
                }
            }
            total_moves += sweep_moves;
            if early_exit {
                if sweep_moves < threshold {
                    low_streak += 1;
                    if low_streak >= 3 {
                        break;
                    }
                } else {
                    low_streak = 0;
                }
            }
        }

        total_moves
    }

    /// Run greedy (argmax) sweeps: each vertex moves to the cluster with
    /// highest conditional score. No stochastic sampling.
    ///
    /// Typically used as a finalization step after Gibbs exploration to
    /// settle into the MAP assignment.
    ///
    /// Returns the total number of vertex moves across all sweeps.
    pub fn run_greedy(
        &mut self,
        tree: &BTree<GammaPoissonParam>,
        stats: &mut SufficientStats,
        adj_list: &[Vec<(usize, f64)>],
        num_sweeps: usize,
        degree_corrected: bool,
    ) -> usize {
        let k = stats.k;
        let n = stats.n;
        self.log_probs.resize(k, 0.0);

        let mut total_moves = 0;

        for _sweep in 0..num_sweeps {
            let mut sweep_moves = 0;
            for v in 0..n {
                let old_c = stats.membership[v];

                compute_log_probs_for_vertex(
                    v,
                    tree,
                    stats,
                    adj_list,
                    degree_corrected,
                    &mut self.log_probs,
                );

                // Argmax: pick cluster with highest delta score
                let new_c = argmax_log(&self.log_probs);

                if new_c != old_c {
                    stats.delta_move(v, old_c, new_c, &adj_list[v]);
                    sweep_moves += 1;
                }
            }
            total_moves += sweep_moves;
            if sweep_moves == 0 {
                break; // converged
            }
        }

        total_moves
    }
}

/// Pick the index with the highest value.
fn argmax_log(log_probs: &[f64]) -> usize {
    let mut best = 0;
    let mut best_val = log_probs[0];
    for (i, &v) in log_probs.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Compute log-probability for assigning vertex `v` to each cluster.
///
/// For each candidate cluster t ≠ current_c(v), computes the change in the
/// total tree score if vertex v were moved from its current cluster to t.
/// The score for the current cluster is set to 0.0 (baseline).
///
/// Complexity: O(K²) per candidate cluster, O(K³) per vertex.
///
/// This is a free function (not a method) so it can be called from parallel
/// contexts where `&SufficientStats` is shared across threads. See
/// [`compute_log_probs_for_vertex_restricted`] for the candidate-subset variant.
///
/// * `vertex` - The vertex to compute proposals for
/// * `tree` - Binary tree with current variational parameters
/// * `stats` - Sufficient statistics (read-only)
/// * `adj_list` - Pre-built adjacency list
/// * `degree_corrected` - Whether to use degree-corrected model
/// * `log_probs` - Output buffer of length K (caller-provided)
pub fn compute_log_probs_for_vertex(
    vertex: usize,
    tree: &BTree<GammaPoissonParam>,
    stats: &SufficientStats,
    adj_list: &[Vec<(usize, f64)>],
    degree_corrected: bool,
    log_probs: &mut [f64],
) {
    compute_log_probs_for_vertex_impl(
        vertex,
        tree,
        stats,
        adj_list,
        degree_corrected,
        None,
        log_probs,
    );
}

/// Candidate-restricted variant of [`compute_log_probs_for_vertex`].
///
/// Evaluates only the clusters listed in `candidates`. Slots not listed are
/// left at `f64::NEG_INFINITY`, and the current cluster is always scored as
/// `0.0`. Use this when the caller has a prior on which destinations are
/// plausible (e.g., BBKNN neighbors, sibling blocks in an external tree) to
/// cut per-vertex work from O(K) to O(|candidates|).
///
/// * `candidates` - Slice of cluster indices in `0..K` to evaluate.
pub fn compute_log_probs_for_vertex_restricted(
    vertex: usize,
    tree: &BTree<GammaPoissonParam>,
    stats: &SufficientStats,
    adj_list: &[Vec<(usize, f64)>],
    degree_corrected: bool,
    candidates: &[usize],
    log_probs: &mut [f64],
) {
    compute_log_probs_for_vertex_impl(
        vertex,
        tree,
        stats,
        adj_list,
        degree_corrected,
        Some(candidates),
        log_probs,
    );
}

/// Per-vertex scratch captured once outside the candidate loop so
/// `evaluate_delta_for_target` stays a small, easy-to-audit signature.
struct VertexCtx<'a> {
    current_c: usize,
    deg: f64,
    edge_to_cluster: &'a [f64],
}

fn evaluate_delta_for_target(
    target: usize,
    k: usize,
    tree: &BTree<GammaPoissonParam>,
    stats: &SufficientStats,
    ctx: &VertexCtx<'_>,
    degree_corrected: bool,
) -> f64 {
    let current_c = ctx.current_c;
    let deg = ctx.deg;
    let edge_to_cluster = ctx.edge_to_cluster;
    if target == current_c {
        return 0.0;
    }
    let mut delta = 0.0;

    // Hypothetical new stats after moving vertex to `target`.
    let new_size_s = stats.cluster_size[current_c] - 1.0;
    let new_size_t = stats.cluster_size[target] + 1.0;
    let new_vol_s = stats.cluster_volume[current_c] - deg;
    let new_vol_t = stats.cluster_volume[target] + deg;

    for ci in 0..k {
        for cj in ci..k {
            if ci != current_c && ci != target && cj != current_c && cj != target {
                continue;
            }

            let lca_node = tree.lca(ci, cj);
            let (a0, b0) = tree.node_params(lca_node);

            let old_edge = stats.edge_stat(ci, cj);
            let old_total = stats.total_stat(ci, cj, degree_corrected);
            let old_s = poisson_score_cpu(a0, b0, old_edge, old_total);

            let mut new_edge = old_edge;

            if ci == current_c && cj == current_c {
                new_edge -= edge_to_cluster[current_c];
            } else if ci == target && cj == target {
                new_edge += edge_to_cluster[target];
            } else if (ci == current_c && cj == target) || (ci == target && cj == current_c) {
                new_edge = old_edge - edge_to_cluster[target] + edge_to_cluster[current_c];
            } else if ci == current_c {
                new_edge -= edge_to_cluster[cj];
            } else if cj == current_c {
                new_edge -= edge_to_cluster[ci];
            } else if ci == target {
                new_edge += edge_to_cluster[cj];
            } else if cj == target {
                new_edge += edge_to_cluster[ci];
            }

            let new_total = if degree_corrected {
                let vol_ci = if ci == current_c {
                    new_vol_s
                } else if ci == target {
                    new_vol_t
                } else {
                    stats.cluster_volume[ci]
                };
                let vol_cj = if cj == current_c {
                    new_vol_s
                } else if cj == target {
                    new_vol_t
                } else {
                    stats.cluster_volume[cj]
                };
                if ci == cj {
                    vol_ci * vol_cj / 2.0
                } else {
                    vol_ci * vol_cj
                }
            } else {
                let sz_ci = if ci == current_c {
                    new_size_s
                } else if ci == target {
                    new_size_t
                } else {
                    stats.cluster_size[ci]
                };
                let sz_cj = if cj == current_c {
                    new_size_s
                } else if cj == target {
                    new_size_t
                } else {
                    stats.cluster_size[cj]
                };
                if ci == cj {
                    sz_ci * (sz_ci - 1.0) / 2.0
                } else {
                    sz_ci * sz_cj
                }
            };

            let new_s = poisson_score_cpu(a0, b0, new_edge, new_total);
            delta += new_s - old_s;
        }
    }

    delta
}

fn compute_log_probs_for_vertex_impl(
    vertex: usize,
    tree: &BTree<GammaPoissonParam>,
    stats: &SufficientStats,
    adj_list: &[Vec<(usize, f64)>],
    degree_corrected: bool,
    candidates: Option<&[usize]>,
    log_probs: &mut [f64],
) {
    let k = stats.k;
    let current_c = stats.membership[vertex];
    let neighbors = &adj_list[vertex];

    let mut edge_to_cluster = vec![0.0f64; k];
    for &(nbr, w) in neighbors {
        edge_to_cluster[stats.membership[nbr]] += w;
    }
    let ctx = VertexCtx {
        current_c,
        deg: stats.vertex_degree[vertex],
        edge_to_cluster: &edge_to_cluster,
    };

    match candidates {
        Some(cand) => {
            log_probs.iter_mut().for_each(|x| *x = f64::NEG_INFINITY);
            log_probs[current_c] = 0.0;
            for &t in cand {
                if t == current_c {
                    continue;
                }
                log_probs[t] = evaluate_delta_for_target(t, k, tree, stats, &ctx, degree_corrected);
            }
        }
        None => {
            for (t, slot) in log_probs.iter_mut().enumerate().take(k) {
                *slot = evaluate_delta_for_target(t, k, tree, stats, &ctx, degree_corrected);
            }
        }
    }
}

/// Sample from a categorical distribution given log-probabilities.
///
/// Uses the log-sum-exp trick for numerical stability.
fn sample_categorical_log(log_probs: &[f64], rng: &mut SmallRng) -> usize {
    let max = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = log_probs.iter().map(|lp| (lp - max).exp()).collect();
    let total: f64 = weights.iter().sum();

    if total <= 0.0 || !total.is_finite() {
        // Fallback: uniform
        return rng.random_range(0..log_probs.len());
    }

    let u: f64 = rng.random::<f64>() * total;
    let mut cum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if cum >= u {
            return i;
        }
    }

    weights.len() - 1
}

/// Build an adjacency list from a weighted edge list.
///
/// Returns `adj_list[v]` = vec of (neighbor, weight) for each vertex v.
pub fn build_adj_list(edges: &[WeightedEdge], n: usize) -> Vec<Vec<(usize, f64)>> {
    let mut adj_list = vec![Vec::new(); n];
    for &(i, j, w) in edges {
        adj_list[i].push((j, w as f64));
        adj_list[j].push((i, w as f64));
    }
    adj_list
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planted_partition_graph(
        n_per_cluster: usize,
        n_clusters: usize,
        p_in: f32,
        p_out: f32,
        seed: u64,
    ) -> (Vec<WeightedEdge>, usize, Vec<usize>) {
        let n = n_per_cluster * n_clusters;
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut edges = Vec::new();
        let mut labels = Vec::with_capacity(n);

        for c in 0..n_clusters {
            for _ in 0..n_per_cluster {
                labels.push(c);
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let p = if labels[i] == labels[j] { p_in } else { p_out };
                if rng.random::<f32>() < p {
                    edges.push((i, j, 1.0f32));
                }
            }
        }

        (edges, n, labels)
    }

    #[test]
    fn test_gibbs_reduces_moves() {
        let (edges, n, _true_labels) = planted_partition_graph(20, 2, 0.6, 0.05, 42);

        let tree = BTree::with_gamma_poisson(2, 1.0, 1.0);
        let k = tree.num_leaves();

        // Random initial labels
        let mut rng = SmallRng::seed_from_u64(123);
        let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..k)).collect();

        let mut stats = SufficientStats::from_edges(&edges, n, k, &labels);
        let adj_list = build_adj_list(&edges, n);

        let mut sampler = GibbsSampler::new(SmallRng::seed_from_u64(456));

        // First sweep should have many moves
        let moves1 = sampler.run(&tree, &mut stats, &adj_list, 1, false);

        // After several sweeps, moves should decrease (convergence)
        let _moves_mid = sampler.run(&tree, &mut stats, &adj_list, 10, false);
        let moves_late = sampler.run(&tree, &mut stats, &adj_list, 1, false);

        // The sampler should converge — fewer moves in later sweeps
        assert!(
            moves_late <= moves1 || moves1 == 0,
            "Expected convergence: early_moves={}, late_moves={}",
            moves1,
            moves_late
        );
    }

    /// Brute-force computation of the full tree score from sufficient statistics.
    ///
    /// For each cluster pair (ci, cj) with ci <= cj, compute the Poisson score
    /// at the LCA node and sum them up.
    fn brute_force_tree_score(
        tree: &BTree<GammaPoissonParam>,
        stats: &SufficientStats,
        degree_corrected: bool,
    ) -> f64 {
        use crate::model::poisson_score_cpu;
        let k = stats.k;
        let mut score = 0.0;
        for ci in 0..k {
            for cj in ci..k {
                let lca = tree.lca(ci, cj);
                let (a0, b0) = tree.node_params(lca);
                let e = stats.edge_stat(ci, cj);
                let t = stats.total_stat(ci, cj, degree_corrected);
                score += poisson_score_cpu(a0, b0, e, t);
            }
        }
        score
    }

    /// Test that compute_log_probs delta scores match brute-force recomputation.
    ///
    /// For each vertex v, for each target cluster t != current_c(v):
    ///   delta(v, t) from compute_log_probs should equal
    ///   full_tree_score(after moving v to t) - full_tree_score(before move)
    #[test]
    fn test_delta_score_matches_brute_force() {
        // Build a small graph: 3 clusters of 5 vertices each
        let (edges, n, _true_labels) = planted_partition_graph(5, 3, 0.8, 0.1, 42);
        let tree = BTree::with_gamma_poisson(3, 2.0, 1.5); // depth 3 → K=4 leaves
        let k = tree.num_leaves();

        // Use a fixed labeling (not necessarily aligned with truth)
        let labels: Vec<usize> = (0..n).map(|v| v % k).collect();
        let stats = SufficientStats::from_edges(&edges, n, k, &labels);
        let adj_list = build_adj_list(&edges, n);

        let mut log_probs = vec![0.0f64; k];
        let mut max_err = 0.0f64;

        for v in 0..n {
            let current_c = stats.membership[v];

            // Get delta scores from compute_log_probs
            compute_log_probs_for_vertex(v, &tree, &stats, &adj_list, false, &mut log_probs);

            // The score for current_c should be 0 (no move)
            assert!(
                log_probs[current_c].abs() < 1e-10,
                "v={}: log_prob for current cluster should be 0, got {}",
                v,
                log_probs[current_c]
            );

            // Compute the baseline (before-move) tree score
            let score_before = brute_force_tree_score(&tree, &stats, false);

            for (t, &computed_delta) in log_probs.iter().enumerate().take(k) {
                if t == current_c {
                    continue;
                }

                // Clone stats and actually perform the move
                let mut stats_moved = stats.clone();
                stats_moved.delta_move(v, current_c, t, &adj_list[v]);

                // Compute the after-move tree score
                let score_after = brute_force_tree_score(&tree, &stats_moved, false);

                let expected_delta = score_after - score_before;
                let err = (computed_delta - expected_delta).abs();

                if err > max_err {
                    max_err = err;
                }

                assert!(
                    err < 1e-8,
                    "v={}, current_c={}, t={}: delta mismatch: computed={:.10}, expected={:.10}, err={:.2e}",
                    v, current_c, t, computed_delta, expected_delta, err
                );
            }
        }

        println!("Delta score max error: {:.2e}", max_err);
    }

    /// Same test but with degree-corrected model.
    #[test]
    fn test_delta_score_degree_corrected_matches_brute_force() {
        let (edges, n, _true_labels) = planted_partition_graph(5, 2, 0.7, 0.1, 99);
        let tree = BTree::with_gamma_poisson(2, 1.0, 1.0); // depth 2 → K=2 leaves
        let k = tree.num_leaves();

        let labels: Vec<usize> = (0..n).map(|v| v % k).collect();
        let stats = SufficientStats::from_edges(&edges, n, k, &labels);
        let adj_list = build_adj_list(&edges, n);

        let mut log_probs = vec![0.0f64; k];
        let mut max_err = 0.0f64;

        for v in 0..n {
            let current_c = stats.membership[v];
            compute_log_probs_for_vertex(v, &tree, &stats, &adj_list, true, &mut log_probs);

            let score_before = brute_force_tree_score(&tree, &stats, true);

            for (t, &computed_delta) in log_probs.iter().enumerate().take(k) {
                if t == current_c {
                    continue;
                }

                let mut stats_moved = stats.clone();
                stats_moved.delta_move(v, current_c, t, &adj_list[v]);

                let score_after = brute_force_tree_score(&tree, &stats_moved, true);
                let expected_delta = score_after - score_before;
                let err = (computed_delta - expected_delta).abs();

                if err > max_err {
                    max_err = err;
                }

                assert!(
                    err < 1e-8,
                    "DC v={}, current_c={}, t={}: delta mismatch: computed={:.10}, expected={:.10}, err={:.2e}",
                    v, current_c, t, computed_delta, expected_delta, err
                );
            }
        }

        println!("DC Delta score max error: {:.2e}", max_err);
    }

    /// Test with non-uniform tree parameters to ensure the LCA-based
    /// parameter lookup in compute_log_probs is correct.
    #[test]
    fn test_delta_score_nonuniform_tree_params() {
        let (edges, n, _true_labels) = planted_partition_graph(4, 2, 0.6, 0.1, 77);
        let mut tree = BTree::with_gamma_poisson(3, 1.0, 1.0); // depth 3 → K=4 leaves
        let k = tree.num_leaves();

        // Set different parameters at each node
        for node in 1..=tree.num_nodes() {
            let a0 = 0.5 + 0.3 * (node as f64);
            let b0 = 0.1 + 0.2 * (node as f64);
            tree.set_node_ln_params(node, a0.ln(), b0.ln());
        }

        let labels: Vec<usize> = (0..n).map(|v| v % k).collect();
        let stats = SufficientStats::from_edges(&edges, n, k, &labels);
        let adj_list = build_adj_list(&edges, n);

        let mut log_probs = vec![0.0f64; k];
        let mut max_err = 0.0f64;

        for v in 0..n {
            let current_c = stats.membership[v];
            compute_log_probs_for_vertex(v, &tree, &stats, &adj_list, false, &mut log_probs);

            let score_before = brute_force_tree_score(&tree, &stats, false);

            for (t, &computed_delta) in log_probs.iter().enumerate().take(k) {
                if t == current_c {
                    continue;
                }

                let mut stats_moved = stats.clone();
                stats_moved.delta_move(v, current_c, t, &adj_list[v]);

                let score_after = brute_force_tree_score(&tree, &stats_moved, false);
                let expected_delta = score_after - score_before;
                let err = (computed_delta - expected_delta).abs();

                if err > max_err {
                    max_err = err;
                }

                assert!(
                    err < 1e-8,
                    "NonUnif v={}, current_c={}, t={}: delta mismatch: computed={:.10}, expected={:.10}, err={:.2e}",
                    v, current_c, t, computed_delta, expected_delta, err
                );
            }
        }

        println!("Non-uniform params delta score max error: {:.2e}", max_err);
    }

    #[test]
    fn test_sample_categorical_log() {
        let mut rng = SmallRng::seed_from_u64(42);

        // Strongly peaked distribution — should almost always return index 1
        let log_probs = vec![-100.0, 0.0, -100.0];
        let mut counts = [0usize; 3];
        for _ in 0..1000 {
            let idx = sample_categorical_log(&log_probs, &mut rng);
            counts[idx] += 1;
        }
        assert!(counts[1] > 990, "Expected mostly index 1, got {:?}", counts);
    }

    /// Test that parallel Gibbs produces valid clustering on a planted partition graph.
    #[test]
    fn test_parallel_gibbs_convergence() {
        let (edges, n, _true_labels) = planted_partition_graph(20, 2, 0.6, 0.05, 42);

        let tree = BTree::with_gamma_poisson(2, 1.0, 1.0);
        let k = tree.num_leaves();

        // Random initial labels
        let mut rng = SmallRng::seed_from_u64(123);
        let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..k)).collect();

        let mut stats = SufficientStats::from_edges(&edges, n, k, &labels);
        let adj_list = build_adj_list(&edges, n);

        let mut sampler = GibbsSampler::new(SmallRng::seed_from_u64(789));

        // First parallel sweep should have many moves
        let moves1 = sampler.run_parallel(&tree, &mut stats, &adj_list, 1, false);

        // After several sweeps, moves should decrease (convergence)
        let _moves_mid = sampler.run_parallel(&tree, &mut stats, &adj_list, 10, false);
        let moves_late = sampler.run_parallel(&tree, &mut stats, &adj_list, 1, false);

        // The parallel sampler should also converge
        assert!(
            moves_late <= moves1 || moves1 == 0,
            "Parallel: Expected convergence: early_moves={}, late_moves={}",
            moves1,
            moves_late
        );

        // The result should have a reasonable number of non-empty clusters
        let nonempty = stats.cluster_size.iter().filter(|&&s| s > 0.0).count();
        assert!(nonempty >= 1 && nonempty <= k);
    }

    #[test]
    fn test_restricted_kernel_matches_unrestricted_on_candidates() {
        let (edges, n, _) = planted_partition_graph(5, 3, 0.8, 0.1, 42);
        let tree = BTree::with_gamma_poisson(3, 2.0, 1.5);
        let k = tree.num_leaves();
        let labels: Vec<usize> = (0..n).map(|v| v % k).collect();
        let stats = SufficientStats::from_edges(&edges, n, k, &labels);
        let adj_list = build_adj_list(&edges, n);

        let mut full = vec![0f64; k];
        let mut restricted = vec![0f64; k];
        let candidates: Vec<usize> = (0..k).collect();

        for v in 0..n {
            compute_log_probs_for_vertex(v, &tree, &stats, &adj_list, false, &mut full);
            compute_log_probs_for_vertex_restricted(
                v,
                &tree,
                &stats,
                &adj_list,
                false,
                &candidates,
                &mut restricted,
            );
            for t in 0..k {
                assert!(
                    (full[t] - restricted[t]).abs() < 1e-10,
                    "v={}, t={}: full={}, restricted={}",
                    v,
                    t,
                    full[t],
                    restricted[t]
                );
            }
        }
    }

    #[test]
    fn test_restricted_kernel_narrow_candidates_leaves_others_neg_inf() {
        let (edges, n, _) = planted_partition_graph(5, 3, 0.8, 0.1, 42);
        let tree = BTree::with_gamma_poisson(3, 2.0, 1.5);
        let k = tree.num_leaves();
        let labels: Vec<usize> = (0..n).map(|v| v % k).collect();
        let stats = SufficientStats::from_edges(&edges, n, k, &labels);
        let adj_list = build_adj_list(&edges, n);

        let mut log_probs = vec![0f64; k];
        let candidates = vec![1usize];
        compute_log_probs_for_vertex_restricted(
            0,
            &tree,
            &stats,
            &adj_list,
            false,
            &candidates,
            &mut log_probs,
        );

        let current_c = stats.membership[0];
        for (t, &lp) in log_probs.iter().enumerate() {
            if t == current_c {
                assert!(lp.abs() < 1e-10);
            } else if candidates.contains(&t) {
                assert!(lp.is_finite());
            } else {
                assert!(lp.is_infinite() && lp < 0.0);
            }
        }
    }

    #[test]
    fn test_gibbs_stagnation_early_exit() {
        // Run Gibbs twice: once with stagnation=0 (fixed sweeps), once with
        // stagnation=0.5 (should early-exit once the chain is quiet).
        let (edges, n, _) = planted_partition_graph(20, 2, 0.6, 0.05, 42);
        let tree = BTree::with_gamma_poisson(2, 1.0, 1.0);
        let k = tree.num_leaves();
        let mut rng = SmallRng::seed_from_u64(123);
        let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..k)).collect();
        let mut stats_a = SufficientStats::from_edges(&edges, n, k, &labels);
        let mut stats_b = SufficientStats::from_edges(&edges, n, k, &labels);
        let adj_list = build_adj_list(&edges, n);

        let mut sampler_a = GibbsSampler::new(SmallRng::seed_from_u64(7));
        let mut sampler_b = GibbsSampler::new(SmallRng::seed_from_u64(7)).with_stagnation(0.5);

        // Run first 5 sweeps on both to converge.
        sampler_a.run(&tree, &mut stats_a, &adj_list, 5, false);
        sampler_b.run(&tree, &mut stats_b, &adj_list, 5, false);

        // Now both are near-converged. Ask for 100 more sweeps; early-exit
        // should return fewer moves (fewer sweeps actually executed).
        let moves_a = sampler_a.run(&tree, &mut stats_a, &adj_list, 100, false);
        let moves_b = sampler_b.run(&tree, &mut stats_b, &adj_list, 100, false);
        assert!(
            moves_b <= moves_a,
            "stagnation early-exit should not do more work: a={}, b={}",
            moves_a,
            moves_b
        );
    }
}
