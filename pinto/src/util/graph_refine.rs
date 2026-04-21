use crate::util::common::*;
use crate::util::knn_graph::KnnGraph;
use std::collections::VecDeque;

/// Reusable scratch buffers for the articulation-point BFS inside
/// `refine_labels`. Keeping them across moves avoids per-call
/// `HashSet`/`VecDeque` allocations in the hot path.
#[derive(Default)]
pub(super) struct ArticulationScratch {
    nbrs_in_cluster: Vec<usize>,
    visited: HashSet<usize>,
    remaining: HashSet<usize>,
    queue: VecDeque<usize>,
}

impl ArticulationScratch {
    fn reset(&mut self) {
        self.nbrs_in_cluster.clear();
        self.visited.clear();
        self.remaining.clear();
        self.queue.clear();
    }
}

/// Would removing node `i` disconnect the induced subgraph of `cluster`?
///
/// BFS from one in-cluster neighbour of `i`, blocking `i` itself and staying
/// within nodes labelled `cluster`. If every other in-cluster neighbour of `i`
/// is reachable, `i` is not an articulation point of the cluster's subgraph.
pub(super) fn would_disconnect_cluster(
    i: usize,
    cluster: usize,
    graph: &KnnGraph,
    labels: &[usize],
    scratch: &mut ArticulationScratch,
) -> bool {
    scratch.reset();
    scratch
        .nbrs_in_cluster
        .extend(graph.neighbors(i).iter().copied().filter(|&j| labels[j] == cluster));

    if scratch.nbrs_in_cluster.len() <= 1 {
        return false;
    }

    let start = scratch.nbrs_in_cluster[0];
    scratch.visited.insert(start);
    scratch.queue.push_back(start);
    scratch
        .remaining
        .extend(scratch.nbrs_in_cluster[1..].iter().copied());

    while let Some(node) = scratch.queue.pop_front() {
        for &nb in graph.neighbors(node) {
            if nb == i || labels[nb] != cluster || !scratch.visited.insert(nb) {
                continue;
            }
            scratch.remaining.remove(&nb);
            if scratch.remaining.is_empty() {
                return false;
            }
            scratch.queue.push_back(nb);
        }
    }

    !scratch.remaining.is_empty()
}

/// Refine cluster labels by Leiden-style local moving.
///
/// Each sweep visits all nodes in random order. For node `i` in cluster `A`,
/// the move to a graph-adjacent cluster `B` is accepted iff
/// `cos(x_i, centroid_B) > cos(x_i, centroid_A)` and removing `i` does not
/// disconnect `A`'s induced subgraph (BFS articulation check).
///
/// Both centroid similarities use the *current* (pre-move) centroids, which
/// includes `i`'s contribution to `A`. This is the standard convergence-safe
/// criterion: each accepted move strictly increases the within-cluster cosine
/// sum (M-step optimality of normalised mean), so the loop terminates.
///
/// * `features` – `[D × N]` L2-normalised feature matrix
/// * `graph` – KNN graph on the same N nodes
/// * `labels` – cluster labels, mutated in place
/// * `max_iter` – safety cap on sweeps; converges earlier when no node moves
/// * `rng_seed` – seed for the random node ordering
///
/// Returns total number of moves accepted.
pub fn refine_labels(
    features: &Mat,
    graph: &KnnGraph,
    labels: &mut [usize],
    max_iter: usize,
    rng_seed: u64,
) -> usize {
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let n = labels.len();
    let dim = features.nrows();
    debug_assert_eq!(features.ncols(), n);
    debug_assert_eq!(graph.n_nodes, n);

    if n == 0 || max_iter == 0 {
        return 0;
    }

    let n_clusters = labels.iter().max().copied().map_or(0, |m| m + 1);
    if n_clusters <= 1 {
        return 0;
    }

    // Unnormalised centroid sums and sizes; cosine denominator = ||sum||.
    let mut centroid_sum = Mat::zeros(dim, n_clusters);
    let mut cluster_sizes = vec![0usize; n_clusters];
    for (i, &c) in labels.iter().enumerate() {
        cluster_sizes[c] += 1;
        centroid_sum
            .column_mut(c)
            .axpy(1.0, &features.column(i), 1.0);
    }
    let mut centroid_norm: Vec<f32> = (0..n_clusters)
        .map(|c| centroid_sum.column(c).norm())
        .collect();

    let mut rng = SmallRng::seed_from_u64(rng_seed);
    let mut node_order: Vec<usize> = (0..n).collect();
    let mut nbr_set: HashSet<usize> = Default::default();
    let mut articulation = ArticulationScratch::default();
    let mut total_moves = 0usize;

    for _iter in 0..max_iter {
        node_order.shuffle(&mut rng);
        let mut moves = 0usize;

        for &i in &node_order {
            let current = labels[i];

            // Don't drain a singleton — we'd lose the cluster id.
            if cluster_sizes[current] <= 1 {
                continue;
            }

            let dot_curr = features.column(i).dot(&centroid_sum.column(current));
            let norm_curr = centroid_norm[current];
            let sim_curr = if norm_curr > 0.0 {
                dot_curr / norm_curr
            } else {
                0.0
            };

            nbr_set.clear();
            for &nb in graph.neighbors(i) {
                let nbc = labels[nb];
                if nbc != current {
                    nbr_set.insert(nbc);
                }
            }
            if nbr_set.is_empty() {
                continue;
            }

            let mut best_c = current;
            let mut best_sim = sim_curr;
            for &c in &nbr_set {
                let norm = centroid_norm[c];
                if norm <= 0.0 {
                    continue;
                }
                let dot = features.column(i).dot(&centroid_sum.column(c));
                let sim = dot / norm;
                if sim > best_sim {
                    best_sim = sim;
                    best_c = c;
                }
            }

            if best_c == current {
                continue;
            }

            // Connectivity: skip moves that would split the source cluster.
            if would_disconnect_cluster(i, current, graph, labels, &mut articulation) {
                continue;
            }

            centroid_sum
                .column_mut(current)
                .axpy(-1.0, &features.column(i), 1.0);
            centroid_sum
                .column_mut(best_c)
                .axpy(1.0, &features.column(i), 1.0);
            cluster_sizes[current] -= 1;
            cluster_sizes[best_c] += 1;
            centroid_norm[current] = centroid_sum.column(current).norm();
            centroid_norm[best_c] = centroid_sum.column(best_c).norm();

            labels[i] = best_c;
            moves += 1;
        }

        if moves == 0 {
            break;
        }
        total_moves += moves;
    }

    if total_moves > 0 {
        info!(
            "Refinement: {} moves over up to {} sweeps",
            total_moves, max_iter
        );
    }

    total_moves
}
