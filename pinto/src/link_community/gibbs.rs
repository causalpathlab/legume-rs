//! Collapsed Gibbs sampler for the link community model.
//!
//! Adapted from `hsblock/src/gibbs.rs` but simplified: flat K communities
//! (no tree/LCA), and edge-level rather than vertex-level assignments.

use crate::link_community::model::*;
use crate::util::cell_pairs::connected_components;
use crate::util::common::new_progress_bar;
use crate::util::knn_graph::KnnGraph;
use log::info;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Controls how edges are reassigned in component-partitioned sweeps.
#[derive(Clone, Copy)]
enum MoveStrategy {
    /// Gibbs sampling: stochastic, runs all sweeps.
    Sample,
    /// Greedy argmax: deterministic, early exit when converged.
    Greedy,
}

/// Parameters for component-partitioned Gibbs/greedy methods.
pub struct ComponentGibbsArgs<'a> {
    pub graph: &'a KnnGraph,
    pub edges: &'a [(usize, usize)],
    pub k: usize,
    /// Dirichlet concentration for mixing weight prior (0.0 = no prior).
    pub alpha: f64,
    /// Optional collapsed K×K incidence (Poisson-Gamma) configuration.
    pub incidence: Option<IncidenceConfig<'a>>,
}

/// Frozen incidence configuration consumed by [`run_components`]. Both
/// `propensity` (`[n_cells × k]` row-major soft propensity, equivalently
/// the variational `q(c_v)`) and `log_incidence` (`[k × k]` row-major
/// `E_q[log B[k, k']]` under a `Gamma(a, b)` posterior) are computed once
/// from the post-V-cycle labels and **frozen** for the duration of EM and
/// greedy. There is no per-move bookkeeping for either.
#[derive(Copy, Clone)]
pub struct IncidenceConfig<'a> {
    pub propensity: &'a [f64],
    pub log_incidence: &'a [f64],
}

/// Collapsed Gibbs sampler for link community assignments.
pub struct LinkGibbsSampler {
    rng: SmallRng,
    parallel_seed: u64,
}

impl LinkGibbsSampler {
    /// Create a new sampler with the given RNG.
    pub fn new(rng: SmallRng) -> Self {
        LinkGibbsSampler {
            rng,
            parallel_seed: 0,
        }
    }

    /// Run `num_sweeps` sequential Gibbs sweeps over all edges.
    ///
    /// Returns the total number of edge moves across all sweeps.
    #[cfg(test)]
    pub fn run(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        num_sweeps: usize,
    ) -> usize {
        let k = stats.k;
        let n = stats.n_edges;
        let mut log_probs = vec![0.0f64; k];

        let mut total_moves = 0;

        let pb = new_progress_bar(
            num_sweeps as u64,
            "Gibbs {bar:40} {pos}/{len} sweeps ({eta})",
        );

        for _sweep in 0..num_sweeps {
            for e in 0..n {
                let old_c = stats.membership[e];

                compute_log_probs_for_edge(e, stats, profiles, None, None, &mut log_probs);

                let new_c = sample_categorical_log(&log_probs, &mut self.rng);

                if new_c != old_c {
                    stats.delta_move(e, old_c, new_c, profiles);
                    total_moves += 1;
                }
            }
            pb.inc(1);
        }
        pb.finish_and_clear();

        total_moves
    }

    /// Run `num_sweeps` Gibbs sweeps with parallel proposal computation.
    ///
    /// Jacobi-style: all edges compute proposals against a frozen snapshot
    /// of the stats using rayon `par_chunks`, then moves are applied sequentially.
    ///
    /// `alpha` is the Dirichlet concentration on community mixing weights:
    /// `alpha > 0` enables the prior (log-weights recomputed each sweep from
    /// the frozen snapshot), `alpha == 0` disables it. Match the prior used
    /// by the downstream component-EM stage for consistency across levels.
    ///
    /// Returns the total number of edge moves across all sweeps.
    #[cfg(test)]
    pub fn run_parallel(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        num_sweeps: usize,
        alpha: f64,
    ) -> usize {
        self.run_parallel_with_observer(stats, profiles, num_sweeps, alpha, |_, _| {})
    }

    /// [`Self::run_parallel`] with a per-sweep observer closure called after
    /// each sweep's stats are patched. The observer receives `(sweep_idx,
    /// &stats)` — use it to record score / MI / whatever per-sweep diagnostic
    /// is useful. Called zero-cost when the closure is a no-op.
    pub fn run_parallel_with_observer<F>(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        num_sweeps: usize,
        alpha: f64,
        mut observer: F,
    ) -> usize
    where
        F: FnMut(usize, &LinkCommunityStats),
    {
        let k = stats.k;
        let n = stats.n_edges;

        if self.parallel_seed == 0 {
            self.parallel_seed = self.rng.random::<u64>() | 1;
        }
        let base_seed = self.parallel_seed;

        let mut total_moves = 0;

        let edge_order: Vec<usize> = (0..n).collect();
        let chunk_size = std::cmp::max(256, n / rayon::current_num_threads().max(1));

        let pb = new_progress_bar(
            num_sweeps as u64,
            "Gibbs {bar:40} {pos}/{len} sweeps ({eta})",
        );

        for sweep in 0..num_sweeps {
            let sweep_seed = base_seed.wrapping_mul(sweep as u64 + 1);

            let lw_snap = (alpha > 0.0).then(|| stats.compute_log_weights(alpha));
            let lw_ref = lw_snap.as_deref();

            let proposals: Vec<usize> = edge_order
                .par_chunks(chunk_size)
                .flat_map(|chunk| {
                    let mut log_probs = vec![0.0f64; k];
                    chunk
                        .iter()
                        .map(|&e| {
                            compute_log_probs_for_edge(
                                e,
                                stats,
                                profiles,
                                lw_ref,
                                None,
                                &mut log_probs,
                            );
                            let vertex_seed = sweep_seed ^ (e as u64).wrapping_mul(2654435761);
                            let mut rng = SmallRng::seed_from_u64(vertex_seed);
                            sample_categorical_log(&log_probs, &mut rng)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            for (e, &new_c) in proposals.iter().enumerate() {
                let old_c = stats.membership[e];
                if new_c != old_c {
                    stats.delta_move(e, old_c, new_c, profiles);
                    total_moves += 1;
                }
            }
            observer(sweep, stats);
            pb.inc(1);
        }
        pb.finish_and_clear();

        total_moves
    }

    /// Run greedy (argmax) sweeps over all edges with early exit on convergence.
    ///
    /// Deterministic, sequential argmax — each edge is reassigned to the
    /// community with the highest log-probability. Used for per-level
    /// finalization in the V-cycle cascade where the graph's connected-
    /// component structure is irrelevant (super-edge resolution has no
    /// meaningful cell-graph walk). For full fine-resolution greedy with
    /// component-balanced parallelism, use [`Self::run_greedy_by_components`].
    ///
    /// `alpha` is the Dirichlet concentration on community mixing weights;
    /// pass `0.0` to disable the prior.
    ///
    /// Returns the total number of edge moves across all sweeps.
    #[cfg(test)]
    pub fn run_greedy_plain(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        max_sweeps: usize,
        alpha: f64,
    ) -> usize {
        self.run_greedy_plain_with_observer(stats, profiles, max_sweeps, alpha, |_, _| {})
    }

    /// [`Self::run_greedy_plain`] with a per-sweep observer closure called
    /// after each sweep; the observer fires even on the final converged
    /// sweep so the caller sees the end state of the early-exit.
    pub fn run_greedy_plain_with_observer<F>(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        max_sweeps: usize,
        alpha: f64,
        mut observer: F,
    ) -> usize
    where
        F: FnMut(usize, &LinkCommunityStats),
    {
        let k = stats.k;
        let n = stats.n_edges;
        let mut log_probs = vec![0.0f64; k];

        let mut total_moves = 0;

        let pb = new_progress_bar(
            max_sweeps as u64,
            "Greedy {bar:40} {pos}/{len} sweeps ({eta})",
        );

        for sweep in 0..max_sweeps {
            let lw_snap = (alpha > 0.0).then(|| stats.compute_log_weights(alpha));
            let lw_ref = lw_snap.as_deref();

            let mut sweep_moves = 0;
            for e in 0..n {
                let old_c = stats.membership[e];

                compute_log_probs_for_edge(e, stats, profiles, lw_ref, None, &mut log_probs);

                let new_c = argmax_log(&log_probs);

                if new_c != old_c {
                    stats.delta_move(e, old_c, new_c, profiles);
                    sweep_moves += 1;
                }
            }
            total_moves += sweep_moves;
            observer(sweep, stats);
            pb.inc(1);
            if sweep_moves == 0 {
                break;
            }
        }
        pb.finish_and_clear();

        total_moves
    }

    /// Memoized component-partitioned sweeps with configurable move strategy.
    ///
    /// Edges are partitioned by connected component (balanced across threads).
    /// Each sweep: partitions run sequential edge updates in parallel using a
    /// frozen snapshot of global stats, then global stats are patched with
    /// per-partition deltas.
    ///
    /// `strategy` controls how edges are reassigned:
    /// - `Sample`: Gibbs sampling (stochastic)
    /// - `Greedy`: argmax (deterministic, early exit when no moves)
    fn run_components(
        &mut self,
        membership: &mut [usize],
        profiles: &LinkProfileStore,
        args: &ComponentGibbsArgs,
        num_sweeps: usize,
        strategy: MoveStrategy,
    ) -> usize {
        let ComponentGibbsArgs {
            graph,
            edges,
            k,
            alpha,
            incidence,
        } = args;
        let (k, alpha) = (*k, *alpha);
        let incidence_cfg = *incidence;

        let (comp_labels, n_comp) = connected_components(graph);
        let comp_edges = partition_edges_balanced(edges, &comp_labels, n_comp);
        let n_parts = comp_edges.len();

        let sub_stores: Vec<LinkProfileStore> = comp_edges
            .iter()
            .map(|indices| profiles.subset(indices))
            .collect();

        // Per-component endpoint slice (indexed by local edge id), only
        // needed when incidence is enabled — avoids an unconditional
        // ~|E|·sizeof((usize,usize)) allocation in the baseline path.
        let local_edges: Vec<Vec<(usize, usize)>> = if incidence_cfg.is_some() {
            comp_edges
                .iter()
                .map(|indices| indices.iter().map(|&e| edges[e]).collect())
                .collect()
        } else {
            Vec::new()
        };

        let label = match strategy {
            MoveStrategy::Sample => "EM Gibbs",
            MoveStrategy::Greedy => "Greedy",
        };

        if matches!(strategy, MoveStrategy::Sample) {
            info!(
                "Memoized {}: {} components -> {} partitions (edges: {})",
                label,
                n_comp,
                n_parts,
                comp_edges
                    .iter()
                    .map(|e| e.len().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        let mut global_stats = LinkCommunityStats::from_profiles(profiles, k, membership);

        let mut memo_stats: Vec<_> = comp_edges
            .iter()
            .map(|indices| LinkCommunityStats::component_stats(profiles, k, indices, membership))
            .collect();

        let mut total_moves = 0usize;
        let base_seed = self.rng.random::<u64>() | 1;

        let pb = new_progress_bar(
            (num_sweeps * n_parts) as u64,
            &format!("{label} {{bar:40}} {{pos}}/{{len}} jobs ({{eta}})"),
        );

        for sweep in 0..num_sweeps {
            let sweep_seed = base_seed.wrapping_mul(sweep as u64 + 1);
            let gene_sum_snap = &global_stats.gene_sum;
            let size_sum_snap = &global_stats.size_sum;
            let edge_count_snap = &global_stats.edge_count;
            let log_gene_snap = &global_stats.log_gene;
            let log_size_offset_snap = &global_stats.log_size_offset;

            // Dirichlet log-weights snapshot (recomputed each sweep)
            let lw_snap = if alpha > 0.0 {
                Some(global_stats.compute_log_weights(alpha))
            } else {
                None
            };
            let lw_ref = lw_snap.as_deref();

            let local_results: Vec<ComponentResult> = (0..n_parts)
                .into_par_iter()
                .map(|c| {
                    let indices = &comp_edges[c];
                    if indices.is_empty() {
                        pb.inc(1);
                        return ComponentResult::empty(k, profiles.m);
                    }

                    let mut local_stats = LinkCommunityStats {
                        k,
                        m: profiles.m,
                        n_edges: sub_stores[c].n_edges,
                        gene_sum: gene_sum_snap.clone(),
                        size_sum: size_sum_snap.clone(),
                        edge_count: edge_count_snap.clone(),
                        membership: indices.iter().map(|&e| membership[e]).collect(),
                        log_gene: log_gene_snap.clone(),
                        log_size_offset: log_size_offset_snap.clone(),
                    };

                    // Frozen incidence context: log_B and propensity are
                    // both global read-only weights; the only per-component
                    // piece is the endpoint slice indexed by local edge id.
                    let inc_ctx = incidence_cfg.map(|cfg| IncidenceCtx {
                        edges: local_edges[c].as_slice(),
                        propensity: cfg.propensity,
                        log_incidence: cfg.log_incidence,
                    });

                    let comp_seed = sweep_seed ^ (c as u64).wrapping_mul(2654435761);
                    let mut rng = SmallRng::seed_from_u64(comp_seed);
                    let mut log_probs = vec![0.0f64; k];
                    let mut moves = 0usize;

                    for e in 0..indices.len() {
                        let old_c = local_stats.membership[e];
                        compute_log_probs_for_edge(
                            e,
                            &local_stats,
                            &sub_stores[c],
                            lw_ref,
                            inc_ctx,
                            &mut log_probs,
                        );
                        let new_c = match strategy {
                            MoveStrategy::Sample => sample_categorical_log(&log_probs, &mut rng),
                            MoveStrategy::Greedy => argmax_log(&log_probs),
                        };
                        if new_c != old_c {
                            local_stats.delta_move(e, old_c, new_c, &sub_stores[c]);
                            moves += 1;
                        }
                    }

                    let new_comp_stats =
                        LinkCommunityStats::local_stats(&sub_stores[c], k, &local_stats.membership);

                    pb.inc(1);

                    ComponentResult {
                        membership: local_stats.membership,
                        new_stats: new_comp_stats,
                        moves,
                    }
                })
                .collect();

            let mut sweep_moves = 0usize;
            for (c, result) in local_results.into_iter().enumerate() {
                sweep_moves += result.moves;

                for (local_e, &global_e) in comp_edges[c].iter().enumerate() {
                    membership[global_e] = result.membership[local_e];
                }

                let old = (
                    memo_stats[c].0.as_slice(),
                    memo_stats[c].1.as_slice(),
                    memo_stats[c].2.as_slice(),
                );
                let new = (
                    result.new_stats.0.as_slice(),
                    result.new_stats.1.as_slice(),
                    result.new_stats.2.as_slice(),
                );
                global_stats.apply_delta(old, new);
                memo_stats[c] = result.new_stats;
            }

            total_moves += sweep_moves;
            if matches!(strategy, MoveStrategy::Greedy) && sweep_moves == 0 {
                break;
            }
        }
        pb.finish_and_clear();

        total_moves
    }

    /// Memoized EM Gibbs sampling across balanced partitions.
    pub fn run_components_em(
        &mut self,
        membership: &mut [usize],
        profiles: &LinkProfileStore,
        args: &ComponentGibbsArgs,
        num_sweeps: usize,
    ) -> usize {
        self.run_components(membership, profiles, args, num_sweeps, MoveStrategy::Sample)
    }

    /// Memoized greedy finalization across balanced partitions.
    pub fn run_greedy_by_components(
        &mut self,
        membership: &mut [usize],
        profiles: &LinkProfileStore,
        args: &ComponentGibbsArgs,
        max_sweeps: usize,
    ) -> usize {
        self.run_components(membership, profiles, args, max_sweeps, MoveStrategy::Greedy)
    }
}

/// Result from a single component's Gibbs/greedy sweep.
struct ComponentResult {
    membership: Vec<usize>,
    new_stats: (Vec<f64>, Vec<f64>, Vec<usize>),
    moves: usize,
}

impl ComponentResult {
    fn empty(k: usize, m: usize) -> Self {
        ComponentResult {
            membership: Vec::new(),
            new_stats: (vec![0.0; k * m], vec![0.0; k], vec![0; k]),
            moves: 0,
        }
    }
}

/// Partition edge indices by connected component.
///
/// Returns `n_comp` vectors, each containing the global edge indices belonging
/// to that component. Edge `(i, j)` is assigned to the component of node `i`.
fn partition_edges_by_component(
    edges: &[(usize, usize)],
    comp_labels: &[usize],
    n_comp: usize,
) -> Vec<Vec<usize>> {
    let mut comp_edges: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for (e, &(i, _j)) in edges.iter().enumerate() {
        comp_edges[comp_labels[i]].push(e);
    }
    comp_edges
}

/// Balanced partitioning: partition by connected component, then split large
/// components so no partition exceeds `total_edges / n_threads`.
///
/// This ensures the memoized EM approach works even for single-component graphs
/// and avoids bottlenecks from imbalanced component sizes.
fn partition_edges_balanced(
    edges: &[(usize, usize)],
    comp_labels: &[usize],
    n_comp: usize,
) -> Vec<Vec<usize>> {
    let n_threads = rayon::current_num_threads().max(1);
    let target_size = (edges.len() / n_threads).max(256);

    let mut comp_edges = partition_edges_by_component(edges, comp_labels, n_comp);

    let mut balanced = Vec::new();
    for group in comp_edges.drain(..) {
        if group.len() <= target_size {
            if !group.is_empty() {
                balanced.push(group);
            }
        } else {
            for chunk in group.chunks(target_size) {
                balanced.push(chunk.to_vec());
            }
        }
    }
    balanced
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

/// Sample from a categorical distribution given log-probabilities.
///
/// Uses the log-sum-exp trick for numerical stability.
pub(crate) fn sample_categorical_log(log_probs: &[f64], rng: &mut SmallRng) -> usize {
    let max = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute total weight without allocating a Vec
    let total: f64 = log_probs.iter().map(|lp| (lp - max).exp()).sum();

    if total <= 0.0 || !total.is_finite() {
        return rng.random_range(0..log_probs.len());
    }

    let u: f64 = rng.random::<f64>() * total;
    let mut cum = 0.0;
    for (i, lp) in log_probs.iter().enumerate() {
        cum += (lp - max).exp();
        if cum >= u {
            return i;
        }
    }

    log_probs.len() - 1
}
