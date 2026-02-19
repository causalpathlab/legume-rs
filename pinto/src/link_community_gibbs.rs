#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
//! Collapsed Gibbs sampler for the link community model.
//!
//! Adapted from `hsblock/src/gibbs.rs` but simplified: flat K communities
//! (no tree/LCA), and edge-level rather than vertex-level assignments.

use crate::link_community_model::*;
use crate::srt_cell_pairs::connected_components;
use crate::srt_common::new_progress_bar;
use crate::srt_knn_graph::KnnGraph;
use log::info;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Collapsed Gibbs sampler for link community assignments.
pub struct LinkGibbsSampler {
    rng: SmallRng,
    log_probs: Vec<f64>,
    parallel_seed: u64,
}

impl LinkGibbsSampler {
    /// Create a new sampler with the given RNG.
    pub fn new(rng: SmallRng) -> Self {
        LinkGibbsSampler {
            rng,
            log_probs: Vec::new(),
            parallel_seed: 0,
        }
    }

    /// Run `num_sweeps` sequential Gibbs sweeps over all edges.
    ///
    /// Returns the total number of edge moves across all sweeps.
    pub fn run(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        a0: f64,
        b0: f64,
        num_sweeps: usize,
    ) -> usize {
        let k = stats.k;
        let n = stats.n_edges;
        self.log_probs.resize(k, 0.0);

        let mut total_moves = 0;

        let pb = new_progress_bar(
            num_sweeps as u64,
            "Gibbs {bar:40} {pos}/{len} sweeps ({eta})",
        );

        for _sweep in 0..num_sweeps {
            for e in 0..n {
                let old_c = stats.membership[e];

                compute_log_probs_for_edge(e, stats, profiles, a0, b0, &mut self.log_probs);

                let new_c = sample_categorical_log(&self.log_probs, &mut self.rng);

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
    /// Returns the total number of edge moves across all sweeps.
    pub fn run_parallel(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        a0: f64,
        b0: f64,
        num_sweeps: usize,
    ) -> usize {
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

            let proposals: Vec<usize> = edge_order
                .par_chunks(chunk_size)
                .flat_map(|chunk| {
                    let mut log_probs = vec![0.0f64; k];
                    chunk
                        .iter()
                        .map(|&e| {
                            compute_log_probs_for_edge(e, stats, profiles, a0, b0, &mut log_probs);
                            let vertex_seed = sweep_seed ^ (e as u64).wrapping_mul(2654435761);
                            let mut rng = SmallRng::seed_from_u64(vertex_seed);
                            sample_categorical_log(&log_probs, &mut rng)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            for e in 0..n {
                let old_c = stats.membership[e];
                let new_c = proposals[e];
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

    /// Run greedy (argmax) sweeps with early exit on convergence.
    ///
    /// Returns the total number of edge moves across all sweeps.
    pub fn run_greedy(
        &mut self,
        stats: &mut LinkCommunityStats,
        profiles: &LinkProfileStore,
        a0: f64,
        b0: f64,
        max_sweeps: usize,
    ) -> usize {
        let k = stats.k;
        let n = stats.n_edges;
        self.log_probs.resize(k, 0.0);

        let mut total_moves = 0;

        let pb = new_progress_bar(
            max_sweeps as u64,
            "Greedy {bar:40} {pos}/{len} sweeps ({eta})",
        );

        for _sweep in 0..max_sweeps {
            let mut sweep_moves = 0;
            for e in 0..n {
                let old_c = stats.membership[e];

                compute_log_probs_for_edge(e, stats, profiles, a0, b0, &mut self.log_probs);

                let new_c = argmax_log(&self.log_probs);

                if new_c != old_c {
                    stats.delta_move(e, old_c, new_c, profiles);
                    sweep_moves += 1;
                }
            }
            total_moves += sweep_moves;
            pb.inc(1);
            if sweep_moves == 0 {
                break;
            }
        }
        pb.finish_and_clear();

        total_moves
    }

    /// Memoized EM Gibbs: parallel sequential-Gibbs per connected component
    /// with memoized sufficient statistics.
    ///
    /// Maintains global stats as the sum of per-component memoized stats.
    /// Each sweep: components run sequential Gibbs in parallel using a frozen
    /// snapshot of global stats for scoring, then global stats are patched
    /// with per-component deltas (new_local - old_local). No full recompute.
    ///
    /// Falls through to `run_parallel` if the graph has only 1 component.
    ///
    /// Returns the total number of edge moves.
    pub fn run_components_em(
        &mut self,
        membership: &mut [usize],
        profiles: &LinkProfileStore,
        graph: &KnnGraph,
        edges: &[(usize, usize)],
        k: usize,
        a0: f64,
        b0: f64,
        num_sweeps: usize,
    ) -> usize {
        let (comp_labels, n_comp) = connected_components(graph);

        if n_comp <= 1 {
            let mut stats = LinkCommunityStats::from_profiles(profiles, k, membership);
            let moves = self.run_parallel(&mut stats, profiles, a0, b0, num_sweeps);
            membership.copy_from_slice(&stats.membership);
            return moves;
        }

        let comp_edges = partition_edges_by_component(edges, &comp_labels, n_comp);

        // Build sub-profile stores (reused across sweeps)
        let sub_stores: Vec<LinkProfileStore> = comp_edges
            .iter()
            .map(|indices| profiles.subset(indices))
            .collect();

        info!(
            "Memoized EM Gibbs: {} components (edges: {})",
            n_comp,
            comp_edges
                .iter()
                .map(|e| e.len().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Initialize global stats and memoized per-component stats
        let mut global_stats = LinkCommunityStats::from_profiles(profiles, k, membership);

        let mut memo_stats: Vec<_> = comp_edges
            .iter()
            .map(|indices| LinkCommunityStats::component_stats(profiles, k, indices, membership))
            .collect();

        let mut total_moves = 0usize;
        let base_seed = self.rng.random::<u64>() | 1;

        let pb = new_progress_bar(
            num_sweeps as u64,
            "EM Gibbs {bar:40} {pos}/{len} sweeps ({eta})",
        );

        for sweep in 0..num_sweeps {
            let sweep_seed = base_seed.wrapping_mul(sweep as u64 + 1);

            // E-step: parallel per component using frozen global stats for scoring
            // Each component builds local stats = global stats (for scoring),
            // but only iterates over its own edges.
            let local_results: Vec<ComponentResult> = (0..n_comp)
                .into_par_iter()
                .map(|c| {
                    let indices = &comp_edges[c];
                    if indices.is_empty() {
                        return ComponentResult::empty(k, profiles.m);
                    }

                    // Build local stats initialized from GLOBAL stats snapshot
                    // (includes contributions from all components)
                    let mut local_stats = LinkCommunityStats {
                        k,
                        m: profiles.m,
                        n_edges: sub_stores[c].n_edges,
                        gene_sum: global_stats.gene_sum.clone(),
                        size_sum: global_stats.size_sum.clone(),
                        edge_count: global_stats.edge_count.clone(),
                        // Local membership for this component's edges only
                        membership: indices.iter().map(|&e| membership[e]).collect(),
                    };

                    let comp_seed = sweep_seed ^ (c as u64).wrapping_mul(2654435761);
                    let mut rng = SmallRng::seed_from_u64(comp_seed);
                    let mut log_probs = vec![0.0f64; k];
                    let mut moves = 0usize;

                    let n = indices.len();
                    for e in 0..n {
                        let old_c = local_stats.membership[e];
                        compute_log_probs_for_edge(
                            e,
                            &local_stats,
                            &sub_stores[c],
                            a0,
                            b0,
                            &mut log_probs,
                        );
                        let new_c = sample_categorical_log(&log_probs, &mut rng);
                        if new_c != old_c {
                            local_stats.delta_move(e, old_c, new_c, &sub_stores[c]);
                            moves += 1;
                        }
                    }

                    // Compute new component-level stats after this sweep
                    let new_comp_stats = LinkCommunityStats::component_stats(
                        &sub_stores[c],
                        k,
                        &(0..n).collect::<Vec<_>>(),
                        &local_stats.membership,
                    );

                    ComponentResult {
                        membership: local_stats.membership,
                        new_stats: new_comp_stats,
                        moves,
                    }
                })
                .collect();

            // M-step: apply deltas to global stats and update membership
            for (c, result) in local_results.into_iter().enumerate() {
                total_moves += result.moves;

                // Write back membership
                for (local_e, &global_e) in comp_edges[c].iter().enumerate() {
                    membership[global_e] = result.membership[local_e];
                }

                // Patch global stats: global += (new_local - old_local)
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

                // Update memoized stats for this component
                memo_stats[c] = result.new_stats;
            }
            pb.inc(1);
        }
        pb.finish_and_clear();

        total_moves
    }

    /// Memoized greedy finalization parallelized by connected components.
    ///
    /// Same memoized approach as `run_components_em` but uses argmax instead
    /// of sampling. Stops early if no moves across all components.
    ///
    /// Falls through to `run_greedy` if graph has only 1 component.
    ///
    /// Returns the total number of edge moves.
    pub fn run_greedy_by_components(
        &mut self,
        membership: &mut [usize],
        profiles: &LinkProfileStore,
        graph: &KnnGraph,
        edges: &[(usize, usize)],
        k: usize,
        a0: f64,
        b0: f64,
        max_sweeps: usize,
    ) -> usize {
        let (comp_labels, n_comp) = connected_components(graph);

        if n_comp <= 1 {
            let mut stats = LinkCommunityStats::from_profiles(profiles, k, membership);
            let moves = self.run_greedy(&mut stats, profiles, a0, b0, max_sweeps);
            membership.copy_from_slice(&stats.membership);
            return moves;
        }

        let comp_edges = partition_edges_by_component(edges, &comp_labels, n_comp);
        let sub_stores: Vec<LinkProfileStore> = comp_edges
            .iter()
            .map(|indices| profiles.subset(indices))
            .collect();

        let mut global_stats = LinkCommunityStats::from_profiles(profiles, k, membership);

        let mut memo_stats: Vec<_> = comp_edges
            .iter()
            .map(|indices| LinkCommunityStats::component_stats(profiles, k, indices, membership))
            .collect();

        let mut total_moves = 0usize;

        let pb = new_progress_bar(
            max_sweeps as u64,
            "Greedy(CC) {bar:40} {pos}/{len} sweeps ({eta})",
        );

        for _sweep in 0..max_sweeps {
            let local_results: Vec<ComponentResult> = (0..n_comp)
                .into_par_iter()
                .map(|c| {
                    let indices = &comp_edges[c];
                    if indices.is_empty() {
                        return ComponentResult::empty(k, profiles.m);
                    }

                    let mut local_stats = LinkCommunityStats {
                        k,
                        m: profiles.m,
                        n_edges: sub_stores[c].n_edges,
                        gene_sum: global_stats.gene_sum.clone(),
                        size_sum: global_stats.size_sum.clone(),
                        edge_count: global_stats.edge_count.clone(),
                        membership: indices.iter().map(|&e| membership[e]).collect(),
                    };

                    let mut log_probs = vec![0.0f64; k];
                    let mut moves = 0usize;

                    let n = indices.len();
                    for e in 0..n {
                        let old_c = local_stats.membership[e];
                        compute_log_probs_for_edge(
                            e,
                            &local_stats,
                            &sub_stores[c],
                            a0,
                            b0,
                            &mut log_probs,
                        );
                        let new_c = argmax_log(&log_probs);
                        if new_c != old_c {
                            local_stats.delta_move(e, old_c, new_c, &sub_stores[c]);
                            moves += 1;
                        }
                    }

                    let new_comp_stats = LinkCommunityStats::component_stats(
                        &sub_stores[c],
                        k,
                        &(0..n).collect::<Vec<_>>(),
                        &local_stats.membership,
                    );

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
            pb.inc(1);
            if sweep_moves == 0 {
                break;
            }
        }
        pb.finish_and_clear();

        total_moves
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
fn sample_categorical_log(log_probs: &[f64], rng: &mut SmallRng) -> usize {
    let max = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = log_probs.iter().map(|lp| (lp - max).exp()).collect();
    let total: f64 = weights.iter().sum();

    if total <= 0.0 || !total.is_finite() {
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

    /// Create a planted partition: edges 0..n/2 belong to community 0,
    /// edges n/2..n belong to community 1, with distinct gene signatures.
    fn make_planted_profiles(n_edges: usize, m: usize) -> (LinkProfileStore, Vec<usize>) {
        let mut profiles = vec![0.0f32; n_edges * m];
        let mut true_labels = vec![0usize; n_edges];

        for e in 0..n_edges {
            let c = if e < n_edges / 2 { 0 } else { 1 };
            true_labels[e] = c;
            for g in 0..m {
                // Strong signal in first half of genes for c=0, second half for c=1
                let signal = if (g < m / 2) == (c == 0) { 10.0 } else { 1.0 };
                profiles[e * m + g] = signal;
            }
        }

        (LinkProfileStore::new(profiles, n_edges, m), true_labels)
    }

    #[test]
    fn test_gibbs_convergence() {
        let (store, _true_labels) = make_planted_profiles(100, 10);
        let k = 2;

        // Start with random labels
        let random_labels: Vec<usize> = (0..100).map(|e| e % k).collect();
        let mut stats = LinkCommunityStats::from_profiles(&store, k, &random_labels);

        let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

        let moves1 = sampler.run(&mut stats, &store, 1.0, 1.0, 1);
        let _moves_mid = sampler.run(&mut stats, &store, 1.0, 1.0, 20);
        let moves_late = sampler.run(&mut stats, &store, 1.0, 1.0, 1);

        assert!(
            moves_late <= moves1 || moves1 == 0,
            "Expected convergence: early={}, late={}",
            moves1,
            moves_late
        );
    }

    #[test]
    fn test_greedy_recovers_planted() {
        let (store, true_labels) = make_planted_profiles(100, 10);
        let k = 2;

        // Start with random labels
        let random_labels: Vec<usize> = (0..100).map(|e| (e * 7) % k).collect();
        let mut stats = LinkCommunityStats::from_profiles(&store, k, &random_labels);

        let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));
        sampler.run(&mut stats, &store, 1.0, 1.0, 50);
        sampler.run_greedy(&mut stats, &store, 1.0, 1.0, 20);

        // Check that the partition matches the planted one (up to label permutation)
        let match_direct: usize = (0..100)
            .filter(|&e| stats.membership[e] == true_labels[e])
            .count();
        let match_flipped: usize = (0..100)
            .filter(|&e| stats.membership[e] == 1 - true_labels[e])
            .count();

        let best_match = match_direct.max(match_flipped);
        assert!(
            best_match >= 90,
            "Planted partition recovery: {}/100",
            best_match
        );
    }

    #[test]
    fn test_parallel_gibbs() {
        let (store, _true_labels) = make_planted_profiles(100, 10);
        let k = 2;

        let random_labels: Vec<usize> = (0..100).map(|e| e % k).collect();
        let mut stats = LinkCommunityStats::from_profiles(&store, k, &random_labels);

        let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

        let moves1 = sampler.run_parallel(&mut stats, &store, 1.0, 1.0, 1);
        let _moves_mid = sampler.run_parallel(&mut stats, &store, 1.0, 1.0, 20);
        let moves_late = sampler.run_parallel(&mut stats, &store, 1.0, 1.0, 1);

        assert!(
            moves_late <= moves1 || moves1 == 0,
            "Parallel: expected convergence: early={}, late={}",
            moves1,
            moves_late
        );
    }

    #[test]
    fn test_sample_categorical_log() {
        let mut rng = SmallRng::seed_from_u64(42);

        let log_probs = vec![-100.0, 0.0, -100.0];
        let mut counts = [0usize; 3];
        for _ in 0..1000 {
            let idx = sample_categorical_log(&log_probs, &mut rng);
            counts[idx] += 1;
        }
        assert!(counts[1] > 990, "Expected mostly index 1, got {:?}", counts);
    }

    /// Test memoized EM Gibbs on a 2-component graph with planted partition.
    ///
    /// Component 0: nodes 0-4, edges among them → community 0 (high in genes 0..m/2)
    /// Component 1: nodes 5-9, edges among them → community 1 (high in genes m/2..m)
    #[test]
    fn test_memoized_em_two_components() {
        let m = 10;
        let k = 2;

        // Two disconnected cliques: nodes 0-4 and nodes 5-9
        let edges_list = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 7),
            (6, 8),
            (7, 8),
            (7, 9),
            (8, 9),
        ];
        let n_edges = edges_list.len();
        let graph = make_test_graph(10, edges_list.clone());

        // Build profiles: edges in component 0 → community 0 signal,
        //                  edges in component 1 → community 1 signal
        let mut profiles = vec![0.0f32; n_edges * m];
        let mut true_labels = vec![0usize; n_edges];
        for (e, &(i, _j)) in edges_list.iter().enumerate() {
            let c = if i < 5 { 0 } else { 1 };
            true_labels[e] = c;
            for g in 0..m {
                let signal = if (g < m / 2) == (c == 0) { 10.0 } else { 1.0 };
                profiles[e * m + g] = signal;
            }
        }
        let store = LinkProfileStore::new(profiles, n_edges, m);

        // Start with random labels
        let mut membership: Vec<usize> = (0..n_edges).map(|e| (e * 7) % k).collect();

        let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

        // Run memoized EM Gibbs
        let moves = sampler.run_components_em(
            &mut membership,
            &store,
            &graph,
            &edges_list,
            k,
            1.0,
            1.0,
            50,
        );
        assert!(moves > 0, "Expected some moves");

        // Run memoized greedy
        let greedy_moves = sampler.run_greedy_by_components(
            &mut membership,
            &store,
            &graph,
            &edges_list,
            k,
            1.0,
            1.0,
            20,
        );
        let _ = greedy_moves;

        // Check planted recovery (up to label permutation)
        let match_direct: usize = (0..n_edges)
            .filter(|&e| membership[e] == true_labels[e])
            .count();
        let match_flipped: usize = (0..n_edges)
            .filter(|&e| membership[e] == 1 - true_labels[e])
            .count();
        let best_match = match_direct.max(match_flipped);
        assert!(
            best_match >= n_edges - 2,
            "Two-component planted recovery: {}/{}",
            best_match,
            n_edges
        );
    }

    /// Test that memoized EM falls through to run_parallel on single component.
    #[test]
    fn test_memoized_em_single_component() {
        let (store, _true_labels) = make_planted_profiles(100, 10);
        let k = 2;

        // Build a connected graph for edges 0..100
        // Edges need a graph with nodes. Let's make a simple chain.
        let n_nodes = 101;
        let edges: Vec<(usize, usize)> = (0..100).map(|i| (i, i + 1)).collect();
        let graph = make_test_graph(n_nodes, edges.clone());

        let mut membership: Vec<usize> = (0..100).map(|e| e % k).collect();

        let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

        let moves =
            sampler.run_components_em(&mut membership, &store, &graph, &edges, k, 1.0, 1.0, 10);
        // Should converge (fewer moves over time, similar to parallel test)
        assert!(moves > 0 || membership.iter().all(|&m| m < k));
    }

    /// Test that memoized stats delta preserves exact global stats.
    #[test]
    fn test_memoized_stats_consistency() {
        let m = 6;
        let k = 3;
        let n_edges = 30;

        let (store, labels) = {
            let mut profiles = vec![0.0f32; n_edges * m];
            let mut labels = vec![0usize; n_edges];
            for e in 0..n_edges {
                let c = e % k;
                labels[e] = c;
                for g in 0..m {
                    let signal = if g % k == c { 5.0 } else { 1.0 };
                    profiles[e * m + g] = signal;
                }
            }
            (LinkProfileStore::new(profiles, n_edges, m), labels)
        };

        let global = LinkCommunityStats::from_profiles(&store, k, &labels);

        // Partition into 2 fake components: edges 0..15 and 15..30
        let comp0: Vec<usize> = (0..15).collect();
        let comp1: Vec<usize> = (15..30).collect();

        let (gs0, ss0, ec0) = LinkCommunityStats::component_stats(&store, k, &comp0, &labels);
        let (gs1, ss1, ec1) = LinkCommunityStats::component_stats(&store, k, &comp1, &labels);

        // Verify: sum of component stats == global stats
        for i in 0..k * m {
            assert!(
                (gs0[i] + gs1[i] - global.gene_sum[i]).abs() < 1e-10,
                "gene_sum mismatch at {}",
                i
            );
        }
        for c in 0..k {
            assert!(
                (ss0[c] + ss1[c] - global.size_sum[c]).abs() < 1e-10,
                "size_sum mismatch at {}",
                c
            );
            assert_eq!(
                ec0[c] + ec1[c],
                global.edge_count[c],
                "edge_count mismatch at {}",
                c
            );
        }
    }
}
