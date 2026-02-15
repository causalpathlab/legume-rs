#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
//! Collapsed Gibbs sampler for the link community model.
//!
//! Adapted from `hsblock/src/gibbs.rs` but simplified: flat K communities
//! (no tree/LCA), and edge-level rather than vertex-level assignments.

use crate::link_community_model::*;
use crate::srt_common::new_progress_bar;
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
        stats: &mut LinkCommunitySuffStats,
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
        stats: &mut LinkCommunitySuffStats,
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
        stats: &mut LinkCommunitySuffStats,
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
        let mut stats = LinkCommunitySuffStats::from_profiles(&store, k, &random_labels);

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
        let mut stats = LinkCommunitySuffStats::from_profiles(&store, k, &random_labels);

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
        let mut stats = LinkCommunitySuffStats::from_profiles(&store, k, &random_labels);

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
}
