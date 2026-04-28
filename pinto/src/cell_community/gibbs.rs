//! Flat-K collapsed Gibbs sampler for cell community assignments.
//!
//! Ported from `link_community::gibbs` with edge→cell substitution.
//! Cells are the sampled units; connected components of the KNN graph
//! partition them for the memoized EM variant.

use super::model::*;
use super::profiles::CellProfileStore;
use crate::util::cell_pairs::connected_components;
use crate::util::common::new_progress_bar;
use crate::util::knn_graph::KnnGraph;
use log::info;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum MoveStrategy {
    Sample,
    Greedy,
}

pub struct ComponentGibbsArgs<'a> {
    pub graph: &'a KnnGraph,
    pub k: usize,
    pub alpha: f64,
}

pub struct CellGibbsSampler {
    rng: SmallRng,
    parallel_seed: u64,
}

impl CellGibbsSampler {
    pub fn new(rng: SmallRng) -> Self {
        Self {
            rng,
            parallel_seed: 0,
        }
    }

    #[cfg(test)]
    pub fn run(
        &mut self,
        stats: &mut CellCommunityStats,
        profiles: &CellProfileStore,
        num_sweeps: usize,
    ) -> usize {
        let k = stats.k;
        let n = stats.n_cells;
        let mut log_probs = vec![0.0f64; k];
        let mut total_moves = 0;
        for _ in 0..num_sweeps {
            for u in 0..n {
                let old_c = stats.membership[u];
                compute_log_probs_for_cell(u, stats, profiles, None, &mut log_probs);
                let new_c = sample_categorical_log(&log_probs, &mut self.rng);
                if new_c != old_c {
                    stats.delta_move(u, old_c, new_c, profiles);
                    total_moves += 1;
                }
            }
        }
        total_moves
    }

    /// Parallel proposal + sequential apply (Jacobi).
    pub fn run_parallel(
        &mut self,
        stats: &mut CellCommunityStats,
        profiles: &CellProfileStore,
        num_sweeps: usize,
    ) -> usize {
        let k = stats.k;
        let n = stats.n_cells;
        if self.parallel_seed == 0 {
            self.parallel_seed = self.rng.random::<u64>() | 1;
        }
        let base_seed = self.parallel_seed;
        let mut total_moves = 0;
        let order: Vec<usize> = (0..n).collect();
        let chunk_size = std::cmp::max(256, n / rayon::current_num_threads().max(1));

        let pb = new_progress_bar(
            num_sweeps as u64,
            "Cell Gibbs {bar:40} {pos}/{len} sweeps ({eta})",
        );

        for sweep in 0..num_sweeps {
            let sweep_seed = base_seed.wrapping_mul(sweep as u64 + 1);
            let proposals: Vec<usize> = order
                .par_chunks(chunk_size)
                .flat_map(|chunk| {
                    let mut log_probs = vec![0.0f64; k];
                    chunk
                        .iter()
                        .map(|&u| {
                            compute_log_probs_for_cell(u, stats, profiles, None, &mut log_probs);
                            let vertex_seed = sweep_seed ^ (u as u64).wrapping_mul(2654435761);
                            let mut rng = SmallRng::seed_from_u64(vertex_seed);
                            sample_categorical_log(&log_probs, &mut rng)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            for (u, &new_c) in proposals.iter().enumerate() {
                let old_c = stats.membership[u];
                if new_c != old_c {
                    stats.delta_move(u, old_c, new_c, profiles);
                    total_moves += 1;
                }
            }
            pb.inc(1);
        }
        pb.finish_and_clear();
        total_moves
    }

    /// Parallel greedy (argmax) with early exit on convergence.
    #[cfg(test)]
    pub fn run_greedy(
        &mut self,
        stats: &mut CellCommunityStats,
        profiles: &CellProfileStore,
        max_sweeps: usize,
    ) -> usize {
        let k = stats.k;
        let n = stats.n_cells;
        let mut total_moves = 0;
        let order: Vec<usize> = (0..n).collect();
        let chunk_size = std::cmp::max(256, n / rayon::current_num_threads().max(1));

        let pb = new_progress_bar(
            max_sweeps as u64,
            "Cell greedy {bar:40} {pos}/{len} sweeps ({eta})",
        );
        for _ in 0..max_sweeps {
            let proposals: Vec<usize> = order
                .par_chunks(chunk_size)
                .flat_map(|chunk| {
                    let mut log_probs = vec![0.0f64; k];
                    chunk
                        .iter()
                        .map(|&u| {
                            compute_log_probs_for_cell(u, stats, profiles, None, &mut log_probs);
                            argmax_log(&log_probs)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let mut sweep_moves = 0;
            for (u, &new_c) in proposals.iter().enumerate() {
                let old_c = stats.membership[u];
                if new_c != old_c {
                    stats.delta_move(u, old_c, new_c, profiles);
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

    /// Memoized component-partitioned sweeps.
    ///
    /// Cells are partitioned by connected component of `args.graph`; balanced so
    /// no partition exceeds total_cells / n_threads.
    fn run_components(
        &mut self,
        membership: &mut [usize],
        profiles: &CellProfileStore,
        args: &ComponentGibbsArgs,
        num_sweeps: usize,
        strategy: MoveStrategy,
    ) -> usize {
        let ComponentGibbsArgs { graph, k, alpha } = args;
        let (k, alpha) = (*k, *alpha);

        let (comp_labels, n_comp) = connected_components(graph);
        let comp_cells = partition_cells_balanced(profiles.n_cells, &comp_labels, n_comp);
        let n_parts = comp_cells.len();

        let sub_stores: Vec<CellProfileStore> = comp_cells
            .iter()
            .map(|indices| subset_store(profiles, indices))
            .collect();

        let label = match strategy {
            MoveStrategy::Sample => "Cell EM Gibbs",
            MoveStrategy::Greedy => "Cell greedy",
        };
        if matches!(strategy, MoveStrategy::Sample) {
            info!(
                "Memoized {}: {} components -> {} partitions (cells: {})",
                label,
                n_comp,
                n_parts,
                comp_cells
                    .iter()
                    .map(|c| c.len().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        let mut global_stats = CellCommunityStats::from_profiles(profiles, k, membership);
        let mut memo_stats: Vec<_> = comp_cells
            .iter()
            .map(|indices| CellCommunityStats::component_stats(profiles, k, indices, membership))
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
            let cell_count_snap = &global_stats.cell_count;
            let log_gene_snap = &global_stats.log_gene;
            let log_size_offset_snap = &global_stats.log_size_offset;

            let lw_snap = if alpha > 0.0 {
                Some(global_stats.compute_log_weights(alpha))
            } else {
                None
            };
            let lw_ref = lw_snap.as_deref();

            let local_results: Vec<ComponentResult> = (0..n_parts)
                .into_par_iter()
                .map(|c| {
                    let indices = &comp_cells[c];
                    if indices.is_empty() {
                        pb.inc(1);
                        return ComponentResult::empty(k, profiles.m);
                    }
                    let mut local_stats = CellCommunityStats {
                        k,
                        m: profiles.m,
                        n_cells: sub_stores[c].n_cells,
                        gene_sum: gene_sum_snap.clone(),
                        size_sum: size_sum_snap.clone(),
                        cell_count: cell_count_snap.clone(),
                        membership: indices.iter().map(|&u| membership[u]).collect(),
                        log_gene: log_gene_snap.clone(),
                        log_size_offset: log_size_offset_snap.clone(),
                    };

                    let comp_seed = sweep_seed ^ (c as u64).wrapping_mul(2654435761);
                    let mut rng = SmallRng::seed_from_u64(comp_seed);
                    let mut log_probs = vec![0.0f64; k];
                    let mut moves = 0usize;

                    for u in 0..indices.len() {
                        let old_c = local_stats.membership[u];
                        compute_log_probs_for_cell(
                            u,
                            &local_stats,
                            &sub_stores[c],
                            lw_ref,
                            &mut log_probs,
                        );
                        let new_c = match strategy {
                            MoveStrategy::Sample => sample_categorical_log(&log_probs, &mut rng),
                            MoveStrategy::Greedy => argmax_log(&log_probs),
                        };
                        if new_c != old_c {
                            local_stats.delta_move(u, old_c, new_c, &sub_stores[c]);
                            moves += 1;
                        }
                    }

                    let new_comp_stats = CellCommunityStats::local_stats(
                        &sub_stores[c],
                        k,
                        &local_stats.membership,
                    );
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
                for (local_u, &global_u) in comp_cells[c].iter().enumerate() {
                    membership[global_u] = result.membership[local_u];
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

    pub fn run_components_em(
        &mut self,
        membership: &mut [usize],
        profiles: &CellProfileStore,
        args: &ComponentGibbsArgs,
        num_sweeps: usize,
    ) -> usize {
        self.run_components(membership, profiles, args, num_sweeps, MoveStrategy::Sample)
    }

    pub fn run_greedy_by_components(
        &mut self,
        membership: &mut [usize],
        profiles: &CellProfileStore,
        args: &ComponentGibbsArgs,
        max_sweeps: usize,
    ) -> usize {
        self.run_components(membership, profiles, args, max_sweeps, MoveStrategy::Greedy)
    }
}

struct ComponentResult {
    membership: Vec<usize>,
    new_stats: (Vec<f64>, Vec<f64>, Vec<usize>),
    moves: usize,
}

impl ComponentResult {
    fn empty(k: usize, m: usize) -> Self {
        Self {
            membership: Vec::new(),
            new_stats: (vec![0.0; k * m], vec![0.0; k], vec![0; k]),
            moves: 0,
        }
    }
}

/// Partition cell indices by connected component, then split large components.
fn partition_cells_balanced(
    n_cells: usize,
    comp_labels: &[usize],
    n_comp: usize,
) -> Vec<Vec<usize>> {
    debug_assert_eq!(comp_labels.len(), n_cells);
    let n_threads = rayon::current_num_threads().max(1);
    let target_size = (n_cells / n_threads).max(256);

    let mut comp_cells: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for (u, &c) in comp_labels.iter().enumerate() {
        comp_cells[c].push(u);
    }

    let mut balanced = Vec::new();
    for group in comp_cells.drain(..) {
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

fn subset_store(profiles: &CellProfileStore, indices: &[usize]) -> CellProfileStore {
    let n = indices.len();
    let m = profiles.m;
    let mut out = vec![0.0f32; n * m];
    for (local, &global) in indices.iter().enumerate() {
        out[local * m..(local + 1) * m].copy_from_slice(profiles.profile(global));
    }
    CellProfileStore::new(out, n, m)
}

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

fn sample_categorical_log(log_probs: &[f64], rng: &mut SmallRng) -> usize {
    let max = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_planted(n_per: usize, m: usize, k: usize) -> (CellProfileStore, Vec<usize>) {
        let n = n_per * k;
        let mut profiles = vec![0.0f32; n * m];
        let mut labels = vec![0usize; n];
        for c in 0..k {
            for i in 0..n_per {
                let u = c * n_per + i;
                labels[u] = c;
                for g in 0..m {
                    profiles[u * m + g] = if g % k == c { 10.0 } else { 1.0 };
                }
            }
        }
        (CellProfileStore::new(profiles, n, m), labels)
    }

    #[test]
    fn test_parallel_gibbs_converges() {
        let (store, _truth) = make_planted(20, 8, 3);
        let init: Vec<usize> = (0..store.n_cells).map(|i| i % 3).collect();
        let mut stats = CellCommunityStats::from_profiles(&store, 3, &init);
        let mut sampler = CellGibbsSampler::new(SmallRng::seed_from_u64(1));
        let _m1 = sampler.run_parallel(&mut stats, &store, 30);
        let m_late = sampler.run_parallel(&mut stats, &store, 1);
        assert!(m_late < stats.n_cells);
    }

    #[test]
    fn test_greedy_recovers_planted() {
        let (store, truth) = make_planted(20, 8, 3);
        let init: Vec<usize> = (0..store.n_cells).map(|i| (i * 7) % 3).collect();
        let mut stats = CellCommunityStats::from_profiles(&store, 3, &init);
        let mut sampler = CellGibbsSampler::new(SmallRng::seed_from_u64(1));
        sampler.run_parallel(&mut stats, &store, 30);
        sampler.run_greedy(&mut stats, &store, 20);

        let mut best = 0usize;
        for c in 0..3 {
            let mut counts = [0usize; 3];
            for u in 0..stats.n_cells {
                if truth[u] == c {
                    counts[stats.membership[u]] += 1;
                }
            }
            best += *counts.iter().max().unwrap();
        }
        assert!(
            best >= stats.n_cells * 9 / 10,
            "planted recovery: {best}/{}",
            stats.n_cells
        );
    }
}
