//! V-cycle cascade through the coarsening pyramid for link community fitting.
//!
//! At each coarsening level `l = 0..L` (coarsest to finest), build super-edges
//! from the pyramid's cell partition, build edge profiles, and run Gibbs +
//! plain greedy. Labels propagate coarser → finer through the fine-edge
//! intermediary: because the pyramid's cell partitions are nested, every
//! fine edge inside a super-edge at level `l` shares the same super-edge at
//! level `l-1`, so the label inheritance is a clean injection (no majority
//! vote).
//!
//! The finest pyramid level's labels are broadcast to full fine edges and
//! returned as the warm start for the downstream component-EM stage.

use crate::link_community::gibbs::LinkGibbsSampler;
use crate::link_community::model::*;
use crate::link_community::outputs::{write_level_outputs, ScoreEntry};
use crate::link_community::profiles::*;
use crate::util::common::*;

/// Gene-pair profile context for building edge profiles from an external
/// gene-gene network. Set up once at the coarsest level (elbow filter, module
/// collapse) and reused for every cascade level + the final EM stage.
pub struct GenePairProfileState {
    pub gene_adj: Vec<Vec<(usize, usize)>>,
    pub gene_means: DVec,
    pub n_gene_pairs: usize,
    pub module_collapse: Option<Vec<usize>>,
}

/// Result of a V-cycle cascade run.
pub struct CascadeResult {
    /// Full-fine-edge labels obtained by broadcasting the finest pyramid
    /// level's super-edge membership through `transfer_labels`. Warm start
    /// for the final component-EM stage.
    pub fine_labels: Vec<usize>,
    /// One [`ScoreEntry`] per Gibbs + greedy sweep across all cascade levels
    /// that actually ran (skipped too-coarse levels are absent).
    pub score_trace: Vec<ScoreEntry>,
}

/// Knobs for the V-cycle cascade: sweep budgets, Dirichlet prior, and
/// per-level output controls.
pub struct CascadeConfig {
    pub k: usize,
    pub num_gibbs: usize,
    pub num_greedy: usize,
    pub user_alpha: Option<f32>,
    pub block_size: Option<usize>,
    pub no_level_outputs: bool,
}

/// Run Gibbs + greedy at every pyramid level, emitting per-level outputs.
#[allow(clippy::too_many_arguments)]
pub fn run_cascade(
    out_prefix: &str,
    edges: &[(usize, usize)],
    level_cell_labels: &[Vec<usize>],
    data_vec: &SparseIoVec,
    gp_state: Option<&GenePairProfileState>,
    proj_basis: Option<&Mat>,
    cfg: &CascadeConfig,
    sampler: &mut LinkGibbsSampler,
    cell_names: &[Box<str>],
) -> anyhow::Result<CascadeResult> {
    anyhow::ensure!(
        !level_cell_labels.is_empty(),
        "cascade needs at least one level"
    );

    let CascadeConfig {
        k,
        num_gibbs,
        num_greedy,
        user_alpha,
        block_size,
        no_level_outputs,
    } = *cfg;

    let n_cells = data_vec.num_columns();
    let n_levels = level_cell_labels.len();
    // Upper bound on per-sweep trace entries: each level runs up to
    // (num_gibbs + num_greedy) sweeps; pre-size to avoid reallocations.
    let mut score_trace: Vec<ScoreEntry> =
        Vec::with_capacity(n_levels * (num_gibbs + num_greedy + 1));
    let skip_below = (2 * k).max(4);

    // Carried between iterations: previous level's (fine_to_super, super_membership).
    let mut prev_fine_to_super: Option<Vec<usize>> = None;
    let mut prev_super_membership: Option<Vec<usize>> = None;

    for l in 0..n_levels {
        // Super-edges at level l.
        let cell_labels_l = &level_cell_labels[l];
        let (super_edges_l, fine_to_super_l) = build_super_edges(edges, cell_labels_l);
        let n_super_l = super_edges_l.len();
        let n_cell_groups = 1 + cell_labels_l.iter().copied().max().unwrap_or(0);

        info!(
            "Cascade L{}/{}: {} cell groups -> {} super-edges (from {} fine)",
            l,
            n_levels - 1,
            n_cell_groups,
            n_super_l,
            edges.len()
        );

        // Skip levels too coarse for K communities — Gibbs is wasted there.
        if n_super_l < skip_below {
            info!(
                "  L{}: skipping ({} super-edges < 2·K = {}); insufficient resolution for K={}",
                l, n_super_l, skip_below, k
            );
            continue;
        }

        let profiles_l =
            build_profiles_for_edges(data_vec, &super_edges_l, gp_state, proj_basis, block_size)?;

        // Initialize labels: round-robin if this is the first level we run
        // (no previous-level state), transfer from previous level otherwise.
        let is_first_run_level = prev_fine_to_super.is_none();
        let init_labels = init_level_labels(
            &fine_to_super_l,
            n_super_l,
            prev_fine_to_super.as_deref(),
            prev_super_membership.as_deref(),
            k,
        );

        // Alpha auto-scaled from this level's mean size factor (user override
        // is honoured). Dirichlet prior on community mixing weights — matches
        // the signal strength at each resolution.
        let mean_sf = profiles_l.size_factors.iter().sum::<f32>() / n_super_l.max(1) as f32;
        let alpha_l = user_alpha
            .map(|v| v as f64)
            .unwrap_or_else(|| (mean_sf as f64 / k as f64).max(0.01));
        info!(
            "  L{}: mean size factor {:.2}, α_l={:.4}",
            l, mean_sf, alpha_l
        );

        // Fit stats, Gibbs, greedy. The first run level gets the full
        // `num_gibbs` budget (cold start); subsequent levels are warm-started
        // and use a smaller budget.
        let mut stats_l = LinkCommunityStats::from_profiles(&profiles_l, k, &init_labels);

        let sweeps = if is_first_run_level {
            num_gibbs
        } else {
            (num_gibbs / 5).max(10)
        };

        // Per-sweep tracing: Gibbs sweeps first (0..num_gibbs-1), then greedy
        // sweeps continue from the same cursor. The observer fires AFTER
        // each sweep, so stats reflect the post-sweep state — the final
        // emitted entry for this level is its end summary.
        let level_i32 = l as i32;
        let mut sweep_cursor: i32 = 0;
        let mut push_entry = |stats: &LinkCommunityStats, cursor: &mut i32| {
            let (score, mi) = stats.score_and_mi();
            score_trace.push(ScoreEntry {
                level: level_i32,
                sweep: *cursor,
                score,
                n_edges: n_super_l,
                total_mass: stats.size_sum.iter().sum(),
                mutual_information: mi,
            });
            *cursor += 1;
        };

        let moves = sampler.run_parallel_with_observer(
            &mut stats_l,
            &profiles_l,
            sweeps,
            alpha_l,
            |_, stats| push_entry(stats, &mut sweep_cursor),
        );
        info!("  L{}: Gibbs {} sweeps, {} moves", l, sweeps, moves);

        let greedy_moves = sampler.run_greedy_plain_with_observer(
            &mut stats_l,
            &profiles_l,
            num_greedy,
            alpha_l,
            |_, stats| push_entry(stats, &mut sweep_cursor),
        );
        info!("  L{}: greedy {} moves", l, greedy_moves);

        // Summary log — the tail entry in `score_trace` is this level's end state.
        if let Some(last) = score_trace.last() {
            info!(
                "  L{}: final score {:.4e}, mass {:.4e}, score/mass {:.4e}, MI {:.4} nats",
                l,
                last.score,
                last.total_mass,
                if last.total_mass > 0.0 {
                    last.score / last.total_mass
                } else {
                    0.0
                },
                last.mutual_information
            );
        }

        let labels_l = stats_l.membership.clone();

        // Per-level outputs (optional).
        if !no_level_outputs {
            let fine_labels_l = transfer_labels(&fine_to_super_l, &labels_l);
            write_level_outputs(
                out_prefix,
                l,
                edges,
                &fine_labels_l,
                n_cells,
                k,
                cell_names,
                data_vec,
                block_size,
            )?;
        }

        prev_fine_to_super = Some(fine_to_super_l);
        prev_super_membership = Some(labels_l);
    }

    // Broadcast finest-level super-edge labels to full fine edges. If every
    // pyramid level was below the skip threshold (rare: would mean K is large
    // relative to the finest pyramid resolution), fall back to a round-robin
    // init so the downstream component-EM still has something to work with.
    let fine_labels = match (prev_fine_to_super.as_ref(), prev_super_membership.as_ref()) {
        (Some(f2s), Some(mem)) => transfer_labels(f2s, mem),
        _ => {
            info!(
                "Cascade ran zero levels (all below 2·K={} super-edge threshold); \
                 initializing fine edges round-robin",
                skip_below
            );
            (0..edges.len()).map(|e| e % k).collect()
        }
    };

    Ok(CascadeResult {
        fine_labels,
        score_trace,
    })
}

/// Dispatch to gene-pair vs projection-basis profile building.
fn build_profiles_for_edges(
    data: &SparseIoVec,
    super_edges: &[(usize, usize)],
    gp_state: Option<&GenePairProfileState>,
    proj_basis: Option<&Mat>,
    block_size: Option<usize>,
) -> anyhow::Result<LinkProfileStore> {
    match (gp_state, proj_basis) {
        (Some(gp), _) => {
            let raw = build_edge_profiles_by_gene_pairs(
                data,
                super_edges,
                &gp.gene_adj,
                &gp.gene_means,
                gp.n_gene_pairs,
                block_size,
            )?;
            Ok(match gp.module_collapse.as_ref() {
                Some(collapse) => raw.collapse_modules(collapse),
                None => raw,
            })
        }
        (None, Some(basis)) => build_edge_profiles(data, super_edges, basis, block_size),
        (None, None) => {
            anyhow::bail!("cascade requires either gene-pair state or projection basis")
        }
    }
}

/// Initialize super-edge labels at level l.
///
/// `l == 0` (no previous level): round-robin `e % k`.
///
/// `l > 0`: transfer from the previous (coarser) level via the fine-edge
/// intermediary. Because the pyramid partitions are nested, every fine edge
/// inside a super-edge at level l maps to the same super-edge at level l-1 —
/// so we just take the label of the first fine edge belonging to each
/// super-edge at level l (no majority vote).
fn init_level_labels(
    fine_to_super_l: &[usize],
    n_super_l: usize,
    prev_fine_to_super: Option<&[usize]>,
    prev_super_membership: Option<&[usize]>,
    k: usize,
) -> Vec<usize> {
    match (prev_fine_to_super, prev_super_membership) {
        (Some(prev_f2s), Some(prev_mem)) => {
            let fine_labels_prev = transfer_labels(prev_f2s, prev_mem);
            let mut init = vec![usize::MAX; n_super_l];
            for (e, &se) in fine_to_super_l.iter().enumerate() {
                if init[se] == usize::MAX {
                    init[se] = fine_labels_prev[e];
                }
            }
            for (s, label) in init.iter_mut().enumerate() {
                if *label == usize::MAX {
                    *label = s % k;
                }
            }
            init
        }
        _ => (0..n_super_l).map(|e| e % k).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nested-partition label cascade: super-edges at a finer level inherit
    /// the coarser level's community via the fine-edge intermediary.
    #[test]
    fn level_to_level_label_injection() {
        // 6 fine edges over 6 cells.
        let fine_edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)];

        // Coarser partition: cells {0,1,2} → 0, {3,4,5} → 1. 3 super-edges:
        //   edges (0,1),(0,2),(1,2) → (0,0)  super 0
        //   edge  (2,3)             → (0,1)  super 1
        //   edges (3,4),(4,5)       → (1,1)  super 2
        let coarse_cell_labels = vec![0, 0, 0, 1, 1, 1];
        let (_coarse_super_edges, coarse_f2s) = build_super_edges(&fine_edges, &coarse_cell_labels);
        // Plant coarse labels: super 0 → community 7, super 1 → 8, super 2 → 9.
        let coarse_membership = vec![7, 8, 9];

        // Finer partition: refines the coarse one. {0,1}→0, {2}→1, {3,4}→2, {5}→3.
        // fine super-edges:
        //   (0,1): (0,0) super 0
        //   (0,2): (0,1) super 1
        //   (1,2): (0,1) super 1
        //   (2,3): (1,2) super 2
        //   (3,4): (2,2) super 3
        //   (4,5): (2,3) super 4
        let fine_cell_labels = vec![0, 0, 1, 2, 2, 3];
        let (_fine_super_edges, fine_f2s) = build_super_edges(&fine_edges, &fine_cell_labels);

        let init = init_level_labels(
            &fine_f2s,
            5, // 5 fine super-edges
            Some(&coarse_f2s),
            Some(&coarse_membership),
            10,
        );

        // Expected inheritance:
        //   super 0 (edge (0,1))      → coarse super 0 → 7
        //   super 1 (edges (0,2),(1,2)) → coarse super 0 → 7
        //   super 2 (edge (2,3))      → coarse super 1 → 8
        //   super 3 (edge (3,4))      → coarse super 2 → 9
        //   super 4 (edge (4,5))      → coarse super 2 → 9
        assert_eq!(init, vec![7, 7, 8, 9, 9]);
    }

    /// At the coarsest level, fall back to round-robin initialization.
    #[test]
    fn level_zero_round_robin() {
        let init = init_level_labels(&[0, 1, 2, 3], 4, None, None, 3);
        assert_eq!(init, vec![0, 1, 2, 0]);
    }
}
