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

/// Per-cell module-pair state carried across cascade levels.
///
/// Computed once at setup (fit.rs step 4-pre) and referenced by the
/// module-pair profile builder at every level. Super-cell expression is
/// aggregated from `module_expr` per level via [`coarsen_module_expression`].
pub struct ModulePairContext {
    pub basis: ModulePairBasis,
    /// `n_modules × n_cells` dense per-cell module expression.
    pub module_expr: Mat,
    /// Per-cell total (`Σ_m module_expr[m, c]`). Used as the null scale.
    pub cell_totals: Vec<f32>,
}

/// Which basis to use when building edge profiles.
///
/// - `ModulePair` is the gene-network path: per-cell module expression plus
///   a precomputed module-pair basis. Per-level super-cell expression is
///   rebuilt from the fine-cell matrix (fast column-sum aggregation) and
///   fed to [`build_module_pair_profiles_for_edges`].
/// - `Projection` is the no-network default: Gaussian random projection
///   basis over genes, with edge profiles computed directly from the
///   sparse expression matrix at each level.
pub enum ProfileMode<'a> {
    ModulePair {
        basis: &'a ModulePairBasis,
        module_expr: &'a Mat,
        cell_totals: &'a [f32],
    },
    Projection {
        basis: &'a Mat,
    },
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
    /// 0-based indices `l` for which `{out_prefix}.L{l}.*` files were
    /// actually written (cascade levels that ran AND were not skipped via
    /// `no_level_outputs`). Empty if the cascade never wrote per-level
    /// outputs.
    pub written_level_indices: Vec<usize>,
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
///
/// `gene_weights` is the precomputed NB Fisher-info weight vector; when
/// `Some` it is forwarded into per-level `compute_gene_community_stat` calls
/// to avoid re-fitting the dispersion trend at every cascade level.
#[allow(clippy::too_many_arguments)]
pub fn run_cascade(
    out_prefix: &str,
    edges: &[(usize, usize)],
    level_cell_labels: &[Vec<usize>],
    data_vec: &SparseIoVec,
    mode: &ProfileMode<'_>,
    cfg: &CascadeConfig,
    sampler: &mut LinkGibbsSampler,
    cell_names: &[Box<str>],
    gene_weights: Option<&[f32]>,
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
    let mut score_trace: Vec<ScoreEntry> =
        Vec::with_capacity(n_levels * (num_gibbs + num_greedy + 1));
    let skip_below = (2 * k).max(4);

    let mut prev_fine_to_super: Option<Vec<usize>> = None;
    let mut prev_super_membership: Option<Vec<usize>> = None;
    let mut written_level_indices: Vec<usize> = Vec::new();

    for (l, cell_labels_l) in level_cell_labels.iter().enumerate() {
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

        if n_super_l < skip_below {
            info!(
                "  L{}: skipping ({} super-edges < 2·K = {}); insufficient resolution for K={}",
                l, n_super_l, skip_below, k
            );
            continue;
        }

        let super_edge_indices: Vec<usize> = (0..n_super_l).collect();
        let profiles_l = build_level_profiles(
            data_vec,
            &super_edges_l,
            &super_edge_indices,
            mode,
            cell_labels_l,
            n_cell_groups,
            block_size,
        )?;

        let is_first_run_level = prev_fine_to_super.is_none();
        let init_labels = init_level_labels(
            &fine_to_super_l,
            n_super_l,
            prev_fine_to_super.as_deref(),
            prev_super_membership.as_deref(),
            k,
        );

        let mean_sf = profiles_l.size_factors.iter().sum::<f32>() / n_super_l.max(1) as f32;
        let alpha_l = user_alpha
            .map(|v| v as f64)
            .unwrap_or_else(|| (mean_sf as f64 / k as f64).max(0.01));
        info!(
            "  L{}: mean size factor {:.2}, α_l={:.4}",
            l, mean_sf, alpha_l
        );

        let mut stats_l = LinkCommunityStats::from_profiles(&profiles_l, k, &init_labels);

        let sweeps = if is_first_run_level {
            num_gibbs
        } else {
            (num_gibbs / 5).max(10)
        };

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
                gene_weights,
                block_size,
            )?;
            written_level_indices.push(l);
        }

        prev_fine_to_super = Some(fine_to_super_l);
        prev_super_membership = Some(labels_l);
    }

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
        written_level_indices,
    })
}

/// Dispatch to module-pair vs projection profile building at a level.
#[allow(clippy::too_many_arguments)]
fn build_level_profiles(
    data: &SparseIoVec,
    super_edges: &[(usize, usize)],
    super_edge_indices: &[usize],
    mode: &ProfileMode<'_>,
    cell_labels: &[usize],
    n_super_cells: usize,
    block_size: Option<usize>,
) -> anyhow::Result<LinkProfileStore> {
    match *mode {
        ProfileMode::ModulePair {
            basis,
            module_expr,
            cell_totals: _,
        } => {
            // Aggregate fine-cell module expression to super-cells. Each
            // super-edge connects two super-cells; feed the aggregated
            // module expression directly to the module-pair builder.
            let (super_expr, super_totals) =
                coarsen_module_expression(module_expr, cell_labels, n_super_cells);
            Ok(build_module_pair_profiles_for_edges(
                &super_expr,
                &super_totals,
                super_edges,
                super_edge_indices,
                basis,
            ))
        }
        ProfileMode::Projection { basis } => {
            // Coarsen fine-cell raw expression to super-cells, then project
            // super-cell aggregates through the basis. Previously this path
            // passed fine-cell data with cluster-label indices, which read
            // arbitrary fine cells as if they were super-cells — a bug that
            // made super-edge profiles decoupled from super-cell biology.
            let super_expr =
                coarsen_cell_expression_dense(data, cell_labels, n_super_cells, block_size)?;
            Ok(build_super_edge_projection_profiles(
                &super_expr,
                super_edges,
                super_edge_indices,
                basis,
            ))
        }
    }
}

/// Initialize super-edge labels at level l.
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
        let fine_edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)];
        let coarse_cell_labels = vec![0, 0, 0, 1, 1, 1];
        let (_coarse_super_edges, coarse_f2s) = build_super_edges(&fine_edges, &coarse_cell_labels);
        let coarse_membership = vec![7, 8, 9];
        let fine_cell_labels = vec![0, 0, 1, 2, 2, 3];
        let (_fine_super_edges, fine_f2s) = build_super_edges(&fine_edges, &fine_cell_labels);

        let init = init_level_labels(
            &fine_f2s,
            5,
            Some(&coarse_f2s),
            Some(&coarse_membership),
            10,
        );
        assert_eq!(init, vec![7, 7, 8, 9, 9]);
    }

    #[test]
    fn level_zero_round_robin() {
        let init = init_level_labels(&[0, 1, 2, 3], 4, None, None, 3);
        assert_eq!(init, vec![0, 1, 2, 0]);
    }
}
