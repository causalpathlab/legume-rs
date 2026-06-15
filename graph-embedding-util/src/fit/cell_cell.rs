use super::config::CellCellConfig;
use super::samplers::DEGENERATE_PB_RATIO;
use crate::data::UnifiedData;
use crate::loss::{build_per_batch_cell_samplers, PbChainFilter, PerBatchCellSampler};
use log::{info, warn};

pub(crate) struct CellCellPrepared {
    pub(crate) samplers: Vec<Option<PerBatchCellSampler>>,
    pub(crate) edges: Vec<(u32, u32)>,
    pub(crate) lambda: f32,
    pub(crate) n_negatives: usize,
    pub(crate) chain: CellCellPreparedChain,
}

pub(crate) struct CellCellPreparedChain {
    /// Indexes into `cell_to_pb_per_level` (coarsest-first); kept owned
    /// so [`CellCellTraining`] can hand out `&[usize]`.
    pub(crate) levels: Vec<usize>,
    /// Same length as `levels`; user-supplied or uniform 1.0.
    pub(crate) lambdas: Vec<f32>,
}

pub(crate) fn build_cell_cell_training(
    unified: &UnifiedData,
    n_cells: usize,
    alpha_neg: f32,
    cell_cell: Option<CellCellConfig>,
    cell_to_pb_per_level: &[Vec<usize>],
) -> Option<CellCellPrepared> {
    let cc = match cell_cell {
        Some(cc) if cc.lambda > 0.0 && !cc.edges.is_empty() => cc,
        _ => return None,
    };

    let chain = resolve_pb_chain(
        cc.pb_levels.as_deref(),
        cc.lambda_per_level.as_deref(),
        cell_to_pb_per_level,
        n_cells,
    );

    let pb_filter = PbChainFilter {
        cell_to_pb_per_level,
        levels: &chain.levels,
    };

    let (samplers, stats) = build_per_batch_cell_samplers(
        &cc.edges,
        &unified.batch_membership,
        unified.n_batches(),
        n_cells,
        alpha_neg,
        Some(pb_filter),
    );
    let n_active = samplers.iter().filter(|s| s.is_some()).count();
    if stats.cross_batch_dropped > 0 {
        info!(
            "Cell-cell loss: dropped {} cross-batch edges; {} batch(es) have within-batch edges",
            stats.cross_batch_dropped, n_active
        );
    }
    if stats.pb_mismatch_dropped > 0 {
        info!(
            "Cell-cell loss: dropped {} edges whose endpoints disagree on pb at one of the chain levels",
            stats.pb_mismatch_dropped
        );
    }
    if n_active == 0 {
        warn!(
            "Cell-cell loss requested (λ={}) but no batch retained any \
             within-batch edges — falling back to bipartite-only.",
            cc.lambda
        );
        return None;
    }
    info!(
        "Cell-cell chain enabled: λ={}, K={}, levels={:?}, λ_per_level={:?}, {} active batch(es), {} edges total",
        cc.lambda,
        cc.n_negatives,
        chain.levels,
        chain.lambdas,
        n_active,
        cc.edges.len(),
    );
    Some(CellCellPrepared {
        samplers,
        edges: cc.edges,
        lambda: cc.lambda,
        n_negatives: cc.n_negatives,
        chain,
    })
}

/// Resolve caller-facing `(pb_levels, lambda_per_level)` into owned
/// `CellCellPreparedChain`. `pb_levels: None` expands to every
/// available level (coarsest-first, matching `cell_to_pb_per_level`).
/// Out-of-range indices are dropped with a warning.
///
/// Levels that are effectively per-cell partitions (pb count >
/// `DEGENERATE_PB_RATIO * n_cells`) are also dropped — at those levels
/// `pb(u) == pb(v)` implies `u == v`, so requiring positives to share
/// pb wipes out the entire edge set and yields no useful signal.
fn resolve_pb_chain(
    user_levels: Option<&[usize]>,
    user_lambdas: Option<&[f32]>,
    cell_to_pb_per_level: &[Vec<usize>],
    n_cells: usize,
) -> CellCellPreparedChain {
    let n_levels = cell_to_pb_per_level.len();
    let raw_levels: Vec<usize> = match user_levels {
        Some(ls) => ls.to_vec(),
        None => (0..n_levels).collect(),
    };

    let mut levels: Vec<usize> = Vec::with_capacity(raw_levels.len());
    let mut keep_mask: Vec<bool> = Vec::with_capacity(raw_levels.len());
    let mut out_of_range: Vec<usize> = Vec::new();
    let mut degenerate: Vec<(usize, usize)> = Vec::new();

    for l in raw_levels {
        if l >= n_levels {
            out_of_range.push(l);
            keep_mask.push(false);
            continue;
        }
        let n_pbs = pb_count(&cell_to_pb_per_level[l]);
        if (n_pbs as f32) > DEGENERATE_PB_RATIO * (n_cells.max(1) as f32) {
            degenerate.push((l, n_pbs));
            keep_mask.push(false);
            continue;
        }
        levels.push(l);
        keep_mask.push(true);
    }
    if !out_of_range.is_empty() {
        warn!(
            "Cell-cell chain: dropping out-of-range pb-level indices {:?} (have {} levels)",
            out_of_range, n_levels
        );
    }
    if !degenerate.is_empty() {
        warn!(
            "Cell-cell chain: dropping degenerate pb levels (pb count > {:.0}% of {} cells, \
             so pb membership ≈ identity): {:?}. Lower --pb-samples to produce chunkier pb's, \
             or pick specific coarser levels via --cell-cell-pb-levels.",
            DEGENERATE_PB_RATIO * 100.0,
            n_cells,
            degenerate
        );
    }

    let lambdas: Vec<f32> = match user_lambdas {
        Some(ls) if ls.len() == keep_mask.len() => ls
            .iter()
            .zip(keep_mask.iter())
            .filter_map(|(&l, &k)| k.then_some(l))
            .collect(),
        Some(ls) => {
            warn!(
                "Cell-cell chain: lambda_per_level length {} doesn't match input levels {} — using uniform 1.0",
                ls.len(),
                keep_mask.len()
            );
            vec![1.0; levels.len()]
        }
        None => vec![1.0; levels.len()],
    };
    CellCellPreparedChain { levels, lambdas }
}

/// Distinct pb count in a compacted `cell_to_pb` map. The multilevel
/// collapse runs `compact_labels` so ids are dense `0..k`; we still
/// take the `max + 1` rather than trust the contract.
fn pb_count(cell_to_pb: &[usize]) -> usize {
    cell_to_pb.iter().copied().max().map(|m| m + 1).unwrap_or(0)
}
