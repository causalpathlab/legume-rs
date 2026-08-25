//! The PB training frame: everything the SGD needs once the trained
//! unit is a finest-level super-cell (pseudobulk) rather than a cell.
//!
//! Built from arm-independent inputs (the coarsening labels and the
//! pair graph), so both gate modes train the same way; the sampled
//! arm's collapse SVD enters only as the optional warm start, wired in
//! `fit.rs`. Cell-cell edges appear here exactly once, at build time,
//! to be folded into PB-PB super edges; the training loop never sees a
//! cell id again.

use crate::util::graph_coarsen::{fold_edges_to_super, MultiLevelCoarsenResult};
use log::{info, warn};
use matrix_util::knn_graph::KnnGraph;

#[derive(Debug)]
pub struct PbFrame {
    /// Cell -> finest-level PB label, `[n_cells]`.
    pub cell_to_finest_pb: Vec<usize>,
    /// Number of finest-level PBs (the trained unit count).
    pub n_pb: usize,
    /// PB-PB super edges at the finest level, `(min, max)` deduped.
    pub super_edges: Vec<(u32, u32)>,
    /// Per COARSENING level except the finest: finest-PB -> super-cell
    /// label at that level. Indexed by the same level ids as
    /// `--chain-levels`, which the CLI restricts to coarser-than-finest
    /// (the finest level IS the trained unit).
    pub pb_parent_maps: Vec<Vec<usize>>,
    /// Finest-PB -> experimental batch id (majority over member cells).
    pub pb_exp_batch: Vec<u32>,
}

/// Build the frame from the multilevel coarsening, the pair graph, and
/// the per-cell batch membership. Returns the frame plus the
/// fine-edge -> super-edge map (`None` = intra-PB), separately so the
/// caller can drop the map once per-gene activity has been folded
/// through it — it is `O(n_fine_edges)` and needed exactly once.
///
/// Nesting is required: every finest PB must map to exactly one
/// super-cell at each coarser level (the coarsener's pass-2 re-nesting
/// guarantees it; violated input is an error, not a warning). Batch
/// purity is NOT guaranteed for user-supplied batch labels, so the
/// batch id is a majority vote and impurity is only warned about.
#[allow(clippy::type_complexity)]
pub fn build_pb_frame(
    ml: &MultiLevelCoarsenResult,
    graph: &KnnGraph,
    batch_membership: &[u32],
    n_batches: usize,
) -> anyhow::Result<(PbFrame, Vec<Option<usize>>)> {
    let finest = ml
        .all_cell_labels
        .last()
        .ok_or_else(|| anyhow::anyhow!("coarsening produced no levels"))?
        .clone();
    let n_cells = finest.len();
    anyhow::ensure!(n_cells > 0, "pb frame: no cells");
    anyhow::ensure!(
        batch_membership.len() == n_cells,
        "pb frame: batch membership length {} != n_cells {}",
        batch_membership.len(),
        n_cells
    );
    let n_pb = finest.iter().max().map(|&m| m + 1).unwrap_or(0);
    // Dense labels are load-bearing: a gap would create a phantom PB with
    // a usize::MAX parent (colliding with the sampler's sentinel space)
    // and an untrained random row in the trained table.
    {
        let mut seen = vec![false; n_pb];
        for &p in &finest {
            seen[p] = true;
        }
        if let Some(gap) = seen.iter().position(|&s| !s) {
            anyhow::bail!(
                "finest coarsening labels are not dense: pb {} of {} has no member cell",
                gap,
                n_pb
            );
        }
    }

    // Cross-batch fine edges may neither FORM a super edge nor feed
    // activity into one: the old cell-level sampler dropped them by
    // construction, and folding them through an impure PB would train
    // on cross-section co-activity. They map to `None` like intra-PB
    // edges, so the activity fold drops them for free.
    let same_batch: Vec<bool> = graph
        .edges
        .iter()
        .map(|&(i, j)| batch_membership[i] == batch_membership[j])
        .collect();
    let n_cross_batch = same_batch.iter().filter(|&&k| !k).count();
    if n_cross_batch > 0 {
        info!(
            "{} of {} fine edges are cross-batch and fold into no super edge",
            n_cross_batch,
            graph.edges.len()
        );
    }
    let (super_edges_usize, fine_to_super) =
        fold_edges_to_super(&finest, &graph.edges, Some(&same_batch));
    let super_edges: Vec<(u32, u32)> = super_edges_usize
        .into_iter()
        .map(|(a, b)| (a as u32, b as u32))
        .collect();

    // Parent maps for every level COARSER than the finest; nesting makes
    // each entry unique. The finest level's map would be the identity and
    // nothing may read it (the CLI rejects finest chain levels).
    let n_levels = ml.all_cell_labels.len();
    let mut pb_parent_maps: Vec<Vec<usize>> =
        vec![vec![usize::MAX; n_pb]; n_levels.saturating_sub(1)];
    for (lvl, labels) in ml.all_cell_labels[..n_levels.saturating_sub(1)]
        .iter()
        .enumerate()
    {
        let parents = &mut pb_parent_maps[lvl];
        for (cell, &p) in finest.iter().enumerate() {
            let lab = labels[cell];
            if parents[p] == usize::MAX {
                parents[p] = lab;
            } else {
                anyhow::ensure!(
                    parents[p] == lab,
                    "coarsening levels do not nest: finest pb {} maps to \
                     both {} and {} at level {}",
                    p,
                    parents[p],
                    lab,
                    lvl
                );
            }
        }
    }

    // Majority batch per PB, on a flat (pb, batch) count grid.
    for (cell, &b) in batch_membership.iter().enumerate() {
        anyhow::ensure!(
            (b as usize) < n_batches,
            "pb frame: cell {} carries batch id {} >= n_batches {}",
            cell,
            b,
            n_batches
        );
    }
    let mut batch_counts = vec![0usize; n_pb * n_batches.max(1)];
    for (cell, &p) in finest.iter().enumerate() {
        batch_counts[p * n_batches.max(1) + batch_membership[cell] as usize] += 1;
    }
    let mut impure = 0usize;
    let pb_exp_batch: Vec<u32> = (0..n_pb)
        .map(|p| {
            let row = &batch_counts[p * n_batches.max(1)..(p + 1) * n_batches.max(1)];
            if row.iter().filter(|&&n| n > 0).count() > 1 {
                impure += 1;
            }
            row.iter()
                .enumerate()
                .max_by_key(|&(_, &n)| n)
                .map(|(b, _)| b as u32)
                .unwrap_or(0)
        })
        .collect();
    if impure > 0 {
        warn!(
            "{} of {} finest PBs span more than one experimental batch; \
             each was assigned its majority batch",
            impure, n_pb
        );
    }
    info!(
        "PB frame: {} PBs over {} cells, {} super edges, {} levels",
        n_pb,
        n_cells,
        super_edges.len(),
        n_levels
    );

    Ok((
        PbFrame {
            cell_to_finest_pb: finest,
            n_pb,
            super_edges,
            pb_parent_maps,
            pb_exp_batch,
        },
        fine_to_super,
    ))
}
