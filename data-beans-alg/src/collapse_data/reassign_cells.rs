//! Reassign cells: the first pass over the initial hash. Cells move
//! between the top nodes under the DC-Poisson likelihood of the node
//! profiles, so a node the marginal sketch left mixed across lineages
//! becomes one lineage before the tree grows below it. Cells only move within their batch: the entities are cells,
//! the groups are (node, batch) pairs, and a cell's candidates are the
//! groups of its own batch.

use super::*;
use crate::dc_poisson::{
    refine_with_candidates, FeatureWeighting, Profiles, RefineContext, RefineParams,
};
use nalgebra_sparse::CscMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct ReassignCellsParams {
    pub num_gibbs: usize,
    pub num_greedy: usize,
    pub feature_weighting: FeatureWeighting,
    pub seed: u64,
}

impl Default for ReassignCellsParams {
    fn default() -> Self {
        Self {
            num_gibbs: 3,
            num_greedy: 10,
            feature_weighting: FeatureWeighting::FisherInfoNb,
            seed: 42,
        }
    }
}

/// Move active cells between nodes in place; returns the number of moves.
/// `csc` holds every cell's counts (genes x cells, global column order);
/// `node_of_cell` holds compact node ids and is rewritten for active cells.
pub(crate) fn reassign_cells_to_nodes(
    csc: &CscMatrix<f32>,
    col_to_batch: &[usize],
    num_batches: usize,
    active: &[bool],
    node_of_cell: &mut [usize],
    params: &ReassignCellsParams,
) -> usize {
    let n = csc.ncols();
    assert_eq!(col_to_batch.len(), n);
    assert_eq!(active.len(), n);
    assert_eq!(node_of_cell.len(), n);
    let num_batches = num_batches.max(1);
    let num_nodes = node_of_cell.iter().copied().max().map_or(0, |m| m + 1);
    if num_nodes < 2 || params.num_gibbs + params.num_greedy == 0 {
        return 0;
    }
    let cells: Vec<usize> = (0..n).filter(|&c| active[c]).collect();
    if cells.is_empty() {
        return 0;
    }

    ////////////////////////////////////
    // Groups = occupied (node, batch) //
    ////////////////////////////////////

    let mut group_of_pair = vec![usize::MAX; num_nodes * num_batches];
    let mut group_pair: Vec<(usize, usize)> = Vec::new();
    let mut labels: Vec<usize> = Vec::with_capacity(cells.len());
    for &c in &cells {
        let key = node_of_cell[c] * num_batches + col_to_batch[c];
        if group_of_pair[key] == usize::MAX {
            group_of_pair[key] = group_pair.len();
            group_pair.push((node_of_cell[c], col_to_batch[c]));
        }
        labels.push(group_of_pair[key]);
    }
    let k = group_pair.len();
    let mut groups_of_batch: Vec<Vec<usize>> = vec![Vec::new(); num_batches];
    for (g, &(_, b)) in group_pair.iter().enumerate() {
        groups_of_batch[b].push(g);
    }
    let candidates: Vec<Vec<usize>> = cells
        .iter()
        .map(|&c| groups_of_batch[col_to_batch[c]].clone())
        .collect();

    //////////////////////////////
    // Cell profiles and sweeps //
    //////////////////////////////

    let rows: Vec<Vec<(usize, f32)>> = cells
        .iter()
        .map(|&c| {
            let col = csc.col(c);
            col.row_indices()
                .iter()
                .zip(col.values())
                .map(|(&g, &v)| (g, v))
                .collect()
        })
        .collect();
    let mut profiles = Profiles::from_gene_sums(&rows, csc.nrows());
    profiles.apply_feature_weighting(params.feature_weighting);
    let refine = RefineParams {
        num_gibbs: params.num_gibbs,
        num_greedy: params.num_greedy,
        feature_weighting: params.feature_weighting,
        seed: params.seed,
        ..RefineParams::default()
    };
    let mut rng = SmallRng::seed_from_u64(params.seed);
    let moves = refine_with_candidates(
        &mut labels,
        &candidates,
        &mut rng,
        &RefineContext {
            profiles: &profiles,
            k,
            params: &refine,
            level_label: "reassign cells",
        },
    );
    for (i, &c) in cells.iter().enumerate() {
        node_of_cell[c] = group_pair[labels[i]].0;
    }
    info!(
        "reassign cells: {} of {} cells moved between {} top nodes ({} node x batch groups)",
        moves,
        cells.len(),
        num_nodes,
        k
    );
    moves
}

#[cfg(test)]
#[path = "reassign_cells_tests.rs"]
mod tests;
