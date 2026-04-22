//! BBKNN-based Poisson refinement over multi-level super-cell partitions.
//!
//! Given a hash-initialized hierarchy of super-cell → group mappings and a
//! `SuperCellLayout` (HNSW over centroids, batch assignments), refine each
//! level from coarsest to finest by proposing moves that keep each super-cell
//! under the same parent group (sibling-constrained) and are drawn from the
//! batch-balanced KNN neighborhood. Moves are scored by a DC-Poisson
//! log-likelihood with NB Fisher-info gene weighting (see
//! [`crate::dc_poisson::GeneWeighting`]).
//!
//! Candidate-set construction is BBKNN-specific and lives here ([`BbknnProposer`]).
//! The generic scoring core lives in [`crate::dc_poisson`] and is shared with
//! other front-ends (e.g. pinto's spatial-graph proposer).
//!
//! Entry point: [`refine_assignments`].

#![allow(dead_code)]

use crate::collapse_data::SuperCellLayout;
use crate::dc_poisson::{
    compute_sibling_sets, intersect_with_siblings_fallback, refine_with_proposer,
    CandidateProposer, ProfileSource, Profiles, RefineContext,
};
use log::{debug, info};
use rand::rngs::SmallRng;
use rand::SeedableRng;

pub use crate::collapse_data::GeneSums;
// Re-export the DC-Poisson core items so existing callers / integration
// tests that expect `refine_multilevel::{RefineParams, compact_labels, ...}`
// keep working after the core moved.
pub use crate::dc_poisson::{compact_labels, RefineParams};

////////////////////////////////////////////////////////////////////////////////
// Public configuration
////////////////////////////////////////////////////////////////////////////////

/// Per-level refined super-cell → group mapping.
///
/// `sc_to_group[level][sc_idx] = refined group id at that level`. Level order
/// matches the existing collapse convention (`level_dims[0]` = finest).
#[derive(Debug, Clone)]
pub struct RefinedAssignment {
    pub sc_to_group: Vec<Vec<usize>>,
    pub num_groups_per_level: Vec<usize>,
}

////////////////////////////////////////////////////////////////////////////////
// Candidate set construction (siblings ∩ BBKNN with sibling fallback)
////////////////////////////////////////////////////////////////////////////////

/// Per-super-cell BBKNN proposals: for every non-own batch, query that
/// batch's cell-level HNSW (`SparseIoVec::batch_knn_lookup`) with the
/// super-cell's centroid, dedup returned cells to their owning super-cells
/// via `layout.cell_to_sc`, and keep up to `knn` distinct super-cells per
/// other batch.
///
/// Total candidate set per super-cell is up to `knn · (num_batches − 1)`.
fn build_bbknn_neighbors(
    layout: &SuperCellLayout,
    batch_knn_lookup: &[matrix_util::knn_match::ColumnDict<usize>],
    knn: usize,
) -> anyhow::Result<Vec<Vec<usize>>> {
    use matrix_util::knn_match::MakeVecPoint;
    let num_sc = layout.cell_counts.len();
    let cell_oversample = (knn * 4 + 1).max(knn);

    let mut result = Vec::with_capacity(num_sc);
    for sc in 0..num_sc {
        let sc_batch = layout.super_cell_to_batch[sc];
        let centroid = layout.centroids.column(sc).to_vp();
        let mut all_scs: Vec<usize> = Vec::new();
        for (b, bknn) in batch_knn_lookup.iter().enumerate() {
            if b == sc_batch {
                continue;
            }
            let (cells, _dists) = bknn.search_by_query_data(&centroid, cell_oversample)?;
            let mut seen: Vec<usize> = Vec::new();
            for &c in &cells {
                let other_sc = layout.cell_to_sc[c];
                if other_sc == usize::MAX || other_sc == sc {
                    continue;
                }
                if !seen.contains(&other_sc) {
                    seen.push(other_sc);
                    if seen.len() >= knn {
                        break;
                    }
                }
            }
            all_scs.extend(seen);
        }
        result.push(all_scs);
    }
    Ok(result)
}

/// Candidate set per super-cell = siblings ∩ (groups of BBKNN neighbors),
/// via the shared [`intersect_with_siblings_fallback`] helper.
fn build_candidate_sets(
    siblings: &[Vec<usize>],
    bbknn: &[Vec<usize>],
    sc_to_group_at_level: &[usize],
) -> Vec<Vec<usize>> {
    siblings
        .iter()
        .enumerate()
        .map(|(sc, sib)| {
            let mut neighbor_groups: Vec<usize> =
                bbknn[sc].iter().map(|&j| sc_to_group_at_level[j]).collect();
            neighbor_groups.sort_unstable();
            neighbor_groups.dedup();
            intersect_with_siblings_fallback(sib, &neighbor_groups, sc_to_group_at_level[sc])
        })
        .collect()
}

////////////////////////////////////////////////////////////////////////////////
// BBKNN-based CandidateProposer
////////////////////////////////////////////////////////////////////////////////

/// Proposes move candidates for super-cells via cross-batch BBKNN.
///
/// For each super-cell, the candidate set is
/// `siblings ∩ (groups of BBKNN neighbors)`, with sibling fallback when
/// the intersection is empty. Siblings are the other super-cells sharing
/// the same parent group at the next-coarser level (passed in at
/// construction).
pub struct BbknnProposer<'a> {
    siblings: Vec<Vec<usize>>,
    bbknn: &'a [Vec<usize>],
}

impl<'a> BbknnProposer<'a> {
    pub fn new(siblings: Vec<Vec<usize>>, bbknn: &'a [Vec<usize>]) -> Self {
        Self { siblings, bbknn }
    }
}

impl<'a> CandidateProposer for BbknnProposer<'a> {
    fn propose(&self, labels: &[usize]) -> Vec<Vec<usize>> {
        build_candidate_sets(&self.siblings, self.bbknn, labels)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Refinement driver
////////////////////////////////////////////////////////////////////////////////

/// Fixed inputs describing *what* to refine. `batch_knn_lookup` is the
/// per-batch HNSW over cells, typically obtained from
/// `SparseIoVec::batch_knn_lookup()` after `build_hnsw_per_batch`.
/// `k_per_batch` is the BBKNN fan-out — up to this many distinct
/// super-cells are drawn from **each** non-own batch as move candidates.
#[derive(Clone, Copy)]
pub struct RefineInputs<'a> {
    pub layout: &'a SuperCellLayout,
    pub gene_sums: &'a GeneSums,
    pub num_genes: usize,
    pub super_cell_to_cells: &'a [Vec<usize>],
    pub batch_knn_lookup: &'a [matrix_util::knn_match::ColumnDict<usize>],
    pub k_per_batch: usize,
    pub initial_sc_to_group_per_level: &'a [Vec<usize>],
}

/// Top-down BBKNN + Poisson refinement.
///
/// Levels in `inputs.initial_sc_to_group_per_level` follow the existing
/// finest → coarsest convention. The returned `RefinedAssignment` uses the
/// same ordering.
pub fn refine_assignments(
    inputs: &RefineInputs<'_>,
    params: &RefineParams,
) -> anyhow::Result<RefinedAssignment> {
    let RefineInputs {
        layout,
        gene_sums,
        num_genes,
        super_cell_to_cells,
        batch_knn_lookup,
        k_per_batch,
        initial_sc_to_group_per_level,
    } = *inputs;
    let num_levels = initial_sc_to_group_per_level.len();
    if num_levels == 0 {
        return Err(anyhow::anyhow!("no levels"));
    }
    let num_sc = layout.cell_counts.len();
    for (i, lvl) in initial_sc_to_group_per_level.iter().enumerate() {
        if lvl.len() != num_sc {
            return Err(anyhow::anyhow!(
                "level {} has {} entries, expected {}",
                i,
                lvl.len(),
                num_sc
            ));
        }
    }

    // Build profiles once.
    let mut profiles = match &params.profile_source {
        ProfileSource::Raw => Profiles::from_gene_sums(gene_sums, num_genes),
        ProfileSource::Projected { basis } => Profiles::from_projection(basis, super_cell_to_cells),
    };
    if matches!(params.profile_source, ProfileSource::Raw) {
        profiles.apply_gene_weighting(params.gene_weighting);
    }

    // BBKNN proposals via the shared per-batch cell HNSW.
    let bbknn = build_bbknn_neighbors(layout, batch_knn_lookup, k_per_batch)?;

    // Compact each level's initial labels to a dense 0..K range.
    let mut refined: Vec<Vec<usize>> = Vec::with_capacity(num_levels);
    let mut ks: Vec<usize> = Vec::with_capacity(num_levels);
    for lvl in initial_sc_to_group_per_level {
        let (compact, k) = compact_labels(lvl);
        refined.push(compact);
        ks.push(k);
    }

    let mut rng = SmallRng::seed_from_u64(params.seed);

    // Walk coarsest → finest (highest level index down to 0).
    for level in (0..num_levels).rev() {
        let k = ks[level];
        debug!("refining level {} (k={}, num_sc={})", level, k, num_sc);
        let siblings = compute_sibling_sets(&refined, level, k);

        let proposer = BbknnProposer::new(siblings, &bbknn);

        let sc_to_group = &mut refined[level];
        let label = format!("Refine L{}/{}", num_levels - level, num_levels);
        let moves = refine_with_proposer(
            sc_to_group,
            &proposer,
            &mut rng,
            &RefineContext {
                profiles: &profiles,
                k,
                params,
                level_label: &label,
            },
        );
        info!("  level {} refined: {} moves; k={} groups", level, moves, k);

        // Compact in case greedy emptied groups (keeps K monotone without gaps).
        let (compact, new_k) = compact_labels(sc_to_group);
        *sc_to_group = compact;
        ks[level] = new_k;
    }

    Ok(RefinedAssignment {
        sc_to_group: refined,
        num_groups_per_level: ks,
    })
}

////////////////////////////////////////////////////////////////////////////////
// Tests (BBKNN-specific)
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_candidate_sets_fallback() {
        // sc 0 has siblings [0,1] but BBKNN lands only on a non-sibling group.
        let siblings = vec![vec![0usize, 1], vec![0, 1]];
        let bbknn = vec![vec![1usize], vec![0]];
        let sc_to_group = vec![0usize, 1];
        // Neighbor of sc0 is sc1 (group 1) → intersection {1}.
        // sc0 is in group 0; intersection lacks 0 so 0 is appended.
        let cand = build_candidate_sets(&siblings, &bbknn, &sc_to_group);
        assert_eq!(cand[0], vec![0, 1]);
        assert_eq!(cand[1], vec![0, 1]);
    }
}
