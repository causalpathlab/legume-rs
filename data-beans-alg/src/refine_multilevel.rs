//! BBKNN-based Poisson refinement over multi-level pb-sample partitions.
//!
//! Given a hash-initialized hierarchy of pb-sample → group mappings and a
//! `PbSampleLayout` (HNSW over centroids, batch assignments), refine each
//! level from coarsest to finest by proposing moves that keep each pb-sample
//! under the same parent group (sibling-constrained) and are drawn from the
//! batch-balanced KNN neighborhood. Moves are scored by a DC-Poisson
//! log-likelihood with NB Fisher-info feature weighting (see
//! [`crate::dc_poisson::FeatureWeighting`]).
//!
//! Candidate-set construction is BBKNN-specific and lives here ([`BbknnProposer`]).
//! The generic scoring core lives in [`crate::dc_poisson`] and is shared with
//! other front-ends (e.g. pinto's spatial-graph proposer).
//!
//! Entry point: [`refine_assignments`].

#![allow(dead_code)]

use crate::collapse_data::PbSampleLayout;
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

/// Per-level refined pb-sample → group mapping.
///
/// `pbsamp_to_group[level][pbsamp_idx] = refined group id at that level`. Level order
/// matches the existing collapse convention (`level_dims[0]` = finest).
#[derive(Debug, Clone)]
pub struct RefinedAssignment {
    pub pbsamp_to_group: Vec<Vec<usize>>,
    pub num_groups_per_level: Vec<usize>,
}

////////////////////////////////////////////////////////////////////////////////
// Candidate set construction (siblings ∩ BBKNN with sibling fallback)
////////////////////////////////////////////////////////////////////////////////

/// Per-pb-sample BBKNN proposals: for every non-own batch, query that
/// batch's cell-level HNSW (`SparseIoVec::batch_knn_lookup`) with the
/// pb-sample's centroid, dedup returned cells to their owning pb-samples
/// via `layout.cell_to_pbsamp`, and keep up to `knn` distinct pb-samples per
/// other batch.
///
/// Total candidate set per pb-sample is up to `knn · (num_batches − 1)`.
fn build_bbknn_neighbors(
    layout: &PbSampleLayout,
    batch_knn_lookup: &[matrix_util::knn_match::ColumnDict<usize>],
    knn: usize,
) -> anyhow::Result<Vec<Vec<usize>>> {
    use matrix_util::knn_match::MakeVecPoint;
    use rayon::prelude::*;
    let num_pb = layout.cell_counts.len();
    let cell_oversample = (knn * 4 + 1).max(knn);

    (0..num_pb)
        .into_par_iter()
        .map(|pbsamp| -> anyhow::Result<Vec<usize>> {
            let pbsamp_batch = layout.pb_sample_to_batch[pbsamp];
            let centroid = layout.centroids.column(pbsamp).to_vp();
            let mut all_scs: Vec<usize> = Vec::new();
            for (b, bknn) in batch_knn_lookup.iter().enumerate() {
                if b == pbsamp_batch {
                    continue;
                }
                let (cells, _dists) = bknn.search_by_query_data(&centroid, cell_oversample)?;
                let mut seen: Vec<usize> = Vec::new();
                for &c in &cells {
                    let other_pbsamp = layout.cell_to_pbsamp[c];
                    if other_pbsamp == usize::MAX || other_pbsamp == pbsamp {
                        continue;
                    }
                    if !seen.contains(&other_pbsamp) {
                        seen.push(other_pbsamp);
                        if seen.len() >= knn {
                            break;
                        }
                    }
                }
                all_scs.extend(seen);
            }
            Ok(all_scs)
        })
        .collect()
}

/// Candidate set per pb-sample = siblings ∩ (groups of BBKNN neighbors),
/// via the shared [`intersect_with_siblings_fallback`] helper.
fn build_candidate_sets(
    siblings: &[Vec<usize>],
    bbknn: &[Vec<usize>],
    pbsamp_to_group_at_level: &[usize],
) -> Vec<Vec<usize>> {
    siblings
        .iter()
        .enumerate()
        .map(|(pbsamp, sib)| {
            let mut neighbor_groups: Vec<usize> = bbknn[pbsamp]
                .iter()
                .map(|&j| pbsamp_to_group_at_level[j])
                .collect();
            neighbor_groups.sort_unstable();
            neighbor_groups.dedup();
            intersect_with_siblings_fallback(
                sib,
                &neighbor_groups,
                pbsamp_to_group_at_level[pbsamp],
            )
        })
        .collect()
}

////////////////////////////////////////////////////////////////////////////////
// BBKNN-based CandidateProposer
////////////////////////////////////////////////////////////////////////////////

/// Proposes move candidates for pb-samples via cross-batch BBKNN.
///
/// For each pb-sample, the candidate set is
/// `siblings ∩ (groups of BBKNN neighbors)`, with sibling fallback when
/// the intersection is empty. Siblings are the other pb-samples sharing
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
/// pb-samples are drawn from **each** non-own batch as move candidates.
#[derive(Clone, Copy)]
pub struct RefineInputs<'a> {
    pub layout: &'a PbSampleLayout,
    pub gene_sums: &'a GeneSums,
    pub num_genes: usize,
    pub pb_sample_to_cells: &'a [Vec<usize>],
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
        pb_sample_to_cells,
        batch_knn_lookup,
        k_per_batch,
        initial_sc_to_group_per_level,
    } = *inputs;
    let num_levels = initial_sc_to_group_per_level.len();
    if num_levels == 0 {
        return Err(anyhow::anyhow!("no levels"));
    }
    let num_pb = layout.cell_counts.len();
    for (i, lvl) in initial_sc_to_group_per_level.iter().enumerate() {
        if lvl.len() != num_pb {
            return Err(anyhow::anyhow!(
                "level {} has {} entries, expected {}",
                i,
                lvl.len(),
                num_pb
            ));
        }
    }

    // Compact each level's initial labels to a dense 0..K range.
    let mut refined: Vec<Vec<usize>> = Vec::with_capacity(num_levels);
    let mut ks: Vec<usize> = Vec::with_capacity(num_levels);
    for lvl in initial_sc_to_group_per_level {
        let (compact, k) = compact_labels(lvl);
        refined.push(compact);
        ks.push(k);
    }

    // Both sweep counts zero ⇒ no DC-Poisson moves can happen; skip the
    // (expensive) profile build, NB-Fisher weighting, and BBKNN construction
    // and return the compacted initial labels.
    if params.num_gibbs == 0 && params.num_greedy == 0 {
        info!("Skipping DC-Poisson refinement: --pb-refine-gibbs=0 and --pb-refine-greedy=0");
        return Ok(RefinedAssignment {
            pbsamp_to_group: refined,
            num_groups_per_level: ks,
        });
    }

    // Build profiles once.
    info!(
        "Building DC-Poisson profiles ({} pb-samples × {} genes) ...",
        num_pb, num_genes
    );
    let mut profiles = match &params.profile_source {
        ProfileSource::Raw => Profiles::from_gene_sums(gene_sums, num_genes),
        ProfileSource::Projected { basis } => Profiles::from_projection(basis, pb_sample_to_cells),
    };
    if matches!(params.profile_source, ProfileSource::Raw)
        && !matches!(
            params.feature_weighting,
            crate::dc_poisson::FeatureWeighting::None
        )
    {
        info!("Computing NB Fisher-info feature weights ...");
        profiles.apply_feature_weighting(params.feature_weighting);
    }

    // BBKNN proposals via the shared per-batch cell HNSW.
    info!(
        "Building BBKNN candidate sets (knn={} per non-own batch) ...",
        k_per_batch
    );
    let bbknn = build_bbknn_neighbors(layout, batch_knn_lookup, k_per_batch)?;

    let mut rng = SmallRng::seed_from_u64(params.seed);

    // Walk coarsest → finest (highest level index down to 0).
    for level in (0..num_levels).rev() {
        // Refining the coarser level moves individual pb-samples across
        // parents, which can leave this (not-yet-refined) finer level's
        // initial groups straddling two *refined* parents. Re-project the
        // level against its already-refined parent so it is a strict
        // refinement before sibling sets are computed — otherwise a
        // straddling group lands in two parents' sibling sets, both
        // halves are then free to stay put, and the refined hierarchy is
        // broken (violating `fine_to_coarse_from_refined`'s invariant).
        if level + 1 < num_levels {
            let (reprojected, new_k) = project_to_refinement(&refined[level], &refined[level + 1]);
            refined[level] = reprojected;
            ks[level] = new_k;
        }
        let k = ks[level];
        debug!("refining level {} (k={}, num_pb={})", level, k, num_pb);
        let siblings = compute_sibling_sets(&refined, level, k);

        let proposer = BbknnProposer::new(siblings, &bbknn);

        let pbsamp_to_group = &mut refined[level];
        let label = format!("Refine L{}/{}", num_levels - level, num_levels);
        let moves = refine_with_proposer(
            pbsamp_to_group,
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
        let (compact, new_k) = compact_labels(pbsamp_to_group);
        *pbsamp_to_group = compact;
        ks[level] = new_k;
    }

    Ok(RefinedAssignment {
        pbsamp_to_group: refined,
        num_groups_per_level: ks,
    })
}

/// Re-label `child` so it is a strict refinement of `parent`: two entities
/// share an output label iff they agree on **both** their current child
/// label and their parent label. Any child group that straddles multiple
/// parents is split. Returns the relabeled vector and its group count
/// (already compact `0..k` — `compact_labels` over the `(child, parent)`
/// key does the first-appearance dense relabel).
fn project_to_refinement(child: &[usize], parent: &[usize]) -> (Vec<usize>, usize) {
    debug_assert_eq!(child.len(), parent.len());
    let pairs: Vec<(usize, usize)> = child.iter().copied().zip(parent.iter().copied()).collect();
    compact_labels(&pairs)
}

////////////////////////////////////////////////////////////////////////////////
// Tests (BBKNN-specific)
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_candidate_sets_fallback() {
        // pbsamp 0 has siblings [0,1] but BBKNN lands only on a non-sibling group.
        let siblings = vec![vec![0usize, 1], vec![0, 1]];
        let bbknn = vec![vec![1usize], vec![0]];
        let pbsamp_to_group = vec![0usize, 1];
        // Neighbor of sc0 is sc1 (group 1) → intersection {1}.
        // sc0 is in group 0; intersection lacks 0 so 0 is appended.
        let cand = build_candidate_sets(&siblings, &bbknn, &pbsamp_to_group);
        assert_eq!(cand[0], vec![0, 1]);
        assert_eq!(cand[1], vec![0, 1]);
    }

    #[test]
    fn test_project_to_refinement_splits_straddling_groups() {
        // child group 0 straddles parents A=0 and B=1; group 1 sits wholly
        // under parent B. Re-projection must split group 0 into two and
        // produce a strict refinement of `parent`.
        let child = vec![0usize, 0, 0, 1, 1];
        let parent = vec![0usize, 0, 1, 1, 1];
        let (reproj, k) = project_to_refinement(&child, &parent);

        // 3 distinct (child, parent) pairs: (0,0), (0,1), (1,1).
        assert_eq!(k, 3);
        // Entities 0,1 → (0,0); entity 2 → (0,1); entities 3,4 → (1,1).
        assert_eq!(reproj[0], reproj[1]);
        assert_ne!(reproj[0], reproj[2]);
        assert_eq!(reproj[3], reproj[4]);
        assert_ne!(reproj[2], reproj[3]);

        // Strict refinement: every reprojected group maps to one parent.
        let mut group_parent = std::collections::HashMap::new();
        for (&g, &p) in reproj.iter().zip(parent.iter()) {
            assert_eq!(*group_parent.entry(g).or_insert(p), p);
        }
    }
}
