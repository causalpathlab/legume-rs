//! Nested multilevel feature coarsening with top-down DC-Poisson refinement.
//!
//! Where [`crate::feature_coarsening::compute_feature_coarsening`] runs a single
//! SVD-binary-hash pass and is invoked once per level *independently* by senna's
//! topic pipelines, this module produces a *nested* hierarchy plus an iterative
//! refinement of group membership. Each fine group is a strict subset of its
//! parent coarse group; the parent map enables sibling-constrained Gibbs/greedy
//! moves, mirroring the cell-side [`crate::refine_multilevel`].
//!
//! ## Two phases
//!
//! 1. **Bottom-up rough coarsening** ([`compute_multilevel_feature_coarsening`])
//!    builds a feature-feature KNN, sorts edges by distance, and union-find
//!    merges in that order. Snapshot UF labels every time `group_count` falls
//!    to a level's target K. If KNN edges are exhausted before reaching the
//!    coarsest target (disconnected components), force inter-component
//!    centroid-distance merges to finish.
//! 2. **Top-down refinement** ([`refine_multilevel_feature_coarsening`]) walks
//!    levels coarsest → finest; builds per-feature candidate sets as
//!    `siblings ∩ neighbor-groups`; calls
//!    [`crate::dc_poisson::refine_with_proposer`] for the Gibbs+greedy sweeps.
//!    After each level's refinement, finer levels are tuple-relabeled
//!    `(finer_label, parent_label)` so nesting is preserved by construction —
//!    sibling-constrained moves alone don't enforce it.
//!
//! ## Conventions
//!
//! - [`MultilevelFeatureCoarsening::levels`] is **coarsest-first** to match
//!   senna's existing `Vec<Option<FeatureCoarsening>>` ordering. The internal
//!   calls to `dc_poisson::compute_sibling_sets` expect finest-first; the
//!   refinement driver flips ordering at the boundary.
//! - The "feature axis" inside [`crate::dc_poisson::Profiles`] is the *sample*
//!   axis here (each entity is a feature; its profile is a row of `sketch_ds`).
//!   [`crate::dc_poisson::FeatureWeighting`] therefore weights *samples*; default
//!   `None` is recommended (NB Fisher in this direction is non-standard).

use crate::dc_poisson::{
    compact_labels, compute_sibling_sets, intersect_with_siblings_fallback, refine_with_proposer,
    CandidateProposer, FeatureWeighting, Profiles, RefineContext, RefineParams,
};
use crate::feature_coarsening::{compute_feature_coarsening, FeatureCoarsening};
use crate::union_find::UnionFind;
use log::{debug, info};
use matrix_util::knn_match::ColumnDict;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::cmp::Ordering;

//////////////////
// Public types //
//////////////////

/// Per-feature top-k KNN cache shared between the bottom-up coarsening pass
/// and top-down refinement. Built once via [`Self::from_sketch`]; the HNSW
/// index is dropped after queries complete, so only the result vectors are
/// retained.
pub struct FeatureKnnContext {
    /// Per-feature top-k neighbor indices (excludes self).
    pub neighbors: Vec<Vec<usize>>,
    /// Per-feature top-k neighbor distances, parallel to `neighbors`.
    pub distances: Vec<Vec<f32>>,
}

impl FeatureKnnContext {
    /// Build from a `[D, S]` pseudobulk sketch using L2 distance over the
    /// sample-dimension profile of each feature. Queries are parallelized
    /// across features; the HNSW index is dropped before returning.
    pub fn from_sketch(sketch_ds: &DMatrix<f32>, knn_k: usize) -> anyhow::Result<Self> {
        let d = sketch_ds.nrows();
        if d == 0 {
            return Err(anyhow::anyhow!("sketch has zero features"));
        }
        let knn_query = knn_k.min(d.saturating_sub(1)).max(1);

        let sketch_sd = sketch_ds.transpose();
        let names: Vec<usize> = (0..d).collect();
        let knn_dict =
            ColumnDict::<usize>::from_dvector_views(sketch_sd.column_iter().collect(), names);

        let pairs: Vec<(Vec<usize>, Vec<f32>)> = (0..d)
            .into_par_iter()
            .map(|f| -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
                let (nbrs, dists) = knn_dict.search_others(&f, knn_query)?;
                let mut keep_n = Vec::with_capacity(nbrs.len());
                let mut keep_d = Vec::with_capacity(dists.len());
                for (n, dist) in nbrs.into_iter().zip(dists) {
                    // Filter NaN/Inf — degenerate (e.g. all-zero) profiles can
                    // produce non-finite cosine/L2 distances that break sort
                    // ordering and union-find tiebreaks downstream.
                    if n != f && dist.is_finite() {
                        keep_n.push(n);
                        keep_d.push(dist);
                    }
                }
                Ok((keep_n, keep_d))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let (neighbors, distances) = pairs.into_iter().unzip();
        Ok(Self {
            neighbors,
            distances,
        })
    }

    pub fn num_features(&self) -> usize {
        self.neighbors.len()
    }
}

/// Nested per-level feature coarsening.
///
/// `levels[0]` is the coarsest, `levels.last()` is the finest. At level `L+1`
/// every coarse group is a strict subset of exactly one level-`L` group, so
/// the parent of group `g` at level `L` is well defined (computed lazily by
/// [`Self::parent_at`]).
#[derive(Clone)]
pub struct MultilevelFeatureCoarsening {
    pub levels: Vec<FeatureCoarsening>,
}

impl MultilevelFeatureCoarsening {
    /// Parent map `parent[group_at_level] = group_id at level-1` for the
    /// supplied (non-coarsest) level. Returns `Ok(None)` for level 0 (no
    /// parent) or out-of-range. Errors if nesting is broken.
    pub fn parent_at(&self, level: usize) -> anyhow::Result<Option<Vec<usize>>> {
        if level == 0 || level >= self.levels.len() {
            return Ok(None);
        }
        let fine = &self.levels[level];
        let coarse = &self.levels[level - 1];
        derive_parent_map(
            &fine.fine_to_coarse,
            &coarse.fine_to_coarse,
            fine.num_coarse,
        )
        .map(Some)
    }
}

/// Knobs for [`refine_multilevel_feature_coarsening`].
#[derive(Clone, Debug)]
pub struct MultilevelRefineParams {
    /// DC-Poisson sweep parameters (Gibbs/greedy counts, RNG seed, weighting).
    /// `feature_weighting` weights *samples* in this context; `None` recommended.
    pub dc_poisson: RefineParams,
}

impl Default for MultilevelRefineParams {
    fn default() -> Self {
        Self {
            dc_poisson: RefineParams {
                feature_weighting: FeatureWeighting::None,
                ..RefineParams::default()
            },
        }
    }
}

////////////////////////////////
// Bottom-up rough coarsening //
////////////////////////////////

/// Build a nested feature coarsening hierarchy from a `[D, S]` pseudobulk
/// sketch via union-find on KNN edges.
///
/// `level_targets` lists target K values from coarsest (smallest) to finest
/// (largest); they must be non-decreasing. Duplicates are allowed (and merged
/// into one snapshot shared across the duplicate slots).
pub fn compute_multilevel_feature_coarsening(
    sketch_ds: &DMatrix<f32>,
    level_targets: &[usize],
    knn: &FeatureKnnContext,
) -> anyhow::Result<MultilevelFeatureCoarsening> {
    let num_levels = level_targets.len();
    if num_levels == 0 {
        return Err(anyhow::anyhow!("level_targets must be non-empty"));
    }
    for w in level_targets.windows(2) {
        if w[0] > w[1] {
            return Err(anyhow::anyhow!(
                "level_targets must be non-decreasing (coarsest → finest)"
            ));
        }
    }
    let d = sketch_ds.nrows();
    if d == 0 {
        return Err(anyhow::anyhow!("sketch has zero features"));
    }
    if knn.num_features() != d {
        return Err(anyhow::anyhow!(
            "knn context has {} features, expected {d}",
            knn.num_features()
        ));
    }
    let coarsest_target = level_targets[0].max(1);
    let finest_target = (*level_targets.last().unwrap()).min(d).max(1);

    info!(
        "Multilevel feature coarsening: D={} features, S={} samples, levels={:?}",
        d,
        sketch_ds.ncols(),
        level_targets,
    );

    // For very wide D, the existing SVD-binary-hash gives a quick coarse
    // initial partition that union-find can then refine; without it many
    // singletons may stay unmerged when KNN edges don't bridge them.
    let prehash: Option<FeatureCoarsening> = if d > finest_target * 2 {
        let bins = (finest_target * 4).min(d);
        Some(compute_feature_coarsening(sketch_ds, bins)?)
    } else {
        None
    };

    // Sorted (distance, a, b) edges with a < b.
    let mut edges: Vec<(f32, u32, u32)> = (0..d)
        .into_par_iter()
        .flat_map_iter(|f| {
            knn.neighbors[f]
                .iter()
                .zip(knn.distances[f].iter())
                .filter_map(move |(&nb, &dist)| {
                    if nb == f {
                        return None;
                    }
                    let (a, b) = if f < nb { (f, nb) } else { (nb, f) };
                    Some((dist, a as u32, b as u32))
                })
        })
        .collect();
    edges.sort_by(|x, y| {
        x.0.partial_cmp(&y.0)
            .unwrap_or(Ordering::Equal)
            .then(x.1.cmp(&y.1))
            .then(x.2.cmp(&y.2))
    });
    debug!("  built {} KNN edges", edges.len());

    let mut uf = UnionFind::new(d);
    let mut group_count = d;

    if let Some(fc) = &prehash {
        for group in &fc.coarse_to_fine {
            for w in group.windows(2) {
                let ra = uf.find(w[0]);
                let rb = uf.find(w[1]);
                if ra != rb {
                    uf.union(ra, rb);
                    group_count -= 1;
                }
            }
        }
        debug!(
            "  prehash seeded UF: {} → {} initial groups",
            d, group_count
        );
    }

    // Targets sorted (descending K, finest first) so we can capture each as
    // group_count crosses it from above. Each entry retains its original
    // level index so `snapshots[level_idx]` writes back to the right slot,
    // even when `level_targets` is non-monotone.
    let mut targets_ftc: Vec<(usize, usize)> = level_targets
        .iter()
        .copied()
        .enumerate()
        .map(|(i, t)| (i, t.max(1)))
        .collect();
    targets_ftc.sort_by(|a, b| b.1.cmp(&a.1));

    let mut snapshots: Vec<Option<Vec<usize>>> = vec![None; num_levels];
    let mut snap_idx = 0usize;

    capture_snapshots(
        &mut uf,
        d,
        &targets_ftc,
        &mut snapshots,
        &mut snap_idx,
        group_count,
    );

    for &(_, a, b) in &edges {
        if snap_idx >= targets_ftc.len() {
            break;
        }
        let (a, b) = (a as usize, b as usize);
        let ra = uf.find(a);
        let rb = uf.find(b);
        if ra == rb {
            continue;
        }
        uf.union(ra, rb);
        group_count -= 1;
        capture_snapshots(
            &mut uf,
            d,
            &targets_ftc,
            &mut snapshots,
            &mut snap_idx,
            group_count,
        );
    }

    if snap_idx < targets_ftc.len() {
        debug!(
            "  KNN edges exhausted at {} groups; falling back to centroid merges",
            group_count
        );
        let mut state = ForcedMergeState {
            uf: &mut uf,
            group_count: &mut group_count,
            snapshots: &mut snapshots,
            snap_idx: &mut snap_idx,
        };
        forced_centroid_merges(sketch_ds, &mut state, &targets_ftc, coarsest_target);
    }

    let levels = snapshots_to_levels(snapshots, num_levels)?;
    info!(
        "Multilevel feature coarsening built: levels K = {:?}",
        levels.iter().map(|l| l.num_coarse).collect::<Vec<_>>()
    );
    Ok(MultilevelFeatureCoarsening { levels })
}

/// Snapshot UF state into every target slot whose K threshold is currently
/// satisfied. Idempotent: skipping already-captured slots is implicit in
/// `snap_idx` advancing.
fn capture_snapshots(
    uf: &mut UnionFind,
    d: usize,
    targets_ftc: &[(usize, usize)],
    snapshots: &mut [Option<Vec<usize>>],
    snap_idx: &mut usize,
    group_count: usize,
) {
    if *snap_idx >= targets_ftc.len() || group_count > targets_ftc[*snap_idx].1 {
        return;
    }
    let labels = flat_labels(uf, d);
    while *snap_idx < targets_ftc.len() && group_count <= targets_ftc[*snap_idx].1 {
        snapshots[targets_ftc[*snap_idx].0] = Some(labels.clone());
        *snap_idx += 1;
    }
}

/// Snapshot of UF labels via one path-compression pass + direct parent reads.
fn flat_labels(uf: &mut UnionFind, d: usize) -> Vec<usize> {
    uf.flatten();
    (0..d).map(|i| uf.parent(i)).collect()
}

/// Build `parent[group_at_this_level] = group_at_coarser_level`. Errors if
/// the nesting invariant is broken (a fine group's members map to multiple
/// coarse parents).
fn derive_parent_map(
    fine_labels: &[usize],
    coarse_labels: &[usize],
    num_fine: usize,
) -> anyhow::Result<Vec<usize>> {
    let mut parent = vec![usize::MAX; num_fine];
    for (f, (&fine, &coarse)) in fine_labels.iter().zip(coarse_labels.iter()).enumerate() {
        if parent[fine] == usize::MAX {
            parent[fine] = coarse;
        } else if parent[fine] != coarse {
            return Err(anyhow::anyhow!(
                "nesting invariant violated at feature {f}: \
                 fine group {fine} maps to both coarse {} and {}",
                parent[fine],
                coarse
            ));
        }
    }
    Ok(parent)
}

fn snapshots_to_levels(
    snapshots: Vec<Option<Vec<usize>>>,
    num_levels: usize,
) -> anyhow::Result<Vec<FeatureCoarsening>> {
    snapshots
        .into_iter()
        .enumerate()
        .take(num_levels)
        .map(|(level_idx, snap_slot)| {
            let raw = snap_slot
                .ok_or_else(|| anyhow::anyhow!("level {level_idx} did not receive a snapshot"))?;
            let (compact, num_coarse) = compact_labels(&raw);
            let mut coarse_to_fine: Vec<Vec<usize>> = vec![Vec::new(); num_coarse];
            for (f, &c) in compact.iter().enumerate() {
                coarse_to_fine[c].push(f);
            }
            Ok(FeatureCoarsening {
                fine_to_coarse: compact,
                coarse_to_fine,
                num_coarse,
            })
        })
        .collect()
}

struct ForcedMergeState<'a> {
    uf: &'a mut UnionFind,
    group_count: &'a mut usize,
    snapshots: &'a mut [Option<Vec<usize>>],
    snap_idx: &'a mut usize,
}

/// Force merges by smallest centroid-pair distance until every remaining
/// target is hit. Centroids and pairwise distances use nalgebra column
/// views over a single `[S, K_active]` `DMatrix` so the inner numerics
/// reduce to BLAS-style ops on contiguous columns.
fn forced_centroid_merges(
    sketch_ds: &DMatrix<f32>,
    state: &mut ForcedMergeState<'_>,
    targets_ftc: &[(usize, usize)],
    coarsest_target: usize,
) {
    let d = sketch_ds.nrows();
    let s = sketch_ds.ncols();

    while *state.snap_idx < targets_ftc.len() && *state.group_count > coarsest_target {
        let max_active = *state.group_count;
        let mut centroids = DMatrix::<f32>::zeros(s, max_active);
        let mut sizes = vec![0usize; max_active];
        let mut roots: Vec<usize> = Vec::with_capacity(max_active);
        let mut root_to_idx: HashMap<usize, usize> = HashMap::default();

        // Accumulate per-component centroid sums via column axpy.
        for f in 0..d {
            let r = state.uf.find(f);
            let idx = *root_to_idx.entry(r).or_insert_with(|| {
                roots.push(r);
                roots.len() - 1
            });
            centroids
                .column_mut(idx)
                .axpy(1.0, &sketch_ds.row(f).transpose(), 1.0);
            sizes[idx] += 1;
        }
        let num_active = roots.len();

        // Mean per centroid: scale each active column by 1 / size.
        let mut means = DMatrix::<f32>::zeros(s, num_active);
        for (i, &n_i) in sizes.iter().enumerate().take(num_active) {
            let inv = 1.0 / n_i as f32;
            means
                .column_mut(i)
                .zip_apply(&centroids.column(i), |dst, c| {
                    *dst = c * inv;
                });
        }

        // Closest pair (squared L2). The column-difference allocates one
        // temporary per pair; for the rare-fallback path that's acceptable.
        let mut best: Option<(f32, usize, usize)> = None;
        for i in 0..num_active {
            for j in (i + 1)..num_active {
                let d_sq = (means.column(i) - means.column(j)).norm_squared();
                if best.is_none_or(|b| d_sq < b.0) {
                    best = Some((d_sq, i, j));
                }
            }
        }
        let Some((_, i, j)) = best else { break };
        state.uf.union(roots[i], roots[j]);
        *state.group_count -= 1;

        capture_snapshots(
            state.uf,
            d,
            targets_ftc,
            state.snapshots,
            state.snap_idx,
            *state.group_count,
        );
    }
}

/////////////////////////
// Top-down refinement //
/////////////////////////

/// Per-feature KNN-driven candidate proposer.
///
/// Candidate set per feature = `siblings ∩ groups-of-KNN` with sibling
/// fallback when the intersection is empty — same shape as `BbknnProposer`
/// and pinto's `GraphProposer`, just over the feature axis.
struct FeatureKnnProposer<'a> {
    siblings: Vec<Vec<usize>>,
    knn_neighbors: &'a [Vec<usize>],
}

impl CandidateProposer for FeatureKnnProposer<'_> {
    fn propose(&self, labels: &[usize]) -> Vec<Vec<usize>> {
        self.siblings
            .iter()
            .enumerate()
            .map(|(f, sib)| {
                let mut neighbor_groups: Vec<usize> =
                    self.knn_neighbors[f].iter().map(|&j| labels[j]).collect();
                neighbor_groups.sort_unstable();
                neighbor_groups.dedup();
                intersect_with_siblings_fallback(sib, &neighbor_groups, labels[f])
            })
            .collect()
    }
}

/// Apply DC-Poisson refinement top-down across the multilevel hierarchy.
///
/// At each level (coarsest → finest), each feature can move only into a
/// sibling group (one sharing its parent at the next-coarser level), drawn
/// from the candidate set its KNN neighbors point to. After each level's
/// refinement, finer levels are tuple-relabeled `(finer_label, parent_label)`
/// to preserve nesting by construction — sibling-constrained moves alone
/// don't enforce it: a coarser-level move changes f's parent, but the same
/// f's finer-level group can still hold members with disagreeing parents
/// until the propagation step splits them.
pub fn refine_multilevel_feature_coarsening(
    sketch_ds: &DMatrix<f32>,
    init: MultilevelFeatureCoarsening,
    knn: &FeatureKnnContext,
    params: &MultilevelRefineParams,
) -> anyhow::Result<MultilevelFeatureCoarsening> {
    let d = sketch_ds.nrows();
    let s = sketch_ds.ncols();
    let num_levels = init.levels.len();
    if num_levels == 0 {
        return Ok(init);
    }
    if init.levels.iter().any(|fc| fc.fine_to_coarse.len() != d) {
        return Err(anyhow::anyhow!(
            "init levels' fine_to_coarse length mismatch (D={d})"
        ));
    }
    if knn.num_features() != d {
        return Err(anyhow::anyhow!(
            "knn context has {} features, expected {d}",
            knn.num_features()
        ));
    }

    info!("Multilevel feature refinement: D={d}, S={s}, levels={num_levels}");

    // Build per-feature dense profiles → Profiles (entity = feature, inner
    // feature axis = sample index). FeatureWeighting::None short-circuits
    // the NB Fisher computation.
    let rows: Vec<Vec<(usize, f32)>> = (0..d)
        .map(|f| {
            (0..s)
                .filter_map(|j| {
                    let v = sketch_ds[(f, j)];
                    (v > 0.0).then_some((j, v))
                })
                .collect()
        })
        .collect();
    let mut profiles = Profiles::from_gene_sums(&rows, s);
    profiles.apply_feature_weighting(params.dc_poisson.feature_weighting);

    // Working state in dc_poisson convention (refined[0] = finest,
    // refined[len-1] = coarsest).
    let mut refined: Vec<Vec<usize>> = init
        .levels
        .iter()
        .rev()
        .map(|fc| fc.fine_to_coarse.clone())
        .collect();

    let mut rng = SmallRng::seed_from_u64(params.dc_poisson.seed);

    for level in (0..num_levels).rev() {
        let (_, k) = compact_labels(&refined[level]);
        let siblings = compute_sibling_sets(&refined, level, k);
        let proposer = FeatureKnnProposer {
            siblings,
            knn_neighbors: &knn.neighbors,
        };
        let label = format!("FeatRefine L{}/{}", num_levels - level, num_levels);
        let labels = &mut refined[level];
        let moves = refine_with_proposer(
            labels,
            &proposer,
            &mut rng,
            &RefineContext {
                profiles: &profiles,
                k,
                params: &params.dc_poisson,
                level_label: &label,
            },
        );
        info!("  feature level {level}: {moves} moves; k={k} groups");

        let (compact, _) = compact_labels(labels);
        *labels = compact;

        // Propagate split to all finer levels so they strictly nest within
        // the just-refined `level`. Tuple-relabel `(finer_label, parent_label)`
        // to dense ids; finer K may grow but never shrink past the original.
        for finer in 0..level {
            let (parent_lbls, finer_lbls) = {
                let (lo, hi) = refined.split_at_mut(level);
                (&hi[0], &mut lo[finer])
            };
            let mut pair_to_id: HashMap<(usize, usize), usize> = HashMap::default();
            *finer_lbls = finer_lbls
                .iter()
                .zip(parent_lbls.iter())
                .map(|(&a, &b)| {
                    let next = pair_to_id.len();
                    *pair_to_id.entry((a, b)).or_insert(next)
                })
                .collect();
        }
    }

    let coarsest_first: Vec<Vec<usize>> = refined.into_iter().rev().collect();
    let levels: Vec<FeatureCoarsening> = coarsest_first
        .into_iter()
        .map(|labels| {
            let (_, num_coarse) = compact_labels(&labels);
            let mut coarse_to_fine: Vec<Vec<usize>> = vec![Vec::new(); num_coarse];
            for (f, &c) in labels.iter().enumerate() {
                coarse_to_fine[c].push(f);
            }
            FeatureCoarsening {
                fine_to_coarse: labels,
                coarse_to_fine,
                num_coarse,
            }
        })
        .collect();

    info!(
        "Refined feature coarsening: levels K = {:?}",
        levels.iter().map(|l| l.num_coarse).collect::<Vec<_>>()
    );
    Ok(MultilevelFeatureCoarsening { levels })
}

///////////
// Tests //
///////////

#[cfg(test)]
#[path = "feature_coarsening_multilevel_tests.rs"]
mod tests;
