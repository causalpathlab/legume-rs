//! Build the **two** frozen contrastive indices a pseudobulk-phase Gibbs needs:
//! one anchored on feature rows (genes) with the pb side frozen, and one anchored
//! on pseudobulk columns with the feature side frozen.
//!
//! [`super::index::build_index`] hard-codes the other orientation — anchors are
//! genes, the other side is *cells*, and the negative slate is drawn from cells.
//! That is the right shape for a post-hoc pass over a fitted model, but a Gibbs
//! that alternates sides needs the transpose as well, and it needs both built from
//! the same data so the two conditionals describe one joint.
//!
//! ## Why pseudobulks and not cells
//!
//! The pb axis is small — `[10, 7, 4]` sort dims give ≤1024/128/16 columns per
//! level, ~1168 stacked — so the Poisson rate normalizer over it is **exact**:
//! `n_partition` clamps to the column count and `partition_scale` is `1.0`. The
//! cell-anchored builder cannot say that; on a 8.3k-cell input it samples a
//! 1024-of-8313 slate. Note the exactness is a property of the **gene** block
//! only: the pb block's other side is the feature axis (tens of thousands of
//! rows), so it still draws a frozen slate.
//!
//! ## Exposure — the part that is easy to get wrong
//!
//! The collapse emits Gamma-posterior **rates** (per-cell means), not counts, and
//! a Poisson fit to a rate is a quasi-Poisson with a per-column dispersion. The
//! measured consequence is recorded on [`StackedPb`]: `Spearman(LRT, detection) =
//! +0.60`, lower quantiles collapsed to zero, `σ̂² → 0`, and 59% of dropped genes
//! called live against `π̂₀ = 0.81`. A spike-and-slab is *exactly* the consumer
//! that breaks on — `σ₀h²` and `π₀h` are what it samples. So this module takes its
//! edges on the count scale (`rate · size_p`) and its pb biases with the
//! `log(size_p)` exposure offset already folded in, both straight off
//! [`StackedPb`], rather than re-deriving either here.

use super::index::ContrastiveIndex;
use crate::fit::feature_projection::StackedPb;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Draw `k` distinct indices out of `0..n` uniformly at random.
///
/// `partial_shuffle` puts the sample at the **END** of the slice and returns it as
/// the first element of the pair — rand's own doctest asserts
/// `sampled == &y[len - amount..]`. Discarding that return value and calling
/// `truncate(k)` therefore keeps the UNSELECTED head, which is a sorted prefix of
/// the axis, not a sample. That silently turns the frozen negative slate into
/// "the first k rows in backend order" while `partition_scale` still folds it up
/// as if it were uniform, so every rate estimate is biased toward whatever those
/// rows happen to be. Use the returned slice.
fn sample_slate(n: usize, k: usize, rng: &mut StdRng) -> Vec<u32> {
    let mut all: Vec<u32> = (0..n as u32).collect();
    let (selected, _rest) = all.partial_shuffle(rng, k);
    selected.to_vec()
}

/// The two conditionals of one pb joint, built from a single pass over the counts.
pub struct PbIndexPair {
    /// Anchors are feature rows (or genes under a grouping); the frozen other side
    /// is the stacked pb axis, and the partition over it is exact.
    pub by_feature: ContrastiveIndex,
    /// Anchors are the stacked pb columns; the frozen other side is the per-row
    /// feature loading, with a sampled slate over rows.
    pub by_pb: ContrastiveIndex,
    /// Observed edges kept, i.e. the length of one direction's edge list. Both
    /// directions hold the same edges, so the pair costs `2 × n_edges` entries.
    pub n_edges: usize,
}

/// Inputs describing the feature (row) side that the pb-anchored index freezes.
pub struct FeatureSide<'a> {
    /// `[n_features × h]` row-major **effective** per-row loading — for gem this
    /// is the materialized `β_g` on spliced rows and `β_g + δ_g` on unspliced
    /// ones, so the pb conditional needs no track logic of its own.
    pub e_feat: &'a [f32],
    /// `[n_features]` per-row bias.
    pub b_feat: &'a [f32],
    /// Unified feature id → backend row, since [`StackedPb`] keeps its count
    /// matrices on the **full backend** axis while the model is on the unified one.
    pub feature_to_backend_row: &'a [usize],
}

/// How feature rows pool into gene-side anchors. `None` = one anchor per row.
pub struct AnchorMap<'a> {
    /// Row → anchor, length `n_features`; `u32::MAX` drops the row from the
    /// **gene** side only. A dropped row still contributes to every pb's edge
    /// list: a pseudobulk is scored against the whole feature axis regardless of
    /// which track the gene block happens to be updating, and thinning it there
    /// would silently misspecify the pb conditional.
    pub row_to_anchor: &'a [u32],
    pub n_anchors: usize,
}

/// Build both conditionals in one pass over the stacked pseudobulk counts.
///
/// `n_partition` caps **both** slates; `0` (or a value at least as large as the
/// axis) sums that side exactly. In practice the pb side is usually exact and the
/// feature side is not, because the two axes differ by an order of magnitude —
/// but the pb axis grows with the cell count, so it is not exact by construction.
/// `seed` fixes each slate once, shared across anchors and sweeps: a slate that
/// moves between sweeps is not a valid frozen normalizer.
pub(crate) fn build_pb_index_pair(
    pb: &StackedPb<'_>,
    feat: &FeatureSide<'_>,
    anchors: Option<&AnchorMap<'_>>,
    h: usize,
    n_partition: usize,
    seed: u64,
) -> anyhow::Result<PbIndexPair> {
    let n_features = feat.b_feat.len();
    let n_pb = pb.n_pb_total();
    anyhow::ensure!(
        feat.e_feat.len() == n_features * h,
        "feature side is {} floats but {n_features} rows × {h} dims = {}",
        feat.e_feat.len(),
        n_features * h
    );
    anyhow::ensure!(
        feat.feature_to_backend_row.len() == n_features,
        "feature_to_backend_row has {} entries but the feature axis has {n_features}",
        feat.feature_to_backend_row.len()
    );
    anyhow::ensure!(
        pb.theta.len() == n_pb * h,
        "pb side is {} floats but {n_pb} columns × {h} dims = {}",
        pb.theta.len(),
        n_pb * h
    );
    if let Some(a) = anchors {
        anyhow::ensure!(
            a.row_to_anchor.len() == n_features,
            "row_to_anchor has {} entries but the feature axis has {n_features}",
            a.row_to_anchor.len()
        );
    }
    let n_anchors = anchors.map_or(n_features, |a| a.n_anchors);

    // One pass, both directions. `by_feature` pools a gene's rows (so one pb can
    // appear twice for a gene whose spliced and unspliced rows both map to it —
    // the Poisson data term is a sum over edges, so repeats add exactly as pooling
    // should). `by_pb` keeps every row, dropped-or-not.
    let mut pos_feat: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_anchors];
    let mut pos_pb: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_pb];
    let mut n_edges = 0usize;

    for uid in 0..n_features {
        let brow = feat.feature_to_backend_row[uid];
        let anchor = anchors.map_or(uid as u32, |a| a.row_to_anchor[uid]);
        for (level, m) in pb.counts.iter().enumerate() {
            if brow >= m.nrows() {
                continue; // row not present in this level's backend matrix
            }
            let base = pb.offsets[level];
            for s in 0..m.ncols() {
                let rate = m[(brow, s)];
                if rate == 0.0 {
                    continue;
                }
                // Count scale: `rate · size_p`. See the module doc — this is the
                // half of the exposure correction that lives on the data; the
                // other half (`log size_p`) is already inside `pb.bias`.
                let v = rate * pb.sizes[level][s];
                let gp = (base + s) as u32;
                if anchor != u32::MAX {
                    pos_feat[anchor as usize].push((gp, v));
                }
                pos_pb[base + s].push((uid as u32, v));
                n_edges += 1;
            }
        }
    }

    // Gene-side anchor bias: the first row mapping to it. Unused by the profile
    // likelihood (which maximizes the intercept out) but carried for a caller
    // that runs the fixed-intercept Poisson instead.
    let anchor_b_feat = match anchors {
        None => feat.b_feat.to_vec(),
        Some(a) => {
            let mut b = vec![0.0f32; n_anchors];
            let mut seen = vec![false; n_anchors];
            for (uid, &an) in a.row_to_anchor.iter().enumerate() {
                if an != u32::MAX && !seen[an as usize] {
                    b[an as usize] = feat.b_feat[uid];
                    seen[an as usize] = true;
                }
            }
            b
        }
    };

    // pb side: sum exactly when it is small enough to be free, else draw a slate
    // like any other axis.
    //
    // Measured, and NOT what the design assumed: the collapse does not size a
    // level at `2^sort_dim`. It builds the finest level from a kNN ratio — 2000
    // cells gave 838/494/128 pb, Σ = 1460, i.e. ~0.73 × n_cells stacked. So the pb
    // axis grows WITH the cell count and "exact is free" only holds at small n.
    // Above the cap an exact sum would cost more per likelihood evaluation than
    // the cell-side slate this replaces.
    let (partition_pb, scale_pb) = if n_partition == 0 || n_partition >= n_pb {
        ((0..n_pb as u32).collect::<Vec<u32>>(), 1.0)
    } else {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x51ED_270B);
        (
            sample_slate(n_pb, n_partition, &mut rng),
            n_pb as f64 / n_partition as f64,
        )
    };

    // Feature side needs a slate: the row axis is tens of thousands wide.
    let (partition_feat, scale_feat) = if n_partition == 0 || n_partition >= n_features {
        ((0..n_features as u32).collect::<Vec<u32>>(), 1.0)
    } else {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9);
        (
            sample_slate(n_features, n_partition, &mut rng),
            n_features as f64 / n_partition as f64,
        )
    };

    Ok(PbIndexPair {
        by_feature: ContrastiveIndex {
            other_e: pb.theta.clone(),
            other_b: pb.bias.clone(),
            h,
            pos: pos_feat,
            anchor_b: anchor_b_feat,
            partition: partition_pb,
            partition_scale: scale_pb,
            anchor_offset: None,
        },
        by_pb: ContrastiveIndex {
            other_e: feat.e_feat.to_vec(),
            other_b: feat.b_feat.to_vec(),
            h,
            pos: pos_pb,
            anchor_b: pb.bias.clone(),
            partition: partition_feat,
            partition_scale: scale_feat,
            anchor_offset: None,
        },
        n_edges,
    })
}

#[cfg(test)]
#[path = "pb_index_tests.rs"]
mod pb_index_tests;
