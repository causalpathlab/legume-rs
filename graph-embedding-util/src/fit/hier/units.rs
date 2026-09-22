use crate::data::Triplet;
use crate::fit::config::TrackSpec;
use crate::fit::projection::CellBatchFold;

/// One sparse feature-count axis (genes, peaks, …).
///
/// Per unit, `feats[u]` is ascending and matches `counts[u]` in length.
/// `total[u]` is the sum of `counts[u]`; `weight[u]` is `sqrt(total)`
/// normalized to mean 1 over all units (all zero when that mean is zero).
/// Never a dense `U × F` block.
pub struct FeatureAxis {
    pub n_features: usize,
    pub feats: Vec<Vec<u32>>,
    pub counts: Vec<Vec<f32>>,
    pub total: Vec<f32>,
    pub weight: Vec<f32>,
}

/// Every unit's sparse count axes, plus axis-0 track exposure.
///
/// Gene-only fits have `axes.len() == 1`. Multiome holds genes then peaks
/// (or more) as homogeneous [`FeatureAxis`] entries — not `peak_*` fields.
///
/// `tracks` / [`Self::total`] / [`Self::weight`] describe axis 0 only (the
/// TrackSpec gene axis): totals and weights are `[n_units × n_tracks]`
/// row-major, indexed by [`UnitTable::total_of`] / [`UnitTable::weight_of`].
/// Additional axes use their own per-unit [`FeatureAxis::total`] /
/// [`FeatureAxis::weight`].
pub struct UnitTable {
    /// Sparse count axes. Gene-only: length 1.
    pub axes: Vec<FeatureAxis>,
    /// row → (track, gene) for `axes[0]`.
    pub tracks: TrackSpec,
    /// `[n_units × n_tracks]` row-major for axis 0.
    pub total: Vec<f32>,
    /// `[n_units × n_tracks]` row-major: `sqrt(total)` normalized to mean 1
    /// over ALL units within its track (all zero when that mean is zero).
    pub weight: Vec<f32>,
    pub level: Vec<u8>,
    pub source_index: Vec<u32>,
}

impl UnitTable {
    pub fn n_units(&self) -> usize {
        self.level.len()
    }

    pub fn n_axes(&self) -> usize {
        self.axes.len()
    }

    /// Feature count on axis 0 (genes / TrackSpec rows).
    pub fn n_features(&self) -> usize {
        self.axes[0].n_features
    }

    pub fn n_tracks(&self) -> usize {
        self.tracks.n_tracks()
    }

    /// Unit `u`'s total count on track `t` (axis 0).
    pub fn total_of(&self, u: usize, t: usize) -> f32 {
        self.total[u * self.n_tracks() + t]
    }

    /// Unit `u`'s loss weight on track `t`; `0` when it has no counts there.
    pub fn weight_of(&self, u: usize, t: usize) -> f32 {
        self.weight[u * self.n_tracks() + t]
    }

    /// The plain gene axis: [`Self::from_pseudobulks_and_cells_tracked`] with
    /// [`TrackSpec::base`]. `fit` always builds a spec (base or not) and calls
    /// the tracked constructor, so inside this crate this is the tests' handle
    /// on the untracked path — the parity guard that the two agree.
    ///
    /// Pseudobulk levels first (coarsest → finest, each level's pb index
    /// order), then cells. Every pseudobulk index in `0..n_pb_per_level[l]`
    /// gets a row at level `l`, even if it never appears in that level's edge
    /// list (empty row). Counts ≤ 0 are dropped; cell counts are divided by
    /// their batch's fold when one is given.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_pseudobulks_and_cells(
        pb_blobs: &[&[Triplet]],
        n_pb_per_level: &[usize],
        cells: &[(u32, &[u32], &[f32])],
        fold: Option<CellBatchFold<'_>>,
        n_features: usize,
    ) -> Self {
        Self::from_pseudobulks_and_cells_tracked(
            pb_blobs,
            n_pb_per_level,
            cells,
            fold,
            n_features,
            TrackSpec::base(n_features),
        )
    }

    /// [`Self::from_pseudobulks_and_cells`] on an axis whose rows are tracks of
    /// genes: totals and weights are accumulated per track. Yields `n_axes == 1`.
    pub(crate) fn from_pseudobulks_and_cells_tracked(
        pb_blobs: &[&[Triplet]],
        n_pb_per_level: &[usize],
        cells: &[(u32, &[u32], &[f32])],
        fold: Option<CellBatchFold<'_>>,
        n_features: usize,
        tracks: TrackSpec,
    ) -> Self {
        assert_eq!(
            pb_blobs.len(),
            n_pb_per_level.len(),
            "pb_blobs and n_pb_per_level must have the same length"
        );

        let mut feats: Vec<Vec<u32>> = Vec::new();
        let mut counts: Vec<Vec<f32>> = Vec::new();
        let mut level: Vec<u8> = Vec::new();
        let mut source_index: Vec<u32> = Vec::new();

        for (l, (blob, &n_pb)) in pb_blobs.iter().zip(n_pb_per_level).enumerate() {
            let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_pb];
            for t in blob.iter().filter(|t| t.count > 0.0) {
                assert!(
                    (t.cell as usize) < n_pb,
                    "triplet cell {} exceeds level {}'s count {}",
                    t.cell,
                    l,
                    n_pb
                );
                rows[t.cell as usize].push((t.feature, t.count));
            }
            for (p, mut row) in rows.into_iter().enumerate() {
                row.sort_unstable_by_key(|&(f, _)| f);
                let (f, c): (Vec<u32>, Vec<f32>) = row.into_iter().unzip();
                feats.push(f);
                counts.push(c);
                level.push(l as u8);
                source_index.push(p as u32);
            }
        }
        let cell_level = pb_blobs.len() as u8;
        for &(cell, f, c) in cells {
            let mut row: Vec<(u32, f32)> = crate::fit::projection::cell_edges(cell, f, c, fold)
                .filter(|&(_, n)| n > 0.0)
                .collect();
            row.sort_unstable_by_key(|&(f, _)| f);
            let (f, c): (Vec<u32>, Vec<f32>) = row.into_iter().unzip();
            feats.push(f);
            counts.push(c);
            level.push(cell_level);
            source_index.push(cell);
        }

        let axis = finish_axis(n_features, feats, counts);
        let (n_u, n_t) = (axis.feats.len(), tracks.n_tracks());
        let (total, weight) = track_totals_weights(&axis.feats, &axis.counts, &tracks, n_u, n_t);
        Self {
            axes: vec![axis],
            tracks,
            total,
            weight,
            level,
            source_index,
        }
    }

    /// Frozen multi-axis units from one pb triplet blob list per feature axis.
    ///
    /// Same level layout as [`Self::from_pseudobulks_and_cells`]: each level's
    /// `n_pb_per_level[l]` rows in index order (empty rows kept). No cell
    /// subsample — callers that freeze pb samples after multilevel collapse
    /// pass only those pb rows. Axis 0 uses [`TrackSpec::base`]; further axes
    /// are plain sparse count axes (not TrackSpec tracks of genes).
    /// Frozen multi-axis units from one pb triplet blob list per feature axis.
    pub fn from_pseudobulk_axes(
        axis_blobs: &[&[&[Triplet]]],
        n_pb_per_level: &[usize],
        n_features: &[usize],
    ) -> Self {
        assert!(
            !axis_blobs.is_empty(),
            "at least one feature axis is required"
        );
        assert_eq!(
            axis_blobs.len(),
            n_features.len(),
            "axis_blobs and n_features must have the same length"
        );
        for (a, blobs) in axis_blobs.iter().enumerate() {
            assert_eq!(
                blobs.len(),
                n_pb_per_level.len(),
                "axis {a}: blobs and n_pb_per_level must have the same length"
            );
        }

        let mut level: Vec<u8> = Vec::new();
        let mut source_index: Vec<u32> = Vec::new();
        let mut per_axis_feats: Vec<Vec<Vec<u32>>> = vec![Vec::new(); axis_blobs.len()];
        let mut per_axis_counts: Vec<Vec<Vec<f32>>> = vec![Vec::new(); axis_blobs.len()];

        for (l, &n_pb) in n_pb_per_level.iter().enumerate() {
            let mut rows_per_axis: Vec<Vec<Vec<(u32, f32)>>> =
                axis_blobs.iter().map(|_| vec![Vec::new(); n_pb]).collect();
            for (a, blobs) in axis_blobs.iter().enumerate() {
                for t in blobs[l].iter().filter(|t| t.count > 0.0) {
                    assert!(
                        (t.cell as usize) < n_pb,
                        "axis {a} triplet cell {} exceeds level {l}'s count {n_pb}",
                        t.cell
                    );
                    rows_per_axis[a][t.cell as usize].push((t.feature, t.count));
                }
            }
            let mut axis_iters: Vec<_> = rows_per_axis
                .into_iter()
                .map(|rows| rows.into_iter())
                .collect();
            for p in 0..n_pb {
                for (a, it) in axis_iters.iter_mut().enumerate() {
                    let mut row = it.next().expect("one row per pb index");
                    row.sort_unstable_by_key(|&(f, _)| f);
                    let (f, c): (Vec<u32>, Vec<f32>) = row.into_iter().unzip();
                    per_axis_feats[a].push(f);
                    per_axis_counts[a].push(c);
                }
                level.push(l as u8);
                source_index.push(p as u32);
            }
        }

        let axes: Vec<FeatureAxis> = (0..axis_blobs.len())
            .map(|a| {
                finish_axis(
                    n_features[a],
                    std::mem::take(&mut per_axis_feats[a]),
                    std::mem::take(&mut per_axis_counts[a]),
                )
            })
            .collect();

        let tracks = TrackSpec::base(n_features[0]);
        let n_u = axes[0].feats.len();
        let n_t = tracks.n_tracks();
        let (total, weight) =
            track_totals_weights(&axes[0].feats, &axes[0].counts, &tracks, n_u, n_t);
        Self {
            axes,
            tracks,
            total,
            weight,
            level,
            source_index,
        }
    }
}

fn finish_axis(n_features: usize, feats: Vec<Vec<u32>>, counts: Vec<Vec<f32>>) -> FeatureAxis {
    let total: Vec<f32> = counts.iter().map(|c| c.iter().sum::<f32>()).collect();
    let raw: Vec<f32> = total.iter().map(|t| t.sqrt()).collect();
    let n_u = feats.len();
    let mean = raw.iter().sum::<f32>() / n_u.max(1) as f32;
    let weight = if mean > 0.0 {
        raw.iter().map(|r| r / mean).collect()
    } else {
        vec![0.0; n_u]
    };
    FeatureAxis {
        n_features,
        feats,
        counts,
        total,
        weight,
    }
}

fn track_totals_weights(
    feats: &[Vec<u32>],
    counts: &[Vec<f32>],
    tracks: &TrackSpec,
    n_u: usize,
    n_t: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut total = vec![0f32; n_u * n_t];
    for (u, (f, c)) in feats.iter().zip(counts).enumerate() {
        for (&row, &x) in f.iter().zip(c) {
            total[u * n_t + tracks.track_of_row[row as usize] as usize] += x;
        }
    }
    let raw: Vec<f32> = total.iter().map(|t| t.sqrt()).collect();
    let mut weight = vec![0f32; n_u * n_t];
    for t in 0..n_t {
        let mean = (0..n_u).map(|u| raw[u * n_t + t]).sum::<f32>() / n_u.max(1) as f32;
        if mean > 0.0 {
            for u in 0..n_u {
                weight[u * n_t + t] = raw[u * n_t + t] / mean;
            }
        }
    }
    (total, weight)
}

#[cfg(test)]
#[path = "units_tests.rs"]
mod units_tests;
