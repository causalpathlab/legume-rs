use crate::data::Triplet;
use crate::fit::config::TrackSpec;
use crate::fit::projection::CellBatchFold;

/// Every unit's feature rows and counts, plus its exposure per TRACK.
///
/// Invariants: `feats[u]` is ascending and matches `counts[u]` in length;
/// `total` and `weight` are `[n_units × n_tracks]` row-major, indexed by
/// [`UnitTable::total_of`] / [`UnitTable::weight_of`]; `tracks` describes the
/// same `n_features`-row axis every entry of `feats` indexes.
pub struct UnitTable {
    pub n_features: usize,
    /// row → (track, gene) for the axis `feats` indexes.
    pub tracks: TrackSpec,
    pub feats: Vec<Vec<u32>>,
    pub counts: Vec<Vec<f32>>,
    /// `[n_units × n_tracks]` row-major.
    pub total: Vec<f32>,
    /// `[n_units × n_tracks]` row-major: `sqrt(total)` normalized to mean 1
    /// over ALL units within its track (all zero when that mean is zero).
    pub weight: Vec<f32>,
    pub level: Vec<u8>,
    pub source_index: Vec<u32>,
}

impl UnitTable {
    pub fn n_units(&self) -> usize {
        self.feats.len()
    }

    pub fn n_tracks(&self) -> usize {
        self.tracks.n_tracks()
    }

    /// Unit `u`'s total count on track `t`.
    pub fn total_of(&self, u: usize, t: usize) -> f32 {
        self.total[u * self.n_tracks() + t]
    }

    /// Unit `u`'s loss weight on track `t`; `0` when it has no counts there.
    pub fn weight_of(&self, u: usize, t: usize) -> f32 {
        self.weight[u * self.n_tracks() + t]
    }

    /// Pseudobulk levels first (coarsest → finest, each level's pb index
    /// order), then cells. Every pseudobulk index in `0..n_pb_per_level[l]`
    /// gets a row at level `l`, even if it never appears in that level's edge
    /// list (empty row). Counts ≤ 0 are dropped; cell counts are divided by
    /// their batch's fold when one is given.
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
    /// genes: totals and weights are accumulated per track.
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

        // Exposure per (unit, track): a unit's counts split by the track its
        // rows belong to. One track reproduces the plain per-unit total.
        let (n_u, n_t) = (feats.len(), tracks.n_tracks());
        let mut total = vec![0f32; n_u * n_t];
        for (u, (f, c)) in feats.iter().zip(&counts).enumerate() {
            for (&row, &x) in f.iter().zip(c) {
                total[u * n_t + tracks.track_of_row[row as usize] as usize] += x;
            }
        }
        // `weight[u, t] = sqrt(total[u, t]) / mean_u sqrt(total[u, t])`, the mean
        // taken over ALL units (a unit with nothing on track `t` still counts in
        // the denominator, and gets weight 0).
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
        Self {
            n_features,
            tracks,
            feats,
            counts,
            total,
            weight,
            level,
            source_index,
        }
    }
}

#[cfg(test)]
#[path = "units_tests.rs"]
mod units_tests;
