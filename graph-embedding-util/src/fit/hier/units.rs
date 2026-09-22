use crate::data::Triplet;
use crate::fit::config::TrackSpec;
use crate::fit::projection::CellBatchFold;

/// Every unit's feature rows and counts, plus its exposure per TRACK.
///
/// Invariants: `feats[u]` is ascending and matches `counts[u]` in length;
/// `total` and `weight` are `[n_units × n_tracks]` row-major, indexed by
/// [`UnitTable::total_of`] / [`UnitTable::weight_of`]; `tracks` describes the
/// same `n_features`-row axis every entry of `feats` indexes.
///
/// Peaks are a second count modality, not a [`TrackSpec`] track of genes:
/// `peak_feats[u]` / `peak_counts[u]` index `0..n_peaks`, with per-unit
/// `peak_total` / `peak_weight` (one scalar per unit, not per track).
pub struct UnitTable {
    pub n_features: usize,
    /// Number of peaks on the ATAC axis (`peak_feats` indexes).
    pub n_peaks: usize,
    /// row → (track, gene) for the axis `feats` indexes.
    pub tracks: TrackSpec,
    pub feats: Vec<Vec<u32>>,
    pub counts: Vec<Vec<f32>>,
    /// Per-unit sparse peak ids (ascending), same length as `peak_counts`.
    pub peak_feats: Vec<Vec<u32>>,
    /// Per-unit sparse ATAC counts aligned with `peak_feats`.
    pub peak_counts: Vec<Vec<f32>>,
    /// `[n_units × n_tracks]` row-major.
    pub total: Vec<f32>,
    /// `[n_units × n_tracks]` row-major: `sqrt(total)` normalized to mean 1
    /// over ALL units within its track (all zero when that mean is zero).
    pub weight: Vec<f32>,
    /// Per-unit ATAC total (sum of `peak_counts[u]`).
    pub peak_total: Vec<f32>,
    /// Per-unit ATAC weight: `sqrt(peak_total)` normalized to mean 1 over all
    /// units (all zero when that mean is zero).
    pub peak_weight: Vec<f32>,
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
            n_peaks: 0,
            tracks,
            feats,
            counts,
            peak_feats: vec![Vec::new(); n_u],
            peak_counts: vec![Vec::new(); n_u],
            total,
            weight,
            peak_total: vec![0.0; n_u],
            peak_weight: vec![0.0; n_u],
            level,
            source_index,
        }
    }

    /// Frozen dual-modality units from RNA and ATAC pb triplet blobs.
    ///
    /// Same level layout as [`Self::from_pseudobulks_and_cells`]: each level's
    /// `n_pb_per_level[l]` rows in index order (empty rows kept). No cell
    /// subsample — callers that freeze pb samples after multilevel collapse
    /// pass only those pb rows. Peaks are a second member list / count axis,
    /// not a gene [`TrackSpec`] track.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_dual_pseudobulks(
        rna_blobs: &[&[Triplet]],
        atac_blobs: &[&[Triplet]],
        n_pb_per_level: &[usize],
        n_genes: usize,
        n_peaks: usize,
    ) -> Self {
        assert_eq!(
            rna_blobs.len(),
            n_pb_per_level.len(),
            "rna_blobs and n_pb_per_level must have the same length"
        );
        assert_eq!(
            atac_blobs.len(),
            n_pb_per_level.len(),
            "atac_blobs and n_pb_per_level must have the same length"
        );

        let mut feats: Vec<Vec<u32>> = Vec::new();
        let mut counts: Vec<Vec<f32>> = Vec::new();
        let mut peak_feats: Vec<Vec<u32>> = Vec::new();
        let mut peak_counts: Vec<Vec<f32>> = Vec::new();
        let mut level: Vec<u8> = Vec::new();
        let mut source_index: Vec<u32> = Vec::new();

        for (l, ((&n_pb, rna), atac)) in n_pb_per_level
            .iter()
            .zip(rna_blobs.iter())
            .zip(atac_blobs.iter())
            .enumerate()
        {
            let mut rna_rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_pb];
            for t in rna.iter().filter(|t| t.count > 0.0) {
                assert!(
                    (t.cell as usize) < n_pb,
                    "RNA triplet cell {} exceeds level {}'s count {}",
                    t.cell,
                    l,
                    n_pb
                );
                rna_rows[t.cell as usize].push((t.feature, t.count));
            }
            let mut atac_rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_pb];
            for t in atac.iter().filter(|t| t.count > 0.0) {
                assert!(
                    (t.cell as usize) < n_pb,
                    "ATAC triplet cell {} exceeds level {}'s count {}",
                    t.cell,
                    l,
                    n_pb
                );
                atac_rows[t.cell as usize].push((t.feature, t.count));
            }
            for (p, (mut rna_row, mut atac_row)) in rna_rows.into_iter().zip(atac_rows).enumerate()
            {
                rna_row.sort_unstable_by_key(|&(f, _)| f);
                atac_row.sort_unstable_by_key(|&(f, _)| f);
                let (f, c): (Vec<u32>, Vec<f32>) = rna_row.into_iter().unzip();
                let (pf, pc): (Vec<u32>, Vec<f32>) = atac_row.into_iter().unzip();
                feats.push(f);
                counts.push(c);
                peak_feats.push(pf);
                peak_counts.push(pc);
                level.push(l as u8);
                source_index.push(p as u32);
            }
        }

        let tracks = TrackSpec::base(n_genes);
        let n_u = feats.len();
        let n_t = tracks.n_tracks();
        let mut total = vec![0f32; n_u * n_t];
        for (u, (f, c)) in feats.iter().zip(&counts).enumerate() {
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

        let peak_total: Vec<f32> = peak_counts.iter().map(|c| c.iter().sum::<f32>()).collect();
        let peak_raw: Vec<f32> = peak_total.iter().map(|t| t.sqrt()).collect();
        let peak_mean = peak_raw.iter().sum::<f32>() / n_u.max(1) as f32;
        let peak_weight: Vec<f32> = if peak_mean > 0.0 {
            peak_raw.iter().map(|r| r / peak_mean).collect()
        } else {
            vec![0.0; n_u]
        };

        Self {
            n_features: n_genes,
            n_peaks,
            tracks,
            feats,
            counts,
            peak_feats,
            peak_counts,
            total,
            weight,
            peak_total,
            peak_weight,
            level,
            source_index,
        }
    }
}

#[cfg(test)]
#[path = "units_tests.rs"]
mod units_tests;
