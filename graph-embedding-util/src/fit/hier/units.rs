use crate::data::Triplet;
use crate::fit::projection::CellBatchFold;

pub struct UnitTable {
    pub n_features: usize,
    pub feats: Vec<Vec<u32>>,
    pub counts: Vec<Vec<f32>>,
    pub total: Vec<f32>,
    pub weight: Vec<f32>,
    pub level: Vec<u8>,
    pub source_index: Vec<u32>,
}

impl UnitTable {
    pub fn n_units(&self) -> usize {
        self.feats.len()
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

        let total: Vec<f32> = counts.iter().map(|c| c.iter().sum()).collect();
        let raw: Vec<f32> = total.iter().map(|t| t.sqrt()).collect();
        let mean = raw.iter().sum::<f32>() / raw.len().max(1) as f32;
        let weight = if mean > 0.0 {
            raw.iter().map(|w| w / mean).collect()
        } else {
            vec![0.0; raw.len()]
        };
        Self {
            n_features,
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
