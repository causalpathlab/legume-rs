//! The hierarchical pseudobulk levels: per-level pb profiles plus the
//! child → parent map between consecutive levels.
//!
//! Levels are finest-first, matching data-beans'
//! `MultilevelCollapseOut::cell_to_pb_per_level`; level 0 is the pb that the
//! link scores are computed on. Profiles are support-masked posterior means:
//! an entry the pb never observed is 0, not the Gamma prior's floor, so a
//! nonzero entry is one the count relations may carry.

use crate::common::Mat;

/// Per-level pseudobulk profiles and the tree between them.
#[derive(Clone, Debug)]
pub struct PbLevels {
    /// RNA per level; `None` when the input is ATAC-only, where the gene
    /// activity surrogate stands in for it on the embed's gene axis.
    pub rna: Option<Vec<Mat>>,
    /// ATAC per level.
    pub atac: Vec<Mat>,
    /// `parent[l][i]` = pb at level `l + 1` holding pb `i` of level `l`;
    /// one entry fewer than there are levels.
    pub parent: Vec<Vec<usize>>,
}

impl PbLevels {
    pub fn n_levels(&self) -> usize {
        self.atac.len()
    }

    pub fn n_pb(&self, level: usize) -> usize {
        self.atac[level].ncols()
    }

    pub fn n_pb_per_level(&self) -> Vec<usize> {
        self.atac.iter().map(Mat::ncols).collect()
    }
}

/// Child → parent map between each pair of consecutive levels. The refined
/// partition nests strictly, so every cell of a fine pb names the same coarse
/// pb; a pb whose cells disagree is an error, not a vote.
pub fn parent_maps(
    cell_to_pb_per_level: &[Vec<usize>],
    n_pb_per_level: &[usize],
) -> anyhow::Result<Vec<Vec<usize>>> {
    anyhow::ensure!(
        cell_to_pb_per_level.len() == n_pb_per_level.len(),
        "{} membership levels vs {} pb counts",
        cell_to_pb_per_level.len(),
        n_pb_per_level.len()
    );
    let n_cells = cell_to_pb_per_level.first().map_or(0, Vec::len);
    for (l, m) in cell_to_pb_per_level.iter().enumerate() {
        anyhow::ensure!(
            m.len() == n_cells,
            "level {l} lists {} cells, level 0 lists {n_cells}",
            m.len()
        );
        anyhow::ensure!(
            m.iter().all(|&p| p < n_pb_per_level[l]),
            "level {l} membership exceeds its {} pbs",
            n_pb_per_level[l]
        );
    }

    let mut parent = Vec::with_capacity(n_pb_per_level.len().saturating_sub(1));
    for l in 0..n_pb_per_level.len().saturating_sub(1) {
        let mut map = vec![usize::MAX; n_pb_per_level[l]];
        for (&fine, &coarse) in cell_to_pb_per_level[l]
            .iter()
            .zip(&cell_to_pb_per_level[l + 1])
        {
            let slot = &mut map[fine];
            if *slot == usize::MAX {
                *slot = coarse;
            }
            anyhow::ensure!(
                *slot == coarse,
                "pb {fine} at level {l} straddles coarse pbs {} and {coarse}",
                *slot
            );
        }
        if let Some(i) = map.iter().position(|&p| p == usize::MAX) {
            anyhow::bail!("pb {i} at level {l} holds no cell");
        }
        parent.push(map);
    }
    Ok(parent)
}
