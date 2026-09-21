//! Cell embedding → clusters for within-cluster peak→gene refine.
//!
//! Wraps `graph_embedding_util::postprocess::cell_clusters` (Leiden) and drops
//! clusters below a min-cell gate.

use graph_embedding_util::postprocess::cell_clusters;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::traits::ConvertMatOps;
use nalgebra::DMatrix;

/// Cluster assignment after the min-cell gate: `label[i]` is dense `0..k` or
/// `None` if the cell belonged to a dropped (too-small) cluster.
#[derive(Clone, Debug)]
pub struct CellClusters {
    pub label: Vec<Option<usize>>,
    pub n_clusters: usize,
    pub sizes: Vec<usize>,
}

/// Leiden-cluster cell rows; drop clusters with size &lt; `min_cells`.
pub fn cluster_cells(
    e_cell: &DMatrix<f32>,
    target_clusters: Option<usize>,
    min_cells: usize,
) -> anyhow::Result<CellClusters> {
    anyhow::ensure!(e_cell.nrows() > 0, "no cells to cluster");
    anyhow::ensure!(e_cell.ncols() > 0, "cell embedding has width 0");
    anyhow::ensure!(min_cells >= 1, "min_cells must be ≥ 1");

    let tensor = e_cell.to_tensor(&Device::Cpu)?;
    let (raw, _target_eff) = cell_clusters(&tensor, target_clusters)?;
    anyhow::ensure!(raw.len() == e_cell.nrows(), "label/row mismatch");

    let raw_k = raw.iter().copied().max().map_or(0, |m| m + 1);
    let mut raw_sizes = vec![0usize; raw_k];
    for &lab in &raw {
        raw_sizes[lab] += 1;
    }

    // Map surviving raw labels → dense 0..k.
    let mut remap = vec![None; raw_k];
    let mut sizes = Vec::new();
    for (lab, &sz) in raw_sizes.iter().enumerate() {
        if sz >= min_cells {
            remap[lab] = Some(sizes.len());
            sizes.push(sz);
        }
    }
    let n_clusters = sizes.len();
    let label: Vec<Option<usize>> = raw.iter().map(|&lab| remap[lab]).collect();

    Ok(CellClusters {
        label,
        n_clusters,
        sizes,
    })
}
