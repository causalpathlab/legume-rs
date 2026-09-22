//! Phase 2: every cell onto the frozen gene / peak dictionaries, one Poisson
//! partition and one intercept per axis, one shared latent. Cells stream from
//! the per-modality backends in groups; nothing dense over cells × features.

use crate::common::Mat;
use data_beans::sparse_io_vector::SparseIoVec;
use graph_embedding_util::fit::projection::{
    project_cells_axes, stream_cell_groups, AxesProjector, AxisDict,
};
use graph_embedding_util::fit::PROJECTION_RIDGE_SGD;
use graph_embedding_util::save_embedding;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::traits::ConvertMatOps;
use log::info;

/// One frozen axis as the hier fit leaves it: composed rows `[F][H]` and bias `[F]`.
pub struct FrozenAxis<'a> {
    pub label: &'a str,
    pub rows: &'a [Vec<f32>],
    pub bias: &'a [f32],
}

pub struct CellEmbedOut {
    /// `[n_cells × H]`, gauge-centred.
    pub theta: Mat,
    pub barcodes: Vec<Box<str>>,
}

/// Project every column of `backends` (one per axis, in `axes` order). The
/// per-axis intercepts are solved but not kept: nothing downstream reads them.
pub fn embed_cells(
    axes: &[FrozenAxis],
    backends: &[&SparseIoVec],
    barcodes: Vec<Box<str>>,
    dim: usize,
    device: &Device,
) -> anyhow::Result<CellEmbedOut> {
    anyhow::ensure!(
        axes.len() == backends.len(),
        "{} frozen axes for {} backends",
        axes.len(),
        backends.len()
    );
    anyhow::ensure!(!axes.is_empty(), "no feature axis to project on");
    let n_cells = backends[0].num_columns();
    anyhow::ensure!(
        barcodes.len() == n_cells,
        "{} barcodes for {n_cells} cells",
        barcodes.len()
    );
    for (ax, b) in axes.iter().zip(backends) {
        anyhow::ensure!(
            ax.rows.len() == b.num_rows() && ax.bias.len() == b.num_rows(),
            "axis {}: dictionary has {} rows, backend has {}",
            ax.label,
            ax.rows.len(),
            b.num_rows()
        );
    }
    let flat: Vec<Vec<f32>> = axes
        .iter()
        .map(|ax| ax.rows.iter().flat_map(|r| r.iter().copied()).collect())
        .collect();
    let dicts: Vec<AxisDict> = axes
        .iter()
        .zip(&flat)
        .map(|(ax, f)| AxisDict {
            label: ax.label,
            feat: f,
            b_feat: ax.bias,
        })
        .collect();
    let projector = AxesProjector::new(&dicts, dim, f64::from(PROJECTION_RIDGE_SGD), device)?;
    info!(
        "Cell embed: {n_cells} cells on {} axes [{}], groups of {}",
        axes.len(),
        axes.iter().map(|a| a.label).collect::<Vec<_>>().join(","),
        projector.group_cells()
    );
    let groups = stream_cell_groups(backends, projector.group_cells())?;
    let out = project_cells_axes(&projector, n_cells, groups)?;
    let theta = Mat::from_row_slice(n_cells, dim, &out.theta);
    Ok(CellEmbedOut { theta, barcodes })
}

/// `{prefix}.cell_embedding.parquet`: rows named by barcode.
pub fn write_cell_parquet(prefix: &str, out: &CellEmbedOut) -> anyhow::Result<()> {
    let path = format!("{prefix}.cell_embedding.parquet");
    save_embedding(
        &path,
        &out.theta.to_tensor(&Device::Cpu)?,
        &out.barcodes,
        "cell",
    )?;
    info!(
        "Wrote {path} ({} × {})",
        out.theta.nrows(),
        out.theta.ncols()
    );
    Ok(())
}

/// Each pb's label is the majority label of its cells (ties: lowest id);
/// `None` for a pb with no labeled cell.
pub fn cluster_labels_to_pb(
    cell_label: &[Option<usize>],
    cell_to_pb: &[usize],
    n_pb: usize,
) -> Vec<Option<usize>> {
    let n_clusters = cell_label
        .iter()
        .flatten()
        .copied()
        .max()
        .map_or(0, |m| m + 1);
    let mut votes = vec![vec![0usize; n_clusters]; n_pb];
    for (&label, &pb) in cell_label.iter().zip(cell_to_pb) {
        if let Some(l) = label {
            votes[pb][l] += 1;
        }
    }
    votes
        .iter()
        .map(|v| {
            let best = v
                .iter()
                .enumerate()
                .max_by_key(|&(i, &n)| (n, std::cmp::Reverse(i)));
            match best {
                Some((i, &n)) if n > 0 => Some(i),
                _ => None,
            }
        })
        .collect()
}
