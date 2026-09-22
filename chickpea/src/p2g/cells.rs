//! Phase 2: every cell onto the frozen gene / peak dictionaries, one Poisson
//! partition and one intercept per axis, one shared latent. Cells stream from
//! the per-modality backends in groups; nothing dense over cells × features.

use crate::common::Mat;
use data_beans::sparse_io_vector::SparseIoVec;
use graph_embedding_util::fit::projection::{
    project_cells_axes, read_cell_group, stream_cell_groups, AxesProjector, AxisDict, CellGroup,
    WarmStart,
};
use graph_embedding_util::fit::{keep_cells_per_pb, majority_batch_per_pb, PROJECTION_RIDGE_SGD};
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

/// Where every cell starts: the phase-1 row of its finest pseudobulk.
pub struct PbWarmStart<'a> {
    /// `[n_pb][H]` as the hier fit leaves them (not L2-normalised).
    pub rows: &'a [Vec<f32>],
    pub cell_to_pb: &'a [usize],
}

/// Project every column of `backends` (one per axis, in `axes` order) into
/// `[n_cells × H]`, gauge-centred, each cell solved from its pseudobulk's row.
/// The per-axis intercepts are solved but not kept: nothing downstream reads
/// them.
pub fn embed_cells(
    axes: &[FrozenAxis],
    backends: &[&SparseIoVec],
    warm: &PbWarmStart,
    dim: usize,
    device: &Device,
) -> anyhow::Result<Mat> {
    anyhow::ensure!(
        axes.len() == backends.len(),
        "{} frozen axes for {} backends",
        axes.len(),
        backends.len()
    );
    anyhow::ensure!(!axes.is_empty(), "no feature axis to project on");
    let n_cells = backends[0].num_columns();
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
    let pb_rows: Vec<f32> = warm.rows.iter().flat_map(|r| r.iter().copied()).collect();
    let start = WarmStart {
        rows: &pb_rows,
        cell_to_row: warm.cell_to_pb,
    };
    let out = project_cells_axes(&projector, n_cells, groups, Some(&start))?;
    Ok(Mat::from_row_slice(n_cells, dim, &out.theta))
}

/// The cells phase 1 trains on next to the pseudobulks, as one group over
/// the gene and peak axes: at most `k` cells per pseudobulk at every level
/// (finest membership `cell_to_pb`, coarser ones through `parent`), unioned,
/// read from `backends` (one per axis in axis order; an ATAC-only run gives
/// the peak backend alone and every cell's gene row is empty). `k = 0` gives
/// no cell units.
pub fn phase1_cell_units(
    backends: &[&SparseIoVec],
    cell_to_pb: &[usize],
    parent: &[Vec<usize>],
    k: usize,
    seed: u64,
) -> anyhow::Result<CellGroup> {
    anyhow::ensure!(
        (1..=2).contains(&backends.len()),
        "{} backends for the gene and peak axes",
        backends.len()
    );
    let mut memberships: Vec<Vec<usize>> = vec![cell_to_pb.to_vec()];
    for map in parent {
        let prev = memberships.last().expect("finest membership");
        memberships.push(prev.iter().map(|&p| map[p]).collect());
    }
    let keep = keep_cells_per_pb(&memberships, k, seed);
    let cells: Vec<usize> = (0..cell_to_pb.len()).filter(|&c| keep[c]).collect();
    let mut group = read_cell_group(backends, &cells)?;
    if backends.len() == 1 {
        group
            .axes
            .insert(0, vec![(Vec::new(), Vec::new()); cells.len()]);
    }
    Ok(group)
}

/// `{prefix}.cell_embedding.parquet`: one row per barcode.
pub fn write_cell_parquet(prefix: &str, theta: &Mat, barcodes: &[Box<str>]) -> anyhow::Result<()> {
    anyhow::ensure!(
        barcodes.len() == theta.nrows(),
        "{} barcodes for {} embedded cells",
        barcodes.len(),
        theta.nrows()
    );
    let path = format!("{prefix}.cell_embedding.parquet");
    save_embedding(&path, &theta.to_tensor(&Device::Cpu)?, barcodes, "cell")?;
    info!("Wrote {path} ({} × {})", theta.nrows(), theta.ncols());
    Ok(())
}

/// Each pb's label is the majority label of its labeled cells (ties: lowest
/// id); `None` for a pb with no labeled cell. Unlabeled cells do not vote.
pub fn cluster_labels_to_pb(
    cell_label: &[Option<usize>],
    cell_to_pb: &[usize],
    n_pb: usize,
) -> Vec<Option<usize>> {
    let (voters, labels): (Vec<usize>, Vec<u32>) = cell_label
        .iter()
        .zip(cell_to_pb)
        .filter_map(|(l, &pb)| l.map(|l| (pb, l as u32)))
        .unzip();
    majority_batch_per_pb(&voters, &labels, n_pb)
        .into_iter()
        .map(|m| (m != u32::MAX).then_some(m as usize))
        .collect()
}
