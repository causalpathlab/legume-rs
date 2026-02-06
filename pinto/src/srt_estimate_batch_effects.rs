use crate::srt_common::*;
use data_beans_alg::collapse_data::*;
use data_beans_alg::random_projection::*;
use matrix_param::dmatrix_gamma::GammaMatrix;

pub struct EstimateBatchArgs {
    pub proj_dim: usize,
    pub sort_dim: usize,
    pub block_size: usize,
    pub knn_batches: usize,
    pub knn_cells: usize,
    pub down_sample: Option<usize>,
}

pub fn estimate_batch(
    data_vec: &mut SparseIoVec,
    batch_membership: &[Box<str>],
    args: EstimateBatchArgs,
) -> anyhow::Result<Option<GammaMatrix>> {
    let batch_hash: HashSet<Box<str>> = batch_membership.iter().cloned().collect();
    let nbatch = batch_hash.len();
    if nbatch < 2 {
        return Ok(None);
    }

    let proj_out = data_vec.project_columns_with_batch_correction(
        args.proj_dim,
        Some(args.block_size),
        Some(batch_membership),
    )?;

    let proj_kn = proj_out.proj;
    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    let nsamp =
        data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), args.down_sample)?;

    info!("Assigning to random {} samples", nsamp);

    data_vec.build_hnsw_per_batch(&proj_kn, batch_membership)?;

    let collapse_out =
        data_vec.collapse_columns(Some(args.knn_batches), Some(args.knn_cells), None, None)?;

    Ok(collapse_out.delta)
}
