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

    let collapse_out = collapse_columns_multilevel_impl(
        data_vec,
        &proj_kn,
        batch_membership,
        Some(args.knn_cells),
        None, // default num_levels
        Some(args.sort_dim),
        None, // default opt_iter
    )?;

    Ok(collapse_out.delta)
}
