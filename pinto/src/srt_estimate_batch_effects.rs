use crate::srt_common::*;
use data_beans_alg::collapse_data::*;
use data_beans_alg::random_projection::*;
use matrix_param::dmatrix_gamma::GammaMatrix;

pub struct EstimateBatchArgs {
    pub proj_dim: usize,
    pub sort_dim: usize,
    pub block_size: usize,
    pub knn_cells: usize,
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

    let collapse_out = data_vec.collapse_columns_multilevel(
        &proj_kn,
        batch_membership,
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            sort_dim: args.sort_dim,
            ..MultilevelParams::new(proj_kn.nrows())
        },
    )?;

    Ok(collapse_out.delta)
}
