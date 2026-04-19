//! `senna layout phate` — raw-gene-space PB landmarks laid out
//! with PHATE diffusion embedding over the log1p-CPM PB features.
//! Cells placed by cheap Nyström in proj space.

use super::fit_visualize_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, PhateCliArgs, VisualizeCommonArgs,
};
use super::viz_prep::apply_svd_preprocessing;
use crate::embed_common::*;
use crate::geometry::phate::phate_layout_2d;

#[derive(Args, Debug)]
pub struct VisualizePhateArgs {
    #[clap(flatten)]
    common: VisualizeCommonArgs,

    #[clap(flatten)]
    phate: PhateCliArgs,

    #[arg(
        long,
        default_value_t = 0,
        help = "SVD preprocessing: keep top N components (0 = skip, use raw features)"
    )]
    svd_dims: usize,

    #[arg(
        long,
        default_value_t = 2000,
        help = "Number of highly variable genes to select (0 = use all genes)"
    )]
    n_hvg: usize,
}

pub fn fit_visualize_phate(args: &VisualizePhateArgs) -> anyhow::Result<()> {
    let mut resolved = resolve_inputs(&args.common)?;
    let prep = preprocess_layout_data(&args.common, &resolved)?;

    // HVG selection: keep top N genes by residual variance (mean-variance corrected)
    let features = if args.n_hvg > 0 && args.n_hvg < prep.log_expr_pb.ncols() {
        info!(
            "Selecting top {} HVGs from {} genes (mean-variance corrected)",
            args.n_hvg,
            prep.log_expr_pb.ncols()
        );
        data_beans_alg::hvg_selection::select_hvg(&prep.log_expr_pb, args.n_hvg, None)
    } else {
        prep.log_expr_pb.clone()
    };

    let features = if args.svd_dims > 0 {
        apply_svd_preprocessing(&features, args.svd_dims)?
    } else {
        features
    };

    let pb_coords = phate_layout_2d(&features, &(&args.phate).into());
    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords)
}
