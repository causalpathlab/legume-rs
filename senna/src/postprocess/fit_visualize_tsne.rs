//! `senna layout tsne` — raw-gene-space PB landmarks laid out with
//! an Rtsne-style t-SNE (PHATE-initialized) over the PB-PB cosine
//! similarity. Cells placed by cheap Nyström in proj space.

use super::fit_visualize_common::{
    finalize_viz, prepare_viz, resolve_inputs, PhateCliArgs, VisualizeCommonArgs,
};
use super::viz_prep::apply_svd_preprocessing;
use crate::embed_common::*;
use crate::geometry::phate::phate_layout_2d;
use crate::geometry::tsne::{similarity_to_distance, TSne};

#[derive(Args, Debug)]
pub struct VisualizeTsneArgs {
    #[clap(flatten)]
    common: VisualizeCommonArgs,

    #[arg(long, default_value_t = 30.0, help = "Perplexity for t-SNE")]
    perplexity: f32,

    #[arg(long, default_value_t = 1000, help = "Number of iterations for t-SNE")]
    tsne_iter: usize,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "densMAP density-preservation strength (λ)"
    )]
    tsne_density_lambda: f32,

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

    #[clap(flatten)]
    phate: PhateCliArgs,
}

pub fn fit_visualize_tsne(args: &VisualizeTsneArgs) -> anyhow::Result<()> {
    let mut resolved = resolve_inputs(&args.common)?;
    let prep = prepare_viz(&args.common, &resolved)?;

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

    info!("Running PHATE to initialize t-SNE ...");
    let init = phate_layout_2d(&features, &(&args.phate).into());
    info!("PHATE init done");

    let n = prep.pb_similarity.nrows();

    // nalgebra's DMatrix is column-major; TSne::fit expects row-major flat.
    let mut init_flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        init_flat.push(init[(i, 0)]);
        init_flat.push(init[(i, 1)]);
    }

    let mut sim_flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            sim_flat.push(prep.pb_similarity[(i, j)]);
        }
    }
    let distances = similarity_to_distance(&sim_flat, n);

    let tsne = TSne::default()
        .perplexity(args.perplexity)
        .n_iter(args.tsne_iter)
        .weights(&prep.pb_size)
        .density_lambda(args.tsne_density_lambda);
    info!(
        "Running t-SNE (perplexity={}, iter={}) on {} PBs ...",
        args.perplexity, args.tsne_iter, n
    );
    let result = tsne
        .fit(&distances, n, Some(&init_flat))
        .map_err(|e| anyhow::anyhow!("t-SNE failed: {e}"))?;

    let mut pb_coords = Mat::zeros(n, 2);
    for i in 0..n {
        pb_coords[(i, 0)] = result[i * 2];
        pb_coords[(i, 1)] = result[i * 2 + 1];
    }
    info!("t-SNE done");

    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords)
}
