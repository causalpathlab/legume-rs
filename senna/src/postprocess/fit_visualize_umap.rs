//! `senna layout umap` — UMAP-style SGD on the PB-PB fuzzy kNN graph.
//! Same prep/finalize pipeline as `layout tsne`; differs only in the
//! 2D layout algorithm.

use super::fit_visualize_common::{
    finalize_viz, preprocess_layout_data, random_init_2d, resolve_inputs, VisualizeCommonArgs,
};
use crate::embed_common::*;
use crate::geometry::umap::Umap;

#[derive(Args, Debug)]
pub struct VisualizeUmapArgs {
    #[clap(flatten)]
    common: VisualizeCommonArgs,

    #[arg(long, default_value_t = 500, help = "Number of SGD epochs")]
    umap_epochs: usize,

    #[arg(
        long,
        default_value_t = 20,
        help = "Negative samples per attractive step"
    )]
    umap_negative_rate: usize,

    #[arg(long, default_value_t = 1.0, help = "Initial learning rate")]
    umap_lr: f32,
}

pub fn fit_visualize_umap(args: &VisualizeUmapArgs) -> anyhow::Result<()> {
    let mut resolved = resolve_inputs(&args.common)?;
    let prep = preprocess_layout_data(&args.common, &resolved)?;

    let n = prep.pb_similarity.nrows();
    let edges = extract_upper_edges(&prep.pb_similarity);
    info!(
        "UMAP: {} PBs, {} edges (mean deg = {:.1})",
        n,
        edges.len(),
        2.0 * edges.len() as f32 / n.max(1) as f32
    );

    // UMAP reference inits in [-10, 10] so gradient clamps at ±4 don't
    // dominate the layout scale. random_init_2d returns [-1, 1].
    const INIT_SCALE: f32 = 10.0;
    let init = random_init_2d(n, args.common.seed);
    let mut init_flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        init_flat.push(init[(i, 0)] * INIT_SCALE);
        init_flat.push(init[(i, 1)] * INIT_SCALE);
    }

    let umap = Umap {
        n_epochs: args.umap_epochs,
        negative_sample_rate: args.umap_negative_rate,
        learning_rate: args.umap_lr,
        seed: args.common.seed,
    };
    info!(
        "Running UMAP SGD (epochs={}, neg={}) ...",
        args.umap_epochs, args.umap_negative_rate
    );
    let result = umap.fit(&edges, n, &init_flat);

    let mut pb_coords = Mat::zeros(n, 2);
    for i in 0..n {
        pb_coords[(i, 0)] = result[i * 2];
        pb_coords[(i, 1)] = result[i * 2 + 1];
    }
    info!("UMAP done");

    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords)
}

/// Collect upper-triangle non-zero entries as `(i, j, w)` with `i < j`.
/// Self-loop reg added by `regularize_similarity` is ignored (i == j).
fn extract_upper_edges(sim: &Mat) -> Vec<(usize, usize, f32)> {
    let n = sim.nrows();
    let mut edges = Vec::with_capacity(n * 15);
    for i in 0..n {
        for j in (i + 1)..n {
            let w = sim[(i, j)];
            if w > 0.0 {
                edges.push((i, j, w));
            }
        }
    }
    edges
}
