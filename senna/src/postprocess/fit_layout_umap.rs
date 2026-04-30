//! `senna layout umap` — UMAP-style SGD on the PB-PB fuzzy kNN graph.
//! Same prep/finalize pipeline as `layout tsne`; differs only in the
//! 2D layout algorithm. Optionally runs a cell-level SGD pass after
//! Nyström to resolve intra-PB structure (`--umap-finetune-epochs > 0`).

use super::fit_layout_common::{
    nystrom_cell_coords, preprocess_layout_data, random_init_2d, resolve_inputs, write_viz_outputs,
    LayoutCommonArgs,
};
use crate::embed_common::*;
use crate::geometry::umap::Umap;
use std::collections::HashMap;

#[derive(Args, Debug)]
pub struct LayoutUmapArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,

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

    #[arg(
        long,
        default_value_t = 100,
        help = "Cell-level UMAP fine-tune epochs after Nyström (0 = disabled)",
        long_help = "After the PB-level UMAP and Nyström cell placement, run \
                     additional UMAP SGD on a cell-cell fuzzy kNN graph in \
                     latent space, warm-started from Nyström. Resolves \
                     intra-PB structure the Nyström interpolation collapses. \
                     Cost scales ~linearly in #cells; 50–100 epochs is usually \
                     enough since the init is already close."
    )]
    umap_finetune_epochs: usize,

    #[arg(
        long,
        default_value_t = 15,
        help = "kNN for cell-cell graph during fine-tune"
    )]
    umap_finetune_knn: usize,
}

pub fn fit_layout_umap(args: &LayoutUmapArgs) -> anyhow::Result<()> {
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

    let mut cell_coords = nystrom_cell_coords(&args.common, &prep, &pb_coords);
    if args.umap_finetune_epochs > 0 {
        finetune_cells_umap(
            &mut cell_coords,
            &prep.cell_proj_kn,
            args.umap_finetune_knn,
            args.umap_finetune_epochs,
            args.umap_negative_rate,
            args.umap_lr,
            args.common.seed,
            args.common.block_size.unwrap_or(1000),
        )?;
    }

    write_viz_outputs(&args.common, &mut resolved, &prep, &pb_coords, &cell_coords)
}

/// Cell-level UMAP fine-tune. Builds a cell-cell fuzzy kNN graph on the
/// same latent features used for Nyström, warm-starts from `coords`,
/// runs `Umap::fit`, and writes back.
#[allow(clippy::too_many_arguments)]
fn finetune_cells_umap(
    coords: &mut Mat,
    feat_kn: &Mat,
    knn: usize,
    epochs: usize,
    negative_rate: usize,
    learning_rate: f32,
    seed: u64,
    block_size: usize,
) -> anyhow::Result<()> {
    let n = coords.nrows();
    assert_eq!(feat_kn.ncols(), n, "feat_kn cols must match cell count");

    info!("Cell fine-tune: building fuzzy kNN (n={n}, knn={knn}) ...");
    let graph = matrix_util::knn_graph::KnnGraph::from_columns(
        feat_kn,
        matrix_util::knn_graph::KnnGraphArgs {
            knn,
            block_size,
            reciprocal: false,
        },
    )?;
    let fuzzy = graph.fuzzy_kernel_weights();

    // The graph may emit both (i,j) and (j,i); collapse to undirected
    // edges keyed by (min,max) with the max weight (fuzzy-union semantics).
    let mut edge_map: HashMap<(usize, usize), f32> = HashMap::with_capacity(graph.edges.len());
    for (&(i, j), &w) in graph.edges.iter().zip(fuzzy.iter()) {
        if i == j || w <= 0.0 {
            continue;
        }
        let key = if i < j { (i, j) } else { (j, i) };
        let slot = edge_map.entry(key).or_insert(0.0);
        if w > *slot {
            *slot = w;
        }
    }
    let edges: Vec<(usize, usize, f32)> =
        edge_map.into_iter().map(|((i, j), w)| (i, j, w)).collect();
    info!(
        "Cell fine-tune: {} cells, {} undirected edges (mean deg = {:.1})",
        n,
        edges.len(),
        2.0 * edges.len() as f32 / n.max(1) as f32
    );

    let mut init_flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        init_flat.push(coords[(i, 0)]);
        init_flat.push(coords[(i, 1)]);
    }

    let umap = Umap {
        n_epochs: epochs,
        negative_sample_rate: negative_rate,
        learning_rate,
        seed: seed.wrapping_add(1),
    };
    info!("Cell fine-tune: UMAP SGD (epochs={epochs}) ...");
    let refined = umap.fit(&edges, n, &init_flat);

    for i in 0..n {
        coords[(i, 0)] = refined[i * 2];
        coords[(i, 1)] = refined[i * 2 + 1];
    }
    info!("Cell fine-tune done");
    Ok(())
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
