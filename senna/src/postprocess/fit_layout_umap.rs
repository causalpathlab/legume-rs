//! `senna layout umap` — UMAP-style SGD on the PB-PB fuzzy kNN graph.
//! Same prep/finalize pipeline as `layout tsne`; differs only in the
//! 2D layout algorithm. Optionally runs a cell-level SGD pass after
//! Nyström to resolve intra-PB structure (`--umap-finetune-epochs > 0`).
//!
//! For `RunKind::Bge` / `RunKind::Fne` manifests, `preprocess_layout_data`
//! returns `LayoutMode::DirectCells` and we skip the PB pass entirely —
//! build a cell-cell fuzzy kNN graph straight from the L2-normalized
//! latent and run UMAP SGD on cells directly. No Nyström, no PB output.

use super::fit_layout_common::{
    nystrom_cell_coords, preprocess_layout_data, random_init_2d, resolve_inputs,
    write_viz_outputs_direct, write_viz_outputs_pb, DirectLayoutPrep, LayoutCommonArgs, LayoutPrep,
    PbLayoutPrep, ResolvedViz,
};
use crate::embed_common::*;
use matrix_util::umap::Umap;
use rayon::prelude::*;

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
        long_help = "After the PB-level UMAP and Nyström cell placement, run\n\
                     additional UMAP SGD on a cell-cell fuzzy kNN graph in\n\
                     latent space, warm-started from Nyström. Resolves\n\
                     intra-PB structure the Nyström interpolation collapses.\n\
                     Cost scales ~linearly in #cells; 50–100 epochs is usually\n\
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

impl Default for LayoutUmapArgs {
    fn default() -> Self {
        Self {
            common: LayoutCommonArgs::default(),
            umap_epochs: 500,
            umap_negative_rate: 20,
            umap_lr: 1.0,
            umap_finetune_epochs: 100,
            umap_finetune_knn: 15,
        }
    }
}

/// Run `senna layout umap` against an existing manifest with all
/// defaults — exactly equivalent to `senna layout umap --from PATH`
/// from the CLI. Used as the auto-layout for `senna plot --from PATH`
/// when no `layout.cell_coords` is populated yet. Mirrors
/// [`crate::postprocess::run_default_phate_layout`] in scope and
/// behavior; differs only in the layout algorithm.
pub fn run_default_umap_layout(manifest_path: &str, preload: bool) -> anyhow::Result<()> {
    use crate::run_manifest::RunManifest;
    use std::path::Path;

    let mut args = LayoutUmapArgs::default();
    args.common.from = Some(Box::from(manifest_path));
    args.common.preload_data = preload;

    let (manifest, manifest_dir) = RunManifest::load(Path::new(manifest_path))?;
    let prefix_base = Path::new(&manifest.prefix)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| manifest.prefix.clone());
    let abs_out = manifest_dir.join(prefix_base);
    args.common.out = Some(abs_out.to_string_lossy().into_owned().into_boxed_str());

    fit_layout_umap(&args)
}

/// UMAP reference inits in [-10, 10] so gradient clamps at ±4 don't
/// dominate the layout scale. `random_init_2d` returns [-1, 1].
const INIT_SCALE: f32 = 10.0;

pub fn fit_layout_umap(args: &LayoutUmapArgs) -> anyhow::Result<()> {
    let mut resolved = resolve_inputs(&args.common)?;
    let prep = preprocess_layout_data(&args.common, &resolved, /*allow_direct_cells=*/ true)?;

    match &prep {
        LayoutPrep::PbThenNystrom(p) => fit_layout_umap_pb(args, &mut resolved, p),
        LayoutPrep::DirectCells(p) => fit_layout_umap_direct(args, &mut resolved, p),
    }
}

fn fit_layout_umap_pb(
    args: &LayoutUmapArgs,
    resolved: &mut ResolvedViz,
    prep: &PbLayoutPrep,
) -> anyhow::Result<()> {
    let n = prep.pb_similarity.nrows();
    let edges = extract_upper_edges(&prep.pb_similarity);
    info!(
        "UMAP: {} PBs, {} edges (mean deg = {:.1})",
        n,
        edges.len(),
        2.0 * edges.len() as f32 / n.max(1) as f32
    );

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

    let mut cell_coords = nystrom_cell_coords(&args.common, prep, &pb_coords);
    if args.umap_finetune_epochs > 0 {
        let edges = build_cell_cell_fuzzy_edges(
            &prep.cell_proj_kn,
            args.umap_finetune_knn,
            args.common.block_size.unwrap_or(1000),
        )?;
        run_cell_level_umap_in_place(
            &mut cell_coords,
            &edges,
            args.umap_finetune_epochs,
            args.umap_negative_rate,
            args.umap_lr,
            args.common.seed.wrapping_add(1),
        );
    }

    write_viz_outputs_pb(&args.common, resolved, prep, &pb_coords, &cell_coords)
}

/// Direct cell-level UMAP for `RunKind::Bge` / `RunKind::Fne` (and any
/// future kind that returns `LayoutPrep::DirectCells`). Skips landmark
/// sampling, fuzzy kNN on PB centroids, and Nyström — the graph-trained
/// latent is already manifold-aware, so we put cells straight on the
/// cell-cell fuzzy kNN graph and run UMAP SGD there. Output: only
/// `cell_coords.parquet`, no `pb_coords`.
fn fit_layout_umap_direct(
    args: &LayoutUmapArgs,
    resolved: &mut ResolvedViz,
    prep: &DirectLayoutPrep,
) -> anyhow::Result<()> {
    let n = prep.cell_proj_kn.ncols();
    info!("UMAP direct cell-level mode: {n} cells");

    let edges = build_cell_cell_fuzzy_edges(
        &prep.cell_proj_kn,
        args.umap_finetune_knn,
        args.common.block_size.unwrap_or(1000),
    )?;

    // RNG-determined init in [-INIT_SCALE, INIT_SCALE]² — keep sequential
    // so the seed contract matches `random_init_2d`'s elsewhere.
    let mut cell_coords = random_init_2d(n, args.common.seed);
    cell_coords *= INIT_SCALE;

    run_cell_level_umap_in_place(
        &mut cell_coords,
        &edges,
        args.umap_epochs,
        args.umap_negative_rate,
        args.umap_lr,
        args.common.seed,
    );

    write_viz_outputs_direct(&args.common, resolved, prep, &cell_coords)
}

/// Build the undirected cell-cell fuzzy kNN edge list used by both the
/// `PbThenNystrom` fine-tune pass and the `DirectCells` cell-level UMAP.
/// `KnnGraph::edges` is already canonical (`i < j`, sorted, deduped)
/// and `fuzzy_kernel_weights` returns one weight per edge.
fn build_cell_cell_fuzzy_edges(
    feat_kn: &Mat,
    knn: usize,
    block_size: usize,
) -> anyhow::Result<Vec<(usize, usize, f32)>> {
    let n = feat_kn.ncols();
    info!("Cell-cell fuzzy kNN (n={n}, knn={knn}) ...");
    let graph = matrix_util::knn_graph::KnnGraph::from_columns(
        feat_kn,
        matrix_util::knn_graph::KnnGraphArgs {
            knn,
            block_size,
            reciprocal: false,
        },
    )?;
    let fuzzy = graph.fuzzy_kernel_weights();

    let edges: Vec<(usize, usize, f32)> = graph
        .edges
        .par_iter()
        .zip(fuzzy.par_iter())
        .filter_map(|(&(i, j), &w)| (w > 0.0).then_some((i, j, w)))
        .collect();
    info!(
        "Cell-cell fuzzy kNN: {} cells, {} edges (mean deg = {:.1})",
        n,
        edges.len(),
        2.0 * edges.len() as f32 / n.max(1) as f32
    );
    Ok(edges)
}

/// Run cell-level UMAP SGD, mutating `coords` in place. Common back-end
/// for both the PB-mode fine-tune pass and (via the shared edge
/// builder) the DirectCells path.
fn run_cell_level_umap_in_place(
    coords: &mut Mat,
    edges: &[(usize, usize, f32)],
    epochs: usize,
    negative_rate: usize,
    learning_rate: f32,
    seed: u64,
) {
    let n = coords.nrows();
    let init_flat: Vec<f32> = (0..n)
        .flat_map(|i| [coords[(i, 0)], coords[(i, 1)]])
        .collect();

    let umap = Umap {
        n_epochs: epochs,
        negative_sample_rate: negative_rate,
        learning_rate,
        seed,
    };
    info!("Cell-level UMAP SGD (epochs={epochs}) ...");
    let refined = umap.fit(edges, n, &init_flat);

    for i in 0..n {
        coords[(i, 0)] = refined[i * 2];
        coords[(i, 1)] = refined[i * 2 + 1];
    }
    info!("Cell-level UMAP done");
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
