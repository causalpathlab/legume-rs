//! Shared SRT preprocessing pipeline.
//!
//! `lc`, `svd`, and `cage` all share the same opening sequence:
//! load → invariants → spatial-or-expression KNN → optional auto-batch →
//! optional batch effects → optional NB Fisher gene weights.
//! `preprocess_srt` extracts that once.
//!
//! `SrtCellPairs<'a>` borrows the `SparseIoVec` and `Mat`, so this bundle
//! returns the *owned* data + graph and the caller builds
//! `SrtCellPairs::with_graph(&pre.data_vec, &pre.coordinates, pre.graph)`.

use crate::util::batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
use crate::util::cell_pairs::{build_expression_graph, build_spatial_graph, SrtCellPairsArgs};
use crate::util::common::*;
use crate::util::input::{
    auto_batch_from_components, read_data_with_coordinates, read_data_without_coordinates, SRTData,
    SrtInputArgs,
};
use crate::util::knn_graph::KnnGraph;
use auxiliary_data::feature_names::FeatureNameKind;
use data_beans_alg::random_projection::RandProjOps;

////////////////////////////////////////////////////////////////
//                                                            //
// Config + result types                                      //
//                                                            //
////////////////////////////////////////////////////////////////

pub struct SrtPreprocessConfig<'a> {
    pub common: &'a SrtInputArgs,
    /// Compute per-gene NB Fisher-info weights. `lc` needs them; `svd`
    /// and (v1) `cage` do not.
    pub fisher_weights: bool,
    /// Estimate per-batch effects. All current subcommands set this.
    pub batch_effects: bool,
    /// Row-name canonicalization strategy. `None` falls back to
    /// `FeatureNameKind::Exact` (strict equality), matching the
    /// historical behaviour of `lc` / `svd`. `cage` passes
    /// `FeatureNameKind::Gene` or an `auto_detect`'d kind so gene
    /// symbols register as aliases of `ENSG..._SYMBOL` row names.
    pub feature_kind: Option<FeatureNameKind>,
}

pub struct SrtPreprocessed {
    pub data_vec: SparseIoVec,
    pub coordinates: Mat,
    pub coordinate_names: Vec<Box<str>>,
    pub batch_membership: Vec<Box<str>>,
    /// Posterior batch-effect mean `[n_genes × n_batches]`. `None`
    /// when single-batch.
    pub batch_effects: Option<Mat>,
    pub graph: KnnGraph,
    pub gene_weights: Option<Vec<f32>>,
    pub n_cells: usize,
    pub n_genes: usize,
}

////////////////////////////////////////////////////////////////
//                                                            //
// preprocess_srt                                             //
//                                                            //
////////////////////////////////////////////////////////////////

/// Run the shared SRT preamble: load, build the cell-cell KNN graph,
/// optionally auto-detect batches from disconnected components, estimate
/// per-batch effects, and (optionally) compute NB Fisher gene weights.
///
/// In expression mode (no `--coord`), the cell embedding used to build
/// the graph is the random-projected count matrix with no batch
/// correction (batch effects haven't been estimated yet at this point).
pub fn preprocess_srt(cfg: SrtPreprocessConfig<'_>) -> anyhow::Result<SrtPreprocessed> {
    let c = cfg.common;

    info!("Loading data files...");
    let has_coords = c.has_coordinates();

    let kind = cfg.feature_kind.unwrap_or(FeatureNameKind::Exact);
    let SRTData {
        data: mut data_vec,
        mut coordinates,
        mut coordinate_names,
        batches: mut batch_membership,
    } = if has_coords {
        read_data_with_coordinates(c.to_read_args_with_kind(kind.clone()))?
    } else {
        info!("No coordinate files provided — using expression mode");
        read_data_without_coordinates(c.to_read_args_with_kind(kind))?
    };

    // Optional shared cell QC — applied before the KNN graph is built so
    // dropped cells never become graph nodes. MAD outliers are dropped from
    // the working set; coordinates + batch labels are filtered in lockstep.
    // (Near-empty cells are kept; the gem-style near-empty output floor is
    // not applied to pinto's per-cell outputs.)
    if let Some(qc_cfg) = c.qc.to_config() {
        let report = data_beans::qc_lib::compute_qc(&data_vec, &qc_cfg, c.block_size)?;
        let n_near_empty = report.near_empty.iter().filter(|&&e| e).count();
        info!(
            "QC: dropped {}/{} cells from the spatial graph ({} near-empty kept)",
            report.n_cells_dropped,
            report.train_keep.len(),
            n_near_empty,
        );
        if report.n_cells_dropped > 0 {
            let kept: Vec<usize> = report
                .train_keep
                .iter()
                .enumerate()
                .filter(|&(_, &k)| k)
                .map(|(i, _)| i)
                .collect();
            // Spatial mode: coordinates [n_cells × n_dims] are consumed by
            // the graph build below, so they must be filtered in lockstep
            // (error loudly on any size mismatch rather than silently skip).
            // Expression mode overwrites `coordinates` with the embedding of
            // the already-masked data_vec, so no filtering is needed there.
            if has_coords {
                anyhow::ensure!(
                    coordinates.nrows() == report.train_keep.len(),
                    "QC: coordinate rows {} != cell count {}",
                    coordinates.nrows(),
                    report.train_keep.len()
                );
                coordinates = coordinates.select_rows(kept.iter());
            }
            data_vec.mask_columns(&report.train_keep)?;
            batch_membership =
                data_beans::qc_lib::filter_by_keep(&batch_membership, &report.train_keep);
        }
    }

    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    anyhow::ensure!(c.proj_dim > 0, "proj_dim must be > 0");
    anyhow::ensure!(c.knn_spatial > 0, "knn_spatial must be > 0");
    anyhow::ensure!(
        c.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({})",
        c.knn_spatial,
        n_cells
    );

    let graph = if has_coords {
        info!("Building spatial KNN graph (k={})...", c.knn_spatial);
        build_spatial_graph(
            &coordinates,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?
    } else {
        info!(
            "Building expression KNN graph (k={}, proj_dim={})...",
            c.knn_spatial, c.proj_dim
        );
        let cell_proj_pre = data_vec.project_columns_with_batch_correction(
            c.proj_dim,
            c.block_size,
            None::<&[Box<str>]>,
        )?;
        let (g, embedding) = build_expression_graph(
            &cell_proj_pre.proj,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?;
        coordinates = embedding;
        coordinate_names = vec!["pc_1".into(), "pc_2".into()];
        g
    };

    if c.auto_batch && c.batch_files.is_none() {
        auto_batch_from_components(&graph, &mut batch_membership);
    }

    let batch_effects = if cfg.batch_effects {
        let batch_sort_dim = c.proj_dim.min(10);
        estimate_and_write_batch_effects(
            &mut data_vec,
            &batch_membership,
            EstimateBatchArgs {
                proj_dim: c.proj_dim,
                sort_dim: batch_sort_dim,
                block_size: c.block_size,
                batch_knn: c.batch_knn,
                num_levels: c.num_levels,
            },
            &c.out,
        )?
    } else {
        None
    };

    let gene_weights = if cfg.fisher_weights {
        info!("Computing NB Fisher-info gene weights for inference...");
        let w = compute_nb_fisher_weights(&data_vec, c.block_size)?;
        info!(
            "Gene weights w_g: min={:.3e}, mean={:.3e}, max={:.3e}",
            w.iter().cloned().fold(f32::INFINITY, f32::min),
            w.iter().sum::<f32>() / (w.len().max(1) as f32),
            w.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
        Some(w)
    } else {
        None
    };

    Ok(SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        batch_effects,
        graph,
        gene_weights,
        n_cells,
        n_genes,
    })
}
