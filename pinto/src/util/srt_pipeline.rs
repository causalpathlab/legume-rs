//! Shared SRT preprocessing pipeline.
//!
//! `lc`, `svd`, and `cage` all share the same opening sequence:
//! load → invariants → spatial-or-expression KNN → optional auto-batch →
//! optional batch effects → optional NB Fisher gene weights.
//! `preprocess_srt` extracts that once.
//!
//! `SrtCellPairs<'a>` borrows the `SparseIoVec` and `Mat`, so this bundle
//! returns the *owned* data + graph and the caller builds
//! `SrtCellPairs::with_graph(&pre.data_vec, &pre.coordinates, &pre.graph)`.

use crate::util::batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
use crate::util::cell_pairs::{
    build_expression_graph, build_expression_knn, build_spatial_graph, SrtCellPairsArgs,
};
use crate::util::common::*;
use crate::util::gene_axis::GeneAxis;
use crate::util::input::{
    auto_batch_from_components, read_data_with_coordinates, read_data_without_coordinates, SRTData,
    SrtInputArgs,
};
use crate::util::knn_graph::KnnGraph;
use matrix_util::knn_graph::{DistanceMerge, EdgeSource};
use auxiliary_data::feature_names::FeatureNameKind;
use data_beans_alg::gene_weighting::fisher_weights_from_stats;
use data_beans_alg::random_projection::RandProjOps;
use data_beans_alg::sparse_streaming::streaming_sparse_running_stats;

///////////////////////////
// Config + result types //
///////////////////////////

pub struct SrtPreprocessConfig<'a> {
    pub common: &'a SrtInputArgs,
    /// Compute per-gene NB Fisher-info weights. `lc` needs them; `svd`
    /// and (v1) `cage` do not.
    pub fisher_weights: bool,
    /// Estimate per-batch effects. All current subcommands set this.
    pub batch_effects: bool,
    /// Whether, and how strictly, to resolve the GENE unit axis from the row
    /// names and fold the running statistics onto it in the same pass.
    pub gene_axis: GeneAxisMode,
    /// Row-name canonicalization strategy. `None` falls back to
    /// `FeatureNameKind::Exact` (strict equality), matching the
    /// historical behaviour of `lc` / `svd`. `cage` passes
    /// `FeatureNameKind::Gene` or an `auto_detect`'d kind so gene
    /// symbols register as aliases of `ENSG..._SYMBOL` row names.
    pub feature_kind: Option<FeatureNameKind>,
    /// Compute the shared post-batch-correction cell projection and hand it
    /// back on [`SrtPreprocessed::cell_proj`]. `lc` and `cage` both need one
    /// for coarsening, so taking it here spares them a second full pass.
    pub cell_projection: bool,
}

/// How a caller wants the feature axis resolved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GeneAxisMode {
    /// Stay on the matrix rows. `svd` stacks its channels there and never
    /// folds, so a gene axis would buy it nothing.
    Rows,
    /// Resolve a gene axis, and FAIL on a feature axis that only partly parses.
    /// For a consumer that pools a gene's tracks: a row whose track is unknown
    /// has no correct pooled answer.
    Strict,
    /// Resolve a gene axis, falling back to one unit per row when the axis is
    /// mixed. For a consumer that only filters and reports per gene, where one
    /// unit per row is a defined answer and aborting would reject multimodal
    /// matrices that used to work.
    Lenient,
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
    /// The pre-augmentation SPATIAL graph, `Some` only when `--knn-expr`
    /// added expression edges. Prefer [`Self::topology_graph`] over reaching
    /// for this directly.
    pub spatial_graph: Option<KnnGraph>,
    /// Which graph each edge of [`Self::graph`] came from, parallel to
    /// `graph.edges`. `None` when no augmentation ran.
    pub edge_source: Option<Vec<EdgeSource>>,
    /// The projection taken here, with the SAME batch argument `lc` and
    /// `cage` would have used, so they can reuse it instead of paying for a
    /// second full pass. `None` unless the config asked for one.
    pub cell_proj: Option<data_beans_alg::random_projection::RandColProjOut>,
    /// The gene unit axis. `Some` iff the config asked for it. Identity (one
    /// gene per row) unless the rows carry splice channels.
    pub gene_axis: Option<GeneAxis>,
    /// Per-ROW NB Fisher-info weights. This is the right axis for the
    /// projection and for anything that reads the matrix directly.
    pub row_weights: Option<Vec<f32>>,
    /// The ROW-axis statistics [`Self::row_weights`] came from. Its `sum()` is
    /// the per-row count total, which spares a caller that needs one a second
    /// full pass over the data.
    pub row_stats: Option<matrix_util::sparse_stat::SparseRunningStatistics<f32>>,
    /// Per-GENE NB Fisher-info weights, `Some` iff both `fisher_weights` and
    /// `gene_axis` were asked for.
    ///
    /// These are NOT a fold of [`Self::row_weights`]. A Fisher weight is a
    /// function of a feature's abundance and mean, and that function is not
    /// additive, so folding the weights would hand a gene a precision no
    /// measurement supports. The fold happens on the statistics instead, and it
    /// happens inside the streaming pass because `npos` and `s2` do not survive
    /// one applied afterwards.
    pub gene_weights: Option<Vec<f32>>,
    /// The gene-axis statistics [`Self::gene_weights`] came from. Carried out
    /// because the dictionary merge needs per-gene DETECTION counts, which is
    /// exactly the statistic a post-hoc fold gets wrong.
    pub gene_stats: Option<matrix_util::sparse_stat::SparseRunningStatistics<f32>>,
    pub n_cells: usize,
    /// Matrix rows. Equal to the gene count only on a non-channelized axis.
    pub n_rows: usize,
}

/// The graph neighbourhood algorithms must navigate, given the modelled graph
/// and the pre-augmentation spatial one.
///
/// The union is the edge set being MODELLED. It is the wrong graph to walk.
/// Seeding floods along adjacency, so over expression edges a seed can jump to
/// a similar cell anywhere in the tissue. Refinement asks whether removing a
/// node would disconnect its cluster, and expression edges can only ever add
/// connections, so the check weakens as they are added. How far it weakens
/// depends on `--knn-expr` and on the data; the point is that neither question
/// is being asked about adjacency any more. Both take the spatial graph.
///
/// A free function rather than a method because every caller destructures
/// [`SrtPreprocessed`] field by field, which is deliberate: adding a field
/// should force each call site to say what it wants.
pub fn topology_graph<'a>(
    graph: &'a KnnGraph,
    spatial_graph: &'a Option<KnnGraph>,
) -> &'a KnnGraph {
    spatial_graph.as_ref().unwrap_or(graph)
}

////////////////////
// preprocess_srt //
////////////////////

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
        // Before masking: the report is indexed by the ORIGINAL cell order, so
        // the names have to be read while they still line up.
        if let Some(path) = c.qc.qc_report.as_deref() {
            data_beans::qc_lib::write_qc_report(path, &data_vec.column_names()?, &report)?;
            info!("Wrote QC report to {}", path);
        }
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

    let n_rows = data_vec.num_rows();
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

    // Augmentation happens AFTER this point, never before, for two reasons.
    //
    // `auto_batch_from_components` above must see the spatial graph. It splits
    // batches by connected component, and expression edges bridge sections
    // freely, so on a multi-section slide the union will typically collapse to
    // one component. Then it reports one batch, batch-effect estimation is
    // skipped, and every tissue core lands in one frame downstream. That
    // failure is silent, which is why the ordering here is load-bearing.
    //
    // And the expression graph should be built on a batch-corrected
    // projection, or its neighbours match batch rather than cell type.
    let batch_arg: Option<&[Box<str>]> = batch_effects
        .is_some()
        .then_some(batch_membership.as_slice());

    let cell_proj = if cfg.cell_projection || (c.knn_expr > 0 && has_coords) {
        Some(data_vec.project_columns_with_batch_correction(c.proj_dim, c.block_size, batch_arg)?)
    } else {
        None
    };

    let (graph, spatial_graph, edge_source) = if c.knn_expr == 0 {
        (graph, None, None)
    } else if !has_coords {
        // The base graph already IS the expression graph, so a union would be
        // a self-union. Say so rather than doing nothing quietly.
        warn!(
            "--knn-expr {} ignored: without --coord the cell-pair graph is already \
             built from expression",
            c.knn_expr
        );
        (graph, None, None)
    } else {
        info!(
            "Adding expression KNN edges (k={}, proj_dim={})...",
            c.knn_expr, c.proj_dim
        );
        let proj = cell_proj
            .as_ref()
            .expect("a projection is taken whenever knn_expr > 0 and coordinates exist");
        let expr_graph = build_expression_knn(
            &proj.proj,
            SrtCellPairsArgs {
                knn: c.knn_expr,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?;
        let (merged, source) = graph.union_with(&expr_graph, DistanceMerge::SourceRank)?;
        let (mut n_spatial, mut n_expr, mut n_both) = (0usize, 0usize, 0usize);
        for s in source.iter() {
            match s {
                EdgeSource::Primary => n_spatial += 1,
                EdgeSource::Secondary => n_expr += 1,
                EdgeSource::Both => n_both += 1,
            }
        }
        info!(
            "{} cell pairs after augmentation: {} spatial, {} expression, {} shared",
            merged.edges.len(),
            n_spatial + n_both,
            n_expr,
            n_both
        );
        (merged, Some(graph), Some(source))
    };

    // What a ROW means, decided once, before anything keyed on a gene runs.
    let gene_axis = match cfg.gene_axis {
        GeneAxisMode::Rows => None,
        GeneAxisMode::Strict => Some(GeneAxis::resolve(&data_vec.row_names()?)?),
        GeneAxisMode::Lenient => Some(GeneAxis::resolve_or_identity(&data_vec.row_names()?)?),
    };

    // One streaming pass, up to four consumers. `compute_nb_fisher_weights`
    // builds these same statistics and returns only the weights, dropping
    // `npos`, which is exactly the per-gene detection count the dictionary
    // merge keys on. Taking the statistics here saves a second full read.
    //
    // On a channelized axis the pass folds as it goes rather than afterwards,
    // because `npos` and `s2` do not survive a post-hoc fold: a cell detected
    // on both tracks would count twice, and the variance would lose its cross
    // term. Those are the two statistics the merge filter and the dispersion
    // trend respectively depend on.
    let mut row_weights = None;
    let mut row_stats = None;
    let mut gene_weights = None;
    let mut gene_stats = None;
    if cfg.fisher_weights {
        info!("Computing NB Fisher-info weights for inference...");
        match gene_axis.as_ref() {
            Some(axis) => {
                let (r_stats, g_stats) =
                    axis.running_stats(&data_vec, c.block_size, "NB-Fisher")?;
                let rw = fisher_weights_from_stats(&r_stats, n_cells);
                let gw = fisher_weights_from_stats(&g_stats, n_cells);
                log_weights("Row weights w_r", &rw);
                log_weights("Gene weights w_g", &gw);
                row_weights = Some(rw);
                row_stats = Some(r_stats);
                gene_weights = Some(gw);
                gene_stats = Some(g_stats);
            }
            None => {
                let stats = streaming_sparse_running_stats(&data_vec, c.block_size, "NB-Fisher")?;
                let rw = fisher_weights_from_stats(&stats, n_cells);
                log_weights("Row weights w_r", &rw);
                row_weights = Some(rw);
                row_stats = Some(stats);
            }
        }
    }

    Ok(SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        batch_effects,
        graph,
        spatial_graph,
        edge_source,
        cell_proj,
        gene_axis,
        row_weights,
        row_stats,
        gene_weights,
        gene_stats,
        n_cells,
        n_rows,
    })
}

/// One log line per weight vector, so a channelized run shows both axes and a
/// reader can see at a glance whether the two disagree.
fn log_weights(label: &str, w: &[f32]) {
    info!(
        "{}: min={:.3e}, mean={:.3e}, max={:.3e} (n={})",
        label,
        w.iter().cloned().fold(f32::INFINITY, f32::min),
        w.iter().sum::<f32>() / (w.len().max(1) as f32),
        w.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        w.len(),
    );
}
