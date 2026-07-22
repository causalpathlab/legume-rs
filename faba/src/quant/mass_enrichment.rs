//! Mass enrichment: group cells by gene expression to concentrate per-stratum
//! read mass for site detection.
//!
//! This is NOT a general cell-typing feature. Cells are grouped (random
//! projection + SVD on gene expression, then Leiden community detection) purely
//! so that stratified editing-site discovery
//! (`editing::pipeline::find_sites_with_celltype_stats`)
//! sees a higher foreground fraction within each stratum: if an edit is specific
//! to a sub-population, pooling all cells dilutes it, whereas grouping
//! concentrates the edited reads and improves the foreground-vs-background
//! contrast. The grouping is an exogenous *instrument* (driven by expression,
//! not by the editing signal), which keeps the per-stratum test honest.

use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::gene_count::collect_all_gene_counts;
use genomic_data::gff::GffRecordMap;

use data_beans::sparse_io_vector::ColumnAlignment;
use data_beans_alg::random_projection::RandProjOps;
use graph_embedding_util::{load_unified_data, LoadUnifiedArgs};
use matrix_util::traits::RandomizedAlgs;
use nalgebra::DMatrix;
use std::sync::Arc;

/// Fixed RNG seed for the Leiden community detection, so a rerun on the same
/// embedding reproduces the same groups.
const ENRICH_SEED: u64 = 42;

/// Parameters for the mass-enrichment grouping.
pub struct MassEnrichmentParams<'a> {
    /// Leiden modularity resolution (higher → more, finer communities). The
    /// group count is not fixed — it emerges from the graph at this resolution.
    resolution: f64,
    /// k for the cell kNN graph fed to Leiden.
    knn: usize,
    /// Embedding dimension: cells are random-projected to `dim` features and the
    /// SVD is taken at the same rank (`proj_dim = svd_dim = dim`), so this is both
    /// the projection width and the principal-component count Leiden clusters on.
    dim: usize,
    pub block_size: usize,
    pub cell_barcode_tag: &'a str,
    pub gene_barcode_tag: &'a str,
    pub allow_prefix_matching: bool,
    /// Minimum non-zeros per row (gene) to keep
    pub min_row_nnz: usize,
    /// Minimum non-zeros per column (cell) to keep
    pub min_col_nnz: usize,
}

/// Group cells by gene expression to enrich per-stratum read mass.
///
/// # Arguments
/// * `bam_files` - BAM files to collect gene counts from (fallback source)
/// * `gff_map` - Gene annotation map
/// * `params` - Mass-enrichment grouping parameters
/// * `matrix_paths` - Gene-count matrices QC already persisted. When non-empty they
///   are reused (no BAM re-scan, no temp backend); empty ⇒ collect from `bam_files`.
///
/// # Returns
/// * `CellMembership` with group (cluster) assignments
pub fn group_cells_for_enrichment(
    bam_files: &[Box<str>],
    gff_map: &GffRecordMap,
    params: &MassEnrichmentParams,
    matrix_paths: &[Box<str>],
) -> anyhow::Result<CellMembership> {
    // Embed cells into a low-dim SVD space (random projection → rSVD), reusing the
    // gene-count matrices QC persisted when available, else collecting from BAMs.
    let (v_nk, cols) = if matrix_paths.is_empty() {
        embed_cells_from_bams(bam_files, gff_map, params)?
    } else {
        embed_cells_from_matrices(matrix_paths, params)?
    };

    if cols.len() < 2 {
        return Err(anyhow::anyhow!(
            "mass enrichment needs at least 2 cells, got {}",
            cols.len()
        ));
    }

    // Group cells by Leiden community detection on the SVD embedding. The count
    // is not fixed: it emerges from the graph at `resolution`. `cosine = false`
    // z-scores the (heterogeneous-scale, unscaled-by-singular-value) SVD columns
    // and takes Euclidean kNN — the correct metric for a raw SVD latent, and
    // more robust than k-means, which on this embedding produced singleton
    // clusters and an unusable Calinski–Harabasz sweep.
    let knn = params.knn.min(cols.len() - 1).max(1);
    info!(
        "Leiden community detection on the SVD embedding (resolution {:.3}, knn {})",
        params.resolution, knn
    );
    let assignments = matrix_util::clustering::leiden_clustering(
        &v_nk,
        knn,
        params.resolution,
        None, // resolution-driven: the community count emerges
        Some(ENRICH_SEED),
        false, // raw SVD dims → z-score + Euclidean kNN
    )?;
    let k = assignments.iter().copied().max().map_or(0, |m| m + 1);
    info!("Leiden found {} cell group(s) over {} cells", k, cols.len());

    log_cluster_sizes(&assignments, k);

    Ok(CellMembership::from_clusters(
        &cols,
        &assignments,
        params.allow_prefix_matching,
    ))
}

/// Tally and log the number of cells in each cluster.
fn log_cluster_sizes(assignments: &[usize], n_clusters: usize) {
    let mut sizes = vec![0usize; n_clusters];
    for &c in assignments {
        sizes[c] += 1;
    }
    info!("Cluster sizes: {:?}", sizes);
}

/// Random-projection + rSVD embedding of a sparse count backend's columns (cells).
/// Returns the `n_cells × dim` right-singular-vector matrix `v_nk`.
fn project_and_embed(
    data_vec: &data_beans::sparse_io_vector::SparseIoVec,
    params: &MassEnrichmentParams,
    n_cells: usize,
) -> anyhow::Result<DMatrix<f32>> {
    // Random-project to `dim` features, then SVD at the same rank — an
    // orthogonal, variance-ordered rotation of the sketch (no truncation). Cap
    // `dim` at the achievable rank: it can exceed neither the cell count nor the
    // feature (row) count, and projecting to more dims than either is wasteful.
    let dim = params.dim.min(n_cells).min(data_vec.num_rows()).max(1);
    info!("Random projection to {dim} dims, then SVD");
    let proj_out = data_vec.project_columns(dim, Some(params.block_size))?;
    let (_, _, v_nk) = proj_out.proj.rsvd(dim)?;
    Ok(v_nk)
}

/// Fresh embedding: scan BAMs → gene counts → temp backend → project. Fallback for
/// when no persisted QC matrix is available (`--skip-gene-qc`, or no `--gff`).
fn embed_cells_from_bams(
    bam_files: &[Box<str>],
    gff_map: &GffRecordMap,
    params: &MassEnrichmentParams,
) -> anyhow::Result<(DMatrix<f32>, Vec<Box<str>>)> {
    info!(
        "Collecting gene counts from {} BAM files for enrichment...",
        bam_files.len()
    );
    let gene_counts = collect_all_gene_counts(
        bam_files,
        gff_map,
        params.cell_barcode_tag,
        params.gene_barcode_tag,
        20, // min_mapq: consistent with ATOI/APA/DART quality filters
    )?;
    if gene_counts.is_empty() {
        return Err(anyhow::anyhow!(
            "No gene counts found for enrichment grouping"
        ));
    }

    let triplets_data = format_data_triplets(gene_counts);
    info!(
        "Gene count matrix: {} genes x {} cells ({} non-zeros)",
        triplets_data.rows.len(),
        triplets_data.cols.len(),
        triplets_data.triplets.len()
    );

    // The temp backend must outlive projection; `v_nk` is owned on return, after
    // which `temp_dir` drops and the staging directory is removed.
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir
        .path()
        .join("enrichment_temp.zarr")
        .to_string_lossy()
        .into_owned();
    let data = triplets_data.to_backend(&temp_path)?;

    if params.min_row_nnz > 0 || params.min_col_nnz > 0 {
        data.qc(SqueezeCutoffs {
            row: params.min_row_nnz,
            column: params.min_col_nnz,
        })?;
    }
    let cols = data.column_names()?;
    info!("After QC: {} cells", cols.len());

    let data_arc: Arc<dyn SparseIo<IndexIter = Vec<usize>>> = Arc::from(data);
    let mut data_vec = data_beans::sparse_io_vector::SparseIoVec::new();
    data_vec.push(data_arc, None)?;
    let v_nk = project_and_embed(&data_vec, params, cols.len())?;
    Ok((v_nk, cols))
}

/// Reuse embedding: load the gene-count matrices QC already persisted (feature-aligned
/// across batches by `load_unified_data`) and project — no BAM re-scan, no temp backend.
/// `Disjoint` keeps each batch's cells as distinct columns (barcodes collide across
/// libraries); the membership is barcode-keyed downstream regardless.
fn embed_cells_from_matrices(
    matrix_paths: &[Box<str>],
    params: &MassEnrichmentParams,
) -> anyhow::Result<(DMatrix<f32>, Vec<Box<str>>)> {
    info!(
        "Reusing {} persisted gene-count matrix(es) for enrichment (no BAM re-scan)",
        matrix_paths.len()
    );
    let unified = load_unified_data(LoadUnifiedArgs {
        data_files: matrix_paths.to_vec(),
        column_alignment: ColumnAlignment::Disjoint,
        ..Default::default()
    })?;
    let cols = unified.barcodes.clone();
    info!(
        "Reused matrix: {} features x {} cells",
        unified.feature_names.len(),
        cols.len()
    );
    let v_nk = project_and_embed(unified.count_backend(), params, cols.len())?;
    Ok((v_nk, cols))
}

//////////////////////
// Shared CLI block //
//////////////////////

/// CLI knobs for the mass-enrichment grouping, shared (via `#[command(flatten)]`)
/// by every subcommand that stratifies detection by cell group (`m6a`, `pipeline`).
#[derive(clap::Args, Debug, Clone, serde::Serialize)]
pub struct MassEnrichmentArgs {
    // The single clustering knob most users touch: this dial turns Leiden auto-grouping
    // on and tunes its granularity. Everything below is advanced embedding/QC tuning,
    // hidden from the short `-h` view.
    #[arg(
        long = "cluster-resolution",
        default_value_t = 0.0,
        help = "Leiden auto-grouping for mass enrichment: OFF by default (bulk); set > 0 to enable (try 0.5; higher → more groups)",
        long_help = "Leiden modularity resolution for mass-enrichment cell grouping.\n\
                     \n\
                     OFF by default (0): discovery runs in bulk, over all cells at once.\n\
                     Set a positive resolution to turn grouping on —\n\
                     cells are embedded (random projection + SVD on gene expression)\n\
                     and Leiden community detection groups them automatically,\n\
                     the group count NOT fixed but emerging from the graph\n\
                     (higher resolution → more, finer groups).\n\
                     Discovery then runs per group,\n\
                     so cell-type-specific edits are not diluted by the cells that do not carry them.\n\
                     Try 0.5.\n\
                     \n\
                     The default is off because grouping is expensive —\n\
                     it adds a cell embedding, a kNN graph, and Leiden,\n\
                     and then multiplies the discovery work by the number of groups.\n\
                     It is a deliberate opt-in for when a rare compartment's edits\n\
                     are being washed out, not something to pay for on every run."
    )]
    pub cluster_resolution: f64,

    #[arg(
        long = "cluster-knn",
        default_value_t = 30,
        hide_short_help = true,
        help = "k for the cell kNN graph fed to Leiden grouping",
        long_help = "Number of nearest neighbours per cell when building the kNN graph\n\
                     that Leiden partitions.\n\
                     Larger k → smoother, coarser communities."
    )]
    pub cluster_knn: usize,

    #[arg(
        long = "cluster-dim",
        default_value_t = 50,
        hide_short_help = true,
        help = "Embedding dimension for grouping (random-projection width = SVD rank)",
        long_help = "Cell embedding dimension for Leiden grouping.\n\
                     Cells are random-projected to this many features\n\
                     and the SVD is taken at the same rank,\n\
                     so it is both the projection width and the principal-component count clustered on.\n\
                     Capped at the cell count. 30–50 is typical."
    )]
    pub cluster_dim: usize,

    #[arg(
        long = "cluster-block-size",
        default_value_t = 100,
        hide_short_help = true,
        help = "Block size for grouping parallel processing",
        long_help = "Number of columns processed per parallel block during the random projection step."
    )]
    pub cluster_block_size: usize,

    #[arg(
        long = "cluster-min-row-nnz",
        default_value_t = 1,
        hide_short_help = true,
        help = "Minimum non-zeros per gene for grouping QC",
        long_help = "Genes with fewer non-zero cells than this are removed\n\
                     before grouping."
    )]
    pub cluster_min_row_nnz: usize,

    #[arg(
        long = "cluster-min-col-nnz",
        default_value_t = 1,
        hide_short_help = true,
        help = "Minimum non-zeros per cell for grouping QC",
        long_help = "Cells with fewer non-zero genes than this are removed\n\
                     before grouping."
    )]
    pub cluster_min_col_nnz: usize,
}

impl MassEnrichmentArgs {
    /// Whether grouping is on. On by default (resolution 0.5); disabled only when
    /// `--cluster-resolution` is set to 0 (or below), which restores bulk detection.
    pub fn enabled(&self) -> bool {
        self.cluster_resolution > 0.0
    }

    /// Generate a `CellMembership` by grouping cells for enrichment, or `None`
    /// when grouping is disabled. When `matrix_paths` is non-empty, enrichment
    /// reuses those persisted gene-count matrices instead of re-scanning `bam_files`.
    pub fn build_membership(
        &self,
        bam_files: &[Box<str>],
        gff_map: &GffRecordMap,
        matrix_paths: &[Box<str>],
        cell_barcode_tag: &str,
        gene_barcode_tag: &str,
        allow_prefix_matching: bool,
    ) -> anyhow::Result<Option<CellMembership>> {
        // `--cluster-resolution` drives Leiden directly. The old k-means count
        // knobs (`--n-clusters` / `--cluster-k-max`) are gone; they were parsed
        // but ignored, so passing one now errors at the CLI instead of silently
        // running at a resolution the caller never chose.
        //
        // Resolution 0 (or below) disables grouping → bulk detection.
        if !self.enabled() {
            return Ok(None);
        }
        let params = MassEnrichmentParams {
            resolution: self.cluster_resolution,
            knn: self.cluster_knn,
            dim: self.cluster_dim,
            block_size: self.cluster_block_size,
            cell_barcode_tag,
            gene_barcode_tag,
            allow_prefix_matching,
            min_row_nnz: self.cluster_min_row_nnz,
            min_col_nnz: self.cluster_min_col_nnz,
        };
        let membership = group_cells_for_enrichment(bam_files, gff_map, &params, matrix_paths)?;
        Ok(Some(membership))
    }
}
