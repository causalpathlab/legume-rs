//! Mass enrichment: group cells by gene expression to concentrate per-stratum
//! read mass for site detection.
//!
//! This is NOT a general cell-typing feature. Cells are grouped (random
//! projection + SVD + k-means on gene expression) purely so that stratified
//! editing-site discovery (`editing::pipeline::find_sites_with_celltype_stats`)
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
use matrix_util::clustering::{Kmeans, KmeansArgs};
use matrix_util::traits::RandomizedAlgs;
use nalgebra::DMatrix;
use std::sync::Arc;

/// Parameters for the mass-enrichment grouping.
pub struct MassEnrichmentParams<'a> {
    cluster_k: ClusterK,
    pub proj_dim: usize,
    pub svd_dim: usize,
    pub block_size: usize,
    pub kmeans_max_iter: usize,
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

    let (assignments, k) = match params.cluster_k {
        ClusterK::Fixed(k) => {
            if cols.len() < k {
                return Err(anyhow::anyhow!(
                    "Number of cells ({}) is less than requested clusters ({})",
                    cols.len(),
                    k
                ));
            }
            info!("Running k-means clustering with {} clusters", k);
            let assignments = v_nk.kmeans_rows(KmeansArgs {
                num_clusters: k,
                max_iter: params.kmeans_max_iter,
            });
            (assignments, k)
        }
        ClusterK::Auto { k_max } => {
            if cols.len() < 2 {
                return Err(anyhow::anyhow!(
                    "auto-K needs at least 2 cells, got {}",
                    cols.len()
                ));
            }
            // Sweep with a capped iteration budget (early-stops anyway), then a
            // full-budget final fit at the chosen K — see `select_k_by_ch`.
            select_k_by_ch(
                &v_nk,
                k_max,
                params.kmeans_max_iter.min(25),
                params.kmeans_max_iter,
            )
        }
    };

    log_cluster_sizes(&assignments, k);

    Ok(CellMembership::from_clusters(
        &cols,
        &assignments,
        params.allow_prefix_matching,
    ))
}

/// Sweep K over `2..=k_max`, cluster the SVD embedding `v_nk` at each, and pick the
/// K that maximizes the Calinski–Harabasz index
/// `CH(k) = (B/(k-1)) / (W/(n-k))`, where `W` is the within-cluster inertia and
/// `B = total − W` the between-cluster inertia. Returns the winning assignments and K.
///
/// The random projection + rSVD that produced `v_nk` ran once; the sweep only
/// repeats the cheap k-means on the small `n × svd_dim` embedding. Rows are
/// materialized once and reused across candidate K (k-means early-stops on
/// convergence), then one full-budget fit runs at the chosen K.
///
/// Reference: Caliński, T. & Harabasz, J. (1974), "A dendrite method for cluster
/// analysis," *Communications in Statistics* 3(1):1–27,
/// <https://doi.org/10.1080/03610927408827101>.
fn select_k_by_ch(
    v_nk: &DMatrix<f32>,
    k_max: usize,
    sweep_iter: usize,
    final_iter: usize,
) -> (Vec<usize>, usize) {
    let n = v_nk.nrows();
    // Materialize rows once so each candidate K reuses the same data (rather than
    // re-cloning the embedding on every call).
    let rows: Vec<Vec<f32>> = v_nk
        .row_iter()
        .map(|r| r.iter().copied().collect())
        .collect();

    let k_hi = k_max.min(n.saturating_sub(1)).max(2);
    let total = total_inertia(&rows);

    let mut best_k = 2;
    let mut best_ch = f64::NEG_INFINITY;
    info!("Auto-K: sweeping K in 2..={} (Calinski–Harabasz)", k_hi);
    for k in 2..=k_hi {
        let clust = clustering::kmeans(k, &rows, sweep_iter);
        let w = within_inertia(&rows, &clust.membership, k);
        let b = (total - w).max(0.0);
        let ch = if w > 0.0 && n > k {
            (b / (k as f64 - 1.0)) / (w / (n as f64 - k as f64))
        } else {
            f64::NEG_INFINITY
        };
        info!("  K={}: CH={:.2} (within-inertia={:.3})", k, ch, w);
        if ch > best_ch {
            best_ch = ch;
            best_k = k;
        }
    }
    info!("Auto-K selected K={} (max Calinski–Harabasz)", best_k);

    // Full-budget final fit at the chosen K.
    let clust = clustering::kmeans(best_k, &rows, final_iter);
    (clust.membership, best_k)
}

/// Total inertia: sum of squared distances of every row to the global centroid.
fn total_inertia(rows: &[Vec<f32>]) -> f64 {
    let n = rows.len();
    if n == 0 {
        return 0.0;
    }
    let d = rows[0].len();
    let mut mean = vec![0f64; d];
    for r in rows {
        for (j, &v) in r.iter().enumerate() {
            mean[j] += v as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    rows.iter().map(|r| sq_dist(r, &mean)).sum()
}

/// Within-cluster inertia: sum of squared distances of each row to its cluster
/// centroid (centroids recomputed from `membership`).
fn within_inertia(rows: &[Vec<f32>], membership: &[usize], k: usize) -> f64 {
    let d = rows.first().map_or(0, Vec::len);
    let mut sums = vec![vec![0f64; d]; k];
    let mut counts = vec![0usize; k];
    for (r, &c) in rows.iter().zip(membership) {
        counts[c] += 1;
        for (j, &v) in r.iter().enumerate() {
            sums[c][j] += v as f64;
        }
    }
    for (centroid, &count) in sums.iter_mut().zip(counts.iter()) {
        if count > 0 {
            for s in centroid.iter_mut() {
                *s /= count as f64;
            }
        }
    }
    rows.iter()
        .zip(membership)
        .map(|(r, &c)| sq_dist(r, &sums[c]))
        .sum()
}

/// Squared Euclidean distance between an `f32` row and an `f64` centroid.
fn sq_dist(row: &[f32], centroid: &[f64]) -> f64 {
    row.iter()
        .zip(centroid)
        .map(|(&v, &m)| {
            let dv = v as f64 - m;
            dv * dv
        })
        .sum()
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
/// Returns the `n_cells × svd_dim` right-singular-vector matrix `v_nk`.
fn project_and_embed(
    data_vec: &data_beans::sparse_io_vector::SparseIoVec,
    params: &MassEnrichmentParams,
    n_cells: usize,
) -> anyhow::Result<DMatrix<f32>> {
    info!("Random projection to {} dims, then SVD", params.proj_dim);
    let proj_out = data_vec.project_columns(params.proj_dim, Some(params.block_size))?;
    let svd_dim = params.svd_dim.min(params.proj_dim).min(n_cells);
    let (_, _, v_nk) = proj_out.proj.rsvd(svd_dim)?;
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

////////////////////////////////////////////////////////////////////////////////
// Shared CLI block                                                             //
////////////////////////////////////////////////////////////////////////////////

/// CLI knobs for the mass-enrichment grouping, shared (via `#[command(flatten)]`)
/// by every subcommand that stratifies detection by cell group (`m6a`, `pipeline`).
#[derive(clap::Args, Debug, Clone)]
pub struct MassEnrichmentArgs {
    #[arg(
        long = "n-clusters",
        default_value_t = 1,
        help = "Number of cell groups for mass enrichment (1 = no grouping)",
        long_help = "Number of cell groups used to concentrate per-stratum read\n\
                     mass for site detection. When 1 (default), all cells are one\n\
                     group (bulk detection). When > 1, cells are grouped via random\n\
                     projection + SVD + k-means on gene expression, and discovery\n\
                     runs per group so cell-type-specific edits are not diluted."
    )]
    pub n_clusters: usize,

    #[arg(
        long = "cluster-proj-dim",
        default_value_t = 50,
        help = "Random projection dimension for grouping",
        long_help = "Dimensionality of the random projection used to compress\n\
                     the gene expression matrix before SVD."
    )]
    pub cluster_proj_dim: usize,

    #[arg(
        long = "cluster-svd-dim",
        default_value_t = 10,
        help = "Number of SVD components for grouping",
        long_help = "Number of leading singular vectors to retain after SVD.\n\
                     These form the feature space for k-means."
    )]
    pub cluster_svd_dim: usize,

    #[arg(
        long = "cluster-block-size",
        default_value_t = 100,
        help = "Block size for grouping parallel processing",
        long_help = "Number of columns processed per parallel block during\n\
                     the random projection step."
    )]
    pub cluster_block_size: usize,

    #[arg(
        long = "cluster-max-iter",
        default_value_t = 100,
        help = "Maximum k-means iterations for grouping",
        long_help = "Maximum number of k-means iterations before convergence\n\
                     is declared."
    )]
    pub cluster_max_iter: usize,

    #[arg(
        long = "cluster-min-row-nnz",
        default_value_t = 1,
        help = "Minimum non-zeros per gene for grouping QC",
        long_help = "Genes with fewer non-zero cells than this are removed\n\
                     before grouping."
    )]
    pub cluster_min_row_nnz: usize,

    #[arg(
        long = "cluster-min-col-nnz",
        default_value_t = 1,
        help = "Minimum non-zeros per cell for grouping QC",
        long_help = "Cells with fewer non-zero genes than this are removed\n\
                     before grouping."
    )]
    pub cluster_min_col_nnz: usize,

    #[arg(
        long = "cluster-k-max",
        default_value_t = 0,
        help = "Auto-select K by sweeping 2..=this (0 = off, use --n-clusters)",
        long_help = "When > 0, the number of cell groups is chosen automatically by\n\
                     sweeping K over 2..=this value on the (cheap) SVD embedding and\n\
                     picking the K that maximizes the Calinski-Harabasz index. This\n\
                     overrides --n-clusters. 0 (default) uses a fixed --n-clusters."
    )]
    pub cluster_k_max: usize,
}

/// Resolved number of enrichment groups. `Bulk` is handled by `build_membership`
/// (returns `None`); the algorithm only ever sees `Fixed` or `Auto`.
#[derive(Clone, Copy, Debug)]
enum ClusterK {
    Fixed(usize),
    Auto { k_max: usize },
}

impl MassEnrichmentArgs {
    /// Whether grouping is on (fixed `--n-clusters > 1` or auto `--cluster-k-max > 1`).
    pub fn enabled(&self) -> bool {
        self.n_clusters > 1 || self.cluster_k_max > 1
    }

    /// Generate a `CellMembership` by grouping cells for enrichment, or `None`
    /// when grouping is disabled (`n_clusters <= 1`). When `matrix_paths` is
    /// non-empty, enrichment reuses those persisted gene-count matrices instead
    /// of re-scanning `bam_files`.
    pub fn build_membership(
        &self,
        bam_files: &[Box<str>],
        gff_map: &GffRecordMap,
        matrix_paths: &[Box<str>],
        cell_barcode_tag: &str,
        gene_barcode_tag: &str,
        allow_prefix_matching: bool,
    ) -> anyhow::Result<Option<CellMembership>> {
        // `--cluster-k-max > 0` auto-selects K (overrides --n-clusters); otherwise
        // a fixed --n-clusters; `<= 1` disables grouping (bulk detection).
        let cluster_k = if self.cluster_k_max > 1 {
            ClusterK::Auto {
                k_max: self.cluster_k_max,
            }
        } else if self.n_clusters > 1 {
            ClusterK::Fixed(self.n_clusters)
        } else {
            return Ok(None);
        };
        let params = MassEnrichmentParams {
            cluster_k,
            proj_dim: self.cluster_proj_dim,
            svd_dim: self.cluster_svd_dim,
            block_size: self.cluster_block_size,
            kmeans_max_iter: self.cluster_max_iter,
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

#[cfg(test)]
mod tests {
    use super::{sq_dist, total_inertia, within_inertia};

    #[test]
    fn total_inertia_matches_hand_calc() {
        // Two points on a line: mean = [1,0]; each is distance 1 away → total 2.
        let rows = vec![vec![0.0f32, 0.0], vec![2.0, 0.0]];
        assert!((total_inertia(&rows) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn within_inertia_uses_per_cluster_centroids() {
        // Two tight pairs; centroids [1,0] and [11,0]; four points each 1 away → 4.
        let rows = vec![
            vec![0.0f32, 0.0],
            vec![2.0, 0.0],
            vec![10.0, 0.0],
            vec![12.0, 0.0],
        ];
        let membership = [0usize, 0, 1, 1];
        assert!((within_inertia(&rows, &membership, 2) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn within_inertia_is_zero_when_each_point_is_its_own_cluster() {
        let rows = vec![vec![3.0f32, -1.0], vec![-2.0, 5.0]];
        assert!(within_inertia(&rows, &[0usize, 1], 2).abs() < 1e-9);
    }

    #[test]
    fn sq_dist_basic() {
        assert!((sq_dist(&[3.0f32, 4.0], &[0.0f64, 0.0]) - 25.0).abs() < 1e-9);
    }
}
