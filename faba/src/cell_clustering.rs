//! Cell clustering for automatic cell type assignment
//!
//! Clusters cells based on gene expression profiles using
//! random projection + SVD + k-means.

use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::gene_count::collect_all_gene_counts;
use genomic_data::gff::GffRecordMap;

use data_beans_alg::random_projection::RandProjOps;
use matrix_util::clustering::{Kmeans, KmeansArgs};
use matrix_util::traits::RandomizedAlgs;
use std::sync::Arc;

/// Parameters for cell clustering
pub struct ClusteringParams<'a> {
    pub n_clusters: usize,
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

/// Cluster cells based on gene expression profiles
///
/// # Arguments
/// * `bam_files` - BAM files to collect gene counts from
/// * `gff_map` - Gene annotation map
/// * `params` - Clustering parameters
///
/// # Returns
/// * `CellMembership` with cluster assignments
pub fn cluster_cells_from_bam(
    bam_files: &[Box<str>],
    gff_map: &GffRecordMap,
    params: &ClusteringParams,
) -> anyhow::Result<CellMembership> {
    info!(
        "Clustering cells into {} clusters using gene expression profiles",
        params.n_clusters
    );

    // Step 1: Collect gene counts per cell
    info!(
        "Collecting gene counts from {} BAM files...",
        bam_files.len()
    );
    let gene_counts = collect_all_gene_counts(
        bam_files,
        gff_map,
        params.cell_barcode_tag,
        params.gene_barcode_tag,
    )?;

    if gene_counts.is_empty() {
        return Err(anyhow::anyhow!("No gene counts found for clustering"));
    }

    // Convert to triplets format
    let triplets_data = format_data_triplets(gene_counts);
    let n_genes = triplets_data.rows.len();
    let n_cells = triplets_data.cols.len();
    let n_nonzeros = triplets_data.triplets.len();

    info!(
        "Gene count matrix: {} genes x {} cells ({} non-zeros)",
        n_genes, n_cells, n_nonzeros
    );

    if n_cells < params.n_clusters {
        return Err(anyhow::anyhow!(
            "Number of cells ({}) is less than requested clusters ({})",
            n_cells,
            params.n_clusters
        ));
    }

    // Step 2: Create temporary sparse backend with metadata
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir
        .path()
        .join("clustering_temp.zarr")
        .to_string_lossy()
        .into_owned();

    info!("Creating temporary sparse matrix");

    let data = triplets_data.to_backend(&temp_path)?;

    // Step 2b: QC - remove sparse rows/columns
    if params.min_row_nnz > 0 || params.min_col_nnz > 0 {
        info!(
            "QC: removing rows with < {} nnz, columns with < {} nnz",
            params.min_row_nnz, params.min_col_nnz
        );
        let cutoffs = SqueezeCutoffs {
            row: params.min_row_nnz,
            column: params.min_col_nnz,
        };
        data.qc(cutoffs)?;
    }

    // Get column names after QC (these are the cells that passed filtering)
    let filtered_cols = data.column_names()?;
    let n_cells_filtered = filtered_cols.len();

    info!(
        "After QC: {} cells remaining (from {} original)",
        n_cells_filtered, n_cells
    );

    if n_cells_filtered < params.n_clusters {
        return Err(anyhow::anyhow!(
            "After QC, number of cells ({}) is less than requested clusters ({})",
            n_cells_filtered,
            params.n_clusters
        ));
    }

    // Step 3: Wrap in SparseIoVec and do random projection
    let data_arc: Arc<dyn SparseIo<IndexIter = Vec<usize>>> = Arc::from(data);
    let mut data_vec = data_beans::sparse_io_vector::SparseIoVec::new();
    data_vec.push(data_arc, None)?;

    info!(
        "Performing random projection to {} dimensions",
        params.proj_dim
    );
    let proj_out = data_vec.project_columns(params.proj_dim, Some(params.block_size))?;
    let proj_kn = proj_out.proj; // k x n matrix

    // Step 4: SVD on the projection
    let svd_dim = params.svd_dim.min(params.proj_dim).min(n_cells_filtered);
    info!("Computing SVD with {} components", svd_dim);
    let (_, _, v_nk) = proj_kn.rsvd(svd_dim)?;

    // Step 5: K-means clustering
    info!(
        "Running k-means clustering with {} clusters",
        params.n_clusters
    );
    let assignments = v_nk.kmeans_rows(KmeansArgs {
        num_clusters: params.n_clusters,
        max_iter: params.kmeans_max_iter,
    });

    // Log cluster sizes
    let mut cluster_sizes = vec![0usize; params.n_clusters];
    for &c in &assignments {
        if c < params.n_clusters {
            cluster_sizes[c] += 1;
        }
    }
    info!("Cluster sizes: {:?}", cluster_sizes);

    // Step 6: Create CellMembership from cluster assignments (using filtered cells)
    let membership =
        CellMembership::from_clusters(&filtered_cols, &assignments, params.allow_prefix_matching);

    // Temp directory is cleaned up when temp_dir goes out of scope
    Ok(membership)
}
