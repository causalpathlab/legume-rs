//! Inputs for the cluster-based annotation pipeline, and the cluster-parquet
//! reader that aligns an existing clustering to the data's cell order.
//!
//! Re-opening the raw counts, resolving where the clustering comes from, and
//! aggregating per-cluster/per-batch gene sums all need the run manifest and
//! the sparse backend, so they live in `senna::annotate_manifest`; what arrives
//! here is the result.

use crate::mat_io::{read_mat, Mat, MatWithNames};
use rustc_hash::FxHashMap as HashMap;

/// Everything [`crate::by_enrichment::run`] needs: the cell/gene dictionaries,
/// the clustering, the batch axis for the sample-permutation null, the marker
/// matrix, and the already-aggregated cluster (and per-batch) expression.
pub struct EnrichmentInputs {
    /// Gene names from the data backend — rows of `profile_gk` / `markers_gc`.
    pub gene_names: Vec<Box<str>>,
    /// Cell names from the data backend — rows of the per-cell outputs.
    pub cell_names: Vec<Box<str>>,
    /// `cluster_labels[n]` = cluster id for cell n, or `usize::MAX` for unassigned.
    pub cluster_labels: Vec<usize>,
    /// Number of distinct (assigned) clusters; ids run 0..`n_clusters`.
    pub n_clusters: usize,
    /// `batch_labels[n]` = batch id for cell n. Used as the pseudobulk
    /// granularity for the sample-permutation null. When the manifest
    /// supplies no batch files, all cells default to batch 0.
    pub batch_labels: Vec<u32>,
    /// Number of distinct batches.
    pub n_batches: usize,
    /// G × C IDF-weighted marker matrix. Empty (C = 0) in GO/GMT gene-set mode.
    pub markers_gc: Mat,
    pub celltype_names: Vec<Box<str>>,
    /// G × K weighted mean cluster expression.
    pub profile_gk: Mat,
    /// G × P weighted mean per-batch expression. `None` in GO/GMT gene-set
    /// mode, which scores the cluster profile directly and runs no permutation.
    pub pb_gene_gp: Option<Mat>,
}

/// Read the cluster parquet (cells × 1 cluster column, NaN for unassigned)
/// and align to the `cell_names` order from the data backend.
pub fn load_cluster_labels(
    clusters_path: &str,
    cell_names: &[Box<str>],
) -> anyhow::Result<(Vec<usize>, usize)> {
    let MatWithNames {
        rows: cluster_cells,
        cols: cluster_cols,
        mat: cluster_mat,
    } = read_mat(clusters_path)?;
    log::info!(
        "Loaded cluster parquet {clusters_path}: {} cells × {} columns",
        cluster_mat.nrows(),
        cluster_mat.ncols()
    );

    anyhow::ensure!(
        cluster_mat.ncols() >= 1,
        "cluster parquet has no value column"
    );
    let label_col = cluster_cols
        .iter()
        .position(|c| c.as_ref() == "cluster")
        .unwrap_or(0);

    // Map cell name → row index in cluster parquet.
    let mut cluster_idx: HashMap<&str, usize> = HashMap::default();
    cluster_idx.reserve(cluster_cells.len());
    for (i, name) in cluster_cells.iter().enumerate() {
        cluster_idx.insert(name.as_ref(), i);
    }

    let mut labels = Vec::with_capacity(cell_names.len());
    let mut max_label: i64 = -1;
    let mut unassigned = 0usize;
    let mut missing = 0usize;
    for cell in cell_names {
        if let Some(&i) = cluster_idx.get(cell.as_ref()) {
            let v = cluster_mat[(i, label_col)];
            if v.is_nan() || v < 0.0 {
                labels.push(usize::MAX);
                unassigned += 1;
            } else {
                let id = v as usize;
                labels.push(id);
                if (id as i64) > max_label {
                    max_label = id as i64;
                }
            }
        } else {
            labels.push(usize::MAX);
            missing += 1;
        }
    }
    anyhow::ensure!(
        missing < cell_names.len(),
        "cluster parquet has no overlap with data cells (missing {} of {})",
        missing,
        cell_names.len()
    );

    let n_clusters = (max_label + 1).max(0) as usize;
    let assigned = cell_names.len() - unassigned - missing;
    log::info!(
        "Clusters: {n_clusters} (assigned: {assigned}, unassigned: {unassigned}, \
         missing-from-parquet: {missing})"
    );
    anyhow::ensure!(
        assigned > 0,
        "cluster parquet aligned but every entry is NaN/missing — every cell is unassigned. \
         Check the parquet's `cluster` column actually contains integer cluster ids."
    );
    Ok((labels, n_clusters))
}

#[cfg(test)]
#[path = "inputs_tests.rs"]
mod tests;
