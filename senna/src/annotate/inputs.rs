//! Manifest + parquet → cluster-based annotation inputs.
//!
//! Loads the raw count matrix (re-opens the input zarr/h5 from
//! `manifest.data.input`), aligns cluster labels from a `senna cluster`
//! parquet to the data column order, and parses the marker TSV.

use crate::cluster::leiden_clustering;
use crate::embed_common::{read_mat, MatWithNames};
use crate::marker_support::build_annotation_matrix;
use crate::run_manifest::{self, RunKind, RunManifest};
use crate::senna_input::{
    read_data_on_shared_columns, ReadSharedColumnsArgs, SparseStackWithBatch,
};
use data_beans::sparse_io_vector::SparseIoVec;
use matrix_util::traits::IoOps;
use rustc_hash::FxHashMap as HashMap;
use std::path::Path;

use crate::embed_common::Mat;

/// CLI passthrough for the internal Leiden fallback (used only when neither
/// `--clusters` nor `manifest.cluster.clusters` is provided).
pub struct LeidenArgs {
    pub knn: usize,
    pub resolution: f64,
    pub num_clusters: Option<usize>,
    pub min_cluster_size: usize,
    pub seed: Option<u64>,
}

/// Loaded inputs for the cluster-based annotation pipeline.
pub struct LoadedInputs {
    /// Owns the `SparseIoVec` (column-block reads done in run.rs).
    pub stack: SparseStackWithBatch,
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
    /// Gene names from the data backend (rows of `data_vec`).
    pub gene_names: Vec<Box<str>>,
    /// Cell names from the data backend (cols of `data_vec`).
    pub cell_names: Vec<Box<str>>,
    /// G × C IDF-weighted marker matrix.
    pub markers_gc: Mat,
    pub celltype_names: Vec<Box<str>>,
}

impl LoadedInputs {
    pub fn data_vec(&self) -> &SparseIoVec {
        &self.stack.data_stack.stack[0]
    }
}

pub fn load_from_manifest(
    manifest_path: &str,
    clusters_override: Option<&str>,
    markers_path: &str,
    preload_data: bool,
    leiden_args: &LeidenArgs,
) -> anyhow::Result<(LoadedInputs, RunManifest, std::path::PathBuf)> {
    let (manifest, manifest_dir) = RunManifest::load(Path::new(manifest_path))?;
    log::info!(
        "Loaded manifest ({}): kind={}",
        manifest_path,
        manifest.kind
    );

    let resolve = |rel: &str| -> String {
        run_manifest::resolve(&manifest_dir, rel)
            .to_string_lossy()
            .into_owned()
    };

    anyhow::ensure!(
        !manifest.data.input.is_empty(),
        "manifest.data.input is empty; cannot re-open raw counts for cluster aggregation"
    );

    let data_files: Vec<Box<str>> = manifest
        .data
        .input
        .iter()
        .map(|s| resolve(s).into_boxed_str())
        .collect();
    let batch_files: Option<Vec<Box<str>>> = if manifest.data.batch.is_empty() {
        None
    } else {
        Some(
            manifest
                .data
                .batch
                .iter()
                .map(|s| resolve(s).into_boxed_str())
                .collect(),
        )
    };

    log::info!("Re-opening raw counts: {} file(s)", data_files.len());
    let stack = read_data_on_shared_columns(ReadSharedColumnsArgs {
        data_files,
        batch_files,
        num_types: 1,
        preload: preload_data,
    })?;
    anyhow::ensure!(
        stack.data_stack.stack.len() == 1,
        "annotate: expected a single data stack, got {}",
        stack.data_stack.stack.len()
    );

    let data_vec = &stack.data_stack.stack[0];
    let n_cells = data_vec.num_columns();
    let n_genes = data_vec.num_rows();
    let cell_names = data_vec.column_names()?;
    let gene_names = data_vec.row_names()?;
    log::info!("Raw counts: {n_genes} genes × {n_cells} cells");

    // Resolve cluster parquet path; fall back to internal Leiden on the
    // manifest's latent matrix when no path is given.
    let clusters_path: Option<String> = match clusters_override {
        Some(p) => Some(p.to_string()),
        None => manifest.cluster.clusters.as_deref().map(&resolve),
    };
    let (cluster_labels, n_clusters) = match clusters_path {
        Some(path) => load_cluster_labels(&path, &cell_names)?,
        None => compute_clusters_from_latent(&manifest, &resolve, &cell_names, leiden_args)?,
    };

    // Build per-cell batch ids (u32) for permutation null.
    let (batch_labels, n_batches) = build_batch_labels(&stack, n_cells)?;
    log::info!("Batches: {n_batches}");

    // Markers aligned to data row order.
    let annot = build_annotation_matrix(markers_path, &gene_names)?;
    log::info!(
        "Marker matrix: {} genes × {} celltypes",
        annot.membership_ga.nrows(),
        annot.membership_ga.ncols()
    );

    Ok((
        LoadedInputs {
            stack,
            cluster_labels,
            n_clusters,
            batch_labels,
            n_batches,
            gene_names,
            cell_names,
            markers_gc: annot.membership_ga,
            celltype_names: annot.annot_names,
        },
        manifest,
        manifest_dir,
    ))
}

/// Run Leiden on the manifest's latent matrix. For topic kinds, log-space
/// latents are exponentiated to probabilities first. Aligns labels back to
/// the data column order.
fn compute_clusters_from_latent<F>(
    manifest: &RunManifest,
    resolve: &F,
    cell_names: &[Box<str>],
    args: &LeidenArgs,
) -> anyhow::Result<(Vec<usize>, usize)>
where
    F: Fn(&str) -> String,
{
    let latent_rel = manifest
        .outputs
        .latent
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("manifest missing outputs.latent"))?;
    let latent_path = resolve(latent_rel);
    let mut latent = Mat::from_parquet_with_row_names(&latent_path, Some(0))?;
    log::info!(
        "Loaded latent {latent_path}: {}×{}",
        latent.mat.nrows(),
        latent.mat.ncols()
    );

    let is_topic = matches!(
        manifest.kind,
        RunKind::Topic | RunKind::Itopic | RunKind::JointTopic
    );
    if is_topic && latent.mat.max() <= 0.0 {
        log::info!("Detected log-space latent (max ≤ 0); exponentiating to probabilities");
        latent.mat = latent.mat.map(f32::exp);
    }

    log::info!(
        "Internal Leiden: knn={}, resolution={:.3}, target_k={:?}, min_cluster_size={}",
        args.knn,
        args.resolution,
        args.num_clusters,
        args.min_cluster_size
    );

    let mut result = leiden_clustering(
        &latent.mat,
        args.knn,
        args.resolution,
        args.num_clusters,
        args.seed,
    )?;
    if args.min_cluster_size > 1 {
        result.remove_small_clusters(args.min_cluster_size);
    }

    // Align by name. The training pipeline writes latents in data column
    // order, so name-mismatched cases are rare but worth a warning.
    let mut idx: HashMap<&str, usize> = HashMap::default();
    idx.reserve(latent.rows.len());
    for (i, name) in latent.rows.iter().enumerate() {
        idx.insert(name.as_ref(), i);
    }
    let mut labels = Vec::with_capacity(cell_names.len());
    let mut missing = 0usize;
    for cell in cell_names {
        match idx.get(cell.as_ref()) {
            Some(&i) => labels.push(result.labels[i]),
            None => {
                labels.push(usize::MAX);
                missing += 1;
            }
        }
    }
    if missing > 0 {
        log::warn!("{missing} cells missing from latent — marked unassigned");
    }
    Ok((labels, result.n_clusters))
}

/// Read the cluster parquet (cells × 1 cluster column, NaN for unassigned)
/// and align to the `cell_names` order from the data backend.
fn load_cluster_labels(
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
        match cluster_idx.get(cell.as_ref()) {
            Some(&i) => {
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
            }
            None => {
                labels.push(usize::MAX);
                missing += 1;
            }
        }
    }
    anyhow::ensure!(
        missing < cell_names.len(),
        "cluster parquet has no overlap with data cells (missing {} of {})",
        missing,
        cell_names.len()
    );

    let n_clusters = (max_label + 1).max(0) as usize;
    log::info!(
        "Clusters: {n_clusters} (assigned: {}, unassigned: {}, missing-from-parquet: {})",
        cell_names.len() - unassigned - missing,
        unassigned,
        missing
    );
    Ok((labels, n_clusters))
}

/// Build per-cell batch ids as compact u32 indices in [0, n_batches).
fn build_batch_labels(
    stack: &SparseStackWithBatch,
    n_cells: usize,
) -> anyhow::Result<(Vec<u32>, usize)> {
    let mut labels: Vec<Box<str>> = Vec::with_capacity(n_cells);
    for v in &stack.batch_stack {
        labels.extend(v.iter().cloned());
    }
    if labels.is_empty() {
        return Ok((vec![0u32; n_cells], 1));
    }
    anyhow::ensure!(
        labels.len() == n_cells,
        "batch labels {} ≠ cell count {}",
        labels.len(),
        n_cells
    );

    let mut name_to_id: HashMap<Box<str>, u32> = HashMap::default();
    let mut next_id: u32 = 0;
    let mut out = Vec::with_capacity(n_cells);
    for name in labels {
        let id = *name_to_id.entry(name).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        out.push(id);
    }
    Ok((out, next_id as usize))
}
