//! Run-manifest glue for the `annotate` crate.
//!
//! `annotate` owns the annotation implementations but must not depend on
//! `senna` — that would cycle, since `senna` re-exports it. So the manifest
//! side of each annotation command lives here: resolve `-f run.senna.json`
//! into the loaded matrices the library takes, then write the artifact paths it
//! returns back into `manifest.annotate.*`. This is the only place that touches
//! [`RunManifest`] on behalf of an annotation command.
//!
//! Re-opening the raw counts and aggregating them per cluster also lives here:
//! the sparse backend and the per-cluster sweeps are senna's, and what
//! `annotate` receives is the cluster × gene expression they produce.

use ::annotate::args::{AnnotateArgs, AnnotateOntologyArgs, AnnotateProjectionArgs};
use ::annotate::inputs::{load_cluster_labels, EnrichmentInputs};
use ::annotate::outputs::AnnotationOutputs;
use ::annotate::{EnrichmentPlan, ProjectionInputs};

use crate::cluster::{leiden_clustering_with_metric, LatentMetric};
use crate::cluster_aggregation::{
    accumulate_gene_sum, accumulate_gene_sum_pair, weighted_mean_profile,
};
use crate::embed_common::Mat;
use crate::marker_embedding::load_marker_feature_embedding_from;
use crate::marker_support::build_annotation_matrix;
use crate::run_manifest::{self, CellSpace, RunManifest};
use crate::senna_input::{
    read_data_on_shared_columns, ReadSharedColumnsArgs, SparseStackWithBatch,
};

use anyhow::{anyhow, Context, Result};
use log::info;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::path::{Path, PathBuf};

/// CLI passthrough for the internal Leiden fallback (used only when neither
/// `--clusters` nor `manifest.cluster.clusters` is provided).
pub struct LeidenArgs {
    pub knn: usize,
    pub resolution: f64,
    pub num_clusters: Option<usize>,
    pub min_cluster_size: usize,
    pub seed: Option<u64>,
}

/// `senna annotate-by-enrichment` / `lupin annotate --method enrichment`:
/// re-open the raw counts the manifest points at, aggregate them per cluster,
/// run the marker enrichment, and record what it wrote.
pub fn annotate_by_enrichment(args: &AnnotateArgs) -> Result<()> {
    let plan = ::annotate::plan_enrichment(args, &run_manifest::derive_out_prefix(&args.from))?;
    let (inputs, mut manifest, manifest_dir) = load_enrichment_inputs(args, &plan)?;
    let outputs = ::annotate::annotate_by_enrichment(args, &plan, &inputs)?;

    let from = Path::new(args.from.as_ref());
    if plan.ontology_mode {
        record_gene_set_signature(&mut manifest, from, &manifest_dir, &outputs)
    } else {
        record_annotation(&mut manifest, from, &manifest_dir, &args.markers, &outputs)
    }
}

/// `senna annotate-by-projection` / `lupin annotate --method projection`: load
/// the co-embedded gene space + the cell embedding off the manifest, score them
/// against the marker panel, and record what it wrote.
pub fn annotate_by_projection(args: &AnnotateProjectionArgs) -> Result<()> {
    let (mut manifest, manifest_dir) = run_manifest::load_for(&args.from)?;
    info!("Loaded manifest ({}): kind={}", args.from, manifest.kind);

    // Feature side: genes on the cell manifold (required for projection). Reads
    // `outputs.feature_coembedding` off the manifest and, for a `gem` run, keeps
    // only the spliced rows re-keyed by gene — see `crate::gem::marker_embedding`.
    let feat = load_marker_feature_embedding_from(&manifest, &manifest_dir, &args.from)
        .with_context(|| {
            "projection needs a co-embedded gene space (a `senna gem` / `bge` / `fne` / \
         `resolve-embedding-space` run). For topic/svd runs use `senna annotate-by-enrichment`."
        })?;
    // Cell side: prefer the explicit cell_embedding; fall back to latent for
    // manifests written before Z moved there unconditionally.
    let cell_rel = manifest.outputs.geometry_latent().ok_or_else(|| {
        anyhow!("manifest has neither `outputs.cell_embedding` nor `outputs.latent`")
    })?;
    let cell_path = resolve(&manifest_dir, cell_rel);
    let cell = DMatrix::<f32>::from_parquet(&cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;

    let outputs = ::annotate::annotate_by_projection(
        args,
        &run_manifest::derive_out_prefix(&args.from),
        &ProjectionInputs {
            feature_embedding: &feat,
            cell_embedding: &cell,
        },
    )?;

    record_annotation(
        &mut manifest,
        Path::new(args.from.as_ref()),
        &manifest_dir,
        &args.markers,
        &outputs,
    )
}

/// `senna annotate-ontology` / `lupin annotate` without markers: walk the CL
/// tree over the cluster × celltype matrix an earlier enrichment run recorded.
pub fn annotate_ontology(args: &AnnotateOntologyArgs) -> Result<()> {
    let from = Path::new(args.from.as_ref());
    let (mut manifest, manifest_dir) = RunManifest::load(from)?;
    let q_rel = manifest
        .annotate
        .cluster_celltype_q
        .clone()
        .ok_or_else(|| {
            anyhow!(
                "manifest has no `annotate.cluster_celltype_q` — run \
             `senna annotate-by-enrichment --from {} -m <markers>` first",
                args.from
            )
        })?;
    let q_abs = resolve(&manifest_dir, &q_rel);

    let outputs =
        ::annotate::annotate_ontology(args, &run_manifest::derive_out_prefix(&args.from), &q_abs)?;

    let rel = |abs: &str| run_manifest::rel_to_manifest(&manifest_dir, abs);
    manifest.annotate.ontology_assignment = outputs.ontology_assignment.as_deref().map(&rel);
    manifest.annotate.ontology_node_mass = outputs.ontology_node_mass.as_deref().map(&rel);
    manifest.save(from)
}

////////////////////////////////////
// manifest → annotation outputs  //
////////////////////////////////////

/// Wire the artifacts into the manifest as paths relative to `manifest_dir`,
/// flip the default plot colour to `annotation`, and save back to `from`. Both
/// marker passes land here, so they keep the same `manifest.annotate.*`
/// contract.
fn record_annotation(
    manifest: &mut RunManifest,
    from: &Path,
    manifest_dir: &Path,
    markers: &str,
    out: &AnnotationOutputs,
) -> Result<()> {
    let rel = |abs: &str| run_manifest::rel_to_manifest(manifest_dir, abs);
    manifest.annotate.argmax = out.argmax.as_deref().map(&rel);
    manifest.annotate.markers = Some(markers.to_string());
    // Always assign (including None) so a projection re-run clears enrichment
    // paths left by an earlier annotate-by-enrichment, matching the ontology
    // clear semantics below.
    manifest.annotate.annotation = out.annotation.as_deref().map(&rel);
    manifest.annotate.cluster_celltype_q = out.cluster_celltype_q.as_deref().map(&rel);
    manifest.annotate.cluster_celltype_es = out.cluster_celltype_es.as_deref().map(&rel);
    manifest.annotate.cluster_expression = out.cluster_expression.as_deref().map(&rel);
    // Overwrite (not conditionally set) so a re-run without ontology clears any
    // stale pointers from a previous standalone `annotate-ontology`.
    manifest.annotate.ontology_assignment = out.ontology_assignment.as_deref().map(&rel);
    manifest.annotate.ontology_node_mass = out.ontology_node_mass.as_deref().map(&rel);
    manifest.defaults.colour_by = Some("annotation".into());
    manifest.save(from)
}

/// The GO/GMT gene-set pass records a per-cluster signature, not cell types:
/// no `argmax`, no `colour_by` flip, and the CL ontology pointers are left
/// alone because this pass never had anything to say about them.
fn record_gene_set_signature(
    manifest: &mut RunManifest,
    from: &Path,
    manifest_dir: &Path,
    out: &AnnotationOutputs,
) -> Result<()> {
    let rel = |abs: &str| run_manifest::rel_to_manifest(manifest_dir, abs);
    manifest.annotate.cluster_expression = out.cluster_expression.as_deref().map(&rel);
    manifest.annotate.ontology_signature = out.ontology_signature.as_deref().map(&rel);
    manifest.annotate.ontology_term_effect = out.ontology_term_effect.as_deref().map(&rel);
    manifest.save(from)
}

/////////////////////////////////////
// manifest → enrichment inputs    //
/////////////////////////////////////

fn resolve(manifest_dir: &Path, rel: &str) -> String {
    run_manifest::resolve(manifest_dir, rel)
        .to_string_lossy()
        .into_owned()
}

/// Re-open the raw counts, resolve the clustering, parse the marker TSV, and
/// aggregate the NB-Fisher-weighted cluster (and, for the marker path,
/// per-batch) expression the enrichment pass scores.
fn load_enrichment_inputs(
    args: &AnnotateArgs,
    plan: &EnrichmentPlan,
) -> Result<(EnrichmentInputs, RunManifest, PathBuf)> {
    let (manifest, manifest_dir) = RunManifest::load(Path::new(args.from.as_ref()))?;
    info!("Loaded manifest ({}): kind={}", args.from, manifest.kind);

    anyhow::ensure!(
        !manifest.data.input.is_empty(),
        "manifest.data.input is empty; cannot re-open raw counts for cluster aggregation"
    );

    let data_files: Vec<Box<str>> = manifest
        .data
        .input
        .iter()
        .map(|s| resolve(&manifest_dir, s).into_boxed_str())
        .collect();
    let batch_files: Option<Vec<Box<str>>> = if manifest.data.batch.is_empty() {
        None
    } else {
        Some(
            manifest
                .data
                .batch
                .iter()
                .map(|s| resolve(&manifest_dir, s).into_boxed_str())
                .collect(),
        )
    };

    info!("Re-opening raw counts: {} file(s)", data_files.len());
    let stack = read_data_on_shared_columns(ReadSharedColumnsArgs {
        data_files,
        batch_files,
        num_types: 1,
        preload: args.preload_data,
        // Annotation maps onto an existing cell set — never drop cells.
        qc: None,
        qc_block_size: None,
        qc_report_out: None,
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
    info!("Raw counts: {n_genes} genes × {n_cells} cells");

    let (cluster_labels, n_clusters) = resolve_clusters(
        args,
        &manifest,
        &manifest_dir,
        &cell_names,
        &LeidenArgs {
            knn: args.knn,
            resolution: args.resolution,
            num_clusters: args.num_clusters,
            min_cluster_size: args.min_cluster_size,
            seed: args.cluster_seed,
        },
    )?;
    anyhow::ensure!(
        n_clusters >= 2,
        "annotate needs ≥ 2 clusters, found {n_clusters}"
    );

    // Build per-cell batch ids (u32) for permutation null.
    let (batch_labels, n_batches) = build_batch_labels(&stack, n_cells)?;
    info!("Batches: {n_batches}");

    // Markers aligned to data row order. Optional: GO/GMT ontology mode supplies
    // gene-sets instead of a curated marker TSV, so an empty path yields an empty
    // marker matrix (the marker-enrichment path is skipped by the caller).
    let (markers_gc, celltype_names) = if args.markers.is_empty() {
        info!("No marker TSV (ontology gene-set mode); skipping marker matrix");
        (Mat::zeros(gene_names.len(), 0), Vec::new())
    } else {
        let annot = build_annotation_matrix(&args.markers, &gene_names)?;
        info!(
            "Marker matrix: {} genes × {} celltypes",
            annot.membership_ga.nrows(),
            annot.membership_ga.ncols()
        );
        (annot.membership_ga, annot.annot_names)
    };

    let nb_fisher = nb_fisher_weights(args, &manifest, &manifest_dir, data_vec, &gene_names)?;
    let (profile_gk, pb_gene_gp) = aggregate_expression(
        args,
        plan,
        data_vec,
        &cluster_labels,
        n_clusters,
        &batch_labels,
        n_batches,
        gene_names.len(),
        &nb_fisher,
    )?;

    Ok((
        EnrichmentInputs {
            gene_names,
            cell_names,
            cluster_labels,
            n_clusters,
            batch_labels,
            n_batches,
            markers_gc,
            celltype_names,
            profile_gk,
            pb_gene_gp,
        },
        manifest,
        manifest_dir,
    ))
}

/// Cluster source priority:
///   1. `--clusters <path>` — user-explicit, hard-fail on missing.
///   2. `manifest.cluster.clusters` if it exists on disk — softly
///      fall back to internal Leiden when stale / relocated, since
///      we can re-cluster from the same latent the manifest points at.
///   3. Internal Leiden on `manifest.outputs.latent`.
fn resolve_clusters(
    args: &AnnotateArgs,
    manifest: &RunManifest,
    manifest_dir: &Path,
    cell_names: &[Box<str>],
    leiden_args: &LeidenArgs,
) -> Result<(Vec<usize>, usize)> {
    let manifest_cluster_path = manifest
        .cluster
        .clusters
        .as_deref()
        .map(|rel| resolve(manifest_dir, rel));
    let resolved_path = args.clusters.as_deref().map(String::from).or_else(|| {
        manifest_cluster_path.filter(|p| {
            let exists = Path::new(p).is_file();
            if !exists {
                log::warn!(
                    "Cluster parquet {p} not found — falling back to internal Leiden \
                         on the manifest's latent."
                );
            }
            exists
        })
    });
    if let Some(path) = resolved_path {
        info!("Resolving cluster source: parquet {path}");
        return load_cluster_labels(&path, cell_names)
            .with_context(|| format!("failed to load cluster parquet {path}"));
    }
    info!(
        "Resolving cluster source: internal Leiden on manifest's cell embedding ({:?})",
        manifest.outputs.geometry_latent()
    );
    let resolve_rel = |rel: &str| resolve(manifest_dir, rel);
    compute_clusters_from_latent(manifest, &resolve_rel, cell_names, leiden_args)
}

/// Per-gene NB-Fisher weights: the cached parquet from training first, falling
/// back to recomputing them off the counts.
fn nb_fisher_weights(
    args: &AnnotateArgs,
    manifest: &RunManifest,
    manifest_dir: &Path,
    data_vec: &data_beans::sparse_io_vector::SparseIoVec,
    gene_names: &[Box<str>],
) -> Result<Vec<f32>> {
    use data_beans_alg::gene_weighting::{compute_nb_fisher_weights, load_fisher_weights};

    let fisher_prefix = resolve(manifest_dir, &manifest.prefix);
    let nb_fisher: Vec<f32> = match load_fisher_weights(&fisher_prefix)? {
        Some((cached_genes, cached_w)) if cached_genes == gene_names => {
            info!(
                "Loaded {} NB-Fisher weights from {fisher_prefix}.fisher_weights.parquet",
                cached_w.len()
            );
            cached_w
        }
        Some((cached_genes, _)) => {
            info!(
                "Cached fisher_weights gene names ({}) don't match data ({}); recomputing",
                cached_genes.len(),
                gene_names.len()
            );
            compute_nb_fisher_weights(data_vec, Some(args.block_size))?
        }
        None => compute_nb_fisher_weights(data_vec, Some(args.block_size))?,
    };
    let (w_min, w_max, w_sum) = nb_fisher.par_iter().map(|&w| (w, w, w)).reduce(
        || (f32::INFINITY, 0.0f32, 0.0f32),
        |(lo, hi, s), (a, b, c)| (lo.min(a), hi.max(b), s + c),
    );
    info!(
        "NB-Fisher weights: min={:.4}, max={:.4}, mean={:.4}",
        w_min,
        w_max,
        w_sum / nb_fisher.len() as f32
    );
    Ok(nb_fisher)
}

/// One fused sweep over the counts for the per-cluster gene sums, plus the
/// per-batch axis the marker path's sample-permutation null needs. The GO/GMT
/// ontology path scores the per-cluster profile directly, so it asks for only
/// the cluster sums.
#[allow(clippy::too_many_arguments)]
fn aggregate_expression(
    args: &AnnotateArgs,
    plan: &EnrichmentPlan,
    data_vec: &data_beans::sparse_io_vector::SparseIoVec,
    cluster_labels: &[usize],
    n_clusters: usize,
    batch_labels: &[u32],
    n_batches: usize,
    g: usize,
    nb_fisher: &[f32],
) -> Result<(Mat, Option<Mat>)> {
    if plan.ontology_mode {
        let gene_sum_kg =
            accumulate_gene_sum(data_vec, cluster_labels, n_clusters, g, args.block_size)?;
        // μ[g, c] = w_NBF[g] · (Σ counts[g, n ∈ c]) / size_sum[c]; Simplex
        // specificity downstream supplies the cross-cluster housekeeping
        // suppression.
        return Ok((
            weighted_mean_profile(&gene_sum_kg, n_clusters, g, nb_fisher),
            None,
        ));
    }

    let batch_labels_usize: Vec<usize> = batch_labels.iter().map(|&b| b as usize).collect();
    let (gene_sum_kg, gene_sum_pg) = accumulate_gene_sum_pair(
        data_vec,
        cluster_labels,
        n_clusters,
        &batch_labels_usize,
        n_batches,
        g,
        args.block_size,
    )?;
    Ok((
        weighted_mean_profile(&gene_sum_kg, n_clusters, g, nb_fisher),
        Some(weighted_mean_profile(&gene_sum_pg, n_batches, g, nb_fisher)),
    ))
}

/// Run Leiden on the manifest's latent matrix. For topic kinds, log-space
/// latents are exponentiated to probabilities first. Aligns labels back to
/// the data column order.
pub fn compute_clusters_from_latent<F>(
    manifest: &RunManifest,
    resolve: &F,
    cell_names: &[Box<str>],
    args: &LeidenArgs,
) -> Result<(Vec<usize>, usize)>
where
    F: Fn(&str) -> String,
{
    let latent_rel = manifest
        .outputs
        .geometry_latent()
        .ok_or_else(|| anyhow!("manifest missing outputs.cell_embedding and outputs.latent"))?;
    let latent_path = resolve(latent_rel);
    let mut latent = Mat::from_parquet_with_row_names(&latent_path, Some(0))
        .with_context(|| format!("failed to load latent {latent_path}"))?;
    info!(
        "Loaded latent {latent_path}: {}×{}",
        latent.mat.nrows(),
        latent.mat.ncols()
    );

    // Gate on the space of the table just READ, not on what `latent.parquet`
    // would hold. A kind's `cell_space()` and its `latent_is_log_simplex()` are
    // separate declarations that do not have to agree, so keying this on
    // `latent_is_log_simplex` risks exponentiating a Euclidean embedding that
    // `geometry_latent` handed back instead of the simplex.
    //
    // This used to sniff `max <= 0.0` instead, which mis-handles masked-vae (raw
    // Gaussian `z`, usually max > 0 so it skipped by luck) and is undefined on an
    // all-NaN latent, where `max()`'s partial ordering decides the branch.
    if manifest.kind.cell_space() == CellSpace::LogSimplex {
        info!(
            "Log-simplex latent ({}); exponentiating to probabilities",
            manifest.kind
        );
        latent.mat = latent.mat.map(f32::exp);
    }

    // An embedding is trained with a dot-product / cosine-style objective, so the
    // kNN graph fed to Leiden should reflect ANGULAR distance. Plain Euclidean on
    // a raw embedding is dominated by the depth axis — `senna gem` documents that
    // about its own output — which is why the choice follows the space rather
    // than a hand-listed pair of kinds.
    let metric = match manifest.kind.cell_space() {
        CellSpace::Embedding => LatentMetric::Cosine,
        CellSpace::LogSimplex | CellSpace::Signed => LatentMetric::ZScoreEuclidean,
    };

    info!(
        "Internal Leiden: knn={}, resolution={:.3}, target_k={:?}, min_cluster_size={}, metric={metric:?}",
        args.knn,
        args.resolution,
        args.num_clusters,
        args.min_cluster_size
    );

    let mut result = leiden_clustering_with_metric(
        &latent.mat,
        args.knn,
        args.resolution,
        args.num_clusters,
        args.seed,
        metric,
    )?;
    if args.min_cluster_size > 1 {
        result.remove_small_clusters(args.min_cluster_size);
    }

    // Align by name. The training pipeline writes latents in data column
    // order, so name-mismatched cases are rare but worth diagnosing
    // loudly — silent misalignment makes every downstream cluster
    // statistic empty.
    let mut idx: HashMap<&str, usize> = HashMap::default();
    idx.reserve(latent.rows.len());
    for (i, name) in latent.rows.iter().enumerate() {
        idx.insert(name.as_ref(), i);
    }
    let mut labels = Vec::with_capacity(cell_names.len());
    let mut missing = 0usize;
    let mut pruned = 0usize;
    for cell in cell_names {
        if let Some(&i) = idx.get(cell.as_ref()) {
            let lab = result.labels[i];
            if lab == usize::MAX {
                pruned += 1;
            }
            labels.push(lab);
        } else {
            labels.push(usize::MAX);
            missing += 1;
        }
    }
    let assigned = cell_names.len() - missing - pruned;
    info!(
        "Cluster alignment: {assigned} assigned, {pruned} pruned by min_cluster_size, \
         {missing} missing from latent (n_clusters={})",
        result.n_clusters
    );
    if missing > 0 {
        let preview_data: Vec<&str> = cell_names
            .iter()
            .take(3)
            .map(std::convert::AsRef::as_ref)
            .collect();
        let preview_latent: Vec<&str> = latent
            .rows
            .iter()
            .take(3)
            .map(std::convert::AsRef::as_ref)
            .collect();
        log::warn!(
            "{missing}/{} cells missing from latent (data examples: {:?}; latent examples: {:?}). \
             Likely cause: latent.parquet was written by a different pipeline run with \
             different barcode formatting (e.g. with vs. without `@<basename>` suffix).",
            cell_names.len(),
            preview_data,
            preview_latent,
        );
    }
    if assigned == 0 {
        anyhow::bail!(
            "Internal Leiden produced 0 assigned cells (missing={missing}, pruned={pruned}). \
             Check that the latent.parquet matches the data backend's cell barcodes."
        );
    }
    Ok((labels, result.n_clusters))
}

/// Build per-cell batch ids as compact u32 indices in [0, `n_batches`).
fn build_batch_labels(stack: &SparseStackWithBatch, n_cells: usize) -> Result<(Vec<u32>, usize)> {
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
