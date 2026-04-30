//! Shared plumbing for `senna layout {tsne, phate}`:
//! - CLI args shared by both layouts.
//! - Build PBs via the existing batch-corrected collapse pipeline
//!   (`topic::common::load_and_collapse`), collect raw-gene log1p-CPM
//!   features, compute PB-PB cosine similarity, apply tail pruning.
//! - Cheap Nyström cell placement in random-projection space.
//! - Shared output writer.
//!
//! Both layout subcommands delegate here for everything except the
//! actual 2D layout algorithm.

use super::viz_prep::{aggregate_features_by_group, select_pb_coverage};
use crate::embed_common::*;
use crate::geometry::cell_layout::project_cells_nystrom;
use crate::geometry::similarity::{
    compute_cosine_similarity, local_scale_similarity, regularize_similarity, threshold_similarity,
};
use crate::run_manifest::{self, RunManifest};
use crate::senna_input::{read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch};
use crate::topic::common::{
    load_and_collapse, preferred_posterior_log_mean, LoadCollapseArgs, PreparedData,
};
use data_beans_alg::random_projection::binary_sort_columns;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Self-loop amount added during similarity regularization to prevent
/// isolated nodes from collapsing the layout.
const SELF_LOOP_REG: f32 = 0.01;

/// Args shared by `senna layout tsne` and `senna layout phate`.
///
/// These drive the PB construction / batch-correction / similarity /
/// cell-placement path. Layout-specific args live on the per-command
/// structs and are not duplicated here.
#[derive(Args, Debug, Clone)]
pub struct LayoutCommonArgs {
    #[arg(
        long,
        short = 'f',
        help = "Run manifest JSON from `senna topic` / `itopic` / `joint-topic` / `svd` / `joint-svd`",
        long_help = "If set, fills in data files, batch files, and (when --out is \
                     absent) the output prefix from the manifest. The manifest is \
                     then updated in place: its `layout.cell_coords`, `layout.pb_coords`, \
                     and `layout.pb_gene_mean` fields are set to the paths just \
                     written, and saved back. Explicit CLI flags still override \
                     manifest values when passed."
    )]
    pub from: Option<Box<str>>,

    #[arg(
        value_delimiter = ',',
        help = "Data files (required unless --from supplies them)",
        long_help = "Sparse backends in `.zarr` or `.h5`. Multiple paths allowed \
                     (space- or comma-separated). Leave empty and pass --from to \
                     inherit from the manifest."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        help = "Output prefix (defaults to the manifest's `prefix` when --from is used)",
        long_help = "Output header for results.\n\n\
		     {out}.pb_coords.parquet:\n\
		       Pseudobulk sample coordinates (n_pb × 2)\n\n\
		     {out}.cell_coords.parquet:\n\
		       Cell coordinates (n_cells × 2 or 3) with optional pb_id / cluster columns\n\n\
		     {out}.pb_gene_mean.parquet:\n\
		       Batch-corrected log1p-CPM per PB (n_pb × n_genes), used as\n\
		       the diagnostic feature matrix for the PB-PB similarity step."
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file."
    )]
    pub batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = false, help = "Preload all columns data")]
    pub preload_data: bool,

    #[arg(
        long,
        default_value_t = 30,
        help = "Random projection dim used to partition cells into PBs"
    )]
    pub proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} projection bits for PB partitioning (≈ 2^d PBs)"
    )]
    pub sort_dim: usize,

    #[arg(long, default_value_t = DEFAULT_KNN, help = "kNN for super-cell matching")]
    pub knn_cells: usize,

    #[arg(
        long,
        default_value_t = DEFAULT_OPT_ITER,
        help = "Iterations for the Gamma posterior optimizer"
    )]
    pub iter_opt: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Number of hierarchical levels (viz uses the finest only)"
    )]
    pub num_levels: usize,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)"
    )]
    pub block_size: Option<usize>,

    #[arg(
        long = "weighting",
        value_enum,
        default_value_t = crate::refine_weighting::WeightingArg::NbFisherInfo,
        help = crate::refine_weighting::WEIGHTING_HELP,
    )]
    pub refine_weighting: crate::refine_weighting::WeightingArg,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Similarity threshold for graph edges",
        long_help = "Edges with similarity below this are zeroed. Default 0 = no threshold."
    )]
    pub similarity_threshold: f32,

    #[arg(
        long,
        help = "Local scaling using k-th neighbor distance",
        long_help = "Zelnik-Manor & Perona local scaling. σ_i = distance to k-th nearest\n\
		     neighbour; S_scaled(i,j) = S(i,j) / sqrt(σ_i σ_j)."
    )]
    pub local_scale_k: Option<usize>,

    #[arg(
        long,
        short = 'k',
        default_value_t = 15,
        help = "Number of nearest PBs for cell projection"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "Alpha-decay kernel exponent for Nyström cell projection"
    )]
    pub kernel_alpha: f32,

    #[arg(
        long,
        default_value_t = 0.95,
        help = "Keep smallest PB set covering this fraction of cells",
        long_help = "Drop PB samples in the long tail until the remaining set\n\
		     contains at least this fraction of the total cells.\n\
		     - 1.0: keep every PB (no filtering)\n\
		     - 0.95 (default): remove minor subpopulations"
    )]
    pub pb_coverage: f32,

    #[arg(
        long,
        help = "Cluster assignments file (from `senna clustering`)",
        long_help = "Optional cluster assignments parquet file (cell × 1 matrix of IDs).\n\
		     If provided, cluster labels are added to cell_coords."
    )]
    pub clusters: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 42,
        help = "RNG seed for random PB-coordinate initialisation (tsne / mst)"
    )]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Landmark count for latent-driven layout",
        long_help = "Only used when a trained latent is available (topic/svd manifest). \
                     Random subsample of cells used as PB landmarks; each cell is \
                     assigned to its nearest landmark. Lower = fewer, denser PBs \
                     (crisper clusters); higher = finer resolution but more blobby. \
                     Ignored by the projection-space fallback path."
    )]
    pub n_landmarks: usize,
}

/// PHATE diffusion-embedding args. Shared by `layout phate` (main
/// layout) and `layout tsne` (PHATE-based t-SNE initialization).
#[derive(Args, Debug, Clone)]
pub struct PhateCliArgs {
    #[arg(long, default_value_t = 20, help = "PHATE diffusion time t")]
    pub phate_t: usize,

    #[arg(long, default_value_t = 5, help = "PHATE adaptive-bandwidth kNN")]
    pub phate_knn: usize,

    #[arg(
        long,
        default_value_t = 40.0,
        help = "PHATE alpha-decay kernel exponent"
    )]
    pub phate_alpha: f32,

    #[arg(long, default_value_t = 300, help = "Max SMACOF iterations")]
    pub phate_mds_iter: usize,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "SMACOF relative-stress tolerance"
    )]
    pub phate_mds_tol: f32,
}

impl From<&PhateCliArgs> for crate::geometry::phate::PhateArgs {
    fn from(a: &PhateCliArgs) -> Self {
        Self {
            t: a.phate_t,
            knn: a.phate_knn,
            alpha: a.phate_alpha,
            mds_iter: a.phate_mds_iter,
            mds_tol: a.phate_mds_tol,
        }
    }
}

/// Inputs resolved from the merge of CLI flags and an optional
/// `--from` run manifest. Held by `preprocess_layout_data` / `finalize_viz` in
/// place of reaching into `LayoutCommonArgs` directly, which keeps
/// the path-resolution logic out of the pipeline body.
///
/// When `manifest` is `Some`, `finalize_viz` updates its `viz{}`
/// section with the files it just wrote and saves back to
/// `manifest_path`. Without `--from` both stay `None`.
pub(crate) struct ResolvedViz {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub out: String,
    pub manifest_path: Option<PathBuf>,
    pub manifest: Option<RunManifest>,
}

pub(crate) fn resolve_inputs(args: &LayoutCommonArgs) -> anyhow::Result<ResolvedViz> {
    let (manifest, manifest_dir, manifest_path) = match &args.from {
        Some(p) => {
            let path = PathBuf::from(p.as_ref());
            let (m, dir) = RunManifest::load(&path)?;
            info!("Loaded run manifest {} (kind: {})", p, m.kind);
            (Some(m), dir, Some(path))
        }
        None => (None, PathBuf::from("."), None),
    };

    let data_files: Vec<Box<str>> = if !args.data_files.is_empty() {
        args.data_files.clone()
    } else if let Some(m) = manifest.as_ref() {
        m.data
            .input
            .iter()
            .map(|s| {
                run_manifest::resolve(&manifest_dir, s)
                    .to_string_lossy()
                    .into_owned()
                    .into_boxed_str()
            })
            .collect()
    } else {
        anyhow::bail!(
            "no data files given and no --from manifest; \
             pass data files positionally or supply --from PATH"
        );
    };
    if data_files.is_empty() {
        anyhow::bail!("manifest {:?} has no data.input entries", manifest_path);
    }

    let batch_files: Option<Vec<Box<str>>> = args.batch_files.clone().or_else(|| {
        manifest.as_ref().and_then(|m| {
            if m.data.batch.is_empty() {
                None
            } else {
                Some(
                    m.data
                        .batch
                        .iter()
                        .map(|s| {
                            run_manifest::resolve(&manifest_dir, s)
                                .to_string_lossy()
                                .into_owned()
                                .into_boxed_str()
                        })
                        .collect(),
                )
            }
        })
    });

    let out: String = args
        .out
        .as_deref()
        .map(String::from)
        .or_else(|| manifest.as_ref().map(|m| m.prefix.clone()))
        .ok_or_else(|| {
            anyhow::anyhow!("no --out given and no manifest prefix available (use --from or -o)")
        })?;
    mkdir_parent(&out)?;

    Ok(ResolvedViz {
        data_files,
        batch_files,
        out,
        manifest_path,
        manifest,
    })
}

/// Everything the layout subcommands need after PB prep.
pub(crate) struct LayoutPrep {
    pub data_vec: SparseIoVec,
    pub pb_size: Vec<usize>,
    pub pb_membership_kept: Vec<usize>,
    /// `(n_pb × D)` per-PB feature matrix in row-per-PB convention,
    /// used for the diagnostic output parquet. Content depends on
    /// `pb_feature_kind`: log1p-CPM gene-space in the recompute path,
    /// proj-space PB centroids in the cached fast path.
    pub pb_features: Mat,
    /// Either `"gene"` (D = n_genes, rows are log1p-CPM per PB) or
    /// `"proj"` (D = proj_dim, rows are mean-projection per PB).
    /// Controls output filename + column naming in `finalize_viz` and
    /// whether `manifest.layout.pb_gene_mean` is populated.
    pub pb_feature_kind: &'static str,
    /// `(n_pb × n_pb)` PB-PB similarity, post threshold / local scaling
    /// / diagonal regularization.
    pub pb_similarity: Mat,
    /// `(proj_dim × n_cells)` per-cell projection features, kept
    /// column-major so each cell is a contiguous slice (SIMD-friendly).
    pub cell_proj_kn: Mat,
    /// `(proj_dim × n_pb)` PB centroid features, matched to `pb_features` order.
    pub pb_proj_kp: Mat,
}

pub(crate) fn preprocess_layout_data(
    args: &LayoutCommonArgs,
    resolved: &ResolvedViz,
) -> anyhow::Result<LayoutPrep> {
    let resolve_from_manifest = |p: &str| -> String {
        let manifest_path = resolved.manifest_path.as_ref().expect("manifest present");
        let manifest_dir = manifest_path
            .parent()
            .filter(|q| !q.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        run_manifest::resolve(manifest_dir, p)
            .to_string_lossy()
            .into_owned()
    };

    let latent_path: Option<(String, crate::run_manifest::RunKind)> = resolved
        .manifest
        .as_ref()
        .and_then(|m| m.outputs.latent.as_ref().map(|p| (p.clone(), m.kind)))
        .filter(|_| resolved.manifest_path.is_some())
        .map(|(p, kind)| (resolve_from_manifest(&p), kind));

    if let Some((p, kind)) = latent_path {
        info!("Layout latent path: PB + similarity from cached latent {p} (kind={kind})");
        return preprocess_layout_data_from_latent(args, resolved, &p, kind);
    }

    let cell_proj_path: Option<String> = resolved
        .manifest
        .as_ref()
        .and_then(|m| m.outputs.cell_proj.as_ref())
        .filter(|_| resolved.manifest_path.is_some())
        .map(|p| resolve_from_manifest(p));

    match cell_proj_path {
        Some(ref p) => {
            info!("Layout fast path: PB partition from cached projection {p}");
            preprocess_layout_data_from_cache(args, resolved, p)
        }
        None => {
            info!(
                "Layout: no cached latent/cell_proj in manifest; running full load_and_collapse \
                 (slow path, includes batch-correction + Gamma posterior)"
            );
            preprocess_layout_data_recompute(args, resolved)
        }
    }
}

/// Latent-driven layout: topic θ is log-softmax on disk, so we apply
/// Hellinger (`exp().sqrt()`) to make cosine ≡ Bhattacharyya; SVD
/// scores are z-scored instead. PBs come from a random landmark
/// subsample (nearest-landmark assignment via HNSW) and PB-PB
/// similarity uses a UMAP-style fuzzy kNN graph — dense cosine on
/// Hellinger-θ saturates and collapses the layout to rank-1.
fn preprocess_layout_data_from_latent(
    args: &LayoutCommonArgs,
    resolved: &ResolvedViz,
    latent_path: &str,
    kind: crate::run_manifest::RunKind,
) -> anyhow::Result<LayoutPrep> {
    let SparseDataWithBatch { data: data_vec, .. } =
        read_data_on_shared_rows(ReadSharedRowsArgs {
            data_files: resolved.data_files.clone(),
            batch_files: resolved.batch_files.clone(),
            preload: args.preload_data,
        })?;

    let MatWithNames {
        rows: cell_names_cached,
        cols: _,
        mat: latent_nk,
    } = Mat::from_parquet_with_row_names(latent_path, Some(0))?;
    let data_cell_names = data_vec.column_names()?;
    anyhow::ensure!(
        data_cell_names == cell_names_cached,
        "cached latent cells don't match data columns (cache={}, data={})",
        cell_names_cached.len(),
        data_cell_names.len()
    );

    let feat_kn: Mat = if kind.is_topic_family() {
        let mut m = latent_nk.transpose();
        for v in m.iter_mut() {
            *v = v.exp().max(0.0).sqrt();
        }
        m
    } else {
        let mut m = latent_nk.transpose();
        m.scale_rows_inplace();
        m
    };
    let n_cells = feat_kn.ncols();
    info!(
        "Loaded latent: {n_cells} cells × {} dims ({})",
        feat_kn.nrows(),
        if kind.is_topic_family() {
            "Hellinger-θ"
        } else {
            "z-scored scores"
        }
    );

    let n_pb_target = args.n_landmarks.min(n_cells).max(1);
    let landmark_cells: Vec<usize> = {
        let mut rng = SmallRng::seed_from_u64(args.seed);
        let mut picked: Vec<usize> = rand::seq::index::sample(&mut rng, n_cells, n_pb_target)
            .into_iter()
            .collect();
        picked.sort_unstable();
        picked
    };
    let n_pb_full = landmark_cells.len();
    info!(
        "Landmark subsample: picked {n_pb_full} cells as PBs (seed={})",
        args.seed
    );

    let landmark_cols: Vec<nalgebra::DVectorView<f32>> =
        landmark_cells.iter().map(|&c| feat_kn.column(c)).collect();
    let landmark_dict = matrix_util::knn_match::ColumnDict::from_dvector_views(
        landmark_cols,
        (0..n_pb_full).collect(),
    );

    let membership_full: Vec<usize> = (0..n_cells)
        .into_par_iter()
        .map_init(
            || matrix_util::knn_match::VecPoint {
                data: Vec::with_capacity(feat_kn.nrows()),
            },
            |query, c| {
                query.data.clear();
                query.data.extend(feat_kn.column(c).iter().copied());
                match landmark_dict.search_by_query_data(query, 1) {
                    Ok((ids, _)) => ids.first().copied().unwrap_or(usize::MAX),
                    Err(_) => usize::MAX,
                }
            },
        )
        .collect();

    let pb_size_full: Vec<usize> = {
        let mut counts = vec![0usize; n_pb_full];
        for &g in &membership_full {
            if g < n_pb_full {
                counts[g] += 1;
            }
        }
        counts
    };
    let kept_indices = select_pb_coverage(&pb_size_full, args.pb_coverage);
    let covered: usize = kept_indices.iter().map(|&i| pb_size_full[i]).sum();
    let total_cells: usize = pb_size_full.iter().sum();
    info!(
        "Coverage filter: kept {} / {} landmarks covering {} / {} cells ({:.1}%)",
        kept_indices.len(),
        n_pb_full,
        covered,
        total_cells,
        100.0 * covered as f32 / total_cells.max(1) as f32
    );

    let mut old_to_new = vec![usize::MAX; n_pb_full];
    for (new_i, &old_i) in kept_indices.iter().enumerate() {
        old_to_new[old_i] = new_i;
    }
    let pb_membership_kept: Vec<usize> = membership_full
        .iter()
        .map(|&g| {
            if g < n_pb_full {
                old_to_new[g]
            } else {
                usize::MAX
            }
        })
        .collect();
    let pb_size: Vec<usize> = kept_indices.iter().map(|&i| pb_size_full[i]).collect();

    let pb_centroids_kp =
        aggregate_features_by_group(&feat_kn, &pb_membership_kept, kept_indices.len());

    let n_pb = pb_centroids_kp.ncols();
    let knn = args.knn.clamp(1, n_pb.saturating_sub(1).max(1));
    info!("Building fuzzy kNN graph on PB centroids: n_pb={n_pb}, knn={knn}");
    let graph = matrix_util::knn_graph::KnnGraph::from_columns(
        &pb_centroids_kp,
        matrix_util::knn_graph::KnnGraphArgs {
            knn,
            block_size: args.block_size.unwrap_or(1000),
            reciprocal: false,
        },
    )?;
    let fuzzy = graph.fuzzy_kernel_weights();

    // √(sᵢ·sⱼ)/mean_size normalises so weighted mean ≈ 1, keeping fuzzy
    // magnitudes comparable to the unweighted case.
    let mean_size = (total_cells as f32) / (n_pb as f32).max(1.0);
    let sqrt_size: Vec<f32> = pb_size.iter().map(|&s| (s as f32).sqrt()).collect();

    let mut sim = Mat::zeros(n_pb, n_pb);
    for (&(i, j), &w) in graph.edges.iter().zip(fuzzy.iter()) {
        let scale = (sqrt_size[i] * sqrt_size[j]) / mean_size.max(1e-6);
        let ws = w * scale;
        sim[(i, j)] = ws;
        sim[(j, i)] = ws;
    }
    let pb_similarity = regularize_similarity(&sim, SELF_LOOP_REG);

    Ok(LayoutPrep {
        data_vec,
        pb_size,
        pb_membership_kept,
        pb_features: pb_centroids_kp.transpose(),
        pb_feature_kind: "proj",
        pb_similarity,
        cell_proj_kn: feat_kn,
        pb_proj_kp: pb_centroids_kp,
    })
}

/// Fast path: given a cached `cell_proj.parquet` from a prior training
/// run, open the sparse backends lightly (no projection, no collapse),
/// partition cells by hashing the cached projection, and build PB
/// features + similarity directly in projection space. Skips super-cell
/// matching and the Gamma posterior optimization that the recompute
/// path runs — those are training-side concerns the layout doesn't need.
fn preprocess_layout_data_from_cache(
    args: &LayoutCommonArgs,
    resolved: &ResolvedViz,
    cell_proj_path: &str,
) -> anyhow::Result<LayoutPrep> {
    // 1. Lightweight data open: no projection, no collapse. Only pays
    //    for barcode / gene-name metadata + any batch annotation, which
    //    we need later for output parquet headers.
    let SparseDataWithBatch { data: data_vec, .. } =
        read_data_on_shared_rows(ReadSharedRowsArgs {
            data_files: resolved.data_files.clone(),
            batch_files: resolved.batch_files.clone(),
            preload: args.preload_data,
        })?;

    // 2. Load the cached projection (written as cells × proj_dim). The
    //    transpose lands us in column-per-cell layout expected by the
    //    rest of the pipeline.
    let MatWithNames {
        rows: cell_names_cached,
        cols: _,
        mat: proj_nk,
    } = Mat::from_parquet_with_row_names(cell_proj_path, Some(0))?;
    let data_cell_names = data_vec.column_names()?;
    anyhow::ensure!(
        data_cell_names == cell_names_cached,
        "cached cell_proj cells don't match data columns (cache={}, data={})",
        cell_names_cached.len(),
        data_cell_names.len()
    );
    let mut proj_kn: Mat = proj_nk.transpose();
    info!(
        "Loaded cell_proj: {} cells × {} proj-dims",
        proj_kn.ncols(),
        proj_kn.nrows()
    );

    // Per-proj-dim z-score across cells so that no single dim dominates
    // the PB-centroid cosine similarity. (scale_rows_inplace standardizes
    // each row to zero-mean/unit-variance across columns.)
    proj_kn.scale_rows_inplace();

    // 3. Partition: binary_sort_columns runs RSVD + sign-hashing on the
    //    projection and returns a per-cell bucket code. Canonicalize
    //    those codes to contiguous PB ids in [0, n_pb). No data_vec
    //    groups needed — the partition is a pure function of proj_kn.
    let n_cells = proj_kn.ncols();
    let kk = args.sort_dim.min(proj_kn.nrows()).min(n_cells);
    let codes = binary_sort_columns(&proj_kn, kk)?;
    let (membership_full, n_pb_full) = canonicalize_codes(&codes);
    info!("Partitioned {} cells into {} PBs", n_cells, n_pb_full);

    // 4. Coverage-prune PBs (same policy as the recompute path).
    let pb_size_full: Vec<usize> = {
        let mut counts = vec![0usize; n_pb_full];
        for &g in &membership_full {
            if g < n_pb_full {
                counts[g] += 1;
            }
        }
        counts
    };
    let kept_indices = select_pb_coverage(&pb_size_full, args.pb_coverage);
    let covered: usize = kept_indices.iter().map(|&i| pb_size_full[i]).sum();
    let total_cells: usize = pb_size_full.iter().sum();
    info!(
        "Coverage filter: kept {} / {} PBs covering {} / {} cells ({:.1}%)",
        kept_indices.len(),
        n_pb_full,
        covered,
        total_cells,
        100.0 * covered as f32 / total_cells.max(1) as f32
    );

    let mut old_to_new = vec![usize::MAX; n_pb_full];
    for (new_i, &old_i) in kept_indices.iter().enumerate() {
        old_to_new[old_i] = new_i;
    }
    let pb_membership_kept: Vec<usize> = membership_full
        .iter()
        .map(|&g| {
            if g < n_pb_full {
                old_to_new[g]
            } else {
                usize::MAX
            }
        })
        .collect();
    let pb_size: Vec<usize> = kept_indices.iter().map(|&i| pb_size_full[i]).collect();

    // 5. PB centroids in proj space (kp = proj_dim × n_pb_kept).
    let pb_centroids_kp =
        aggregate_features_by_group(&proj_kn, &pb_membership_kept, kept_indices.len());

    // 6. PB-PB cosine similarity directly on the proj-space centroids —
    //    columns of `pb_centroids_kp` are the PB vectors, which is the
    //    convention `compute_cosine_similarity` expects.
    let sim = compute_cosine_similarity(&pb_centroids_kp);
    let sim = if args.similarity_threshold > 0.0 {
        threshold_similarity(&sim, args.similarity_threshold)
    } else {
        sim
    };
    let sim = if let Some(k) = args.local_scale_k {
        let scaled = local_scale_similarity(&sim, k);
        info!("Applied local scaling with k={k}");
        scaled
    } else {
        sim
    };
    let pb_similarity = regularize_similarity(&sim, SELF_LOOP_REG);

    Ok(LayoutPrep {
        data_vec,
        pb_size,
        pb_membership_kept,
        pb_features: pb_centroids_kp.transpose(),
        pb_feature_kind: "proj",
        pb_similarity,
        cell_proj_kn: proj_kn,
        pb_proj_kp: pb_centroids_kp,
    })
}

/// Slow path (fallback): the original full pipeline. Runs when no
/// cached projection is available — older manifests, or pure CLI runs
/// without `--from`. Does the full batch-corrected collapse and
/// builds gene-space log1p-CPM PB features.
fn preprocess_layout_data_recompute(
    args: &LayoutCommonArgs,
    resolved: &ResolvedViz,
) -> anyhow::Result<LayoutPrep> {
    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
    } = load_and_collapse(&LoadCollapseArgs {
        data_files: &resolved.data_files,
        batch_files: &resolved.batch_files,
        preload: args.preload_data,
        warm_start_proj_file: None,
        proj_dim: args.proj_dim,
        sort_dim: args.sort_dim,
        knn_cells: args.knn_cells,
        num_levels: args.num_levels.max(1),
        iter_opt: args.iter_opt,
        block_size: args.block_size,
        out: &resolved.out,
        // Layout recompute is the legacy path: match training's default
        // HVG gate so the projection reflects variable biology.
        max_features: 5000,
        feature_list_file: None,
        refine: Some(data_beans_alg::refine_multilevel::RefineParams {
            gene_weighting: args.refine_weighting.into(),
            ..data_beans_alg::refine_multilevel::RefineParams::default()
        }),
    })?;

    let finest = collapsed_levels
        .last()
        .ok_or_else(|| anyhow::anyhow!("no collapsed levels produced"))?;

    let log_expr_full_dp = preferred_posterior_log_mean(finest);
    let n_pb_full = log_expr_full_dp.ncols();
    info!(
        "Built log-expr PB features: {} PBs × {} genes",
        n_pb_full,
        log_expr_full_dp.nrows()
    );

    let n_cells = data_vec.num_columns();
    let membership_full = data_vec.get_group_membership(0..n_cells)?;
    let pb_size_full: Vec<usize> = {
        let mut counts = vec![0usize; n_pb_full];
        for &g in &membership_full {
            if g < n_pb_full {
                counts[g] += 1;
            }
        }
        counts
    };
    let kept_indices = select_pb_coverage(&pb_size_full, args.pb_coverage);
    let covered: usize = kept_indices.iter().map(|&i| pb_size_full[i]).sum();
    let total_cells: usize = pb_size_full.iter().sum();
    info!(
        "Coverage filter: kept {} / {} PBs covering {} / {} cells ({:.1}%)",
        kept_indices.len(),
        n_pb_full,
        covered,
        total_cells,
        100.0 * covered as f32 / total_cells.max(1) as f32
    );

    let log_expr_dp = log_expr_full_dp.select_columns(kept_indices.iter());
    let pb_size: Vec<usize> = kept_indices.iter().map(|&i| pb_size_full[i]).collect();

    let mut old_to_new = vec![usize::MAX; n_pb_full];
    for (new_i, &old_i) in kept_indices.iter().enumerate() {
        old_to_new[old_i] = new_i;
    }
    let pb_membership_kept: Vec<usize> = membership_full
        .iter()
        .map(|&g| {
            if g < n_pb_full {
                old_to_new[g]
            } else {
                usize::MAX
            }
        })
        .collect();

    let pb_centroids_kn =
        aggregate_features_by_group(&proj_kn, &pb_membership_kept, kept_indices.len());

    let sim = compute_cosine_similarity(&log_expr_dp);
    let sim = if args.similarity_threshold > 0.0 {
        threshold_similarity(&sim, args.similarity_threshold)
    } else {
        sim
    };
    let sim = if let Some(k) = args.local_scale_k {
        let scaled = local_scale_similarity(&sim, k);
        info!("Applied local scaling with k={k}");
        scaled
    } else {
        sim
    };
    let pb_similarity = regularize_similarity(&sim, SELF_LOOP_REG);

    Ok(LayoutPrep {
        data_vec,
        pb_size,
        pb_membership_kept,
        pb_features: log_expr_dp.transpose(),
        pb_feature_kind: "gene",
        pb_similarity,
        cell_proj_kn: proj_kn,
        pb_proj_kp: pb_centroids_kn,
    })
}

/// Canonicalize raw bucket codes from `binary_sort_columns` into
/// contiguous PB ids `[0, n_pb)`. Returns `(membership, n_pb)`.
fn canonicalize_codes(codes: &[usize]) -> (Vec<usize>, usize) {
    use std::collections::HashMap;
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut membership = Vec::with_capacity(codes.len());
    for &c in codes {
        let next_id = mapping.len();
        let id = *mapping.entry(c).or_insert(next_id);
        membership.push(id);
    }
    let n = mapping.len();
    (membership, n)
}

/// Load cluster assignments (one-column parquet of cell IDs) and validate length.
pub(crate) fn load_cluster_assignments(
    path: &str,
    n_cells_expected: usize,
) -> anyhow::Result<Vec<usize>> {
    info!("Reading cluster assignments from {path}...");
    let MatWithNames {
        rows: _cluster_cell_names,
        cols: _,
        mat: cluster_mat,
    } = Mat::from_parquet(path)?;
    if cluster_mat.nrows() != n_cells_expected {
        anyhow::bail!(
            "Cluster file has {} cells but data has {} cells",
            cluster_mat.nrows(),
            n_cells_expected
        );
    }
    let clusters: Vec<usize> = (0..cluster_mat.nrows())
        .map(|i| cluster_mat[(i, 0)] as usize)
        .collect();
    info!("Loaded {} cluster assignments", clusters.len());
    Ok(clusters)
}

/// Uniform-random 2D coordinate init for PB-level layouts, in `[-1, 1]²`.
/// Shared by `layout tsne` and `layout mst` so both are reproducible
/// via the common `--seed` flag.
pub(crate) fn random_init_2d(n: usize, seed: u64) -> Mat {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = Mat::zeros(n, 2);
    for i in 0..n {
        out[(i, 0)] = rng.random_range(-1.0..1.0);
        out[(i, 1)] = rng.random_range(-1.0..1.0);
    }
    out
}

/// Cheap Nyström cell placement. Exposed so subcommands that fine-tune
/// cell coords (e.g. `layout umap`) can init from here and hand the
/// refined result to [`write_viz_outputs`] directly.
pub(crate) fn nystrom_cell_coords(
    args: &LayoutCommonArgs,
    prep: &LayoutPrep,
    pb_coords: &Mat,
) -> Mat {
    let coords = project_cells_nystrom(
        &prep.cell_proj_kn,
        &prep.pb_proj_kp,
        pb_coords,
        args.knn,
        args.kernel_alpha,
    );
    info!("Projected cells to 2D via proj-space Nyström");
    coords
}

/// Finalize: place cells via cheap Nyström, write the three output
/// parquet files, and — when a manifest was loaded via `--from` —
/// update its `viz{}` section and save it back in place.
///
/// `pb_coords` comes from the layout subcommand.
pub(crate) fn finalize_viz(
    args: &LayoutCommonArgs,
    resolved: &mut ResolvedViz,
    prep: &LayoutPrep,
    pb_coords: &Mat,
) -> anyhow::Result<()> {
    let cell_coords = nystrom_cell_coords(args, prep, pb_coords);
    write_viz_outputs(args, resolved, prep, pb_coords, &cell_coords)
}

/// Write the three layout parquet files + update manifest. Split out so
/// a caller can supply custom `cell_coords` (e.g. post-refinement).
pub(crate) fn write_viz_outputs(
    args: &LayoutCommonArgs,
    resolved: &mut ResolvedViz,
    prep: &LayoutPrep,
    pb_coords: &Mat,
    cell_coords: &Mat,
) -> anyhow::Result<()> {
    let pb_names: Vec<Box<str>> = (0..pb_coords.nrows())
        .map(|i| format!("PB_{i}").into_boxed_str())
        .collect();
    let cell_names = prep.data_vec.column_names()?;

    let out = &resolved.out;
    let pb_coords_path = format!("{out}.pb_coords.parquet");
    let cell_coords_path = format!("{out}.cell_coords.parquet");

    let coord_cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    pb_coords.to_parquet_with_names(
        &pb_coords_path,
        (Some(&pb_names), Some("pb")),
        Some(&coord_cols),
    )?;

    // PB feature matrix lands in one of two files depending on prep
    // origin: gene-space log1p-CPM (slow path, consumed by
    // `senna annotate`) or proj-space centroids (fast path, diagnostic).
    // Only the gene-space output populates `manifest.layout.pb_gene_mean`.
    let (pb_feat_path, pb_feat_is_gene) = match prep.pb_feature_kind {
        "gene" => (format!("{out}.pb_gene_mean.parquet"), true),
        "proj" => (format!("{out}.pb_proj_mean.parquet"), false),
        other => anyhow::bail!("unexpected pb_feature_kind {other:?}"),
    };
    let feat_col_names: Vec<Box<str>> = if pb_feat_is_gene {
        prep.data_vec.row_names()?
    } else {
        (0..prep.pb_features.ncols())
            .map(|i| format!("p{i}").into_boxed_str())
            .collect()
    };
    prep.pb_features.to_parquet_with_names(
        &pb_feat_path,
        (Some(&pb_names), Some("pb")),
        Some(&feat_col_names),
    )?;

    let cluster_ids = match &args.clusters {
        Some(path) => Some(load_cluster_assignments(
            path,
            prep.pb_membership_kept.len(),
        )?),
        None => None,
    };

    let n_cells = cell_coords.nrows();
    let n_extra = 1 + cluster_ids.as_ref().map_or(0, |_| 1);
    let mut cell_out = Mat::zeros(n_cells, 2 + n_extra);
    for i in 0..n_cells {
        cell_out[(i, 0)] = cell_coords[(i, 0)];
        cell_out[(i, 1)] = cell_coords[(i, 1)];
        let pb_id = prep.pb_membership_kept[i];
        // NaN flags orphan cells (pb dropped by coverage filter); downstream
        // readers can mask with `is_nan` without collision against real IDs.
        cell_out[(i, 2)] = if pb_id == usize::MAX {
            f32::NAN
        } else {
            pb_id as f32
        };
    }
    let mut col_names: Vec<Box<str>> = vec!["x".into(), "y".into(), "pb_id".into()];
    if let Some(ref clusters) = cluster_ids {
        for (i, &c) in clusters.iter().enumerate() {
            cell_out[(i, 3)] = c as f32;
        }
        col_names.push("cluster".into());
    }
    cell_out.to_parquet_with_names(
        &cell_coords_path,
        (Some(&cell_names), Some("cell")),
        Some(&col_names),
    )?;

    info!("Saved {pb_coords_path}, {pb_feat_path}, {cell_coords_path}");

    update_manifest_viz(
        resolved,
        &cell_coords_path,
        &pb_coords_path,
        if pb_feat_is_gene {
            Some(&pb_feat_path)
        } else {
            None
        },
    )?;

    Ok(())
}

/// When `--from` was used, update the loaded manifest's `viz{}` section
/// to point at the files we just wrote, then save it back. Stores each
/// path as a *basename relative to the manifest's directory* so it
/// resolves correctly when the run directory is moved.
fn update_manifest_viz(
    resolved: &mut ResolvedViz,
    cell_coords_path: &str,
    pb_coords_path: &str,
    pb_gene_mean_path: Option<&str>,
) -> anyhow::Result<()> {
    let (Some(manifest), Some(manifest_path)) =
        (resolved.manifest.as_mut(), resolved.manifest_path.as_ref())
    else {
        return Ok(());
    };
    let manifest_dir = manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    manifest.layout.cell_coords = Some(rel_to_manifest(manifest_dir, cell_coords_path));
    manifest.layout.pb_coords = Some(rel_to_manifest(manifest_dir, pb_coords_path));
    // Only the gene-space recompute path produces a proper pb_gene_mean;
    // the fast path writes a proj-space file that `senna annotate`
    // would misread, so don't advertise it.
    manifest.layout.pb_gene_mean = pb_gene_mean_path.map(|p| rel_to_manifest(manifest_dir, p));

    manifest.save(manifest_path)
}

/// Store `written_path` as a basename-relative string when it lives
/// inside `manifest_dir`; otherwise leave as absolute. Keeps the
/// manifest portable across directory moves without accidentally
/// rewriting paths that point elsewhere (e.g. `-o /tmp/foo` while the
/// manifest lives in `~/work/run1/`).
fn rel_to_manifest(manifest_dir: &Path, written_path: &str) -> String {
    let abs = PathBuf::from(written_path);
    let abs = if abs.is_absolute() {
        abs
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&abs))
            .unwrap_or(abs)
    };
    let manifest_abs = manifest_dir
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.to_path_buf());
    let written_abs = abs.canonicalize().unwrap_or(abs);
    match written_abs.strip_prefix(&manifest_abs) {
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_) => written_abs.to_string_lossy().into_owned(),
    }
}
