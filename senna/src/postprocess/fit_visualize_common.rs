//! Shared plumbing for `senna visualize {tsne, phate}`:
//! - CLI args shared by both layouts.
//! - Build PBs via the existing batch-corrected collapse pipeline
//!   (`topic::common::load_and_collapse`), collect raw-gene log1p-CPM
//!   features, compute PB-PB cosine similarity, apply tail pruning.
//! - Cheap Nyström cell placement in random-projection space.
//! - Shared output writer.
//!
//! Both layout subcommands delegate here for everything except the
//! actual 2D layout algorithm.

use super::viz_prep::{aggregate_features_by_group, log1p_cpm_pb, select_pb_coverage};
use crate::embed_common::*;
use crate::geometry::cell_layout::project_cells_nystrom;
use crate::geometry::similarity::{
    compute_cosine_similarity, local_scale_similarity, regularize_similarity, threshold_similarity,
};
use crate::topic::common::{
    load_and_collapse, preferred_posterior_mean, LoadCollapseArgs, PreparedData,
};

/// Counts-per-million target used for log1p-CPM PB features.
const CPM_SCALE: f32 = 1e4;

/// Self-loop amount added during similarity regularization to prevent
/// isolated nodes from collapsing the layout.
const SELF_LOOP_REG: f32 = 0.01;

/// Args shared by `senna visualize tsne` and `senna visualize phate`.
///
/// These drive the PB construction / batch-correction / similarity /
/// cell-placement path. Layout-specific args live on the per-command
/// structs and are not duplicated here.
#[derive(Args, Debug, Clone)]
pub struct VisualizeCommonArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided (space or comma separated)."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header for results.\n\n\
		     {out}.pb_coords.parquet:\n\
		       Pseudobulk sample coordinates (n_pb × 2)\n\n\
		     {out}.cell_coords.parquet:\n\
		       Cell coordinates (n_cells × 2 or 3) with optional pb_id / cluster columns\n\n\
		     {out}.pb_gene_mean.parquet:\n\
		       Batch-corrected log1p-CPM per PB (n_pb × n_genes), used as\n\
		       the diagnostic feature matrix for the PB-PB similarity step."
    )]
    pub out: Box<str>,

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
        default_value_t = 10_000,
        help = "Cell block size for projection"
    )]
    pub block_size: usize,

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
}

/// PHATE diffusion-embedding args. Shared by `visualize phate` (main
/// layout) and `visualize tsne` (PHATE-based t-SNE initialization).
#[derive(Args, Debug, Clone)]
pub struct PhateCliArgs {
    #[arg(long, default_value_t = 20, help = "PHATE diffusion time t")]
    pub phate_t: usize,

    #[arg(long, default_value_t = 5, help = "PHATE adaptive-bandwidth kNN")]
    pub phate_knn: usize,

    #[arg(
        long,
        default_value_t = 10.0,
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

/// Everything the layout subcommands need after PB prep.
pub(crate) struct VizPrep {
    pub data_vec: SparseIoVec,
    pub pb_size: Vec<usize>,
    pub pb_membership_kept: Vec<usize>,
    /// `(n_pb × D)` log1p-CPM PB features in the row-per-PB convention
    /// expected by PHATE and parquet output.
    pub log_cpm_pb: Mat,
    /// `(n_pb × n_pb)` PB-PB similarity, post threshold / local scaling
    /// / diagonal regularization.
    pub pb_similarity: Mat,
    /// `(proj_dim × n_cells)` per-cell projection features, kept
    /// column-major so each cell is a contiguous slice (SIMD-friendly).
    pub cell_proj_kn: Mat,
    /// `(proj_dim × n_pb)` PB centroid features, matched to `log_cpm_pb` order.
    pub pb_proj_kp: Mat,
}

pub(crate) fn prepare_viz(args: &VisualizeCommonArgs) -> anyhow::Result<VizPrep> {
    // 1. Load + project + multi-level collapse (shared w/ topic / svd).
    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
    } = load_and_collapse(&LoadCollapseArgs {
        data_files: &args.data_files,
        batch_files: &args.batch_files,
        preload: args.preload_data,
        warm_start_proj_file: None,
        proj_dim: args.proj_dim,
        sort_dim: args.sort_dim,
        knn_cells: args.knn_cells,
        num_levels: args.num_levels.max(1),
        iter_opt: args.iter_opt,
        block_size: args.block_size,
        out: &args.out,
        oversample: false,
    })?;

    let finest = collapsed_levels
        .last()
        .ok_or_else(|| anyhow::anyhow!("no collapsed levels produced"))?;
    let mu_dp = preferred_posterior_mean(finest);

    // 2. Raw-gene-space log1p-CPM PB features (diagnostic signal).
    //    Build in (D × n_pb) column-major so writes are cache-friendly
    //    and `compute_cosine_similarity` (which wants columns) can be
    //    called directly later with no transpose.
    let log_cpm_full_dp = log1p_cpm_pb(mu_dp, CPM_SCALE);
    let n_pb_full = log_cpm_full_dp.ncols();
    info!(
        "Built log1p-CPM PB features: {} PBs × {} genes",
        n_pb_full,
        log_cpm_full_dp.nrows()
    );

    // 3. Coverage-based tail pruning.
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

    let log_cpm_dp = log_cpm_full_dp.select_columns(kept_indices.iter());
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

    // 4. PB centroids in proj space for cheap Nyström.
    let pb_centroids_kn =
        aggregate_features_by_group(&proj_kn, &pb_membership_kept, kept_indices.len());

    // 5. PB-PB cosine similarity (in log1p-CPM gene space). PBs are
    //    columns of `log_cpm_dp`, which is exactly what
    //    `compute_cosine_similarity` expects.
    let sim = compute_cosine_similarity(&log_cpm_dp);
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

    Ok(VizPrep {
        data_vec,
        pb_size,
        pb_membership_kept,
        log_cpm_pb: log_cpm_dp.transpose(),
        pb_similarity,
        cell_proj_kn: proj_kn,
        pb_proj_kp: pb_centroids_kn,
    })
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

/// Finalize: place cells via cheap Nyström, write the three output parquet files.
///
/// `pb_coords` comes from the layout subcommand.
pub(crate) fn finalize_viz(
    args: &VisualizeCommonArgs,
    prep: &VizPrep,
    pb_coords: &Mat,
) -> anyhow::Result<()> {
    let cell_coords = project_cells_nystrom(
        &prep.cell_proj_kn,
        &prep.pb_proj_kp,
        pb_coords,
        args.knn,
        args.kernel_alpha,
    );
    info!("Projected cells to 2D via proj-space Nyström");

    let pb_names: Vec<Box<str>> = (0..pb_coords.nrows())
        .map(|i| format!("PB_{i}").into_boxed_str())
        .collect();
    let gene_names = prep.data_vec.row_names()?;
    let cell_names = prep.data_vec.column_names()?;

    let coord_cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    pb_coords.to_parquet_with_names(
        &(args.out.to_string() + ".pb_coords.parquet"),
        (Some(&pb_names), Some("pb")),
        Some(&coord_cols),
    )?;

    prep.log_cpm_pb.to_parquet_with_names(
        &(args.out.to_string() + ".pb_gene_mean.parquet"),
        (Some(&pb_names), Some("pb")),
        Some(&gene_names),
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
        &(args.out.to_string() + ".cell_coords.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&col_names),
    )?;

    info!(
        "Saved {}.pb_coords.parquet, {}.pb_gene_mean.parquet, {}.cell_coords.parquet",
        args.out, args.out, args.out
    );
    Ok(())
}
