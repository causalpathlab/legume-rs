use super::phate::{phate_layout_2d, PhateArgs};
use super::viz_cell_layout::{project_cells_nystrom, refine_cells_local, LocalRefineArgs};
use super::viz_prep::{
    compute_pb_latent, compute_whitening, load_dictionary, load_latent_file, select_pb_coverage,
    whiten_pb_features,
};
use super::viz_similarity::{
    compute_cosine_similarity, local_scale_similarity, regularize_similarity, threshold_similarity,
};
use crate::embed_common::*;
use crate::senna_input::*;

#[derive(ValueEnum, Clone, Debug, Default, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum LayoutMethod {
    /// t-SNE (Rtsne-style update rule, PHATE-initialized) in reconstruction
    /// space.
    #[default]
    Tsne,
    /// PHATE diffusion embedding — trajectory / branch structure via
    /// heat-diffusion potential distances + metric MDS.
    Phate,
}

#[derive(Args, Debug)]
pub struct VisualizeArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided (space or comma separated)."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header for results.\n\
		     Generates the following files:\n\n\
		     {out}.pb_coords.parquet:\n\
		       Pseudobulk sample coordinates (n_pb × 2)\n\
		       Columns: row (PB ID), x, y\n\n\
		     {out}.cell_coords.parquet:\n\
		       Cell coordinates (n_cells × 2 or 3)\n\
		       Columns: row (cell ID), x, y, [cluster]\n\
		       Cluster column included if --clusters is provided"
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'l',
        required = true,
        help = "Latent file (cells × K)",
        long_help = "Latent topic proportions or SVD projection (cells × K matrix).\n\
		     Used for partitioning cells into pseudobulk groups\n\
		     and for computing cell-to-PB similarity.\n\n\
		     Expected format from `senna topic`:\n\
		     - .latent.parquet: softmax probabilities in [0,1], rows sum to 1\n\
		     - First column: cell names (must match data files)\n\n\
		     Cell coordinates are computed by:\n\
		     1. Finding k nearest PB samples based on latent similarity\n\
		     2. Weighted average of PB positions using temperature-scaled softmax"
    )]
    latent: Box<str>,

    #[arg(
        long,
        required = true,
        help = "Dictionary file (genes × K)",
        long_help = "Dictionary matrix β (genes × K) from `senna topic` or `senna svd`.\n\
			     PB features are whitened so distances in the K-dim latent\n\
			     space equal reconstruction distances in gene space:\n\
			     z = chol(β^T β) · μ̄_pb.\n\n\
			     Redundant dictionary atoms (near-identical columns) collapse\n\
			     through the Cholesky factor, so similarity is invariant to\n\
			     redundancy. Works for both topic proportions (simplex latent)\n\
			     and Gaussian latents (SVD, autoencoder).\n\n\
			     Expected file: {prefix}.dictionary.parquet"
    )]
    dictionary: Box<str>,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components for partitioning",
        long_help = "Use top {d} components for partitioning cells into PB groups.\n\
		     Number of PB samples will be less than `2^{d}+1`.\n\n\
		     Tuning:\n\
		     - Increase (11-12) for finer-grained PB partitioning\n\
		     - Decrease (8-9) if you have fewer distinct populations\n\
		     - Typical values: 8-12"
    )]
    sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file."
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = false, help = "Preload all columns data")]
    preload_data: bool,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Similarity threshold for graph edges",
        long_help = "Threshold for similarity values.\n\
		     Edges with similarity below this are set to zero.\n\
		     Default 0.0 means no thresholding."
    )]
    similarity_threshold: f32,

    #[arg(
        long,
        help = "Local scaling using k-th neighbor distance",
        long_help = "Apply local scaling (Zelnik-Manor & Perona) to similarity matrix.\n\
		     For each point, compute local scale σ_i = distance to k-th nearest neighbor.\n\
		     Then scale: S_scaled(i,j) = S(i,j) / sqrt(σ_i × σ_j).\n\
		     This spreads out dense regions and compresses sparse ones.\n\
		     Typical values: 5-10. If not set, no local scaling is applied."
    )]
    local_scale_k: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Softmax temperature for cell projection",
        long_help = "Temperature for softmax weighting when projecting cells.\n\
		     Lower values make assignments sharper (closer to nearest PB).\n\
		     Higher values spread cells more evenly.\n\n\
		     Tuning:\n\
		     - 0.01-0.1: sharp assignments, cells cluster tightly around PBs\n\
		     - 0.5-2.0: smoother, cells spread between PBs\n\
		     - If visualization too concentrated, try increasing to 0.5"
    )]
    temperature: f32,

    #[arg(
        long,
        short = 'k',
        default_value_t = 15,
        help = "Number of nearest PB neighbors for cell projection",
        long_help = "For each cell, find k nearest PB samples based on latent similarity.\n\
		     Cell position is weighted average of these k PB positions.\n\
		     Uses HNSW index for fast lookup.\n\n\
		     Tuning:\n\
		     - Increase if visualization is too concentrated\n\
		     - Decrease if visualization is too diffuse\n\
		     - Typical values: 10-30"
    )]
    knn: usize,

    #[arg(
        long,
        value_enum,
        default_value = "tsne",
        help = "Layout method for PB samples",
        long_help = "Layout algorithm for positioning PB samples:\n\n\
		     tsne (default):\n\
		       t-SNE with an Rtsne-aligned update rule (adaptive gains,\n\
		       momentum switch, strong early exaggeration, init rescaled\n\
		       to std = 1e-4). Initialized from a PHATE embedding of the\n\
		       whitened PB features so the global trajectory backbone is\n\
		       preserved while local clusters sharpen.\n\n\
		     phate:\n\
		       Pure PHATE diffusion embedding (Moon et al. 2019). Alpha-\n\
		       decay kernel over whitened reconstruction features, heat\n\
		       diffusion in time, then metric MDS (SMACOF). Reveals\n\
		       continuous trajectories and branching structure without\n\
		       the extra t-SNE cluster sharpening."
    )]
    layout: LayoutMethod,

    #[arg(
        long,
        default_value_t = 30.0,
        help = "Perplexity for t-SNE",
        long_help = "Perplexity parameter for t-SNE (related to number of neighbors).\n\
		     Typical values: 5-50. Larger values consider more neighbors."
    )]
    perplexity: f32,

    #[arg(long, default_value_t = 1000, help = "Number of iterations for t-SNE")]
    tsne_iter: usize,

    #[arg(
        long,
        default_value_t = 0.95,
        help = "Keep the smallest set of PBs that covers this fraction of cells",
        long_help = "Drop PB samples in the long tail until the remaining set\n\
		     contains at least this fraction of the total cells. Removes\n\
		     outlier singletons / near-isolated PBs that would otherwise\n\
		     distort spectral / MDS / diffusion layouts.\n\n\
		     Cells whose dominant PB is dropped still appear in the\n\
		     output — they are re-projected via the Nyström kernel over\n\
		     the surviving PBs.\n\n\
		     - 1.0: keep every PB (no filtering)\n\
		     - 0.99: drop the tail holding 1 % of cells\n\
		     - 0.95 (default): more aggressive, removes minor subpopulations"
    )]
    pb_coverage: f32,

    #[arg(
        long,
        default_value_t = 30,
        help = "Local cell-refinement iterations (parallel kNN fine-tune). 0 = off",
        long_help = "After the global PB layout and Nyström cell initialization,\n\
		     run a local t-SNE-style fine-tune per PB in parallel. For\n\
		     each PB, its cells are refined against cells in the K=5\n\
		     nearest PBs (in the 2D layout) as fixed context anchors, so\n\
		     within-PB and across-PB-boundary structure both sharpen.\n\n\
		     Set to 0 to skip refinement and keep pure Nyström positions."
    )]
    local_iter: usize,

    #[arg(
        long,
        default_value_t = 20.0,
        help = "Learning rate for the local MST-DFS refinement step",
        long_help = "Step size for the manual t-SNE gradient descent inside\n\
		     local cell refinement. Lower than the standard t-SNE rate\n\
		     (200) because each local batch is small (a few hundred cells)\n\
		     and runs for few iterations; large LR otherwise exaggerates\n\
		     outliers, especially on tight t-SNE clusters.\n\n\
		     Typical values: 5–50."
    )]
    local_lr: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "densMAP density-preservation strength (λ) for t-SNE",
        long_help = "Strength of the density-preservation auxiliary loss added\n\
		     to t-SNE (Narayan, Berger, Cho, Nat Biotechnol 2021). For each\n\
		     point, pins the soft low-D local radius R_i = Σ_j P_ij · d_ij(Y)\n\
		     to ∝ 1/√n_i so dense regions stay dense and sparse regions\n\
		     spread out.\n\n\
		     - 0 (default, disabled): pure weighted t-SNE\n\
		     - 0.1–0.5 (mild): subtle density emphasis, safe on t-SNE\n\
		     - >1: aggressive — often dominates the KL and distorts the layout"
    )]
    tsne_density_lambda: f32,

    #[arg(
        long,
        default_value_t = 20,
        help = "PHATE diffusion time t (powers the diffusion operator P^t)",
        long_help = "Number of diffusion steps for PHATE.\n\
		     Larger t smooths more aggressively and exposes global structure;\n\
		     smaller t preserves finer local structure. Typical values: 10-40."
    )]
    phate_t: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "PHATE adaptive-bandwidth neighbor index",
        long_help = "σᵢ = distance from PB i to its knn-th nearest neighbor, used\n\
		     as the local kernel bandwidth. Typical values: 5-15."
    )]
    phate_knn: usize,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "PHATE alpha-decay kernel exponent",
        long_help = "Exponent α in the alpha-decay kernel\n\
		     K[i,j] ∝ exp(-(d/σ)^α).\n\
		     α = 2 gives a Gaussian kernel; α ≥ 10 gives sharper, more\n\
		     locality-preserving affinities (the PHATE default is 10)."
    )]
    phate_alpha: f32,

    #[arg(
        long,
        default_value_t = 300,
        help = "Max SMACOF iterations for PHATE metric MDS"
    )]
    phate_mds_iter: usize,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "Relative-stress tolerance for SMACOF early exit"
    )]
    phate_mds_tol: f32,

    #[arg(
        long,
        help = "Cluster assignments file (from `senna clustering`)",
        long_help = "Optional cluster assignments parquet file.\n\
		     If provided, cluster labels are added to the output cell coordinates.\n\n\
		     Expected format (from `senna clustering`):\n\
		     - {prefix}.clusters.parquet: cell × 1 matrix with cluster IDs\n\
		     - First column: cell names (must match latent file)\n\n\
		     To generate clusters:\n\
		     senna clustering -l latent.parquet -k 10 -o clusters"
    )]
    clusters: Option<Box<str>>,
}

pub fn fit_visualize(args: &VisualizeArgs) -> anyhow::Result<()> {
    // 1. Read the data
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: _batch_membership,
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;

    // 2. Load latent file
    let latent_nk = load_latent_file(&args.latent, &data_vec)?;
    let proj_kn = latent_nk.transpose();

    info!(
        "Loaded latent: {} cells × {} dimensions",
        latent_nk.nrows(),
        latent_nk.ncols()
    );

    // 3. Partition cells into PB groups
    let nsamp = data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), None)?;
    info!("Partitioned cells into {} pseudobulk samples", nsamp);

    // 4. Load dictionary β and compute whitening L such that L L^T = β^T β.
    //    Euclidean on z-space = reconstruction distance in gene space.
    let beta = load_dictionary(&args.dictionary, latent_nk.ncols())?;
    let l_kk = compute_whitening(&beta);

    // 5. Average latent per PB; whiten every cell into reconstruction space.
    let pb_latent_mean_full = compute_pb_latent(&latent_nk, &data_vec)?;
    let cell_z = &latent_nk * l_kk.transpose();

    // 5b. Coverage-based tail pruning. Drop PBs in the long tail until the
    //     surviving set still covers `pb_coverage` of all cells. Orphaned
    //     cells re-project through the Nyström kernel onto the kept PBs.
    let n_pb_full = pb_latent_mean_full.nrows();
    let membership_full = data_vec.get_group_membership(0..latent_nk.nrows())?;
    let pb_size_full: Vec<usize> = {
        let mut counts = vec![0usize; n_pb_full];
        for &g in &membership_full {
            if g < n_pb_full {
                counts[g] += 1;
            }
        }
        counts
    };
    let kept_indices: Vec<usize> = select_pb_coverage(&pb_size_full, args.pb_coverage);
    let dropped = n_pb_full - kept_indices.len();
    let covered: usize = kept_indices.iter().map(|&i| pb_size_full[i]).sum();
    let total_cells: usize = pb_size_full.iter().sum();
    info!(
        "Coverage filter: kept {} / {} PBs covering {} / {} cells ({:.1}%); dropped {}",
        kept_indices.len(),
        n_pb_full,
        covered,
        total_cells,
        100.0 * covered as f32 / total_cells.max(1) as f32,
        dropped
    );

    let pb_latent_mean = pb_latent_mean_full.select_rows(kept_indices.iter());
    let pb_size: Vec<usize> = kept_indices.iter().map(|&i| pb_size_full[i]).collect();
    let pb_z = whiten_pb_features(&pb_latent_mean, &l_kk);

    // Map each cell's original PB index → new index in the kept set.
    // Orphan cells (dominant PB was dropped) get `usize::MAX` and are skipped
    // by the local refinement step.
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

    // 6. Compute PB-PB similarity in reconstruction space (cosine on rows).
    let similarity_pp = compute_cosine_similarity(&pb_z.transpose());
    info!("Computed PB-PB similarity matrix in reconstruction space");

    // 7. Apply threshold if specified
    let similarity_pp = if args.similarity_threshold > 0.0 {
        threshold_similarity(&similarity_pp, args.similarity_threshold)
    } else {
        similarity_pp
    };

    // 7b. Apply local scaling if specified
    let similarity_pp = if let Some(k) = args.local_scale_k {
        let scaled = local_scale_similarity(&similarity_pp, k);
        info!("Applied local scaling with k={}", k);
        scaled
    } else {
        similarity_pp
    };

    // 7c. Regularize similarity: add small self-loops to prevent isolated nodes
    let similarity_pp = regularize_similarity(&similarity_pp, 0.01);

    // 8. K-means clustering on SVD-projected latent
    // Read cluster assignments if provided
    let cell_clusters = if let Some(cluster_file) = &args.clusters {
        info!("Reading cluster assignments from {}...", cluster_file);
        let MatWithNames {
            rows: _cluster_cell_names,
            cols: _,
            mat: cluster_mat,
        } = Mat::from_parquet(cluster_file)?;

        // Verify that cluster file has same number of cells as latent
        if cluster_mat.nrows() != latent_nk.nrows() {
            anyhow::bail!(
                "Cluster file has {} cells but latent has {} cells",
                cluster_mat.nrows(),
                latent_nk.nrows()
            );
        }

        // Extract cluster labels (assume single column with cluster IDs)
        let clusters: Vec<usize> = (0..cluster_mat.nrows())
            .map(|i| cluster_mat[(i, 0)] as usize)
            .collect();

        info!("Loaded {} cluster assignments", clusters.len());
        Some(clusters)
    } else {
        None
    };

    // 10. Compute 2D coordinates for visualization
    //     (reuses pb_latent_mean from step 5 for cell projection below)
    let pb_coords = match args.layout {
        LayoutMethod::Phate => {
            let coords = phate_layout_2d(
                &pb_z,
                &PhateArgs {
                    t: args.phate_t,
                    knn: args.phate_knn,
                    alpha: args.phate_alpha,
                    mds_iter: args.phate_mds_iter,
                    mds_tol: args.phate_mds_tol,
                },
            );
            info!(
                "Computed PHATE embedding (t={}, knn={}, α={}, SMACOF iter≤{})",
                args.phate_t, args.phate_knn, args.phate_alpha, args.phate_mds_iter
            );
            coords
        }
        LayoutMethod::Tsne => {
            use super::visualization_alg::{similarity_to_distance, TSne};

            // Initialize t-SNE from a PHATE embedding of the whitened PB
            // features. PHATE gives a trajectory/branching-aware starting
            // basin — much better global structure than PCA, and cheap at
            // this scale (162 PBs → <100 ms).
            let init = phate_layout_2d(
                &pb_z,
                &PhateArgs {
                    t: args.phate_t,
                    knn: args.phate_knn,
                    alpha: args.phate_alpha,
                    mds_iter: args.phate_mds_iter,
                    mds_tol: args.phate_mds_tol,
                },
            );
            info!("Initialized t-SNE from PHATE embedding on pb_z");

            // nalgebra's DMatrix is column-major, but `TSne::fit` expects a
            // row-major flat Vec of [(x0, y0), (x1, y1), ...]. Pack it
            // explicitly so the init isn't scrambled.
            let n = similarity_pp.nrows();
            let mut init_flat = Vec::with_capacity(n * 2);
            for i in 0..n {
                init_flat.push(init[(i, 0)]);
                init_flat.push(init[(i, 1)]);
            }

            // Similarly, pack `similarity_pp` (column-major DMatrix) into the
            // row-major flat layout `similarity_to_distance` expects.
            let mut sim_flat = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    sim_flat.push(similarity_pp[(i, j)]);
                }
            }
            let distances = similarity_to_distance(&sim_flat, n);

            let tsne = TSne::default()
                .perplexity(args.perplexity)
                .n_iter(args.tsne_iter)
                .weights(&pb_size)
                .density_lambda(args.tsne_density_lambda);
            let result = tsne
                .fit(&distances, n, Some(&init_flat))
                .map_err(|e| anyhow::anyhow!("t-SNE failed: {}", e))?;

            let mut coords = Mat::zeros(n, 2);
            for i in 0..n {
                coords[(i, 0)] = result[i * 2];
                coords[(i, 1)] = result[i * 2 + 1];
            }
            info!(
                "Computed t-SNE for visualization (perplexity={}, iter={})",
                args.perplexity, args.tsne_iter
            );
            coords
        }
    };

    // 12. Project cells to 2D via Nyström: weighted average of PB
    //     coordinates using the same alpha-decay kernel PHATE uses internally.
    let mut cell_coords =
        project_cells_nystrom(&cell_z, &pb_z, &pb_coords, args.phate_knn, args.phate_alpha);

    // 12b. Optional local parallel refinement of cell positions.
    if args.local_iter > 0 {
        refine_cells_local(
            &mut cell_coords,
            &LocalRefineArgs {
                cell_z: &cell_z,
                pb_coords: &pb_coords,
                pb_membership: &pb_membership_kept,
                iters: args.local_iter,
                perplexity: args.perplexity,
                learning_rate: args.local_lr,
            },
        );
        info!(
            "Local parallel refinement: {} iters per PB (lr={})",
            args.local_iter, args.local_lr
        );
    }
    info!("Projected cells to 2D for visualization");

    // 13. Save outputs
    let pb_names: Vec<Box<str>> = (0..pb_coords.nrows())
        .map(|i| format!("PB_{}", i).into_boxed_str())
        .collect();

    let coord_cols = vec![
        "x".to_string().into_boxed_str(),
        "y".to_string().into_boxed_str(),
    ];

    pb_coords.to_parquet_with_names(
        &(args.out.to_string() + ".pb_coords.parquet"),
        (Some(&pb_names), None),
        Some(&coord_cols),
    )?;

    let cell_names = data_vec.column_names()?;

    // Add cluster column if clustering was performed
    if let Some(clusters) = cell_clusters {
        let mut cell_coords_with_cluster = cell_coords.clone();
        let cluster_col: Vec<f32> = clusters.iter().map(|&c| c as f32).collect();
        cell_coords_with_cluster = cell_coords_with_cluster.insert_column(2, 0.0);
        for (i, &c) in cluster_col.iter().enumerate() {
            cell_coords_with_cluster[(i, 2)] = c;
        }
        let coord_cols_with_cluster = vec![
            "x".to_string().into_boxed_str(),
            "y".to_string().into_boxed_str(),
            "cluster".to_string().into_boxed_str(),
        ];
        cell_coords_with_cluster.to_parquet_with_names(
            &(args.out.to_string() + ".cell_coords.parquet"),
            (Some(&cell_names), Some("cell")),
            Some(&coord_cols_with_cluster),
        )?;
    } else {
        cell_coords.to_parquet_with_names(
            &(args.out.to_string() + ".cell_coords.parquet"),
            (Some(&cell_names), Some("cell")),
            Some(&coord_cols),
        )?;
    }

    info!(
        "Saved embeddings to {}.pb_coords.parquet and {}.cell_coords.parquet",
        args.out, args.out
    );

    Ok(())
}
