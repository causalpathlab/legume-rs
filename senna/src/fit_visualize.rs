use crate::embed_common::*;
use crate::senna_input::*;
use matrix_util::knn_match::{ColumnDict, VecPoint};

#[derive(ValueEnum, Clone, Debug, Default, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum LayoutMethod {
    #[default]
    Spectral,
    Tree,
    Tsne,
}

#[derive(Args, Debug)]
pub struct VisualizeArgs {
    #[arg(
        required = true,
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided."
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

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all columns data"
    )]
    preload_data: bool,

    #[arg(long, short, help = "Verbosity")]
    verbose: bool,

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
        default_value_t = 2,
        help = "Number of eigenvectors for spectral embedding",
        long_help = "Number of non-trivial eigenvectors to use.\n\
		     If > 2, eigenvectors are weighted by 1/eigenvalue (diffusion map style)\n\
		     and PCA is applied to reduce to 2D.\n\
		     Default 2 uses 2nd and 3rd smallest eigenvectors directly."
    )]
    num_eigen: usize,

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
        default_value = "spectral",
        help = "Layout method for PB samples",
        long_help = "Layout algorithm for positioning PB samples:\n\n\
		     spectral (default):\n\
		       Best for general-purpose visualization.\n\
		       Uses normalized Laplacian eigenvectors (diffusion map style).\n\
		       Preserves local neighborhood structure.\n\n\
		     tree:\n\
		       Best for hierarchical/developmental trajectories.\n\
		       Creates MST-based radial tree showing branching structure.\n\
		       Root at center, branches spread radially.\n\n\
		     tsne:\n\
		       Best for emphasizing local cluster separation.\n\
		       Initialized with spectral, refined with t-SNE.\n\
		       More computationally expensive."
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

    #[arg(
        long,
        default_value_t = 1000,
        help = "Number of iterations for t-SNE"
    )]
    tsne_iter: usize,

    #[arg(
        long,
        help = "Root node index for tree layout",
        long_help = "Index of the PB sample to use as tree root.\n\
		     If not specified, uses the node with highest total similarity (most central)."
    )]
    tree_root: Option<usize>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Radius scaling for tree layout",
        long_help = "Scale factor for radial distance between tree levels.\n\
		     Larger values spread the tree more."
    )]
    tree_radius: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Decay/growth factor for tree layout",
        long_help = "Factor for radius increment at each level.\n\
		     Level n step = radius × decay^n.\n\
		     - < 1.0: compress outer branches (decay)\n\
		     - > 1.0: expand outer branches (growth)\n\
		     - = 1.0: constant spacing"
    )]
    tree_decay: f32,

    #[arg(
        long,
        help = "Clip outliers beyond ±N standard deviations",
        long_help = "For spectral layout, clip coordinates beyond ±N standard deviations.\n\
		     Helps prevent outlier PB samples from distorting the visualization.\n\
		     Typical values: 2.0 or 3.0. If not set, no clipping is applied."
    )]
    outlier_sd: Option<f32>,

    #[arg(
        long,
        default_value_t = 100,
        help = "Repulsion iterations to spread out points",
        long_help = "After spectral embedding, apply N iterations of repulsion forces\n\
		     to push overlapping points apart. Helps reduce crowding.\n\
		     Typical values: 50-200. Set to 0 to disable."
    )]
    repulsion_iter: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Repulsion strength",
        long_help = "Strength of repulsion force between nearby points.\n\
		     Higher values push points apart more aggressively."
    )]
    repulsion_strength: f32,

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
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // 1. Read the data
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: _batch_membership,
        nbatch: _,
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

    // 4. Collapse to get PB expression (no batch adjustment for simplicity)
    let collapsed = data_vec.collapse_columns(None, None, None, None)?;

    let pb_expr = collapsed
        .mu_adjusted
        .as_ref()
        .unwrap_or(&collapsed.mu_observed);

    let mut pb_log_expr = pb_expr.posterior_log_mean().clone();
    info!(
        "PB expression matrix: {} genes × {} samples",
        pb_log_expr.nrows(),
        pb_log_expr.ncols()
    );

    // 5. Scale columns (center and normalize) before computing similarity
    pb_log_expr.scale_columns_inplace();

    // 6. Compute PB-PB similarity (cosine on scaled log expression)
    let similarity_pp = compute_cosine_similarity(&pb_log_expr);
    info!("Computed PB-PB similarity matrix");

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

    // 9. Compute PB latent for cell projection
    let pb_latent = compute_pb_latent(&latent_nk, &data_vec)?;

    // 10. Compute 2D coordinates for visualization
    let pb_coords = match args.layout {
        LayoutMethod::Spectral => {
            let pb_spectral = spectral_embed(&similarity_pp, args.num_eigen)?;
            let mut coords = reduce_to_2d(&pb_spectral);
            if let Some(sd_threshold) = args.outlier_sd {
                clip_outliers(&mut coords, sd_threshold);
                info!("Clipped outliers beyond ±{} SD", sd_threshold);
            }
            if args.repulsion_iter > 0 {
                apply_repulsion(&mut coords, args.repulsion_iter, args.repulsion_strength);
                info!("Applied {} repulsion iterations", args.repulsion_iter);
            }
            coords
        }
        LayoutMethod::Tree => {
            tree_layout_2d(&similarity_pp, args.tree_root, args.tree_radius, args.tree_decay)?
        }
        LayoutMethod::Tsne => {
            use crate::visualization_alg::{TSne, similarity_to_distance};

            let pb_spectral = spectral_embed(&similarity_pp, args.num_eigen)?;
            let init = reduce_to_2d(&pb_spectral);
            let init_flat: Vec<f32> = init.iter().cloned().collect();

            let n = similarity_pp.nrows();
            let sim_flat: Vec<f32> = similarity_pp.iter().cloned().collect();
            let distances = similarity_to_distance(&sim_flat, n);

            let tsne = TSne::default().perplexity(args.perplexity).n_iter(args.tsne_iter);
            let result = tsne.fit(&distances, n, Some(&init_flat))
                .map_err(|e| anyhow::anyhow!("t-SNE failed: {}", e))?;

            let mut coords = Mat::zeros(n, 2);
            for i in 0..n {
                coords[(i, 0)] = result[i * 2];
                coords[(i, 1)] = result[i * 2 + 1];
            }
            info!("Computed t-SNE for visualization (perplexity={}, iter={})", args.perplexity, args.tsne_iter);
            coords
        }
    };

    // 12. Project cells to 2D for visualization
    let cell_coords = project_cells_to_embedding(&latent_nk, &pb_latent, &pb_coords, args.knn, args.temperature)?;
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

/// Load latent file and validate/reorder to match data columns
fn load_latent_file(path: &str, data_vec: &SparseIoVec) -> anyhow::Result<Mat> {
    use matrix_util::common_io::*;
    use std::collections::HashMap;

    let ext = file_ext(path)?;
    let MatWithNames {
        rows: latent_cells,
        cols: _,
        mat: latent_nk,
    } = match ext.as_ref() {
        "parquet" => Mat::from_parquet_with_row_names(path, Some(0))?,
        _ => Mat::read_data_with_names(path, &['\t', ',', ' '], Some(0), Some(0))?,
    };

    let data_cells = data_vec.column_names()?;

    // Build index map: cell_name -> row index in latent
    let latent_idx: HashMap<&str, usize> = latent_cells
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_ref(), i))
        .collect();

    // Check all data cells exist in latent
    let missing: Vec<_> = data_cells
        .iter()
        .filter(|c| !latent_idx.contains_key(c.as_ref()))
        .take(5)
        .collect();

    if !missing.is_empty() {
        return Err(anyhow::anyhow!(
            "Latent file missing {} cells from data (e.g., {:?})",
            data_cells.iter().filter(|c| !latent_idx.contains_key(c.as_ref())).count(),
            missing
        ));
    }

    // Reorder latent rows to match data column order
    let k = latent_nk.ncols();
    let mut reordered = Mat::zeros(data_cells.len(), k);
    for (i, cell) in data_cells.iter().enumerate() {
        let src_idx = latent_idx[cell.as_ref()];
        reordered.row_mut(i).copy_from(&latent_nk.row(src_idx));
    }

    if latent_cells.len() != data_cells.len() {
        info!(
            "Latent has {} cells, data has {} cells; using {} common cells",
            latent_cells.len(),
            data_cells.len(),
            data_cells.len()
        );
    }

    Ok(reordered)
}

/// Compute cosine similarity between columns of a matrix
fn compute_cosine_similarity(x_dp: &Mat) -> Mat {
    let n = x_dp.ncols();

    // Normalize columns to unit vectors
    let mut x_norm = x_dp.clone();
    for j in 0..n {
        let col = x_norm.column(j);
        let norm = col.norm();
        if norm > 1e-10 {
            x_norm.column_mut(j).scale_mut(1.0 / norm);
        }
    }

    // Similarity = X^T X
    x_norm.transpose() * &x_norm
}

/// Threshold similarity matrix
fn threshold_similarity(s: &Mat, threshold: f32) -> Mat {
    let mut result = s.clone();
    for val in result.iter_mut() {
        if *val < threshold {
            *val = 0.0;
        }
    }
    result
}

/// Local scaling of similarity matrix (Zelnik-Manor & Perona, 2004)
/// Scales similarity by local density: S_scaled(i,j) = S(i,j) / sqrt(σ_i × σ_j)
/// where σ_i = 1 - similarity to k-th most similar neighbor
fn local_scale_similarity(s: &Mat, k: usize) -> Mat {
    let n = s.nrows();
    let k = k.min(n - 1).max(1);

    // For each point, find the k-th highest similarity (excluding self)
    let mut sigma = vec![0.0f32; n];
    for i in 0..n {
        let mut sims: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| s[(i, j)])
            .collect();
        // Sort descending to get k-th highest
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // σ_i = 1 - (similarity to k-th neighbor), i.e., distance to k-th neighbor
        // Use small epsilon to avoid division by zero
        let kth_sim = if k <= sims.len() { sims[k - 1] } else { sims.last().copied().unwrap_or(0.0) };
        sigma[i] = (1.0 - kth_sim).max(1e-6);
    }

    // Apply local scaling
    let mut result = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                result[(i, j)] = 1.0; // Self-similarity stays 1
            } else {
                // S_scaled = S / sqrt(σ_i * σ_j)
                let scale = (sigma[i] * sigma[j]).sqrt();
                result[(i, j)] = s[(i, j)] / scale;
            }
        }
    }

    result
}

/// Clip outliers beyond ±N standard deviations
fn clip_outliers(coords: &mut Mat, sd_threshold: f32) {
    let n = coords.nrows();

    for col in 0..coords.ncols() {
        // Compute mean and std for this coordinate
        let values: Vec<f32> = (0..n).map(|i| coords[(i, col)]).collect();
        let mean: f32 = values.iter().sum::<f32>() / n as f32;
        let variance: f32 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let sd = variance.sqrt();

        let lower = mean - sd_threshold * sd;
        let upper = mean + sd_threshold * sd;

        // Clip values
        for i in 0..n {
            coords[(i, col)] = coords[(i, col)].clamp(lower, upper);
        }
    }
}

/// Apply repulsion forces to spread out overlapping points
fn apply_repulsion(coords: &mut Mat, iterations: usize, strength: f32) {
    let n = coords.nrows();
    if n < 2 {
        return;
    }

    // Compute median pairwise distance for adaptive scaling
    let mut distances = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[(i, 0)] - coords[(j, 0)];
            let dy = coords[(i, 1)] - coords[(j, 1)];
            distances.push((dx * dx + dy * dy).sqrt());
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_dist = distances[distances.len() / 2].max(1e-6);

    for _iter in 0..iterations {
        let mut forces = vec![(0.0f32, 0.0f32); n];

        // Compute repulsion forces (inverse square law)
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = coords[(i, 0)] - coords[(j, 0)];
                let dy = coords[(i, 1)] - coords[(j, 1)];
                let dist_sq = dx * dx + dy * dy;
                let dist = dist_sq.sqrt().max(1e-6);

                // Repulsion force: stronger when closer
                // Scale by median distance so force is relative to point cloud size
                let force = strength * median_dist * median_dist / dist_sq;

                // Unit direction from j to i
                let fx = force * dx / dist;
                let fy = force * dy / dist;

                forces[i].0 += fx;
                forces[i].1 += fy;
                forces[j].0 -= fx;
                forces[j].1 -= fy;
            }
        }

        // Apply forces with damping
        let damping = 0.5;
        for i in 0..n {
            coords[(i, 0)] += damping * forces[i].0;
            coords[(i, 1)] += damping * forces[i].1;
        }
    }

    // Re-center coordinates
    let mean_x: f32 = (0..n).map(|i| coords[(i, 0)]).sum::<f32>() / n as f32;
    let mean_y: f32 = (0..n).map(|i| coords[(i, 1)]).sum::<f32>() / n as f32;
    for i in 0..n {
        coords[(i, 0)] -= mean_x;
        coords[(i, 1)] -= mean_y;
    }
}

/// Helper to sort eigenvalue indices
fn argsort(vals: &DVec, asc: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| {
        let c = vals[a].partial_cmp(&vals[b]).unwrap();
        if asc { c } else { c.reverse() }
    });
    idx
}

/// Regularize similarity matrix: add small self-loops to prevent isolated nodes
/// S_reg = S + ε * I ensures minimum degree ≥ ε
fn regularize_similarity(similarity: &Mat, eps: f32) -> Mat {
    let n = similarity.nrows();
    let mut sim_reg = similarity.clone();
    for i in 0..n {
        sim_reg[(i, i)] += eps;
    }

    // Warn about low-degree nodes
    let low_degree_count = (0..n)
        .filter(|&i| sim_reg.row(i).iter().sum::<f32>() < eps * 2.0)
        .count();
    if low_degree_count > 0 {
        info!(
            "Warning: {} PB samples have very low similarity to others.",
            low_degree_count
        );
    }

    sim_reg
}

/// Spectral embedding: compute k-dimensional embedding from similarity matrix
/// Uses symmetric normalized Laplacian: L_sym = I - D^{-1/2} S D^{-1/2}
/// Returns weighted eigenvectors (1/λ) for clustering
fn spectral_embed(similarity: &Mat, num_eigen: usize) -> anyhow::Result<Mat> {
    let n = similarity.nrows();
    let k = num_eigen.clamp(2, n - 1);
    anyhow::ensure!(n >= k + 1, "Need {} PB samples, got {}", k + 1, n);

    // Build normalized Laplacian: L_sym = I - D^{-1/2} S D^{-1/2}
    let degree: DVec = DVec::from_iterator(n, similarity.row_iter().map(|r| r.sum()));
    let d_inv_sqrt = Mat::from_diagonal(&degree.map(|d| 1.0 / d.sqrt()));
    let laplacian = Mat::identity(n, n) - &d_inv_sqrt * similarity * &d_inv_sqrt;

    // Eigen decomposition, extract k eigenvectors (skip trivial), weight by 1/λ
    let eig = laplacian.symmetric_eigen();
    let idx = argsort(&eig.eigenvalues, true);
    let mut emb = Mat::zeros(n, k);
    for (j, &i) in idx[1..=k].iter().enumerate() {
        let w = 1.0 / eig.eigenvalues[i].max(1e-10);
        emb.column_mut(j).copy_from(&(w * eig.eigenvectors.column(i)));
    }

    Ok(emb)
}

/// Reduce k-dimensional embedding to 2D via PCA
fn reduce_to_2d(emb: &Mat) -> Mat {
    let n = emb.nrows();
    let k = emb.ncols();

    if k == 2 {
        return emb.clone();
    }

    // PCA to 2D
    let mut centered = emb.clone();
    centered.centre_columns_inplace();
    let pca = (centered.transpose() * &centered).symmetric_eigen();
    let pca_idx = argsort(&pca.eigenvalues, false);
    let mut coords = Mat::zeros(n, 2);
    coords.column_mut(0).copy_from(&(&centered * pca.eigenvectors.column(pca_idx[0])));
    coords.column_mut(1).copy_from(&(&centered * pca.eigenvectors.column(pca_idx[1])));
    coords
}


/// MST-based radial tree layout
fn tree_layout_2d(
    similarity: &Mat,
    root: Option<usize>,
    radius_scale: f32,
    growth: f32,
) -> anyhow::Result<Mat> {
    let n = similarity.nrows();
    if n < 2 {
        return Err(anyhow::anyhow!("Need at least 2 nodes for tree layout"));
    }

    // 1. Build MST using Prim's algorithm (maximum spanning tree for similarity)
    let mst = build_mst(similarity);

    // 2. Choose root: user-specified or node with highest total similarity
    let root_node = root.unwrap_or_else(|| {
        (0..n)
            .max_by(|&a, &b| {
                let sum_a: f32 = similarity.row(a).iter().sum();
                let sum_b: f32 = similarity.row(b).iter().sum();
                sum_a.partial_cmp(&sum_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    });

    // 3. Build adjacency list from MST edges
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for &(u, v) in &mst {
        adj[u].push(v);
        adj[v].push(u);
    }

    // 4. Compute tree structure via BFS from root
    let (parent, depth, subtree_size) = tree_structure(&adj, root_node);

    // 5. Radial layout with growth factor
    let coords = radial_layout(&adj, &parent, &depth, &subtree_size, root_node, radius_scale, growth);

    Ok(coords)
}

/// Build maximum spanning tree using Prim's algorithm
fn build_mst(similarity: &Mat) -> Vec<(usize, usize)> {
    let n = similarity.nrows();
    let mut in_tree = vec![false; n];
    let mut edges = Vec::with_capacity(n - 1);

    // Start from node 0
    in_tree[0] = true;
    let mut nodes_in_tree = 1;

    while nodes_in_tree < n {
        // Find the maximum weight edge connecting tree to non-tree
        let mut best_edge: Option<(usize, usize, f32)> = None;

        for u in 0..n {
            if !in_tree[u] {
                continue;
            }
            for v in 0..n {
                if in_tree[v] {
                    continue;
                }
                let w = similarity[(u, v)];
                if best_edge.map_or(true, |(_, _, bw)| w > bw) {
                    best_edge = Some((u, v, w));
                }
            }
        }

        if let Some((u, v, _)) = best_edge {
            edges.push((u, v));
            in_tree[v] = true;
            nodes_in_tree += 1;
        } else {
            // Graph might be disconnected; find an unvisited node and continue
            if let Some(start) = in_tree.iter().position(|&x| !x) {
                in_tree[start] = true;
                nodes_in_tree += 1;
            }
        }
    }

    edges
}

/// Compute tree structure: parent, depth, and subtree sizes via BFS
fn tree_structure(adj: &[Vec<usize>], root: usize) -> (Vec<Option<usize>>, Vec<usize>, Vec<usize>) {
    let n = adj.len();
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut depth = vec![0usize; n];
    let mut subtree_size = vec![1usize; n];

    // BFS to compute parent and depth
    let mut queue = std::collections::VecDeque::new();
    let mut visited = vec![false; n];

    queue.push_back(root);
    visited[root] = true;

    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                parent[v] = Some(u);
                depth[v] = depth[u] + 1;
                queue.push_back(v);
            }
        }
    }

    // Compute subtree sizes (post-order traversal)
    // Sort nodes by depth descending, then accumulate
    let mut nodes_by_depth: Vec<usize> = (0..n).collect();
    nodes_by_depth.sort_by(|&a, &b| depth[b].cmp(&depth[a]));

    for &u in &nodes_by_depth {
        if let Some(p) = parent[u] {
            subtree_size[p] += subtree_size[u];
        }
    }

    (parent, depth, subtree_size)
}

/// Radial layout: root at center, children spread radially
fn radial_layout(
    adj: &[Vec<usize>],
    parent: &[Option<usize>],
    depth: &[usize],
    subtree_size: &[usize],
    root: usize,
    radius_scale: f32,
    decay: f32,
) -> Mat {
    let n = adj.len();
    let mut coords = Mat::zeros(n, 2);

    // Precompute cumulative radius for each depth level
    // r(d) = radius_scale * (1 + decay + decay^2 + ... + decay^(d-1))
    //      = radius_scale * (1 - decay^d) / (1 - decay)  for decay != 1
    let max_depth = *depth.iter().max().unwrap_or(&0);
    let cumulative_radius: Vec<f32> = (0..=max_depth)
        .map(|d| {
            if (decay - 1.0).abs() < 1e-6 {
                // decay ≈ 1.0, use linear
                d as f32 * radius_scale
            } else {
                // Geometric series sum
                radius_scale * (1.0 - decay.powi(d as i32)) / (1.0 - decay)
            }
        })
        .collect();

    // Root at origin
    coords[(root, 0)] = 0.0;
    coords[(root, 1)] = 0.0;

    // Assign angular ranges to each node
    let mut angle_start = vec![0.0f32; n];
    let mut angle_end = vec![std::f32::consts::TAU; n]; // TAU = 2*PI

    // BFS to assign positions
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);

    while let Some(u) = queue.pop_front() {
        let r = cumulative_radius[depth[u]];

        if u != root {
            // Position at midpoint of angular range
            let angle = (angle_start[u] + angle_end[u]) / 2.0;
            coords[(u, 0)] = r * angle.cos();
            coords[(u, 1)] = r * angle.sin();
        }

        // Get children (neighbors that have u as parent)
        let children: Vec<usize> = adj[u]
            .iter()
            .filter(|&&v| parent[v] == Some(u))
            .cloned()
            .collect();

        if children.is_empty() {
            continue;
        }

        // Distribute angular range among children proportional to subtree size
        let total_size: usize = children.iter().map(|&c| subtree_size[c]).sum();
        let mut current_angle = angle_start[u];

        for &child in &children {
            let child_fraction = subtree_size[child] as f32 / total_size as f32;
            let child_range = (angle_end[u] - angle_start[u]) * child_fraction;

            angle_start[child] = current_angle;
            angle_end[child] = current_angle + child_range;
            current_angle += child_range;

            queue.push_back(child);
        }
    }

    coords
}

/// Compute average latent for each PB sample
fn compute_pb_latent(latent_nk: &Mat, data_vec: &SparseIoVec) -> anyhow::Result<Mat> {
    let n_cells = latent_nk.nrows();
    let k = latent_nk.ncols();

    // Get PB membership for all cells
    let pb_membership = data_vec.get_group_membership(0..n_cells)?;
    let n_pb = *pb_membership.iter().max().unwrap_or(&0) + 1;

    // Accumulate latent vectors per PB
    let mut pb_sum = Mat::zeros(n_pb, k);
    let mut pb_count = vec![0usize; n_pb];

    for (cell_idx, &pb_idx) in pb_membership.iter().enumerate() {
        for col in 0..k {
            pb_sum[(pb_idx, col)] += latent_nk[(cell_idx, col)];
        }
        pb_count[pb_idx] += 1;
    }

    // Average
    for pb_idx in 0..n_pb {
        if pb_count[pb_idx] > 0 {
            pb_sum
                .row_mut(pb_idx)
                .scale_mut(1.0 / pb_count[pb_idx] as f32);
        }
    }

    Ok(pb_sum)
}

/// Project cells to 2D using kNN to nearest PB samples
fn project_cells_to_embedding(
    cell_latent: &Mat,
    pb_latent: &Mat,
    pb_coords: &Mat,
    knn: usize,
    temperature: f32,
) -> anyhow::Result<Mat> {
    use rayon::prelude::*;

    let n_cells = cell_latent.nrows();
    let n_pb = pb_latent.nrows();
    let k = knn.min(n_pb);

    // Build HNSW index for PB latent vectors
    let pb_names: Vec<usize> = (0..n_pb).collect();
    let pb_dict: ColumnDict<usize> = ColumnDict::from_dmatrix(pb_latent.transpose(), pb_names);

    // Process cells in parallel
    let coords: Vec<_> = (0..n_cells)
        .into_par_iter()
        .map(|i| {
            // Get cell's latent vector as VecPoint
            let cell_vec: Vec<f32> = cell_latent.row(i).iter().cloned().collect();
            let query = VecPoint { data: cell_vec };

            // Find k nearest PB samples
            let (neighbors, distances) = pb_dict
                .search_by_query_data(&query, k)
                .unwrap_or_else(|_| (vec![], vec![]));

            if neighbors.is_empty() {
                return (0.0f32, 0.0f32);
            }

            // Convert distances to weights (inverse distance with softmax)
            let weights: Vec<f32> = if distances.iter().any(|&d| d < 1e-10) {
                // If any distance is ~0, use hard assignment to that neighbor
                distances
                    .iter()
                    .map(|&d| if d < 1e-10 { 1.0 } else { 0.0 })
                    .collect()
            } else {
                // Softmax on negative distances (closer = higher weight)
                let neg_dists: Vec<f32> = distances.iter().map(|&d| -d / temperature).collect();
                let max_neg = neg_dists.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = neg_dists.iter().map(|&x| (x - max_neg).exp()).collect();
                let sum_exp: f32 = exp_vals.iter().sum();
                if sum_exp < 1e-10 {
                    return (0.0, 0.0);
                }
                exp_vals.iter().map(|&e| e / sum_exp).collect()
            };

            // Weighted average of neighbor PB coordinates
            let mut x = 0.0f32;
            let mut y = 0.0f32;
            for (j, &pb_idx) in neighbors.iter().enumerate() {
                x += weights[j] * pb_coords[(pb_idx, 0)];
                y += weights[j] * pb_coords[(pb_idx, 1)];
            }
            (x, y)
        })
        .collect();

    // Convert to matrix
    let mut cell_coords = Mat::zeros(n_cells, 2);
    for (i, (x, y)) in coords.into_iter().enumerate() {
        cell_coords[(i, 0)] = x;
        cell_coords[(i, 1)] = y;
    }

    Ok(cell_coords)
}
