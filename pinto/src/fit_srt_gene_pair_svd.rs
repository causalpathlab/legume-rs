use crate::edge_profiles::compute_propensity_and_gene_topic_stat;
use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_gene_graph::*;
use crate::srt_gene_pairs::*;
use crate::srt_input::{self, *};
use clap::Parser;
use data_beans_alg::random_projection::*;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::*;

#[derive(Parser, Debug, Clone)]
pub struct SrtGenePairSvdArgs {
    #[command(flatten)]
    pub common: srt_input::SrtInputArgs,

    #[arg(
        long,
        default_value_t = 10,
        help = "Binary sort depth: produces up to 2^S pseudobulk samples",
        long_help = "Number of top projection components for binary sort.\n\
                       Cells are recursively split along each component,\n\
                       producing up to 2^S pseudobulk samples.\n\
                       Must be <= --proj-dim."
    )]
    sort_dim: usize,

    #[arg(
        long,
        default_value_t = 20,
        help = "KNN for gene-gene co-expression graph",
        long_help = "Number of nearest neighbours for building the gene-gene graph.\n\
                       Genes are connected based on similarity of their posterior\n\
                       mean expression across pseudobulk samples.\n\
                       Ignored when --gene-network is provided."
    )]
    knn_gene: usize,

    #[arg(
        long,
        short = 's',
        help = "Max cells per pseudobulk sample (cap group size)"
    )]
    down_sample: Option<usize>,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of SVD components for latent pair representation"
    )]
    n_latent_topics: usize,

    #[arg(
        short = 'k',
        long,
        help = "Number of edge clusters for K-means (defaults to n_latent_topics)"
    )]
    n_edge_clusters: Option<usize>,

    #[arg(
        long,
        help = "External gene-gene network file (two-column TSV: gene1, gene2)",
        long_help = "External gene-gene network file (e.g., from BioGRID).\n\
                       Two-column TSV/CSV with gene1 and gene2 per line.\n\
                       Gene names are matched against data using exact, delimiter-based,\n\
                       and prefix matching. Skips KNN graph construction when provided."
    )]
    gene_network: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Allow prefix matching for gene names in external network"
    )]
    gene_network_allow_prefix: bool,

    #[arg(
        long,
        default_value = "_",
        help = "Delimiter for splitting compound gene names (e.g., ENSG00000141510_TP53)"
    )]
    gene_network_delimiter: Option<char>,

    #[arg(
        long,
        default_value_t = false,
        help = "Use union (non-reciprocal) matching for gene KNN graph",
        long_help = "Use union matching for the gene-gene KNN graph.\n\
                       By default, only reciprocal KNN edges are kept (both genes\n\
                       must be in each other's KNN list). With union matching,\n\
                       an edge is kept if either gene is in the other's list,\n\
                       producing a denser graph."
    )]
    gene_graph_union: bool,
}

/// Gene-gene interaction pipeline:
///
/// 1. Load data + coordinates
/// 2. Build spatial cell-cell KNN graph
/// 3. Assign cell pairs to samples (random projection + binary sort)
/// 4. Preliminary collapse → gene × sample matrix
/// 5. Build gene-gene KNN graph from posterior means
/// 6. Compute gene log means (μ̃_g)
/// 7. Compute gene-pair deltas (δ⁺/δ⁻) by visiting cells
/// 8. Fit Poisson-Gamma on gene-pair stats
/// 9. SVD on concatenated posterior log means
/// 10. Nystrom projection → per-cell → per-pair latent codes
/// 11. Export
pub fn fit_srt_gene_pair_svd(args: &SrtGenePairSvdArgs) -> anyhow::Result<()> {
    let c = &args.common;

    // 1. Load data
    info!("[1/9] Loading data files...");

    let SRTData {
        data: mut data_vec,
        coordinates,
        coordinate_names,
        batches: mut batch_membership,
    } = read_data_with_coordinates(c.to_read_args())?;

    let gene_names = data_vec.row_names()?;
    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    anyhow::ensure!(c.proj_dim > 0, "proj_dim must be > 0");
    anyhow::ensure!(args.sort_dim > 0, "sort_dim must be > 0");
    anyhow::ensure!(
        args.sort_dim <= c.proj_dim,
        "sort_dim ({}) must be <= proj_dim ({})",
        args.sort_dim,
        c.proj_dim
    );
    anyhow::ensure!(c.knn_spatial > 0, "knn_spatial must be > 0");
    anyhow::ensure!(args.knn_gene > 0, "knn_gene must be > 0");
    anyhow::ensure!(args.n_latent_topics > 0, "n_latent_topics must be > 0");
    anyhow::ensure!(
        c.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({})",
        c.knn_spatial,
        n_cells
    );

    // 2. Build spatial cell-cell KNN graph and extract pair info
    info!("[2/9] Building spatial KNN graph (k={})...", c.knn_spatial);
    let cell_pairs: Vec<(usize, usize)>;
    {
        let srt_cell_pairs = SrtCellPairs::new(
            &data_vec,
            &coordinates,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
            },
        )?;

        srt_cell_pairs.to_parquet(
            &(c.out.to_string() + ".coord_pairs.parquet"),
            Some(coordinate_names.clone()),
        )?;

        // Auto-detect batches from connected components when no explicit batch files
        if c.batch_files.is_none() {
            srt_input::auto_batch_from_components(&srt_cell_pairs.graph, &mut batch_membership);
        }

        cell_pairs = srt_cell_pairs
            .pairs
            .iter()
            .map(|p| (p.left, p.right))
            .collect();
    }
    // srt_cell_pairs dropped — data_vec is no longer borrowed

    // 3. Assign individual cells to samples (finest level for gene graph)
    info!("[3/9] Random projection and sample assignment...");

    let cell_proj_out = data_vec.project_columns_with_batch_correction(
        c.proj_dim,
        Some(c.block_size),
        Some(&batch_membership),
    )?;

    let n_samples = data_vec.partition_columns_to_groups(
        &cell_proj_out.proj,
        Some(args.sort_dim),
        args.down_sample,
    )?;

    info!("Assigned cells to {} samples (finest level)", n_samples);

    // 4-5. Build gene-gene graph (external network or KNN from posterior means)
    let mut gene_graph = if let Some(network_file) = &args.gene_network {
        info!(
            "[4/9] Loading external gene network from {}...",
            network_file
        );
        GenePairGraph::from_edge_list(
            network_file,
            gene_names.clone(),
            args.gene_network_allow_prefix,
            args.gene_network_delimiter,
        )?
    } else {
        // 4. Preliminary collapse: gene × sample sums
        let (gene_sum_ds, size_s) = preliminary_collapse(&data_vec, n_genes, n_samples)?;

        // Compute posterior means via Poisson-Gamma
        let (a0, b0) = (1_f32, 1_f32);
        let mut mu_param = GammaMatrix::new((n_genes, n_samples), a0, b0);
        let denom_ds = DVec::from_element(n_genes, 1_f32) * size_s.transpose();
        mu_param.update_stat(&gene_sum_ds, &denom_ds);
        mu_param.calibrate();

        // 5. Build gene-gene KNN graph
        info!(
            "[5/9] Building gene-gene KNN graph (k={})...",
            args.knn_gene
        );
        GenePairGraph::from_posterior_means(
            mu_param.posterior_mean(),
            gene_names.clone(),
            GenePairGraphArgs {
                knn: args.knn_gene,
                block_size: c.block_size,
                reciprocal: !args.gene_graph_union,
            },
        )?
    };

    // 6. Compute gene raw means
    let gene_means = compute_gene_raw_means(&data_vec, c.block_size)?;

    // 7. Multi-level gene-pair interaction deltas
    info!(
        "[6/9] Computing gene-pair interaction deltas ({} levels)...",
        c.num_levels
    );

    let level_dims = compute_level_sort_dims(args.sort_dim, c.num_levels);
    let mut gene_pair_stat = {
        let mut last_stat = None;
        for (level, &level_sort_dim) in level_dims.iter().enumerate() {
            info!(
                "Level {}/{}: sort_dim={}",
                level + 1,
                level_dims.len(),
                level_sort_dim
            );
            let level_n_samples = data_vec.partition_columns_to_groups(
                &cell_proj_out.proj,
                Some(level_sort_dim),
                args.down_sample,
            )?;
            last_stat = Some(compute_gene_interaction_deltas(
                &data_vec,
                &gene_graph,
                &gene_means,
                level_n_samples,
                false,
            )?);
        }
        last_stat.ok_or(anyhow::anyhow!("no levels"))?
    };

    // 7b. Filter empty gene pairs
    let use_elbow = args.gene_graph_union || args.gene_network.is_some();
    let n_removed = gene_pair_stat.filter_empty_edges(&mut gene_graph, use_elbow);
    if n_removed > 0 {
        info!(
            "Filtered {} empty gene pairs ({} remaining)",
            n_removed,
            gene_graph.num_edges()
        );
    }

    gene_graph.to_parquet(&(c.out.to_string() + ".gene_graph.parquet"))?;

    // 8. Fit Poisson-Gamma
    info!("[7/9] Fitting Poisson-Gamma model...");
    let gene_pair_params = gene_pair_stat.optimize(None)?;

    // 9. SVD on positive channel posterior log means
    info!(
        "[8/9] Randomized SVD ({} components)...",
        args.n_latent_topics
    );

    let training_dm = gene_pair_params
        .delta_pos
        .posterior_log_mean()
        .scale_columns();

    // Here, d = 2 x gene-gene interactions
    let (u_dk, s_k, _) = training_dm.rsvd(args.n_latent_topics)?;
    let basis_dk = nystrom_basis(&u_dk, &s_k);

    // Write dictionary
    let dict_row_names = gene_graph.edge_names();
    u_dk.to_parquet_with_names(
        &(c.out.to_string() + ".basis.parquet"),
        (Some(&dict_row_names), Some("gene_pair")),
        None,
    )?;

    // 10. Nystrom projection: per-cell first, then convert to per-pair
    info!("[9/9] Nystrom projection...");

    let cell_proj_kn =
        nystrom_gene_pair_projection(&data_vec, &gene_graph, &gene_means, &basis_dk, c.block_size)?;

    // Convert cell-level projections to pair-level, skipping pairs
    // where both cells have zero projection (no observed gene-pair deltas).
    info!("Converting cell latents to pair latents...");
    let n_topics = args.n_latent_topics;
    let mut kept_pairs: Vec<(usize, DVec)> = Vec::with_capacity(cell_pairs.len());

    for (pair_idx, &(left, right)) in cell_pairs.iter().enumerate() {
        let left_col = cell_proj_kn.column(left);
        let right_col = cell_proj_kn.column(right);
        let avg = (left_col + right_col) * 0.5;
        if avg.norm() > 0.0 {
            kept_pairs.push((pair_idx, avg));
        }
    }

    let n_kept = kept_pairs.len();
    if n_kept < cell_pairs.len() {
        info!(
            "Filtered {} zero-projection pairs ({} -> {})",
            cell_pairs.len() - n_kept,
            cell_pairs.len(),
            n_kept,
        );
    }

    let mut pair_proj_kn = Mat::zeros(n_topics, n_kept);
    let kept_indices: Vec<usize> = kept_pairs
        .iter()
        .enumerate()
        .map(|(new_idx, (_, avg))| {
            pair_proj_kn.column_mut(new_idx).copy_from(avg);
            kept_pairs[new_idx].0
        })
        .collect();

    // L2-normalize each pair's latent vector so downstream clustering
    // is driven by direction rather than magnitude.
    pair_proj_kn.normalize_columns_inplace();

    pair_proj_kn.transpose().to_parquet_with_names(
        &(c.out.to_string() + ".latent.parquet"),
        (None, Some("cell_pair")),
        None,
    )?;

    // Rewrite coord_pairs to include only kept pairs
    if n_kept < cell_pairs.len() {
        let coord_path = c.out.to_string() + ".coord_pairs.parquet";
        filter_parquet_by_indices(&coord_path, &kept_indices)?;
    }

    // Propensity + dictionary
    let kept_edges: Vec<(usize, usize)> = kept_indices.iter().map(|&i| cell_pairs[i]).collect();

    let n_clusters = args.n_edge_clusters.unwrap_or(args.n_latent_topics);
    compute_propensity_and_gene_topic_stat(
        &pair_proj_kn,
        &kept_edges,
        &data_vec,
        n_cells,
        n_clusters,
        c.block_size,
        &c.out,
    )?;

    info!("Done");
    Ok(())
}

/// Nystrom projection: project individual cells onto the gene-pair
/// dictionary to obtain per-cell latent codes.
///
/// For each cell, computes positive delta values (raw count products)
/// for present gene pairs and projects onto the SVD basis.
fn nystrom_gene_pair_projection(
    data_vec: &SparseIoVec,
    gene_graph: &GenePairGraph,
    gene_means: &DVec,
    basis_dk: &Mat,
    block_size: usize,
) -> anyhow::Result<Mat> {
    let n_cells = data_vec.num_columns();
    let n_topics = basis_dk.ncols();
    let n_edges = gene_graph.num_edges();

    info!(
        "Nystrom gene-pair projection: {} cells, {} edges, {} topics",
        n_cells, n_edges, n_topics,
    );

    let gene_adj = gene_graph.build_directed_adjacency();

    let shared_in = NystromSharedInput {
        gene_means: gene_means.clone(),
        gene_adj,
        basis: basis_dk.clone(),
    };

    let mut proj_kn = Mat::zeros(n_topics, n_cells);

    data_vec.visit_columns_by_block(
        &nystrom_gene_pair_visitor,
        &shared_in,
        &mut proj_kn,
        Some(block_size),
    )?;

    Ok(proj_kn)
}

struct NystromSharedInput {
    gene_means: DVec,
    gene_adj: Vec<Vec<(usize, usize)>>,
    basis: Mat,
}

fn nystrom_gene_pair_visitor(
    bound: (usize, usize),
    data_vec: &SparseIoVec,
    shared_in: &NystromSharedInput,
    arc_proj: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let gene_means = &shared_in.gene_means;
    let gene_adj = &shared_in.gene_adj;
    let basis = &shared_in.basis;
    let n_topics = basis.ncols();

    let yy = data_vec.read_columns_csc(lb..ub)?;

    let n_cells_block = ub - lb;
    let mut local_proj = Mat::zeros(n_topics, n_cells_block);

    for (cell_idx, y_j) in yy.col_iter().enumerate() {
        let rows = y_j.row_indices();
        let vals = y_j.values();

        let mut proj_k = DVec::zeros(n_topics);

        visit_gene_pair_deltas(
            rows,
            vals,
            gene_adj,
            gene_means,
            false,
            |edge_idx, delta| {
                if delta > 0.0 {
                    proj_k += delta * &basis.row(edge_idx).transpose();
                }
            },
        );

        local_proj.column_mut(cell_idx).copy_from(&proj_k);
    }

    let mut proj_kn = arc_proj.lock().expect("lock nystrom proj");
    proj_kn.columns_range_mut(lb..ub).copy_from(&local_proj);

    Ok(())
}
