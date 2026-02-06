use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_gene_graph::*;
use crate::srt_gene_pairs::*;
use crate::srt_input::*;
use clap::Parser;
use data_beans_alg::random_projection::*;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::*;

#[derive(Parser, Debug, Clone)]
///
/// PINTO gene-gene interaction analysis by SVD
///
pub struct SrtGenePairSvdArgs {
    /// Data files of either `.zarr` or `.h5` format.
    #[arg(required = true, value_delimiter(','))]
    data_files: Vec<Box<str>>,

    /// Auxiliary cell coordinate files (comma separated).
    #[arg(long = "coord", short = 'c', required = true, value_delimiter(','))]
    coord_files: Vec<Box<str>>,

    /// Cell coordinate column indices in the `coord` files (comma separated)
    #[arg(long = "coord-column-indices", value_delimiter(','))]
    coord_columns: Option<Vec<usize>>,

    /// Column names in the `coord` files (comma separated)
    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres"
    )]
    coord_column_names: Vec<Box<str>>,

    /// Header row in coordinate file (0 if first line is column names)
    #[arg(long)]
    coord_header_row: Option<usize>,

    /// Coordinate embedding dimension
    #[arg(long, default_value_t = 256)]
    coord_emb: usize,

    /// Batch membership files (comma separated).
    #[arg(long, short = 'b', value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// Random projection dimension
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// Use top S components for sample assignment. #samples < 2^S+1.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// k-nearest neighbours for gene-gene graph
    #[arg(long, default_value_t = 20)]
    knn_gene: usize,

    /// k-nearest neighbours for spatial cell pairs
    #[arg(short = 'k', long, default_value_t = 10)]
    knn_spatial: usize,

    /// Downsampling columns per collapsed sample
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Block size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// Number of latent topics
    #[arg(short = 't', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// Preload all column data
    #[arg(long, default_value_t = false)]
    preload_data: bool,

    /// Verbosity
    #[arg(long, short)]
    verbose: bool,
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
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }

    // 1. Load data
    info!("Reading data files...");

    let SRTData {
        data: mut data_vec,
        coordinates,
        coordinate_names,
        batches: batch_membership,
    } = read_data_with_coordinates(SRTReadArgs {
        data_files: args.data_files.clone(),
        coord_files: args.coord_files.clone(),
        preload_data: args.preload_data,
        coord_columns: args.coord_columns.clone().unwrap_or_default(),
        coord_column_names: args.coord_column_names.clone(),
        batch_files: args.batch_files.clone(),
        header_in_coord: args.coord_header_row,
    })?;

    let gene_names = data_vec.row_names()?;
    let n_genes = data_vec.num_rows();

    // 2. Build spatial cell-cell KNN graph and extract pair info
    info!("Constructing spatial nearest neighbourhood graphs");
    let cell_pairs: Vec<(usize, usize)>;
    {
        let srt_cell_pairs = SrtCellPairs::new(
            &data_vec,
            &coordinates,
            SrtCellPairsArgs {
                knn: args.knn_spatial,
                coordinate_emb_dim: args.coord_emb,
                block_size: args.block_size,
            },
        )?;

        srt_cell_pairs.to_parquet(
            &(args.out.to_string() + ".coord_pairs.parquet"),
            Some(coordinate_names.clone()),
        )?;

        cell_pairs = srt_cell_pairs
            .pairs
            .iter()
            .map(|p| (p.left, p.right))
            .collect();
    }
    // srt_cell_pairs dropped — data_vec is no longer borrowed

    // 3. Assign individual cells to samples for gene-pair analysis
    info!("Projecting cells for sample assignment...");

    let cell_proj_out = data_vec.project_columns_with_batch_correction(
        args.proj_dim,
        Some(args.block_size),
        Some(&batch_membership),
    )?;

    let n_samples = data_vec.partition_columns_to_groups(
        &cell_proj_out.proj,
        Some(args.sort_dim),
        args.down_sample,
    )?;

    info!("Assigned cells to {} samples", n_samples);

    // 4. Preliminary collapse: gene × sample sums
    let (gene_sum_ds, size_s) = preliminary_collapse(&data_vec, n_genes, n_samples)?;

    // Compute posterior means via Poisson-Gamma
    let (a0, b0) = (1_f32, 1_f32);
    let mut mu_param = GammaMatrix::new((n_genes, n_samples), a0, b0);
    let denom_ds = DVec::from_element(n_genes, 1_f32) * size_s.transpose();
    mu_param.update_stat(&gene_sum_ds, &denom_ds);
    mu_param.calibrate();

    // 5. Build gene-gene KNN graph
    info!("Building gene-gene KNN graph...");

    let gene_graph = GenePairGraph::from_posterior_means(
        mu_param.posterior_mean(),
        gene_names.clone(),
        GenePairGraphArgs {
            knn: args.knn_gene,
            block_size: args.block_size,
        },
    )?;

    // Write gene graph as edge list with distances
    gene_graph.to_parquet(&(args.out.to_string() + ".gene_graph.parquet"))?;

    // 6. Compute gene log means
    let gene_log_means = compute_gene_log_means(&data_vec, args.block_size)?;

    // 7. Compute gene-pair deltas
    info!("Computing gene-pair interaction deltas...");

    let gene_pair_stat = compute_gene_pair_deltas(
        &data_vec,
        &gene_graph,
        &gene_log_means,
        n_samples,
    )?;

    gene_pair_stat.to_parquet(&(args.out.to_string() + ".gene_pairs.parquet"))?;

    // 8. Fit Poisson-Gamma
    info!("Fitting Poisson-Gamma on gene-pair statistics...");
    let gene_pair_params = gene_pair_stat.optimize(None)?;

    // 9. SVD on vertically concatenated [δ⁺; δ⁻] posterior log means
    info!("Randomized SVD on gene-pair features...");

    let training_dm = concatenate_vertical(&[
        gene_pair_params.delta_pos.posterior_log_mean().clone(),
        gene_pair_params.delta_neg.posterior_log_mean().clone(),
    ])?
    .scale_columns();

    let (u_dk, s_k, _) = training_dm.rsvd(args.n_latent_topics)?;
    let eps = 1e-8;
    let sinv_k = DVec::from_iterator(
        s_k.len(),
        s_k.iter().map(|&s| 1.0 / (s + eps)),
    );
    let basis_dk = &u_dk * Mat::from_diagonal(&sinv_k);

    // Write dictionary
    let dict_row_names = gene_graph.edge_names_with_channels();
    u_dk.to_parquet(
        Some(&dict_row_names),
        None,
        &(args.out.to_string() + ".dictionary.parquet"),
    )?;

    // 10. Nystrom projection: per-cell first, then convert to per-pair
    info!("Nystrom gene-pair projection...");

    let cell_proj_kn = nystrom_gene_pair_projection(
        &data_vec,
        &gene_graph,
        &gene_log_means,
        &basis_dk,
        args.block_size,
    )?;

    // Convert cell-level projections to pair-level:
    // pair_proj = 0.5 * (cell_proj[left] + cell_proj[right])
    info!("Converting cell latents to pair latents...");
    let n_pairs = cell_pairs.len();
    let n_topics = args.n_latent_topics;
    let mut pair_proj_kn = Mat::zeros(n_topics, n_pairs);

    for (pair_idx, &(left, right)) in cell_pairs.iter().enumerate() {
        let left_col = cell_proj_kn.column(left);
        let right_col = cell_proj_kn.column(right);
        let avg = (&left_col + &right_col) * 0.5;
        pair_proj_kn.column_mut(pair_idx).copy_from(&avg);
    }

    pair_proj_kn
        .transpose()
        .to_parquet(None, None, &(args.out.to_string() + ".latent.parquet"))?;

    info!("Done");
    Ok(())
}
