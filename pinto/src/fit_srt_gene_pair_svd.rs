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
pub struct SrtGenePairSvdArgs {
    #[arg(required = true, value_delimiter(','),
          help = "Data files (.zarr or .h5 format, comma separated)")]
    data_files: Vec<Box<str>>,

    #[arg(long = "coord", short = 'c', required = true, value_delimiter(','),
          help = "Spatial coordinate files, one per data file",
          long_help = "Spatial coordinate files, one per data file (comma separated).\n\
                       Each file: barcode, x, y, ... per line.")]
    coord_files: Vec<Box<str>>,

    #[arg(long = "coord-column-indices", value_delimiter(','),
          help = "Column indices for coordinates in coord files",
          long_help = "Column indices for coordinates in coord files (comma separated).\n\
                       Use when coord files have extra columns beyond barcode,x,y.")]
    coord_columns: Option<Vec<usize>>,

    #[arg(long = "coord-column-names", value_delimiter(','),
          default_value = "pxl_row_in_fullres,pxl_col_in_fullres",
          help = "Column names to look up in coord files")]
    coord_column_names: Vec<Box<str>>,

    #[arg(long,
          help = "Header row index in coord files (0 = first line is column names)")]
    coord_header_row: Option<usize>,

    #[arg(long, default_value_t = 256,
          help = "Dimension for spectral embedding of spatial coordinates")]
    coord_emb: usize,

    #[arg(long, short = 'b', value_delimiter(','),
          help = "Batch membership files, one per data file",
          long_help = "Batch membership files, one per data file (comma separated).\n\
                       Each file maps cells to batch labels for batch effect correction.")]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, short = 'p', default_value_t = 50,
          help = "Random projection dimension for pseudobulk sample construction")]
    proj_dim: usize,

    #[arg(long, short = 'd', default_value_t = 10,
          help = "Number of top projection components for binary sort",
          long_help = "Number of top projection components for binary sort.\n\
                       Produces up to 2^S pseudobulk samples.")]
    sort_dim: usize,

    #[arg(long, default_value_t = 20,
          help = "Number of nearest neighbours for gene-gene co-expression graph")]
    knn_gene: usize,

    #[arg(short = 'k', long, default_value_t = 10,
          help = "Number of nearest neighbours for spatial cell-pair graph")]
    knn_spatial: usize,

    #[arg(long, short = 's',
          help = "Maximum cells per pseudobulk sample (downsampling)")]
    down_sample: Option<usize>,

    #[arg(long, short, required = true,
          help = "Output file prefix",
          long_help = "Output file prefix.\n\
                       Generates: {out}.coord_pairs.parquet, {out}.gene_graph.parquet,\n\
                       {out}.gene_pairs.parquet, {out}.dictionary.parquet,\n\
                       {out}.latent.parquet")]
    out: Box<str>,

    #[arg(long, default_value_t = 100,
          help = "Block size for parallel processing of cells")]
    block_size: usize,

    #[arg(short = 't', long, default_value_t = 10,
          help = "Number of SVD components (latent dimensions)")]
    n_latent_topics: usize,

    #[arg(long, default_value_t = false,
          help = "Preload all sparse column data into memory for faster access")]
    preload_data: bool,

    #[arg(long, short,
          help = "Enable verbose logging (sets RUST_LOG=info)")]
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

    // We may consider edges x vertices incidence matrix later in
    // downstream analyses.
    gene_graph.to_parquet(&(args.out.to_string() + ".gene_graph.parquet"))?;

    // 6. Compute gene log means
    let gene_log_means = compute_gene_log_means(&data_vec, args.block_size)?;

    // 7. Compute gene-pair deltas
    info!("Calibrating gene-gene interaction statistics...");

    let gene_pair_stat = compute_gene_interaction_deltas(
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
    info!("Randomized SVD on gene-pair x sample features...");

    let training_dm = concatenate_vertical(&[
        gene_pair_params.delta_pos.posterior_log_mean().clone(),
        gene_pair_params.delta_neg.posterior_log_mean().clone(),
    ])?
    .scale_columns();

    // Here, d = 2 x gene-gene interactions
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

/// Nystrom projection: project individual cells onto the gene-pair
/// dictionary to obtain per-cell latent codes.
///
/// For each cell, computes delta values for present gene pairs and
/// projects onto the SVD basis (split into pos/neg halves).
fn nystrom_gene_pair_projection(
    data_vec: &SparseIoVec,
    gene_graph: &GenePairGraph,
    gene_log_means: &DVec,
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

    // Split basis into pos (top half) and neg (bottom half)
    let basis_pos = basis_dk.rows(0, n_edges).clone_owned();
    let basis_neg = basis_dk.rows(n_edges, n_edges).clone_owned();

    let gene_adj = gene_graph.build_directed_adjacency();

    let shared_in = NystromSharedInput {
        gene_log_means: gene_log_means.clone(),
        gene_adj,
        basis_pos,
        basis_neg,
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
    gene_log_means: DVec,
    gene_adj: Vec<Vec<(usize, usize)>>,
    basis_pos: Mat,
    basis_neg: Mat,
}

fn nystrom_gene_pair_visitor(
    bound: (usize, usize),
    data_vec: &SparseIoVec,
    shared_in: &NystromSharedInput,
    arc_proj: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let gene_log_means = &shared_in.gene_log_means;
    let gene_adj = &shared_in.gene_adj;
    let basis_pos = &shared_in.basis_pos;
    let basis_neg = &shared_in.basis_neg;
    let n_topics = basis_pos.ncols();

    let yy = data_vec.read_columns_csc(lb..ub)?;

    let n_cells_block = ub - lb;
    let mut local_proj = Mat::zeros(n_topics, n_cells_block);

    for (cell_idx, y_j) in yy.col_iter().enumerate() {
        let rows = y_j.row_indices();
        let vals = y_j.values();

        let mut proj_k = DVec::zeros(n_topics);

        visit_gene_pair_deltas(rows, vals, gene_adj, gene_log_means, |edge_idx, delta| {
            if delta > 0.0 {
                proj_k += delta * &basis_pos.row(edge_idx).transpose();
            } else if delta < 0.0 {
                proj_k += (-delta) * &basis_neg.row(edge_idx).transpose();
            }
        });

        local_proj.column_mut(cell_idx).copy_from(&proj_k);
    }

    let mut proj_kn = arc_proj.lock().expect("lock nystrom proj");
    proj_kn.columns_range_mut(lb..ub).copy_from(&local_proj);

    Ok(())
}
