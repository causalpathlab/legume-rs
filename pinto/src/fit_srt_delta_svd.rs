use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::estimate_batch;
use crate::srt_estimate_batch_effects::EstimateBatchArgs;
use crate::srt_input::*;
use crate::srt_random_projection::*;

use clap::Parser;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::io::ParamIo;
use matrix_param::traits::*;

#[derive(Parser, Debug, Clone)]
pub struct SrtDeltaSvdArgs {
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

    #[arg(short = 'k', long, default_value_t = 10,
          help = "Number of nearest neighbours for spatial cell-pair graph")]
    knn_spatial: usize,

    #[arg(long, default_value_t = 10,
          help = "Number of nearest-neighbour batches for batch effect estimation")]
    knn_batches: usize,

    #[arg(long, default_value_t = 10,
          help = "Number of nearest neighbours within each batch for batch estimation")]
    knn_cells: usize,

    #[arg(long, short = 's',
          help = "Maximum cells per pseudobulk sample (downsampling)")]
    down_sample: Option<usize>,

    #[arg(long, short, required = true,
          help = "Output file prefix",
          long_help = "Output file prefix.\n\
                       Generates: {out}.delta.parquet (when multiple batches), {out}.coord_pairs.parquet,\n\
                       {out}.dictionary.parquet, {out}.latent.parquet")]
    out: Box<str>,

    #[arg(long, default_value_t = 100,
          help = "Block size for parallel processing of cell pairs")]
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

/// Accumulated shared/difference statistics per gene per sample.
///
/// For each cell pair (left, right) and each gene g:
///   shared_g = log1p(x_left_g) + log1p(x_right_g)
///   diff_g   = |log1p(x_left_g) - log1p(x_right_g)|
pub(crate) struct PairDeltaCollapsedStat {
    shared_ds: Mat,
    diff_ds: Mat,
    size_s: DVec,
    n_genes: usize,
    n_samples: usize,
}

impl PairDeltaCollapsedStat {
    pub(crate) fn new(n_genes: usize, n_samples: usize) -> Self {
        Self {
            shared_ds: Mat::zeros(n_genes, n_samples),
            diff_ds: Mat::zeros(n_genes, n_samples),
            size_s: DVec::zeros(n_samples),
            n_genes,
            n_samples,
        }
    }

    pub(crate) fn optimize(
        &self,
        hyper_param: Option<(f32, f32)>,
    ) -> anyhow::Result<PairDeltaParameters> {
        let (a0, b0) = hyper_param.unwrap_or((1_f32, 1_f32));
        let shape = (self.n_genes, self.n_samples);

        let mut shared = GammaMatrix::new(shape, a0, b0);
        let mut diff = GammaMatrix::new(shape, a0, b0);

        let size_s = &self.size_s.transpose();
        let sample_size_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        info!("Calibrating pair delta statistics");

        shared.update_stat(&self.shared_ds, &sample_size_ds);
        shared.calibrate();
        diff.update_stat(&self.diff_ds, &sample_size_ds);
        diff.calibrate();

        info!("Resolved pair delta collapsed statistics");

        Ok(PairDeltaParameters { shared, diff })
    }
}

pub(crate) struct PairDeltaParameters {
    pub(crate) shared: GammaMatrix,
    pub(crate) diff: GammaMatrix,
}

/// Cell-pair SVD pipeline with shared/difference channels:
///
/// 1. Load data + coordinates
/// 2. Estimate batch effects
/// 3. Build spatial cell-cell KNN graph
/// 4. Random projection → assign pairs to samples
/// 5. Collapse: compute shared/diff per gene per sample
/// 6. Fit Poisson-Gamma on each channel
/// 7. SVD on vertically stacked [shared; diff] posterior log means
/// 8. Nystrom projection → per-pair latent codes
/// 9. Export dictionary + pair latents
pub fn fit_srt_delta_svd(args: &SrtDeltaSvdArgs) -> anyhow::Result<()> {
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

    // 2. Estimate batch effects
    info!("Checking potential batch effects...");

    let batch_effects = estimate_batch(
        &mut data_vec,
        &batch_membership,
        EstimateBatchArgs {
            proj_dim: args.proj_dim,
            sort_dim: args.sort_dim,
            block_size: args.block_size,
            knn_batches: args.knn_batches,
            knn_cells: args.knn_cells,
            down_sample: args.down_sample,
        },
    )?;

    if let Some(batch_db) = batch_effects.as_ref() {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet(Some(&gene_names), batch_names.as_deref(), &outfile)?;
    }

    // 3. Build spatial KNN graph
    info!("Constructing spatial nearest neighbourhood graphs");

    let mut srt_cell_pairs = SrtCellPairs::new(
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

    // 4. Random projection + sample assignment
    let proj_out = srt_cell_pairs.random_projection(
        args.proj_dim,
        args.block_size,
        Some(&batch_membership),
    )?;

    srt_cell_pairs.assign_pairs_to_samples(
        &proj_out,
        Some(args.sort_dim),
        args.down_sample,
    )?;

    // 5. Collapse: compute shared/diff per gene per sample
    info!("Collecting shared/difference statistics across cell pairs...");

    let batch_db = batch_effects.map(|x| x.posterior_mean().clone());
    let batch_ref = batch_db.as_ref();

    let mut collapsed_stat =
        PairDeltaCollapsedStat::new(n_genes, srt_cell_pairs.num_samples()?);

    srt_cell_pairs.visit_pairs_by_sample(
        &collect_pair_delta_visitor,
        &batch_ref,
        &mut collapsed_stat,
    )?;

    // 6. Fit Poisson-Gamma
    info!("Fitting Poisson-Gamma on shared/difference statistics...");
    let params = collapsed_stat.optimize(None)?;

    // 7. SVD on [shared; diff] posterior log means
    info!("Randomized SVD on pair delta features...");

    let training_dm = concatenate_vertical(&[
        params.shared.posterior_log_mean().clone(),
        params.diff.posterior_log_mean().clone(),
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
    let dict_row_names: Vec<Box<str>> = gene_names
        .iter()
        .map(|g| format!("{}@shared", g).into_boxed_str())
        .chain(
            gene_names
                .iter()
                .map(|g| format!("{}@diff", g).into_boxed_str()),
        )
        .collect();

    u_dk.to_parquet(
        Some(&dict_row_names),
        None,
        &(args.out.to_string() + ".dictionary.parquet"),
    )?;

    // 8. Nystrom projection
    info!("Nystrom pair delta projection...");

    let mut proj_kn = Mat::zeros(args.n_latent_topics, srt_cell_pairs.num_pairs());

    let nystrom_input = NystromPairInput {
        basis_shared: basis_dk.rows(0, n_genes).clone_owned(),
        basis_diff: basis_dk.rows(n_genes, n_genes).clone_owned(),
        batch_effect: batch_db,
    };

    srt_cell_pairs.visit_pairs_by_block(
        &nystrom_pair_delta_visitor,
        &nystrom_input,
        &mut proj_kn,
        args.block_size,
    )?;

    // 9. Export
    proj_kn
        .transpose()
        .to_parquet(None, None, &(args.out.to_string() + ".latent.parquet"))?;

    info!("Done");
    Ok(())
}

/// Collapse visitor: accumulate shared/diff per gene per sample.
pub(crate) fn collect_pair_delta_visitor(
    indices: &[usize],
    data: &SrtCellPairs,
    sample: usize,
    batch_effect: &Option<&Mat>,
    arc_stat: Arc<Mutex<&mut PairDeltaCollapsedStat>>,
) -> anyhow::Result<()> {
    let pairs: Vec<&Pair> = indices
        .iter()
        .filter_map(|&j| data.pairs.get(j))
        .collect();

    let left = pairs.iter().map(|&x| x.left);
    let right = pairs.iter().map(|&x| x.right);

    let mut y_left = data.data.read_columns_csc(left)?;
    let mut y_right = data.data.read_columns_csc(right)?;

    // batch adjustment: divide raw counts by batch effect
    if let Some(delta_db) = *batch_effect {
        let left = pairs.iter().map(|&x| x.left);
        let right = pairs.iter().map(|&x| x.right);
        let left_batches = data.data.get_batch_membership(left);
        y_left.adjust_by_division_of_selected_inplace(delta_db, &left_batches);
        let right_batches = data.data.get_batch_membership(right);
        y_right.adjust_by_division_of_selected_inplace(delta_db, &right_batches);
    }

    let n_genes = y_left.nrows();
    let mut local_shared = DVec::zeros(n_genes);
    let mut local_diff = DVec::zeros(n_genes);
    let mut local_size = 0_f32;

    for (left_col, right_col) in y_left.col_iter().zip(y_right.col_iter()) {
        // Build hashmap for right cell: gene -> log1p(adjusted_count)
        let right_log: HashMap<usize, f32> = right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
            .map(|(&g, &v)| (g, v.ln_1p()))
            .collect();

        let mut left_visited = HashSet::new();

        // Process genes present in left cell
        for (&gene, &val) in left_col
            .row_indices()
            .iter()
            .zip(left_col.values().iter())
        {
            let log_left = val.ln_1p();
            let log_right = right_log.get(&gene).copied().unwrap_or(0.0);
            local_shared[gene] += log_left + log_right;
            local_diff[gene] += (log_left - log_right).abs();
            left_visited.insert(gene);
        }

        // Process genes only present in right cell
        for (&gene, &val) in right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
        {
            if !left_visited.contains(&gene) {
                let log_right = val.ln_1p();
                // log_left = 0 → shared = log_right, diff = log_right
                local_shared[gene] += log_right;
                local_diff[gene] += log_right;
            }
        }

        local_size += 1.0;
    }

    let mut stat = arc_stat.lock().expect("lock pair delta stat");
    let mut col_shared = stat.shared_ds.column_mut(sample);
    col_shared += &local_shared;
    let mut col_diff = stat.diff_ds.column_mut(sample);
    col_diff += &local_diff;
    stat.size_s[sample] += local_size;

    Ok(())
}

/// Shared input for Nystrom pair-delta projection.
struct NystromPairInput {
    basis_shared: Mat,
    basis_diff: Mat,
    batch_effect: Option<Mat>,
}

/// Nystrom projection visitor: project each pair onto the split basis.
///
/// For each pair and each gene present in either cell:
///   shared = log1p(x_left) + log1p(x_right)
///   diff   = |log1p(x_left) - log1p(x_right)|
///   proj  += shared * basis_shared[gene] + diff * basis_diff[gene]
fn nystrom_pair_delta_visitor(
    bound: (usize, usize),
    data: &SrtCellPairs,
    shared_in: &NystromPairInput,
    arc_proj: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let left = pairs.iter().map(|pp| pp.left);
    let right = pairs.iter().map(|pp| pp.right);

    let mut y_left = data.data.read_columns_csc(left)?;
    let mut y_right = data.data.read_columns_csc(right)?;

    // batch adjustment
    if let Some(delta_db) = &shared_in.batch_effect {
        let left = pairs.iter().map(|x| x.left);
        let right = pairs.iter().map(|x| x.right);
        let left_batches = data.data.get_batch_membership(left);
        y_left.adjust_by_division_of_selected_inplace(delta_db, &left_batches);
        let right_batches = data.data.get_batch_membership(right);
        y_right.adjust_by_division_of_selected_inplace(delta_db, &right_batches);
    }

    let n_topics = shared_in.basis_shared.ncols();
    let n_pairs_block = ub - lb;
    let mut local_proj = Mat::zeros(n_topics, n_pairs_block);

    for (pair_idx, (left_col, right_col)) in
        y_left.col_iter().zip(y_right.col_iter()).enumerate()
    {
        let right_log: HashMap<usize, f32> = right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
            .map(|(&g, &v)| (g, v.ln_1p()))
            .collect();

        let mut proj_k = DVec::zeros(n_topics);
        let mut left_visited = HashSet::new();

        for (&gene, &val) in left_col
            .row_indices()
            .iter()
            .zip(left_col.values().iter())
        {
            let log_left = val.ln_1p();
            let log_right = right_log.get(&gene).copied().unwrap_or(0.0);
            let sigma = log_left + log_right;
            let delta = (log_left - log_right).abs();
            proj_k += sigma * &shared_in.basis_shared.row(gene).transpose();
            proj_k += delta * &shared_in.basis_diff.row(gene).transpose();
            left_visited.insert(gene);
        }

        // Right-only genes: log_left = 0 → sigma = log_right, delta = log_right
        for (&gene, _) in right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
        {
            if !left_visited.contains(&gene) {
                let log_right = right_log[&gene];
                proj_k += log_right * &shared_in.basis_shared.row(gene).transpose();
                proj_k += log_right * &shared_in.basis_diff.row(gene).transpose();
            }
        }

        local_proj.column_mut(pair_idx).copy_from(&proj_k);
    }

    let mut proj = arc_proj.lock().expect("lock nystrom pair delta proj");
    proj.columns_range_mut(lb..ub).copy_from(&local_proj);

    Ok(())
}
