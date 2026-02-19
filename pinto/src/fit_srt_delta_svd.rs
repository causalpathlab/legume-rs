use crate::edge_profiles::compute_propensity_and_gene_topic_stat;
use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
use crate::srt_graph_coarsen::*;
use crate::srt_input::{self, *};
use data_beans_alg::random_projection::*;

use clap::Parser;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::*;

#[derive(Parser, Debug, Clone)]
pub struct SrtDeltaSvdArgs {
    #[command(flatten)]
    pub common: srt_input::SrtInputArgs,

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
}

/// Input for fused multi-level pair delta visitor.
struct FusedDeltaInput<'a> {
    batch_effect: Option<&'a Mat>,
    all_pair_to_sample: Vec<Vec<usize>>,
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
    // 1. Load data
    info!("Loading data files...");

    let c = &args.common;

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
    anyhow::ensure!(c.n_pseudobulk > 0, "n_pseudobulk must be > 0");
    anyhow::ensure!(c.knn_spatial > 0, "knn_spatial must be > 0");
    anyhow::ensure!(args.n_latent_topics > 0, "n_latent_topics must be > 0");
    anyhow::ensure!(
        c.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({})",
        c.knn_spatial,
        n_cells
    );

    // 2. Build spatial KNN graph
    info!("Building spatial KNN graph (k={})...", c.knn_spatial);

    let graph = build_spatial_graph(
        &coordinates,
        SrtCellPairsArgs {
            knn: c.knn_spatial,
            block_size: c.block_size,
        },
    )?;

    // Auto-detect batches from connected components when no explicit batch files
    if c.batch_files.is_none() {
        srt_input::auto_batch_from_components(&graph, &mut batch_membership);
    }

    // 3. Estimate batch effects
    let batch_sort_dim = c.proj_dim.min(10);
    let batch_db = estimate_and_write_batch_effects(
        &mut data_vec,
        &batch_membership,
        EstimateBatchArgs {
            proj_dim: c.proj_dim,
            sort_dim: batch_sort_dim,
            block_size: c.block_size,
            batch_knn: c.batch_knn,
        },
        &c.out,
    )?;

    // Wrap graph with data for pair-level operations
    let srt_cell_pairs = SrtCellPairs::with_graph(&data_vec, &coordinates, graph);

    srt_cell_pairs.to_parquet(
        &(c.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    // 4. Per-cell random projection
    info!("Per-cell random projection...");
    let mut cell_proj = data_vec.project_columns_with_batch_correction(
        c.proj_dim,
        Some(c.block_size),
        Some(&batch_membership),
    )?;

    // 5. Graph-constrained coarsening + multi-level assignment
    info!(
        "Graph coarsening + multi-level assignment ({} levels, n_clusters={})...",
        c.num_levels, c.n_pseudobulk
    );

    let batch_ref = batch_db.as_ref();

    let ml = graph_coarsen_multilevel(
        &srt_cell_pairs.graph,
        &mut cell_proj.proj,
        &srt_cell_pairs.pairs,
        c.n_pseudobulk,
        c.num_levels,
    );

    let mut all_stats: Vec<PairDeltaCollapsedStat> = ml
        .all_num_samples
        .iter()
        .map(|&n| PairDeltaCollapsedStat::new(n_genes, n))
        .collect();

    let fused_input = FusedDeltaInput {
        batch_effect: batch_ref,
        all_pair_to_sample: ml.all_pair_to_sample,
    };

    srt_cell_pairs.visit_pairs_by_block(
        &fused_pair_delta_visitor,
        &fused_input,
        &mut all_stats,
        c.block_size,
    )?;

    // 6. Fit Poisson-Gamma (finest level)
    info!("Fitting Poisson-Gamma model...");
    let collapsed_stat = all_stats.last().ok_or(anyhow::anyhow!("no levels"))?;
    let params = collapsed_stat.optimize(None)?;

    // 7. SVD on [shared; diff] posterior log means
    info!("Randomized SVD ({} components)...", args.n_latent_topics);

    let training_dm = concatenate_vertical(&[
        params.shared.posterior_log_mean().scale_columns(),
        params.diff.posterior_log_mean().scale_columns(),
    ])?;

    let (u_dk, s_k, _) = training_dm.rsvd(args.n_latent_topics)?;
    let basis_dk = nystrom_basis(&u_dk, &s_k);

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

    u_dk.to_parquet_with_names(
        &(c.out.to_string() + ".basis.parquet"),
        (Some(&dict_row_names), Some("gene")),
        None,
    )?;

    // 8. Nystrom projection
    info!("Nystrom projection...");

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
        c.block_size,
    )?;

    // 9. Export
    // L2-normalize each pair's latent vector so downstream clustering
    // is driven by direction rather than magnitude.
    proj_kn.normalize_columns_inplace();

    proj_kn.transpose().to_parquet_with_names(
        &(c.out.to_string() + ".latent.parquet"),
        (None, Some("cell_pair")),
        None,
    )?;

    // 10. Propensity + dictionary
    let edges: Vec<(usize, usize)> = srt_cell_pairs
        .pairs
        .iter()
        .map(|p| (p.left, p.right))
        .collect();

    let n_clusters = args.n_edge_clusters.unwrap_or(args.n_latent_topics);
    compute_propensity_and_gene_topic_stat(
        &proj_kn,
        &edges,
        &data_vec,
        n_cells,
        n_clusters,
        c.block_size,
        &c.out,
    )?;

    info!("Done");
    Ok(())
}

/// Fused block-based visitor: read pair data once, accumulate into all levels' stats.
fn fused_pair_delta_visitor(
    bound: (usize, usize),
    data: &SrtCellPairs,
    input: &FusedDeltaInput,
    arc_stats: Arc<Mutex<&mut Vec<PairDeltaCollapsedStat>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let n_pairs = ub - lb;

    let left = pairs.iter().map(|x| x.left);
    let right = pairs.iter().map(|x| x.right);

    let mut y_left = data.data.read_columns_csc(left)?;
    let mut y_right = data.data.read_columns_csc(right)?;

    // batch adjustment
    if let Some(delta_db) = input.batch_effect {
        let left = pairs.iter().map(|x| x.left);
        let right = pairs.iter().map(|x| x.right);
        let left_batches = data.data.get_batch_membership(left);
        y_left.adjust_by_division_of_selected_inplace(delta_db, &left_batches);
        let right_batches = data.data.get_batch_membership(right);
        y_right.adjust_by_division_of_selected_inplace(delta_db, &right_batches);
    }

    let n_genes = y_left.nrows();

    // Compute per-pair shared/diff into dense block matrices
    let mut block_shared = Mat::zeros(n_genes, n_pairs);
    let mut block_diff = Mat::zeros(n_genes, n_pairs);

    for (pair_idx, (left_col, right_col)) in y_left.col_iter().zip(y_right.col_iter()).enumerate() {
        let right_log: HashMap<usize, f32> = right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
            .map(|(&g, &v)| (g, v.ln_1p()))
            .collect();

        let mut left_visited = HashSet::new();

        for (&gene, &val) in left_col.row_indices().iter().zip(left_col.values().iter()) {
            let log_left = val.ln_1p();
            let log_right = right_log.get(&gene).copied().unwrap_or(0.0);
            block_shared[(gene, pair_idx)] += log_left + log_right;
            block_diff[(gene, pair_idx)] += (log_left - log_right).abs();
            left_visited.insert(gene);
        }

        for (&gene, &val) in right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
        {
            if !left_visited.contains(&gene) {
                let log_right = val.ln_1p();
                block_shared[(gene, pair_idx)] += log_right;
                block_diff[(gene, pair_idx)] += log_right;
            }
        }
    }

    // Single lock: distribute to all levels' stats
    let mut stats = arc_stats.lock().expect("lock fused delta stats");
    for (level, pair_to_sample) in input.all_pair_to_sample.iter().enumerate() {
        for local_idx in 0..n_pairs {
            let sample = pair_to_sample[lb + local_idx];
            let mut col_shared = stats[level].shared_ds.column_mut(sample);
            col_shared += &block_shared.column(local_idx);
            let mut col_diff = stats[level].diff_ds.column_mut(sample);
            col_diff += &block_diff.column(local_idx);
            stats[level].size_s[sample] += 1.0;
        }
    }

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

    for (pair_idx, (left_col, right_col)) in y_left.col_iter().zip(y_right.col_iter()).enumerate() {
        let right_log: HashMap<usize, f32> = right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
            .map(|(&g, &v)| (g, v.ln_1p()))
            .collect();

        let mut proj_k = DVec::zeros(n_topics);
        let mut left_visited = HashSet::new();

        for (&gene, &val) in left_col.row_indices().iter().zip(left_col.values().iter()) {
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
