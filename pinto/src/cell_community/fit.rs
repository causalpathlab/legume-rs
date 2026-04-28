//! Cell community pipeline: KNN → multi-level coarsening → flat-K Gibbs.
//!
//! Ported from `link_community::fit` swapping the sampled unit from edges to
//! cells. Coarsening merges neighbouring cells with similar expression, so
//! each super-cell is already a local neighbourhood seed — no extra
//! structural / connectivity term is needed at the objective level.

use super::gibbs::{CellGibbsSampler, ComponentGibbsArgs};
use super::model::CellCommunityStats;
use super::profiles::{build_cell_projection_profiles, coarsen_cell_profiles};
use crate::link_community::fit::{link_community_histogram, write_score_trace};
use crate::link_community::profiles::{
    compute_gene_topic_stat, compute_gene_totals, filter_basis_by_gene_count,
};
use crate::util::batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
use crate::util::cell_pairs::{
    build_expression_graph, build_spatial_graph, SrtCellPairs, SrtCellPairsArgs,
};
use crate::util::common::*;
use crate::util::graph_coarsen::{graph_coarsen_multilevel, SeedingParams};
use crate::util::input::*;
use clap::Parser;
use data_beans_alg::random_projection::RandProjOps;
use matrix_util::parquet::*;
use parquet::basic::Type as ParquetType;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[derive(Parser, Debug, Clone)]
pub struct SrtCellCommunityArgs {
    #[command(flatten)]
    pub common: SrtInputArgs,

    #[arg(long, default_value_t = 20, help = "Number of cell communities (K)")]
    pub n_communities: usize,

    #[arg(long, default_value_t = 100, help = "Gibbs sweeps at coarsest level")]
    pub num_gibbs: usize,

    #[arg(long, default_value_t = 10, help = "Max greedy sweeps at fine level")]
    pub num_greedy: usize,

    #[arg(
        long,
        help = "EM Gibbs sweeps on full cell set (default: num_gibbs/4, min 5)"
    )]
    pub num_em: Option<usize>,

    #[arg(
        long,
        help = "Dirichlet concentration α (default: auto from mean size factor / K)"
    )]
    pub alpha: Option<f32>,

    #[arg(long, default_value_t = 1.0, help = "Min gene total for projection basis")]
    pub min_gene_count: f32,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable IDF background correction (on by default)"
    )]
    pub no_background: bool,
}

pub fn fit_srt_cell_community(args: &SrtCellCommunityArgs) -> anyhow::Result<()> {
    let c = &args.common;
    let k = args.n_communities;

    anyhow::ensure!(k >= 2, "n-communities must be >= 2");
    anyhow::ensure!(c.knn_spatial > 0, "knn-spatial must be > 0");
    anyhow::ensure!(c.proj_dim > 0, "proj-dim must be > 0");

    // 1. Load data.
    info!("Loading data files...");
    let has_coords = c.has_coordinates();
    let SRTData {
        data: mut data_vec,
        mut coordinates,
        mut coordinate_names,
        batches: mut batch_membership,
    } = if has_coords {
        read_data_with_coordinates(c.to_read_args())?
    } else {
        info!("No coordinate files provided — using expression mode");
        read_data_without_coordinates(c.to_read_args())?
    };
    let n_cells = data_vec.num_columns();
    let n_genes = data_vec.num_rows();
    anyhow::ensure!(
        c.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({n_cells})",
        c.knn_spatial
    );
    info!("{} cells, {} genes", n_cells, n_genes);

    // 2. Build KNN graph.
    let graph;
    if has_coords {
        info!("Building spatial KNN graph (k={})...", c.knn_spatial);
        graph = build_spatial_graph(
            &coordinates,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?;
    } else {
        info!(
            "Building expression KNN graph (k={}, proj_dim={})...",
            c.knn_spatial, c.proj_dim
        );
        let cell_proj_pre = data_vec.project_columns_with_batch_correction(
            c.proj_dim,
            c.block_size,
            None::<&[Box<str>]>,
        )?;
        let (g, embedding) = build_expression_graph(
            &cell_proj_pre.proj,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?;
        graph = g;
        coordinates = embedding;
        coordinate_names = vec!["pc_1".into(), "pc_2".into()];
    }

    if c.auto_batch && c.batch_files.is_none() {
        crate::util::input::auto_batch_from_components(&graph, &mut batch_membership);
    }

    // 3. Batch effect estimation (no-op for single batch).
    let batch_sort_dim = c.proj_dim.min(10);
    let batch_db = estimate_and_write_batch_effects(
        &mut data_vec,
        &batch_membership,
        EstimateBatchArgs {
            proj_dim: c.proj_dim,
            sort_dim: batch_sort_dim,
            block_size: c.block_size,
            batch_knn: c.batch_knn,
            num_levels: c.num_levels,
        },
        &c.out,
    )?;

    // 4. Wrap with coordinates + write coord pairs.
    let srt_cell_pairs = SrtCellPairs::with_graph(&data_vec, &coordinates, graph);
    srt_cell_pairs.to_parquet(
        &(c.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;
    let n_edges = srt_cell_pairs.graph.edges.len();
    info!("KNN: {n_cells} cells, {n_edges} edges");

    let mut score_trace: Vec<f64> = Vec::new();

    // 5. Multi-level cell coarsening (spatial-seeded if coords available).
    info!(
        "Graph coarsening ({} levels, {} coarse clusters)...",
        c.num_levels, c.n_pseudobulk
    );
    let batch_arg: Option<&[Box<str>]> = if batch_db.is_some() {
        Some(&batch_membership)
    } else {
        None
    };
    let cell_proj =
        data_vec.project_columns_with_batch_correction(c.proj_dim, c.block_size, batch_arg)?;

    let ml = graph_coarsen_multilevel(
        &srt_cell_pairs.graph,
        &mut cell_proj.proj.clone(),
        &srt_cell_pairs.pairs,
        c.n_pseudobulk,
        c.num_levels,
        has_coords.then(|| SeedingParams {
            coordinates: &coordinates,
            batch_membership: Some(&batch_membership),
        }),
    );
    let coarsest_labels = &ml.all_cell_labels[0];
    let n_super = *ml.all_num_samples.first().unwrap();
    info!("Coarsest level: {n_super} super-cells");

    // 6. Build fine random-projection cell profiles.
    let mut basis = cell_proj.basis.clone();
    if args.min_gene_count > 0.0 {
        info!("Computing gene totals for filtering...");
        let gene_totals = compute_gene_totals(&data_vec, c.block_size)?;
        let n_kept =
            filter_basis_by_gene_count(&mut basis, &gene_totals, args.min_gene_count);
        info!(
            "Kept {n_kept}/{n_genes} genes (min_count={:.0})",
            args.min_gene_count
        );
    }
    info!(
        "Building fine random-projection cell profiles (dim={})...",
        c.proj_dim
    );
    let fine_profiles = build_cell_projection_profiles(&data_vec, &basis, c.block_size)?;
    let m = fine_profiles.m;

    // 7. Aggregate to coarse → IDF → Gibbs on coarsest.
    let mut coarse_profiles = coarsen_cell_profiles(&fine_profiles, coarsest_labels, n_super);
    let bg: Option<Vec<f64>> = if !args.no_background {
        let dist = coarse_profiles.empirical_marginal();
        info!(
            "Background: min={:.2e}, max={:.2e}, effective_dims={:.1}",
            dist.iter().cloned().fold(f64::INFINITY, f64::min),
            dist.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            (-dist
                .iter()
                .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
                .sum::<f64>())
            .exp()
        );
        coarse_profiles.weight_by_idf(&dist);
        Some(dist)
    } else {
        None
    };
    info!(
        "Coarse profiles: {} super-cells × {m} dims",
        coarse_profiles.n_cells
    );

    let init_super_labels: Vec<usize> = (0..n_super).map(|i| i % k).collect();
    let mut coarse_stats =
        CellCommunityStats::from_profiles(&coarse_profiles, k, &init_super_labels);

    let mut sampler = CellGibbsSampler::new(SmallRng::seed_from_u64(c.seed));
    info!("Gibbs on coarsest ({} sweeps)...", args.num_gibbs);
    let moves = sampler.run_parallel(&mut coarse_stats, &coarse_profiles, args.num_gibbs);
    let coarse_score = coarse_stats.total_score();
    info!("Coarsest Gibbs: {moves} moves, score={:.2}", coarse_score);
    score_trace.push(coarse_score);

    // 8. Transfer super-labels to fine cells.
    let mut fine_labels: Vec<usize> = coarsest_labels
        .iter()
        .map(|&s| coarse_stats.membership[s])
        .collect();
    drop(coarse_profiles);
    drop(coarse_stats);

    // 9. IDF on fine profiles using the coarse-derived marginal.
    let mut fine_profiles = fine_profiles;
    if let Some(ref bg_dist) = bg {
        fine_profiles.weight_by_idf(bg_dist);
    }
    info!(
        "Fine profiles: {} cells × {m} dims ({:.1} MB)",
        fine_profiles.n_cells,
        (fine_profiles.profiles.len() * std::mem::size_of::<f32>()) as f64 / 1_048_576.0
    );

    // 10. EM Gibbs + greedy at full resolution, partitioned by KNN components.
    let alpha: f64 = args.alpha.map_or_else(
        || {
            let mean_sf = fine_profiles.size_factors.iter().sum::<f32>()
                / fine_profiles.n_cells.max(1) as f32;
            let v = (mean_sf as f64 / k as f64).max(0.01);
            info!(
                "Auto alpha = {v:.4} (mean_size_factor {mean_sf:.1} / K={k})"
            );
            v
        },
        |v| v as f64,
    );
    let comp_args = ComponentGibbsArgs {
        graph: &srt_cell_pairs.graph,
        k,
        alpha,
    };
    let num_em_sweeps = args.num_em.unwrap_or((args.num_gibbs / 4).max(5));
    if num_em_sweeps > 0 {
        info!("EM Gibbs on full cell set ({num_em_sweeps} sweeps)...");
        let em_moves = sampler.run_components_em(
            &mut fine_labels,
            &fine_profiles,
            &comp_args,
            num_em_sweeps,
        );
        info!("EM Gibbs: {em_moves} moves");
    }

    info!("Greedy finalisation ({} max sweeps)...", args.num_greedy);
    let greedy_moves = sampler.run_greedy_by_components(
        &mut fine_labels,
        &fine_profiles,
        &comp_args,
        args.num_greedy,
    );

    let fine_stats = CellCommunityStats::from_profiles(&fine_profiles, k, &fine_labels);
    let final_score = fine_stats.total_score();
    info!("Greedy: {greedy_moves} moves, score={:.2}", final_score);
    score_trace.push(final_score);

    if log::log_enabled!(log::Level::Info) {
        eprintln!();
        eprintln!("{}", link_community_histogram(&fine_labels, k, 50));
        eprintln!();
    }

    // 11. Outputs.
    let cell_names = data_vec.column_names()?;

    write_cell_community(
        &(c.out.to_string() + ".cell_community.parquet"),
        &cell_names,
        &fine_labels,
    )?;
    write_score_trace(&(c.out.to_string() + ".scores.parquet"), &score_trace)?;

    let cell_propensity =
        Mat::from_fn(n_cells, k, |i, j| if fine_labels[i] == j { 1.0 } else { 0.0 });
    compute_gene_topic_stat(&cell_propensity, &data_vec, c.block_size, &c.out)?;

    info!("Done");
    Ok(())
}

fn write_cell_community(
    file_path: &str,
    cell_names: &[Box<str>],
    labels: &[usize],
) -> anyhow::Result<()> {
    let col_names: Vec<Box<str>> = vec!["community".into()];
    let col_types = vec![ParquetType::FLOAT];
    let writer = ParquetWriter::new(
        file_path,
        (labels.len(), col_names.len()),
        (Some(cell_names), Some(&col_names)),
        Some(&col_types),
        Some("cell"),
    )?;
    let row_names_vec = writer.row_names_vec().clone();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;
    parquet_add_bytearray(&mut row_group, &row_names_vec)?;
    let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
    parquet_add_numeric_column(&mut row_group, &labels_f32)?;
    row_group.close()?;
    writer.close()?;
    Ok(())
}
