#![allow(clippy::needless_range_loop)]
//! Link community model pipeline for spatial transcriptomics.
//!
//! Discovers link communities from spatial cell-cell KNN graphs via
//! collapsed Gibbs sampling on gene-projected edge profiles.

use crate::edge_profiles::*;
use crate::link_community_gibbs::LinkGibbsSampler;
use crate::link_community_model::LinkCommunityStats;
use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::{estimate_batch, EstimateBatchArgs};
use crate::srt_graph_coarsen::*;
use crate::srt_input::{self, *};

use clap::Parser;
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::io::ParamIo;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[derive(Parser, Debug, Clone)]
pub struct SrtLinkCommunityArgs {
    #[arg(
        required = true,
        value_delimiter(','),
        help = "Data files (.zarr or .h5 format, comma separated)"
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long = "coord",
        short = 'c',
        required = true,
        value_delimiter(','),
        help = "Spatial coordinate files, one per data file",
        long_help = "Spatial coordinate files, one per data file (comma separated).\n\
                       Format: CSV, TSV, or space-delimited text (or .parquet).\n\
                       First column: cell/barcode names (must match data file).\n\
                       Subsequent columns: spatial coordinates (x, y, etc.).\n\
                       Header row is auto-detected or use --coord-header-row to specify."
    )]
    coord_files: Vec<Box<str>>,

    #[arg(
        long = "coord-column-indices",
        value_delimiter(','),
        help = "Column indices for coordinates in coord files"
    )]
    coord_columns: Option<Vec<usize>>,

    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres",
        help = "Column names to look up in coord files"
    )]
    coord_column_names: Vec<Box<str>>,

    #[arg(
        long,
        help = "Header row index in coord files (0 = first line is column names)"
    )]
    coord_header_row: Option<usize>,

    #[arg(
        long,
        short = 'b',
        value_delimiter(','),
        help = "Batch membership files, one per data file",
        long_help = "Batch membership files, one per data file (comma separated).\n\
                       Format: plain text file, one batch label per line.\n\
                       Must have one line for each cell in the corresponding data file.\n\
                       If not provided, each data file is treated as a separate batch."
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 300,
        help = "Random projection dimension (M)"
    )]
    proj_dim: usize,

    #[arg(
        short = 'k',
        long,
        default_value_t = 20,
        help = "Number of link communities (K)"
    )]
    n_communities: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Number of gene modules for edge profiles (0 = same as K)"
    )]
    n_gene_modules: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of nearest neighbours for spatial cell-pair graph"
    )]
    knn_spatial: usize,

    #[arg(
        long = "batch-knn",
        default_value_t = 10,
        help = "KNN for cross-batch super-cell matching during batch correction",
        long_help = "Number of nearest neighbours for cross-batch super-cell matching.\n\
                       During batch effect estimation, cells are coarsened into super-cells\n\
                       (one per batch x pseudobulk group). Each super-cell finds its batch-knn\n\
                       nearest neighbors from other batches via HNSW on centroids. These matches\n\
                       provide counterfactual expression estimates for batch effect decomposition.\n\
                       Searches are over coarsened centroids, not individual cells, so this stays small."
    )]
    batch_knn: usize,

    #[arg(long, default_value_t = 100, help = "Number of Gibbs sweeps per level")]
    num_sweeps: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of greedy finalization sweeps"
    )]
    num_greedy: usize,

    #[arg(
        long,
        default_value_t = 2,
        help = "Number of multi-level coarsening levels"
    )]
    num_levels: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 1024,
        help = "Target number of coarse clusters for graph coarsening"
    )]
    n_coarse_clusters: usize,

    #[arg(long, default_value_t = 1.0, help = "Gamma shape prior (a0)")]
    a0: f32,

    #[arg(long, default_value_t = 1.0, help = "Gamma rate prior (b0)")]
    b0: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    seed: u64,

    #[arg(long, short, required = true, help = "Output file prefix")]
    out: Box<str>,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing of edges"
    )]
    block_size: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse column data into memory"
    )]
    preload_data: bool,
}

/// Link community model pipeline.
///
/// 1.  Load data + coordinates
/// 2.  Estimate batch effects
/// 3.  Build spatial KNN graph
/// 4.  Multi-level cell coarsening
/// 5.  Gene module discovery (sketch + K-means)
/// 6.  Build edge profiles as module counts
/// 7.  Gibbs on coarsest → transfer → refine at each finer level
/// 8.  Extract and write outputs
pub fn fit_srt_link_community(args: &SrtLinkCommunityArgs) -> anyhow::Result<()> {
    let a0 = args.a0 as f64;
    let b0 = args.b0 as f64;
    let k = args.n_communities;
    let n_gm = if args.n_gene_modules > 0 {
        args.n_gene_modules
    } else {
        k
    };

    // 1. Load data + coordinates
    info!("Loading data files...");

    let SRTData {
        data: mut data_vec,
        coordinates,
        coordinate_names,
        batches: mut batch_membership,
    } = read_data_with_coordinates(SRTReadArgs {
        data_files: args.data_files.clone(),
        coord_files: args.coord_files.clone(),
        preload_data: args.preload_data,
        coord_columns: args.coord_columns.clone().unwrap_or_default(),
        coord_column_names: args.coord_column_names.clone(),
        batch_files: args.batch_files.clone(),
        header_in_coord: args.coord_header_row,
    })?;

    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    anyhow::ensure!(args.proj_dim > 0, "proj_dim must be > 0");
    anyhow::ensure!(args.n_communities > 0, "n_communities must be > 0");
    anyhow::ensure!(args.knn_spatial > 0, "knn_spatial must be > 0");
    anyhow::ensure!(
        args.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({})",
        args.knn_spatial,
        n_cells
    );

    // 2. Build spatial KNN graph
    info!("Building spatial KNN graph (k={})...", args.knn_spatial);

    let graph = build_spatial_graph(
        &coordinates,
        SrtCellPairsArgs {
            knn: args.knn_spatial,
            block_size: args.block_size,
        },
    )?;

    // Auto-detect batches from connected components when no explicit batch files
    if args.batch_files.is_none() {
        srt_input::auto_batch_from_components(&graph, &mut batch_membership);
    }

    // 3. Estimate batch effects (only when multiple batches exist)
    let uniq_batches: HashSet<&Box<str>> = batch_membership.iter().collect();
    let n_batches = uniq_batches.len();
    drop(uniq_batches);

    let batch_effects = if n_batches > 1 {
        info!("Estimating batch effects ({} batches)...", n_batches);
        let batch_sort_dim = args.proj_dim.min(10);
        estimate_batch(
            &mut data_vec,
            &batch_membership,
            EstimateBatchArgs {
                proj_dim: args.proj_dim,
                sort_dim: batch_sort_dim,
                block_size: args.block_size,
                batch_knn: args.batch_knn,
            },
        )?
    } else {
        None
    };

    if let Some(batch_db) = batch_effects.as_ref() {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet_with_names(
            &outfile,
            (Some(&gene_names), Some("gene")),
            batch_names.as_deref(),
        )?;
    }

    // Wrap graph with data for pair-level operations
    let srt_cell_pairs = SrtCellPairs::with_graph(&data_vec, &coordinates, graph);

    srt_cell_pairs.to_parquet(
        &(args.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    let edges = &srt_cell_pairs.graph.edges;
    let n_edges = edges.len();
    info!("{} cells, {} edges", n_cells, n_edges);

    // Track scores
    let mut score_trace: Vec<f64> = Vec::new();

    // 4. Multi-level cell coarsening
    info!(
        "Graph coarsening ({} levels, {} coarse clusters)...",
        args.num_levels, args.n_coarse_clusters
    );

    let batch_arg: Option<&[Box<str>]> = if n_batches > 1 {
        Some(&batch_membership)
    } else {
        None
    };

    let cell_proj = data_vec.project_columns_with_batch_correction(
        args.proj_dim.min(50),
        Some(args.block_size),
        batch_arg,
    )?;

    let ml = graph_coarsen_multilevel(
        &srt_cell_pairs.graph,
        &mut cell_proj.proj.clone(),
        &srt_cell_pairs.pairs,
        args.n_coarse_clusters,
        args.num_levels,
    );

    // 5. Gene module discovery via sketch + K-means
    let finest_cell_labels = &ml.all_cell_labels[ml.all_cell_labels.len() - 1];
    let n_finest_clusters = ml.all_num_samples[ml.all_num_samples.len() - 1];

    let sketch_dim = args.proj_dim.min(50);
    info!(
        "Gene module sketch ({} clusters, {} sketch dims)...",
        n_finest_clusters, sketch_dim
    );
    let gene_embed = compute_gene_module_sketch(
        &data_vec,
        finest_cell_labels,
        n_finest_clusters,
        sketch_dim,
        args.block_size,
    )?;

    info!(
        "K-means on gene embeddings ({} → {} modules)...",
        n_genes, n_gm
    );
    let gene_to_module = gene_embed.kmeans_rows(KmeansArgs {
        num_clusters: n_gm,
        max_iter: 100,
    });

    // 6. Build edge profiles as module counts
    info!("Building module-count edge profiles...");
    let edge_profiles =
        build_edge_profiles_by_module(&data_vec, edges, &gene_to_module, n_gm, args.block_size)?;

    info!(
        "Edge profiles: {} edges × {} modules, mean size factor: {:.1}",
        edge_profiles.n_edges,
        edge_profiles.m,
        edge_profiles.size_factors.iter().sum::<f32>() / edge_profiles.n_edges as f32
    );

    // 7. Gibbs: coarsest → transfer → refine at each finer level
    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(args.seed));

    // Use cell labels from coarsening for edge profile coarsening
    let coarsest_labels = &ml.all_cell_labels[0];
    let (coarse_profiles, fine_to_super) =
        coarsen_edge_profiles(&edge_profiles, edges, coarsest_labels);

    info!(
        "Coarsest level: {} super-edges from {} fine edges",
        coarse_profiles.n_edges, n_edges
    );

    // Random initial labels for coarsest
    let init_labels: Vec<usize> = (0..coarse_profiles.n_edges).map(|e| e % k).collect();

    let mut coarse_stats = LinkCommunityStats::from_profiles(&coarse_profiles, k, &init_labels);

    info!("Gibbs on coarsest ({} sweeps)...", args.num_sweeps);
    let moves = sampler.run_parallel(&mut coarse_stats, &coarse_profiles, a0, b0, args.num_sweeps);
    info!(
        "Coarsest Gibbs: {} total moves, score={:.2}",
        moves,
        coarse_stats.total_score(a0, b0)
    );
    score_trace.push(coarse_stats.total_score(a0, b0));

    // Transfer to fine level
    let mut current_labels = transfer_labels(&fine_to_super, &coarse_stats.membership);

    // Refine at finer levels
    for level in 1..args.num_levels {
        info!("Refining at level {}...", level);

        let level_labels = &ml.all_cell_labels[level.min(ml.all_cell_labels.len() - 1)];
        let (level_profiles, level_f2s) =
            coarsen_edge_profiles(&edge_profiles, edges, level_labels);

        // Transfer current labels to this level's super-edges (majority vote)
        let n_super = level_profiles.n_edges;
        let mut super_label_votes = vec![vec![0usize; k]; n_super];
        for (fine_e, &se) in level_f2s.iter().enumerate() {
            let c = current_labels[fine_e];
            super_label_votes[se][c] += 1;
        }
        let super_init: Vec<usize> = super_label_votes
            .iter()
            .map(|votes| {
                votes
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &count)| count)
                    .map(|(c, _)| c)
                    .unwrap_or(0)
            })
            .collect();

        let mut level_stats =
            LinkCommunityStats::from_profiles(&level_profiles, k, &super_init);

        let sweeps = args.num_sweeps / 2; // fewer sweeps at finer levels
        let moves = sampler.run_parallel(&mut level_stats, &level_profiles, a0, b0, sweeps.max(10));
        info!(
            "Level {} Gibbs: {} moves, score={:.2}",
            level,
            moves,
            level_stats.total_score(a0, b0)
        );
        score_trace.push(level_stats.total_score(a0, b0));

        current_labels = transfer_labels(&level_f2s, &level_stats.membership);
    }

    // Final refinement on original edges
    info!(
        "Final Gibbs on full edge set ({} sweeps)...",
        args.num_sweeps / 2
    );
    let mut fine_stats = LinkCommunityStats::from_profiles(&edge_profiles, k, &current_labels);

    let moves = sampler.run_parallel(
        &mut fine_stats,
        &edge_profiles,
        a0,
        b0,
        (args.num_sweeps / 2).max(10),
    );
    info!(
        "Fine Gibbs: {} moves, score={:.2}",
        moves,
        fine_stats.total_score(a0, b0)
    );

    // Greedy finalization
    info!("Greedy finalization ({} max sweeps)...", args.num_greedy);
    let greedy_moves = sampler.run_greedy(&mut fine_stats, &edge_profiles, a0, b0, args.num_greedy);
    info!(
        "Greedy: {} moves, final score={:.2}",
        greedy_moves,
        fine_stats.total_score(a0, b0)
    );
    score_trace.push(fine_stats.total_score(a0, b0));

    // Recompute for drift correction
    fine_stats.recompute(&edge_profiles);
    let final_membership = fine_stats.membership.clone();

    // 9. Extract and write outputs
    let gene_names = data_vec.row_names()?;
    let cell_names = data_vec.column_names()?;

    // 9a. cell propensity [N × K]
    info!("Computing cell propensity...");
    let cell_propensity = compute_node_membership(edges, &final_membership, n_cells, k);

    let topic_names: Vec<Box<str>> = (0..k).map(|i| i.to_string().into_boxed_str()).collect();
    cell_propensity.to_parquet_with_names(
        &(args.out.to_string() + ".propensity.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&topic_names),
    )?;

    // 9b. Gene module assignments [G × 1]
    info!("Writing gene module assignments ({} modules)...", n_gm);
    let gene_modules = Mat::from_fn(n_genes, 1, |g, _| gene_to_module[g] as f32);
    let module_col_names: Vec<Box<str>> = vec!["module".into()];
    gene_modules.to_parquet_with_names(
        &(args.out.to_string() + ".gene_modules.parquet"),
        (Some(&gene_names), Some("gene")),
        Some(&module_col_names),
    )?;

    // 9c. Link community assignments
    info!("Writing link community assignments...");
    write_link_communities(
        &(args.out.to_string() + ".link_community.parquet"),
        edges,
        &final_membership,
        &cell_names,
    )?;

    // 9d. Score trace
    info!("Writing score trace...");
    write_score_trace(&(args.out.to_string() + ".scores.parquet"), &score_trace)?;

    info!("Done");
    Ok(())
}

/// Write link community assignments to parquet.
fn write_link_communities(
    file_path: &str,
    edges: &[(usize, usize)],
    membership: &[usize],
    cell_names: &[Box<str>],
) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    use parquet::basic::Type as ParquetType;

    let n_edges = edges.len();
    let left_cells: Vec<Box<str>> = edges.iter().map(|&(i, _)| cell_names[i].clone()).collect();
    let right_cells: Vec<Box<str>> = edges.iter().map(|&(_, j)| cell_names[j].clone()).collect();
    let cluster_f32: Vec<f32> = membership.iter().map(|&k| k as f32).collect();

    let col_names: Vec<Box<str>> =
        vec!["left_cell".into(), "right_cell".into(), "community".into()];
    let col_types = vec![
        ParquetType::BYTE_ARRAY,
        ParquetType::BYTE_ARRAY,
        ParquetType::FLOAT,
    ];

    let writer = ParquetWriter::new(
        file_path,
        (n_edges, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("edge"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_string_column(&mut row_group, &left_cells)?;
    parquet_add_string_column(&mut row_group, &right_cells)?;
    parquet_add_numeric_column(&mut row_group, &cluster_f32)?;

    row_group.close()?;
    writer.close()?;

    Ok(())
}

/// Write score trace to parquet.
fn write_score_trace(file_path: &str, scores: &[f64]) -> anyhow::Result<()> {
    let mat = Mat::from_fn(scores.len(), 1, |i, _| scores[i] as f32);
    let col_names = vec!["score".to_string().into_boxed_str()];
    let row_names: Vec<Box<str>> = (0..scores.len())
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    mat.to_parquet_with_names(
        file_path,
        (Some(&row_names), Some("step")),
        Some(&col_names),
    )
}
