#![allow(clippy::needless_range_loop)]
//! Link community model pipeline for spatial transcriptomics.
//!
//! Discovers link communities from spatial cell-cell KNN graphs via
//! collapsed Gibbs sampling on gene-projected edge profiles.

use crate::edge_profiles::*;
use crate::link_community_gibbs::LinkGibbsSampler;
use crate::link_community_model::LinkCommunitySuffStats;
use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::{estimate_batch, EstimateBatchArgs};
use crate::srt_graph_coarsen::*;
use crate::srt_input::*;

use clap::Parser;
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::io::ParamIo;
use matrix_util::utils::generate_minibatch_intervals;
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
        help = "Spatial coordinate files, one per data file"
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
        help = "Batch membership files, one per data file"
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
        default_value_t = 0,
        help = "Number of gene modules (0 = same as K). When larger than K, \
                K-means on edge profiles produces finer gene modules"
    )]
    n_gene_modules: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of nearest neighbours for spatial cell-pair graph"
    )]
    knn_spatial: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of nearest neighbours within each batch for batch estimation"
    )]
    knn_cells: usize,

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
        default_value_t = 1,
        help = "Number of projection refinement rounds"
    )]
    num_refine_rounds: usize,

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

    #[arg(long, short, help = "Enable verbose logging (sets RUST_LOG=info)")]
    verbose: bool,
}

/// Link community model pipeline.
///
/// 1.  Load data + coordinates
/// 2.  Estimate batch effects
/// 3.  Build spatial KNN graph
/// 4.  Random projection basis
/// 5.  Build edge profiles
/// 6.  Multi-level coarsening
/// 7.  Gibbs on coarsest → transfer → refine at each finer level
/// 8.  Projection refinement rounds
/// 9.  Extract and write outputs
pub fn fit_srt_link_community(args: &SrtLinkCommunityArgs) -> anyhow::Result<()> {
    init_logger(args.verbose);

    let a0 = args.a0 as f64;
    let b0 = args.b0 as f64;
    let k = args.n_communities;
    let m = args.proj_dim;
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

    // 2. Estimate batch effects
    info!("Estimating batch effects...");

    let batch_sort_dim = args.proj_dim.min(10);
    let batch_effects = estimate_batch(
        &mut data_vec,
        &batch_membership,
        EstimateBatchArgs {
            proj_dim: args.proj_dim,
            sort_dim: batch_sort_dim,
            block_size: args.block_size,
            knn_cells: args.knn_cells,
        },
    )?;

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

    // 3. Build spatial KNN graph
    info!("Building spatial KNN graph (k={})...", args.knn_spatial);

    let srt_cell_pairs = SrtCellPairs::new(
        &data_vec,
        &coordinates,
        SrtCellPairsArgs {
            knn: args.knn_spatial,
            block_size: args.block_size,
        },
    )?;

    srt_cell_pairs.to_parquet(
        &(args.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    let edges = &srt_cell_pairs.graph.edges;
    let n_edges = edges.len();
    info!("{} cells, {} edges", n_cells, n_edges);

    // 4. Random projection basis
    info!("Building random projection basis ({} → {})...", n_genes, m);
    let mut basis = Mat::rnorm(n_genes, m);

    // Normalize basis columns
    for j in 0..m {
        let norm: f32 = (0..n_genes)
            .map(|i| basis[(i, j)] * basis[(i, j)])
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 {
            for i in 0..n_genes {
                basis[(i, j)] /= norm;
            }
        }
    }

    // Track scores across rounds
    let mut score_trace: Vec<f64> = Vec::new();
    let mut final_membership: Vec<usize> = vec![0; n_edges];
    let mut last_edge_profiles: Option<crate::link_community_model::LinkProfileStore> = None;

    for refine_round in 0..=args.num_refine_rounds {
        info!(
            "=== Refinement round {}/{} ===",
            refine_round, args.num_refine_rounds
        );

        // 5. Build edge profiles
        info!("Building edge profiles...");
        let edge_profiles = build_edge_profiles(
            &data_vec,
            edges,
            &basis,
            None, // batch_effect handled in data_vec already
            args.block_size,
        )?;

        info!(
            "Edge profiles: {} edges × {} dims, mean size factor: {:.1}",
            edge_profiles.n_edges,
            edge_profiles.m,
            edge_profiles.size_factors.iter().sum::<f32>() / edge_profiles.n_edges as f32
        );

        // 6. Multi-level coarsening
        info!(
            "Graph coarsening ({} levels, {} coarse clusters)...",
            args.num_levels, args.n_coarse_clusters
        );

        let cell_proj = data_vec.project_columns_with_batch_correction(
            args.proj_dim.min(50),
            Some(args.block_size),
            Some(&batch_membership),
        )?;

        let ml = graph_coarsen_multilevel(
            &srt_cell_pairs.graph,
            &mut cell_proj.proj.clone(),
            &srt_cell_pairs.pairs,
            args.n_coarse_clusters,
            args.num_levels,
        );

        // 7. Gibbs: coarsest → transfer → refine at each finer level
        let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(
            args.seed.wrapping_add(refine_round as u64),
        ));

        // Extract cell labels at each level for edge profile coarsening
        let level_cell_labels: Vec<Vec<usize>> = ml
            .all_pair_to_sample
            .iter()
            .map(|p2s| {
                // Build cell labels from pair-to-sample
                let mut cell_labels = vec![0usize; n_cells];
                for (pair_idx, &sample) in p2s.iter().enumerate() {
                    let p = &srt_cell_pairs.pairs[pair_idx];
                    cell_labels[p.left] = sample;
                    cell_labels[p.right] = sample;
                }
                cell_labels
            })
            .collect();

        // Coarsest level
        let coarsest_labels = &level_cell_labels[0];
        let (coarse_profiles, fine_to_super) =
            coarsen_edge_profiles(&edge_profiles, edges, coarsest_labels);

        info!(
            "Coarsest level: {} super-edges from {} fine edges",
            coarse_profiles.n_edges, n_edges
        );

        // Random initial labels for coarsest
        let init_labels: Vec<usize> = (0..coarse_profiles.n_edges).map(|e| e % k).collect();

        let mut coarse_stats =
            LinkCommunitySuffStats::from_profiles(&coarse_profiles, k, &init_labels);

        info!("Gibbs on coarsest ({} sweeps)...", args.num_sweeps);
        let moves =
            sampler.run_parallel(&mut coarse_stats, &coarse_profiles, a0, b0, args.num_sweeps);
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

            let level_labels = &level_cell_labels[level.min(level_cell_labels.len() - 1)];
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
                LinkCommunitySuffStats::from_profiles(&level_profiles, k, &super_init);

            let sweeps = args.num_sweeps / 2; // fewer sweeps at finer levels
            let moves =
                sampler.run_parallel(&mut level_stats, &level_profiles, a0, b0, sweeps.max(10));
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
        let mut fine_stats =
            LinkCommunitySuffStats::from_profiles(&edge_profiles, k, &current_labels);

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
        let greedy_moves =
            sampler.run_greedy(&mut fine_stats, &edge_profiles, a0, b0, args.num_greedy);
        info!(
            "Greedy: {} moves, final score={:.2}",
            greedy_moves,
            fine_stats.total_score(a0, b0)
        );
        score_trace.push(fine_stats.total_score(a0, b0));

        // Recompute for drift correction
        fine_stats.recompute(&edge_profiles);
        final_membership = fine_stats.membership.clone();

        // Projection refinement (skip on last round)
        if refine_round < args.num_refine_rounds {
            info!("Refining projection basis...");
            let centroids = compute_community_centroids(
                &data_vec,
                edges,
                &final_membership,
                k,
                args.block_size,
            )?;
            basis = refine_projection_basis(&centroids, m)?;
            info!("Projection basis refined");
        }

        last_edge_profiles = Some(edge_profiles);
    }

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

    // 9b. Gene modules [G × n_gm]: accumulate per-module gene sums
    let gene_module_membership = if n_gm != k {
        info!(
            "K-means on edge profiles ({} → {} gene modules)...",
            k, n_gm
        );
        let ep = last_edge_profiles.as_ref().expect("edge profiles");
        let prof_mat = Mat::from_fn(ep.m, ep.n_edges, |g, e| ep.profiles[e * ep.m + g]);
        prof_mat.kmeans_columns(KmeansArgs {
            num_clusters: n_gm,
            max_iter: 100,
        })
    } else {
        final_membership.clone()
    };

    info!("Computing gene modules ({} modules)...", n_gm);
    let gene_modules = compute_gene_modules(
        &data_vec,
        edges,
        &gene_module_membership,
        n_gm,
        args.block_size,
    )?;

    let module_names: Vec<Box<str>> = (0..n_gm).map(|i| i.to_string().into_boxed_str()).collect();
    gene_modules.to_parquet_with_names(
        &(args.out.to_string() + ".gene_modules.parquet"),
        (Some(&gene_names), Some("gene")),
        Some(&module_names),
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

/// Compute gene expression modules per community.
///
/// For each community k, sums the expression of both cells across all edges
/// assigned to k, then normalizes by total community size.
fn compute_gene_modules(
    data: &SparseIoVec,
    edges: &[(usize, usize)],
    membership: &[usize],
    k: usize,
    block_size: usize,
) -> anyhow::Result<Mat> {
    let n_genes = data.num_rows();
    let n_edges = edges.len();

    let jobs = generate_minibatch_intervals(n_edges, block_size);

    let pb = new_progress_bar(
        jobs.len() as u64,
        "Gene modules {bar:40} {pos}/{len} blocks ({eta})",
    );
    let partial_stats: Vec<(Mat, Vec<f64>)> = jobs
        .par_iter()
        .progress_with(pb.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(Mat, Vec<f64>)> {
            let chunk_edges = &edges[lb..ub];
            let chunk_mem = &membership[lb..ub];

            let mut unique_set: HashSet<usize> = HashSet::new();
            for &(i, j) in chunk_edges {
                unique_set.insert(i);
                unique_set.insert(j);
            }
            let mut unique_cells: Vec<usize> = unique_set.into_iter().collect();
            unique_cells.sort_unstable();

            let x_dn = data.read_columns_csc(unique_cells.iter().copied())?;
            let cell_to_col: HashMap<usize, usize> = unique_cells
                .iter()
                .enumerate()
                .map(|(col, &cell)| (cell, col))
                .collect();

            let mut local_sum = Mat::zeros(n_genes, k);
            let mut local_total = vec![0.0f64; k];

            for (&(ci, cj), &c) in chunk_edges.iter().zip(chunk_mem.iter()) {
                let col_i = cell_to_col[&ci];
                let col_j = cell_to_col[&cj];

                let col_slice_i = x_dn.col(col_i);
                for (&row, &val) in col_slice_i
                    .row_indices()
                    .iter()
                    .zip(col_slice_i.values().iter())
                {
                    local_sum[(row, c)] += val;
                }
                let col_slice_j = x_dn.col(col_j);
                for (&row, &val) in col_slice_j
                    .row_indices()
                    .iter()
                    .zip(col_slice_j.values().iter())
                {
                    local_sum[(row, c)] += val;
                }

                local_total[c] += 1.0;
            }

            Ok((local_sum, local_total))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    pb.finish_and_clear();

    let mut total_sum = Mat::zeros(n_genes, k);
    let mut total_count = vec![0.0f64; k];
    for (local_sum, local_total) in partial_stats {
        total_sum += local_sum;
        for c in 0..k {
            total_count[c] += local_total[c];
        }
    }

    // Normalize: gene_modules[g,k] = sum[g,k] / total_count[k]
    for c in 0..k {
        if total_count[c] > 0.0 {
            let scale = 1.0 / total_count[c] as f32;
            total_sum.column_mut(c).scale_mut(scale);
        }
    }

    Ok(total_sum)
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
