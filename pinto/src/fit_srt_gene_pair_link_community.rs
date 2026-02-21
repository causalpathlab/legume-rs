#![allow(clippy::needless_range_loop)]
//! Gene pair link community model pipeline for spatial transcriptomics.
//!
//! Combines gene-pair interaction deltas with collapsed Gibbs link
//! community sampling: for each spatial edge (i,j), the profile is a
//! vector of δ⁺ values across gene pairs, capturing which gene-gene
//! interactions are active on that edge.

use crate::edge_profiles::*;
use crate::fit_srt_link_community::{
    link_community_histogram, write_link_communities, write_score_trace,
};
use crate::link_community_gibbs::{ComponentGibbsArgs, LinkGibbsSampler};
use crate::link_community_model::LinkCommunityStats;
use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use crate::srt_estimate_batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
use crate::srt_gene_graph::*;
use crate::srt_gene_pairs::*;
use crate::srt_graph_coarsen::*;
use crate::srt_input::{self, *};

use clap::Parser;
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[derive(Parser, Debug, Clone)]
pub struct SrtGenePairLinkCommunityArgs {
    #[command(flatten)]
    pub common: srt_input::SrtInputArgs,

    #[arg(
        long,
        default_value_t = 20,
        help = "Number of spatial link communities to discover"
    )]
    n_communities: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Gibbs sampling sweeps per coarsening level"
    )]
    num_sweeps: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Max greedy refinement sweeps after Gibbs"
    )]
    num_greedy: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Gamma shape prior for Poisson-Gamma model"
    )]
    a0: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Gamma rate prior for Poisson-Gamma model"
    )]
    b0: f32,

    #[arg(
        long,
        default_value_t = 10,
        help = "Binary sort depth for cell-to-sample assignment (gene graph construction)"
    )]
    sort_dim: usize,

    #[arg(
        long,
        short = 's',
        help = "Max cells per pseudobulk sample (cap group size)"
    )]
    down_sample: Option<usize>,

    #[arg(
        long,
        default_value_t = 20,
        help = "KNN for gene-gene co-expression graph"
    )]
    knn_gene: usize,

    #[arg(
        long,
        help = "External gene-gene network file (two-column TSV: gene1, gene2)"
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
        help = "Delimiter for splitting compound gene names"
    )]
    gene_network_delimiter: Option<char>,

    #[arg(
        long,
        default_value_t = false,
        help = "Use union (non-reciprocal) matching for gene KNN graph"
    )]
    gene_graph_union: bool,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Max gene-pair dimensions; cluster into modules if exceeded",
        long_help = "Maximum number of gene-pair dimensions for edge profiles.\n\
                       When the number of gene pairs exceeds this threshold,\n\
                       gene pairs are clustered into modules via K-means and\n\
                       edge profiles are summed per module. Set to 0 to disable."
    )]
    n_edge_modules: usize,
}

/// Gene pair link community model pipeline.
///
/// 1.  Load data + coordinates
/// 2.  Build spatial KNN graph
/// 3.  Estimate batch effects
/// 4.  Build gene-gene graph (KNN or external) — needs mutable data_vec
/// 5.  Multi-level cell coarsening
/// 6.  Compute gene raw means
/// 7.  Build gene-pair edge profiles (δ⁺ per spatial edge)
/// 8.  Filter low-signal gene pairs (elbow)
/// 9.  Gibbs on coarsest → transfer → refine at each finer level
/// 10. Extract and write outputs
pub fn fit_srt_gene_pair_link_community(args: &SrtGenePairLinkCommunityArgs) -> anyhow::Result<()> {
    let c = &args.common;
    let a0 = args.a0 as f64;
    let b0 = args.b0 as f64;
    let k = args.n_communities;

    // 1. Load data (with or without coordinates)
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

    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    anyhow::ensure!(c.proj_dim > 0, "proj_dim must be > 0");
    anyhow::ensure!(args.n_communities > 0, "n_communities must be > 0");
    anyhow::ensure!(c.knn_spatial > 0, "knn_spatial must be > 0");
    anyhow::ensure!(args.knn_gene > 0, "knn_gene must be > 0");
    anyhow::ensure!(
        c.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({})",
        c.knn_spatial,
        n_cells
    );

    // 2. Build KNN graph (spatial or expression-based)
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
            Some(c.block_size),
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

    // Auto-detect batches from connected components (opt-in via --auto-batch)
    if c.auto_batch && c.batch_files.is_none() {
        srt_input::auto_batch_from_components(&graph, &mut batch_membership);
    }

    // 3. Estimate batch effects (skipped for single-batch)
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

    // 4. Random projection for coarsening and gene graph
    let batch_arg: Option<&[Box<str>]> = if batch_db.is_some() {
        Some(&batch_membership)
    } else {
        None
    };

    let cell_proj = data_vec.project_columns_with_batch_correction(
        c.proj_dim,
        Some(c.block_size),
        batch_arg,
    )?;

    // 5. Build gene-gene graph — partition_columns_to_groups needs &mut data_vec,
    //    so this must happen before SrtCellPairs borrows data_vec immutably.
    let gene_names = data_vec.row_names()?;

    let mut gene_graph = if let Some(network_file) = &args.gene_network {
        info!("Loading external gene network from {}...", network_file);
        GenePairGraph::from_edge_list(
            network_file,
            gene_names.clone(),
            args.gene_network_allow_prefix,
            args.gene_network_delimiter,
        )?
    } else {
        // Assign cells to samples for gene graph construction
        let n_samples = data_vec.partition_columns_to_groups(
            &cell_proj.proj,
            Some(args.sort_dim),
            args.down_sample,
        )?;
        info!("Assigned cells to {} samples for gene graph", n_samples);

        // Preliminary collapse: gene × sample sums
        let (gene_sum_ds, size_s) = preliminary_collapse(&data_vec, n_genes, n_samples)?;

        // Compute posterior means via Poisson-Gamma
        let (ga0, gb0) = (1_f32, 1_f32);
        let mut mu_param = GammaMatrix::new((n_genes, n_samples), ga0, gb0);
        let denom_ds = DVec::from_element(n_genes, 1_f32) * size_s.transpose();
        mu_param.update_stat(&gene_sum_ds, &denom_ds);
        mu_param.calibrate();

        // Build gene-gene KNN graph
        info!("Building gene-gene KNN graph (k={})...", args.knn_gene);
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

    let n_gene_pairs = gene_graph.num_edges();
    info!("Gene graph: {} gene pairs", n_gene_pairs);

    // Now create SrtCellPairs (borrows data_vec immutably)
    let srt_cell_pairs = SrtCellPairs::with_graph(&data_vec, &coordinates, graph);

    srt_cell_pairs.to_parquet(
        &(c.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    let edges = &srt_cell_pairs.graph.edges;
    let n_edges = edges.len();
    info!("{} cells, {} edges", n_cells, n_edges);

    // Track scores
    let mut score_trace: Vec<f64> = Vec::new();

    // 6. Multi-level cell coarsening
    info!(
        "Graph coarsening ({} levels, {} coarse clusters)...",
        c.num_levels, c.n_pseudobulk
    );

    let ml = graph_coarsen_multilevel(
        &srt_cell_pairs.graph,
        &mut cell_proj.proj.clone(),
        &srt_cell_pairs.pairs,
        c.n_pseudobulk,
        c.num_levels,
    );

    // 7. Compute gene raw means
    let gene_means = compute_gene_raw_means(&data_vec, c.block_size)?;

    // 8. Build gene-pair edge profiles
    info!(
        "Building gene-pair edge profiles ({} edges x {} gene pairs)...",
        n_edges, n_gene_pairs
    );
    let gene_adj = gene_graph.build_directed_adjacency();

    let edge_profiles = build_edge_profiles_by_gene_pairs(
        &data_vec,
        edges,
        &gene_adj,
        &gene_means,
        n_gene_pairs,
        c.block_size,
    )?;

    // 9. Filter low-signal gene pairs via elbow on column sums
    let col_sums: Vec<f32> = (0..n_gene_pairs)
        .map(|p| {
            let mut s = 0.0f32;
            for e in 0..n_edges {
                s += edge_profiles.profile(e)[p];
            }
            s
        })
        .collect();

    let use_elbow = args.gene_graph_union || args.gene_network.is_some();
    let (threshold, elbow_rank) = if use_elbow {
        elbow_threshold(&col_sums)
    } else {
        (0.0, 0)
    };

    let keep_cols: Vec<usize> = (0..n_gene_pairs)
        .filter(|&p| col_sums[p] > threshold)
        .collect();

    let edge_profiles = if keep_cols.len() < n_gene_pairs {
        info!(
            "Elbow threshold: {:.4} (rank {}), kept {}/{} gene pairs",
            threshold,
            elbow_rank,
            keep_cols.len(),
            n_gene_pairs
        );
        gene_graph.filter_edges(&keep_cols);
        filter_profile_columns(&edge_profiles, &keep_cols)
    } else {
        edge_profiles
    };

    gene_graph.to_parquet(&(c.out.to_string() + ".gene_graph.parquet"))?;

    // Collapse gene-pair columns into modules if too many
    let edge_profiles = if args.n_edge_modules > 0 && edge_profiles.m > args.n_edge_modules {
        info!(
            "Clustering {} gene pairs into {} edge modules...",
            edge_profiles.m, args.n_edge_modules
        );
        let (module_profiles, _assignments) =
            collapse_profile_columns_by_module(&edge_profiles, args.n_edge_modules);
        module_profiles
    } else {
        edge_profiles
    };

    info!(
        "Edge profiles: {} edges x {} dims, mean size factor: {:.1}",
        edge_profiles.n_edges,
        edge_profiles.m,
        edge_profiles.size_factors.iter().sum::<f32>() / edge_profiles.n_edges as f32
    );

    // 10. Gibbs: coarsest -> transfer -> refine at each finer level
    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(c.seed));

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
    for level in 1..c.num_levels {
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

        let mut level_stats = LinkCommunityStats::from_profiles(&level_profiles, k, &super_init);

        let sweeps = args.num_sweeps / 2;
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
    let num_fine_sweeps = (args.num_sweeps / 2).max(10);
    info!(
        "Final Gibbs on full edge set ({} sweeps)...",
        num_fine_sweeps
    );

    let comp_args = ComponentGibbsArgs {
        graph: &srt_cell_pairs.graph,
        edges,
        k,
        a0,
        b0,
    };

    let moves = sampler.run_components_em(
        &mut current_labels,
        &edge_profiles,
        &comp_args,
        num_fine_sweeps,
    );
    let fine_score =
        LinkCommunityStats::from_profiles(&edge_profiles, k, &current_labels).total_score(a0, b0);
    info!("Fine Gibbs: {} moves, score={:.2}", moves, fine_score);

    // Greedy finalization
    info!("Greedy finalization ({} max sweeps)...", args.num_greedy);
    let greedy_moves = sampler.run_greedy_by_components(
        &mut current_labels,
        &edge_profiles,
        &comp_args,
        args.num_greedy,
    );
    let mut fine_stats = LinkCommunityStats::from_profiles(&edge_profiles, k, &current_labels);
    info!(
        "Greedy: {} moves, final score={:.2}",
        greedy_moves,
        fine_stats.total_score(a0, b0)
    );
    score_trace.push(fine_stats.total_score(a0, b0));

    // Recompute for drift correction
    fine_stats.recompute(&edge_profiles);
    let final_membership = fine_stats.membership.clone();

    // Display link community size histogram
    if log::log_enabled!(log::Level::Info) {
        eprintln!();
        eprintln!("{}", link_community_histogram(&final_membership, k, 50));
        eprintln!();
    }

    // 11. Extract and write outputs
    let cell_names = data_vec.column_names()?;

    // Cell propensity [N x K]
    info!("Computing cell propensity...");
    let cell_propensity = compute_node_membership(edges, &final_membership, n_cells, k);

    let topic_names: Vec<Box<str>> = (0..k).map(|i| i.to_string().into_boxed_str()).collect();
    cell_propensity.to_parquet_with_names(
        &(c.out.to_string() + ".propensity.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&topic_names),
    )?;

    // Gene-topic statistics
    compute_gene_topic_stat(&cell_propensity, &data_vec, c.block_size, &c.out)?;

    // Link community assignments
    info!("Writing link community assignments...");
    write_link_communities(
        &(c.out.to_string() + ".link_community.parquet"),
        edges,
        &final_membership,
        &cell_names,
    )?;

    // Score trace
    info!("Writing score trace...");
    write_score_trace(&(c.out.to_string() + ".scores.parquet"), &score_trace)?;

    info!("Done");
    Ok(())
}
