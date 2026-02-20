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
use crate::srt_estimate_batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
use crate::srt_graph_coarsen::*;
use crate::srt_input::{self, *};

use clap::Parser;
use data_beans_alg::random_projection::RandProjOps;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[derive(Parser, Debug, Clone)]
pub struct SrtLinkCommunityArgs {
    #[command(flatten)]
    pub common: srt_input::SrtInputArgs,

    #[arg(
        long,
        default_value_t = 20,
        help = "Number of spatial link communities to discover",
        long_help = "Number of link communities (K). Each edge in the spatial graph\n\
                       is assigned to one of K communities via collapsed Gibbs sampling.\n\
                       Communities capture distinct spatial gene expression patterns.\n\
                       Cell propensity = fraction of edges per community."
    )]
    n_communities: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Number of gene modules for edge profiles (0 = skip gene modules)",
        long_help = "Number of gene modules (M) for edge profile construction.\n\
                       Genes are clustered into M modules via K-means on gene embeddings.\n\
                       Edge profiles are M-dimensional module-count vectors.\n\
                       Set to 0 to skip gene modules and use random-projection edge profiles\n\
                       of dimension --proj-dim instead."
    )]
    n_gene_modules: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Sketch dimension for gene module discovery",
        long_help = "Dimension of random sketches for gene module discovery.\n\
                       Genes are projected into this many dimensions per pseudobulk\n\
                       cluster, then clustered via K-means to form gene modules.\n\
                       Independent of --proj-dim (which is for cell embeddings).\n\
                       Only used when --n-gene-modules > 0."
    )]
    sketch_dim: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Gibbs sampling sweeps per coarsening level",
        long_help = "Number of Gibbs sweeps at each multi-level coarsening level.\n\
                       The finest level uses num_sweeps/2 (minimum 10).\n\
                       More sweeps improve convergence but take longer."
    )]
    num_sweeps: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Max greedy refinement sweeps after Gibbs",
        long_help = "Maximum number of greedy (argmax) sweeps after Gibbs sampling.\n\
                       Each sweep deterministically moves edges to their best community.\n\
                       Stops early if no edges move. Typically converges in 2-5 sweeps."
    )]
    num_greedy: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Min total count to include a gene in projection basis",
        long_help = "Genes with total count below this threshold are zeroed out\n\
                       in the projection basis. Only used when --n-gene-modules=0.\n\
                       Set to 0 to include all genes."
    )]
    min_gene_count: f32,

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
    let c = &args.common;
    let a0 = args.a0 as f64;
    let b0 = args.b0 as f64;
    let k = args.n_communities;
    let n_gm = args.n_gene_modules; // 0 = skip gene modules

    // 1. Load data + coordinates
    info!("Loading data files...");

    let SRTData {
        data: mut data_vec,
        coordinates,
        coordinate_names,
        batches: mut batch_membership,
    } = read_data_with_coordinates(c.to_read_args())?;

    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    anyhow::ensure!(c.proj_dim > 0, "proj_dim must be > 0");
    anyhow::ensure!(args.n_communities > 0, "n_communities must be > 0");
    anyhow::ensure!(c.knn_spatial > 0, "knn_spatial must be > 0");
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
            reciprocal: c.reciprocal,
        },
    )?;

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

    // Wrap graph with data for pair-level operations
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

    // 4. Multi-level cell coarsening
    info!(
        "Graph coarsening ({} levels, {} coarse clusters)...",
        c.num_levels, c.n_pseudobulk
    );

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

    let ml = graph_coarsen_multilevel(
        &srt_cell_pairs.graph,
        &mut cell_proj.proj.clone(),
        &srt_cell_pairs.pairs,
        c.n_pseudobulk,
        c.num_levels,
    );

    // 5-6. Build edge profiles: either via gene modules or random projection
    let gene_to_module: Option<Vec<usize>>;
    let edge_profiles;

    if n_gm > 0 {
        // Gene module discovery via sketch + K-means
        let finest_cell_labels = &ml.all_cell_labels[ml.all_cell_labels.len() - 1];
        let n_finest_clusters = finest_cell_labels.iter().copied().max().unwrap_or(0) + 1;

        info!(
            "Gene module sketch ({} clusters, {} sketch dims)...",
            n_finest_clusters, args.sketch_dim
        );
        let gene_embed = compute_gene_module_sketch(
            &data_vec,
            finest_cell_labels,
            n_finest_clusters,
            args.sketch_dim,
            c.block_size,
        )?;

        info!(
            "K-means on gene embeddings ({} → {} modules)...",
            n_genes, n_gm
        );
        let g2m = gene_embed.kmeans_rows(KmeansArgs {
            num_clusters: n_gm,
            max_iter: 100,
        });

        info!("Building module-count edge profiles...");
        edge_profiles =
            build_edge_profiles_by_module(&data_vec, edges, &g2m, n_gm, c.block_size)?;
        gene_to_module = Some(g2m);
    } else {
        // Skip gene modules: use random-projection edge profiles
        let mut basis = cell_proj.basis.clone();

        if args.min_gene_count > 0.0 {
            info!("Computing gene totals for filtering...");
            let gene_totals = compute_gene_totals(&data_vec, c.block_size)?;
            let n_kept =
                filter_basis_by_gene_count(&mut basis, &gene_totals, args.min_gene_count);
            info!(
                "Kept {}/{} genes (min_count={:.0})",
                n_kept, n_genes, args.min_gene_count
            );
        }

        info!(
            "Building random-projection edge profiles (dim={})...",
            c.proj_dim
        );
        edge_profiles = build_edge_profiles(&data_vec, edges, &basis, None, c.block_size)?;
        gene_to_module = None;
    }

    info!(
        "Edge profiles: {} edges × {} dims, mean size factor: {:.1}",
        edge_profiles.n_edges,
        edge_profiles.m,
        edge_profiles.size_factors.iter().sum::<f32>() / edge_profiles.n_edges as f32
    );

    // 7. Gibbs: coarsest → transfer → refine at each finer level
    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(c.seed));

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

    // Final refinement on original edges (memoized EM parallel by connected components)
    let num_fine_sweeps = (args.num_sweeps / 2).max(10);
    info!(
        "Final Gibbs on full edge set ({} sweeps)...",
        num_fine_sweeps
    );

    let moves = sampler.run_components_em(
        &mut current_labels,
        &edge_profiles,
        &srt_cell_pairs.graph,
        edges,
        k,
        a0,
        b0,
        num_fine_sweeps,
    );
    let fine_score =
        LinkCommunityStats::from_profiles(&edge_profiles, k, &current_labels).total_score(a0, b0);
    info!("Fine Gibbs: {} moves, score={:.2}", moves, fine_score);

    // Greedy finalization (memoized parallel by connected components)
    info!("Greedy finalization ({} max sweeps)...", args.num_greedy);
    let greedy_moves = sampler.run_greedy_by_components(
        &mut current_labels,
        &edge_profiles,
        &srt_cell_pairs.graph,
        edges,
        k,
        a0,
        b0,
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

    // 9. Extract and write outputs
    let gene_names = data_vec.row_names()?;
    let cell_names = data_vec.column_names()?;

    // 9a. cell propensity [N × K]
    info!("Computing cell propensity...");
    let cell_propensity = compute_node_membership(edges, &final_membership, n_cells, k);

    let topic_names: Vec<Box<str>> = (0..k).map(|i| i.to_string().into_boxed_str()).collect();
    cell_propensity.to_parquet_with_names(
        &(c.out.to_string() + ".propensity.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&topic_names),
    )?;

    // 9b. Gene-topic statistics: Poisson-Gamma profiles per community [G × K]
    compute_gene_topic_stat(&cell_propensity, &data_vec, c.block_size, &c.out)?;

    // 9c. Gene module assignments [G × 1] (only when gene modules are used)
    if let Some(ref g2m) = gene_to_module {
        info!("Writing gene module assignments ({} modules)...", n_gm);
        let gene_modules = Mat::from_fn(n_genes, 1, |g, _| g2m[g] as f32);
        let module_col_names: Vec<Box<str>> = vec!["module".into()];
        gene_modules.to_parquet_with_names(
            &(c.out.to_string() + ".gene_modules.parquet"),
            (Some(&gene_names), Some("gene")),
            Some(&module_col_names),
        )?;
    }

    // 9c. Link community assignments
    info!("Writing link community assignments...");
    write_link_communities(
        &(c.out.to_string() + ".link_community.parquet"),
        edges,
        &final_membership,
        &cell_names,
    )?;

    // 9d. Score trace
    info!("Writing score trace...");
    write_score_trace(&(c.out.to_string() + ".scores.parquet"), &score_trace)?;

    info!("Done");
    Ok(())
}

/// Write link community assignments to parquet.
pub(crate) fn write_link_communities(
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

/// ASCII histogram of link community sizes, showing communities with > 1% of edges.
pub(crate) fn link_community_histogram(membership: &[usize], k: usize, max_width: usize) -> String {
    let n = membership.len();
    let mut sizes = vec![0usize; k];
    for &c in membership {
        sizes[c] += 1;
    }

    // Sort non-empty communities by size descending
    let mut ranked: Vec<(usize, usize)> = sizes
        .iter()
        .enumerate()
        .filter(|(_, &s)| s > 0)
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(id, &s)| (id, s))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));

    let max_size = ranked.first().map(|&(_, s)| s).unwrap_or(1);
    let min_edges = n / 100; // 1% threshold

    let mut lines = Vec::new();
    lines.push(format!(
        "Link communities ({} edges, {} non-empty of {}):",
        n,
        ranked.len(),
        k
    ));
    lines.push(String::new());

    let mut shown = 0;
    for &(community_id, size) in &ranked {
        if size <= min_edges {
            break;
        }
        let pct = 100.0 * size as f64 / n as f64;
        let bar_len = ((size as f64 / max_size as f64) * max_width as f64) as usize;
        let bar = "\u{2588}".repeat(bar_len.max(1));
        lines.push(format!(
            "  Community {:3}  {:>7} edges ({:>5.1}%)  {}",
            community_id, size, pct, bar
        ));
        shown += 1;
    }

    let hidden = ranked.len() - shown;
    if hidden > 0 {
        let hidden_edges: usize = ranked[shown..].iter().map(|&(_, s)| s).sum();
        let hidden_pct = 100.0 * hidden_edges as f64 / n as f64;
        lines.push(format!(
            "  ... and {} more ({} edges, {:.1}%)",
            hidden, hidden_edges, hidden_pct
        ));
    }

    lines.join("\n")
}

/// Write score trace to parquet.
pub(crate) fn write_score_trace(file_path: &str, scores: &[f64]) -> anyhow::Result<()> {
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
