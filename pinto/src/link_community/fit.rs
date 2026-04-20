//! Link community model pipeline for spatial transcriptomics.
//!
//! Discovers link communities from spatial cell-cell KNN graphs via
//! collapsed Gibbs sampling on gene-projected edge profiles.

use crate::gene_network::graph::*;
use crate::gene_network::pairs::*;
use crate::link_community::gibbs::{ComponentGibbsArgs, LinkGibbsSampler};
use crate::link_community::model::{LinkCommunityStats, LinkProfileStore};
use crate::link_community::module_cluster::{self, KmeansModules, LeidenModules, ModuleClusterer};
use crate::link_community::profiles::*;
use crate::util::batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
use crate::util::cell_pairs::*;
use crate::util::common::*;
use crate::util::graph_coarsen::*;
use crate::util::input::*;

/// Gene-pair profile construction state, built from an external gene network.
struct GenePairProfileState {
    gene_adj: Vec<Vec<(usize, usize)>>,
    gene_means: DVec,
    n_gene_pairs: usize,
    module_collapse: Option<Vec<usize>>,
}

use clap::Parser;
use data_beans_alg::random_projection::RandProjOps;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Module clustering method for gene/gene-pair module discovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ModuleMethod {
    Kmeans,
    Leiden,
}

impl std::fmt::Display for ModuleMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleMethod::Kmeans => write!(f, "kmeans"),
            ModuleMethod::Leiden => write!(f, "leiden"),
        }
    }
}

#[derive(Parser, Debug, Clone)]
pub struct SrtLinkCommunityArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

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
        help = "Number of gene modules (default: sqrt of number of genes)",
        long_help = "Number of gene modules (M) for edge profile construction.\n\
                       Genes are clustered into M modules via --module-method (kmeans\n\
                       or leiden) on gene embeddings. Edge profiles are M-dimensional\n\
                       module-count vectors. Default: auto = sqrt(number of genes).\n\
                       With leiden, this is a target (actual count may differ slightly).\n\
                       Use --no-gene-modules to skip and use random-projection profiles."
    )]
    n_gene_modules: Option<usize>,

    #[arg(
        long,
        conflicts_with = "n_gene_modules",
        help = "Skip gene modules, use projection profiles"
    )]
    no_gene_modules: bool,

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
        help = "Gibbs iterations at the coarsest level",
        long_help = "Number of Gibbs iterations at the coarsest coarsening level.\n\
                       The finest coarsening level uses num_gibbs/5 (minimum 10).\n\
                       Full-resolution EM iterations are controlled by --num-em."
    )]
    num_gibbs: usize,

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
        help = "EM Gibbs sweeps on full edge set",
        long_help = "Number of EM Gibbs sweeps on full-resolution edges.\n\
                       Set to 0 to skip EM entirely and use only greedy refinement.\n\
                       If omitted, defaults to num_gibbs/4 (minimum 5)."
    )]
    num_em: Option<usize>,

    #[arg(
        long,
        help = "Dirichlet concentration for community mixing weights",
        long_help = "Concentration parameter α for the symmetric Dirichlet prior\n\
                       on community mixing weights. Enables variational truncation:\n\
                       communities with few edges are naturally pruned.\n\
                       Set to 0 to disable (uniform prior).\n\
                       If omitted, auto-scaled from edge profile sparsity:\n\
                       α = mean_size_factor / K, so sparser data gets a weaker prior."
    )]
    alpha: Option<f32>,

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
        help = "External gene-gene network file (two-column TSV: gene1, gene2)",
        long_help = "External gene-gene network file (two-column TSV: gene1, gene2).\n\
                       When provided, edge profiles are built from gene-pair interaction\n\
                       deltas instead of gene modules. Each edge e=(i,j) gets a profile\n\
                       y_e[p] = sum of positive co-expression deltas for gene pair p."
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
        default_value_t = 100,
        help = "Max gene-pair dimensions; cluster into modules if exceeded",
        long_help = "Maximum number of gene-pair dimensions for edge profiles.\n\
                       When the number of gene pairs exceeds this threshold,\n\
                       gene pairs are clustered into modules via K-means and\n\
                       edge profiles are summed per module. Set to 0 to disable.\n\
                       Only used with --gene-network."
    )]
    n_edge_modules: usize,

    #[arg(
        long,
        default_value_t = ModuleMethod::Kmeans,
        help = "Module clustering method: kmeans or leiden",
        long_help = "Clustering algorithm for gene module discovery.\n\
                       kmeans: fast, deterministic cluster count (= --n-gene-modules).\n\
                       leiden: graph-based, builds KNN on gene embeddings then\n\
                       runs Leiden with resolution tuning to approximate the\n\
                       target module count. Often finds more biologically coherent\n\
                       modules. Use with --n-outer-iter > 1 for best results."
    )]
    module_method: ModuleMethod,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Resolution for Leiden module clustering",
        long_help = "Modularity resolution for Leiden module clustering.\n\
                       Higher values produce more modules, lower values fewer.\n\
                       When --n-gene-modules is set, resolution is auto-tuned\n\
                       via binary search to approximate the target count.\n\
                       Only used with --module-method leiden."
    )]
    module_resolution: f64,

    #[arg(
        long,
        default_value_t = 10,
        help = "KNN for Leiden module graph",
        long_help = "Number of nearest neighbours for building the gene-gene\n\
                       similarity graph used by Leiden module clustering.\n\
                       Only used with --module-method leiden."
    )]
    module_knn: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Outer EM iterations for module re-estimation",
        long_help = "Outer EM iterations that alternate:\n\
                       E-step: Gibbs + greedy community assignment.\n\
                       M-step: re-cluster gene modules from community centroids.\n\
                       Default 1 = fixed modules (no re-estimation).\n\
                       Set to 2-5 to let modules adapt to discovered communities.\n\
                       Stops early if score plateaus (<0.1% improvement).\n\
                       Applies to gene-module and gene-pair-module paths only;\n\
                       ignored with --no-gene-modules (no M-step to iterate)."
    )]
    n_outer_iter: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable IDF background correction (on by default)",
        long_help = "By default, edge profiles are reweighted by inverse-frequency\n\
                       w_g = -ln(π_bg,g + ε), where π_bg is the empirical gene-module\n\
                       marginal over all edges. Housekeeping genes (high π_bg) get\n\
                       small weight, specific genes (low π_bg) get large weight — so\n\
                       community assignment is driven by distinctive genes, not by\n\
                       bulk expression level (DC-SBM degree correction with θ_g = π_bg).\n\
                       Pass --no-background to disable and use raw counts."
    )]
    no_background: bool,
}

/// Link community model pipeline.
///
/// 1.  Load data + coordinates
/// 2.  Estimate batch effects
/// 3.  Build spatial KNN graph
/// 4.  Multi-level cell coarsening
/// 5.  Gene module discovery (sketch + K-means) — or random-projection / gene-pairs
/// 6.  Build edge profiles; optionally IDF-weight by empirical gene marginal
///     (DC-SBM degree correction ~ housekeeping frequency)
/// 7.  Collapsed Gibbs on coarsest → transfer → refine at finer levels
///     (entropy / multinomial DC-SBM objective)
/// 8.  Extract and write outputs
pub fn fit_srt_link_community(args: &SrtLinkCommunityArgs) -> anyhow::Result<()> {
    let c = &args.common;
    let k = args.n_communities;
    let n_outer_iter = args.n_outer_iter.max(1);
    // n_gene_modules resolved after data loading (may need median cell nnz)

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

    // Auto-detect batches from connected components (opt-in via --auto-batch)
    if c.auto_batch && c.batch_files.is_none() {
        crate::util::input::auto_batch_from_components(&graph, &mut batch_membership);
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

    // Resolve gene module count
    let n_gm: Option<usize> = if args.no_gene_modules {
        None
    } else if let Some(n) = args.n_gene_modules {
        Some(n)
    } else {
        // Auto: sqrt(number of genes)
        let n_genes = data_vec.num_rows();
        let n = (n_genes as f64).sqrt().round() as usize;
        let n = n.max(10);
        info!("Auto n_gene_modules = {} (sqrt of {} genes)", n, n_genes);
        Some(n)
    };

    // Build module clusterer from CLI args
    let make_clusterer = |target: usize| -> Box<dyn ModuleClusterer> {
        match args.module_method {
            ModuleMethod::Leiden => Box::new(LeidenModules {
                knn: args.module_knn,
                resolution: args.module_resolution,
                target_modules: Some(target),
                seed: Some(c.seed),
            }),
            ModuleMethod::Kmeans => Box::new(KmeansModules {
                n_modules: target,
                max_iter: 100,
            }),
        }
    };
    let gene_clusterer = make_clusterer(n_gm.unwrap_or(10));
    let edge_clusterer = make_clusterer(args.n_edge_modules.max(2));

    // Gene-pair mode: load external network before borrowing data_vec
    let mut gene_pair_state: Option<(GenePairGraph, DVec)> = None;
    if let Some(ref network_file) = args.gene_network {
        let gene_names = data_vec.row_names()?;
        info!("Loading external gene network from {}...", network_file);
        let gene_graph = GenePairGraph::from_edge_list(
            network_file,
            gene_names,
            args.gene_network_allow_prefix,
            args.gene_network_delimiter,
        )?;
        anyhow::ensure!(
            gene_graph.num_edges() > 0,
            "Gene network matched 0 gene pairs. Check that gene names in {} \
             match the data gene names (use --gene-network-delimiter or \
             --gene-network-allow-prefix for fuzzy matching).",
            network_file
        );
        info!("Gene graph: {} gene pairs", gene_graph.num_edges());
        let gene_means = compute_gene_raw_means(&data_vec, c.block_size)?;
        gene_pair_state = Some((gene_graph, gene_means));
    }

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

    // 5-6. Build COARSE edge profiles first
    let mut gene_to_module: Option<Vec<usize>>;
    let coarse_profiles;
    let proj_basis: Option<Mat>;

    // Gene-pair profile state: (gene_adj, gene_means, n_gene_pairs, module_collapse)
    let mut gp_profile_state: Option<GenePairProfileState>;

    // Build super-edges at coarsest level
    let coarsest_labels = &ml.all_cell_labels[0];
    let (super_edges, fine_to_super) = build_super_edges(edges, coarsest_labels);
    let n_super = super_edges.len();
    info!(
        "Coarsest level: {} super-edges from {} fine edges",
        n_super, n_edges
    );

    if let Some((ref mut gene_graph, ref gene_means)) = gene_pair_state {
        // Gene-pair profile path
        let n_gene_pairs = gene_graph.num_edges();

        info!("Building COARSE gene-pair profiles...");
        let gene_adj = gene_graph.build_directed_adjacency();
        let raw_profiles = build_edge_profiles_by_gene_pairs(
            &data_vec,
            &super_edges,
            &gene_adj,
            gene_means,
            n_gene_pairs,
            c.block_size,
        )?;

        // Elbow filtering on coarse column sums
        let col_sums: Vec<f32> = (0..n_gene_pairs)
            .map(|p| {
                let mut s = 0.0f32;
                for e in 0..n_super {
                    s += raw_profiles.profile(e)[p];
                }
                s
            })
            .collect();

        let (threshold, elbow_rank) = elbow_threshold(&col_sums);

        let keep_cols: Vec<usize> = (0..n_gene_pairs)
            .filter(|&p| col_sums[p] > threshold)
            .collect();

        let (filtered_profiles, final_n_gene_pairs) = if keep_cols.len() < n_gene_pairs {
            info!(
                "Elbow threshold: {:.4} (rank {}), kept {}/{} gene pairs",
                threshold,
                elbow_rank,
                keep_cols.len(),
                n_gene_pairs
            );
            gene_graph.filter_edges(&keep_cols);
            (
                filter_profile_columns(&raw_profiles, &keep_cols),
                keep_cols.len(),
            )
        } else {
            (raw_profiles, n_gene_pairs)
        };

        // Write gene graph
        gene_graph.to_parquet(&(c.out.to_string() + ".gene_graph.parquet"))?;

        // Rebuild gene adjacency after filtering
        let gene_adj = gene_graph.build_directed_adjacency();

        // Collapse gene-pair columns into modules if too many
        let (final_profiles, module_collapse) =
            if args.n_edge_modules > 0 && final_n_gene_pairs > args.n_edge_modules {
                info!(
                    "Clustering {} gene pairs into {} edge modules...",
                    final_n_gene_pairs, args.n_edge_modules
                );
                let (module_profiles, assignments) =
                    collapse_profile_columns(&filtered_profiles, &*edge_clusterer);
                (module_profiles, Some(assignments))
            } else {
                (filtered_profiles, None)
            };

        coarse_profiles = final_profiles;
        gp_profile_state = Some(GenePairProfileState {
            gene_adj,
            gene_means: gene_means.clone(),
            n_gene_pairs: final_n_gene_pairs,
            module_collapse,
        });
        gene_to_module = None;
        proj_basis = None;
    } else if n_gm.is_some() {
        // Gene module discovery via sketch + clustering
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
            "Clustering gene embeddings ({} genes, method={:?})...",
            n_genes, args.module_method
        );
        let mut g2m = gene_clusterer.cluster(&gene_embed);
        module_cluster::compact_labels(&mut g2m);
        let n_gm_actual = g2m.iter().copied().max().unwrap_or(0) + 1;
        info!("{} gene modules discovered", n_gm_actual);

        info!("Building COARSE module-count edge profiles...");
        coarse_profiles = build_edge_profiles_by_module(
            &data_vec,
            &super_edges,
            &g2m,
            n_gm_actual,
            c.block_size,
        )?;
        gene_to_module = Some(g2m);
        proj_basis = None;
        gp_profile_state = None;
    } else {
        // Skip gene modules: use random-projection edge profiles
        let mut basis = cell_proj.basis.clone();

        if args.min_gene_count > 0.0 {
            info!("Computing gene totals for filtering...");
            let gene_totals = compute_gene_totals(&data_vec, c.block_size)?;
            let n_kept = filter_basis_by_gene_count(&mut basis, &gene_totals, args.min_gene_count);
            info!(
                "Kept {}/{} genes (min_count={:.0})",
                n_kept, n_genes, args.min_gene_count
            );
        }

        info!(
            "Building COARSE random-projection edge profiles (dim={})...",
            c.proj_dim
        );
        coarse_profiles = build_edge_profiles(&data_vec, &super_edges, &basis, c.block_size)?;
        gene_to_module = None;
        proj_basis = Some(basis);
        gp_profile_state = None;
    }

    let mean_sf = coarse_profiles.size_factors.iter().sum::<f32>() / coarse_profiles.n_edges as f32;
    let m = coarse_profiles.m;
    info!(
        "Coarse edge profiles: {} super-edges × {} dims, mean size factor: {:.1}",
        coarse_profiles.n_edges, m, mean_sf
    );

    // Unless --no-background, compute empirical gene-module marginal from coarse
    // profiles and apply IDF weighting in-place: y'_eg = w_g · y_eg with
    // w_g = -ln(π_bg,g + ε). Housekeeping genes (high π_bg) are down-weighted;
    // rare genes are amplified. This is the DC-SBM degree correction (θ_g = π_bg).
    let bg: Option<Vec<f64>> = if !args.no_background {
        let dist = coarse_profiles.empirical_marginal();
        info!(
            "Background distribution: min={:.2e}, max={:.2e}, effective_dims={:.1}",
            dist.iter().cloned().fold(f64::INFINITY, f64::min),
            dist.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            (-dist
                .iter()
                .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
                .sum::<f64>())
            .exp()
        );
        Some(dist)
    } else {
        None
    };

    let mut coarse_profiles = coarse_profiles;
    if let Some(ref bg_dist) = bg {
        coarse_profiles.weight_by_idf(bg_dist);
        info!("Applied IDF weighting to coarse profiles (housekeeping genes down-weighted)");
    }

    // 7. Gibbs on coarsest level
    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(c.seed));
    let init_labels: Vec<usize> = (0..coarse_profiles.n_edges).map(|e| e % k).collect();

    let mut coarse_stats = LinkCommunityStats::from_profiles(&coarse_profiles, k, &init_labels);

    info!("Gibbs on coarsest ({} sweeps)...", args.num_gibbs);
    let moves = sampler.run_parallel(&mut coarse_stats, &coarse_profiles, args.num_gibbs);
    let coarse_score = coarse_stats.total_score();
    info!(
        "Coarsest Gibbs: {} total moves, score={:.2}",
        moves, coarse_score
    );
    score_trace.push(coarse_score);

    // 7b. Transfer labels to full resolution and build profiles
    let mut current_labels = transfer_labels(&fine_to_super, &coarse_stats.membership);
    let all_edge_indices: Vec<usize> = (0..n_edges).collect();

    let (mut full_profiles, full_raw_gp_profiles) = build_full_profiles_inline(
        &data_vec,
        &all_edge_indices,
        edges,
        &gp_profile_state,
        &gene_to_module,
        &proj_basis,
    )?;
    if let Some(ref bg_dist) = bg {
        // Reuse the coarse-derived bg for the full profiles (same M layout when
        // --no-gene-modules; for module paths the M-step recomputes bg below).
        full_profiles.weight_by_idf(bg_dist);
    }

    info!(
        "Full profiles: {} edges × {} dims ({:.1} MB)",
        full_profiles.n_edges,
        full_profiles.m,
        (full_profiles.profiles.len() * std::mem::size_of::<f32>()) as f64 / 1_048_576.0
    );

    // Resolve alpha: auto-scale from full-resolution (not coarse) edge sparsity,
    // so the Dirichlet prior reflects the actual per-edge signal strength.
    let alpha: f64 = args.alpha.map_or_else(
        || {
            let full_mean_sf = full_profiles.size_factors.iter().sum::<f32>()
                / full_profiles.n_edges.max(1) as f32;
            let v = (full_mean_sf as f64 / k as f64).max(0.01);
            info!(
                "Auto alpha = {:.4} (mean_size_factor {:.1} / K={})",
                v, full_mean_sf, k
            );
            v
        },
        |v| v as f64,
    );
    let comp_args = ComponentGibbsArgs {
        graph: &srt_cell_pairs.graph,
        edges,
        k,
        alpha,
    };

    let num_fine_sweeps = args.num_em.unwrap_or((args.num_gibbs / 4).max(5));

    // 8. Outer EM loop: E-step (Gibbs+greedy) → M-step (re-cluster modules).
    //    Without gene modules the M-step is a no-op, so one iteration suffices.
    let has_mstep = gene_to_module.is_some()
        || gp_profile_state
            .as_ref()
            .is_some_and(|gp| gp.module_collapse.is_some());
    let effective_outer = if has_mstep { n_outer_iter } else { 1 };
    let mut prev_score = f64::NEG_INFINITY;

    for outer_iter in 0..effective_outer {
        if effective_outer > 1 {
            info!(
                "=== Outer iteration {}/{} ===",
                outer_iter + 1,
                effective_outer
            );
        }

        // --- E-step: EM Gibbs + greedy ---
        if num_fine_sweeps > 0 {
            info!("EM Gibbs on full edge set ({} sweeps)...", num_fine_sweeps);
            let moves = sampler.run_components_em(
                &mut current_labels,
                &full_profiles,
                &comp_args,
                num_fine_sweeps,
            );
            info!("EM Gibbs: {} moves", moves);
        }

        info!("Greedy finalization ({} max sweeps)...", args.num_greedy);
        let greedy_moves = sampler.run_greedy_by_components(
            &mut current_labels,
            &full_profiles,
            &comp_args,
            args.num_greedy,
        );

        let fine_stats = LinkCommunityStats::from_profiles(&full_profiles, k, &current_labels);
        let score = fine_stats.total_score();
        info!("Greedy: {} moves, score={:.2}", greedy_moves, score);
        score_trace.push(score);

        // Last iteration or no modules to update → done
        if outer_iter + 1 >= n_outer_iter {
            break;
        }

        // Score-based early stopping
        if score <= prev_score * 1.001 && prev_score > f64::NEG_INFINITY {
            info!("Outer EM converged (score plateau), stopping early");
            break;
        }
        prev_score = score;

        // --- M-step: re-estimate modules from community-conditioned stats ---
        if let Some(ref mut g2m) = gene_to_module {
            // Gene-module path: compute [G × K] centroids → re-cluster genes
            info!("M-step: recomputing gene modules from community centroids...");
            let centroids =
                compute_community_centroids(&data_vec, edges, &current_labels, k, c.block_size)?;
            let mut new_g2m = gene_clusterer.cluster(&centroids);
            module_cluster::compact_labels(&mut new_g2m);
            let new_n_gm = new_g2m.iter().copied().max().unwrap_or(0) + 1;
            info!("Re-estimated {} gene modules", new_n_gm);
            *g2m = new_g2m;

            // Rebuild profiles with new modules
            let (new_profiles, _) = build_full_profiles_inline(
                &data_vec,
                &all_edge_indices,
                edges,
                &gp_profile_state,
                &gene_to_module,
                &proj_basis,
            )?;
            full_profiles = new_profiles;
            if bg.is_some() {
                let dist = full_profiles.empirical_marginal();
                full_profiles.weight_by_idf(&dist);
            }
        } else if let (Some(raw), Some(ref mut gp)) = (&full_raw_gp_profiles, &mut gp_profile_state)
        {
            if gp.module_collapse.is_some() {
                // Gene-pair-module path: compute rates from raw stats → re-cluster pairs
                info!("M-step: recomputing gene-pair modules from community rates...");
                let raw_stats = LinkCommunityStats::from_profiles(raw, k, &current_labels);
                let rate_mat = compute_module_rate_matrix(&raw_stats);
                let mut new_collapse = edge_clusterer.cluster(&rate_mat);
                module_cluster::compact_labels(&mut new_collapse);
                let new_n_mod = new_collapse.iter().copied().max().unwrap_or(0) + 1;
                info!("Re-estimated {} edge modules", new_n_mod);

                // Re-collapse raw profiles (no I/O!)
                full_profiles = raw.collapse_modules(&new_collapse);
                if bg.is_some() {
                    let dist = full_profiles.empirical_marginal();
                    full_profiles.weight_by_idf(&dist);
                }
                gp.module_collapse = Some(new_collapse);
            }
        }
        // else: projection path — no modules to update
    }

    // Final stats (from_profiles builds fresh — no drift to correct)
    let fine_stats = LinkCommunityStats::from_profiles(&full_profiles, k, &current_labels);
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
        let n_modules = g2m.iter().copied().max().unwrap_or(0) + 1;
        info!("Writing gene module assignments ({} modules)...", n_modules);
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

/// Build full-resolution edge profiles, optionally keeping raw (un-collapsed)
/// gene-pair profiles for the M-step.
///
/// Returns `(collapsed_profiles, Option<raw_gp_profiles>)`.
fn build_full_profiles_inline(
    data: &SparseIoVec,
    edge_indices: &[usize],
    all_edges: &[(usize, usize)],
    gp_state: &Option<GenePairProfileState>,
    gene_to_module: &Option<Vec<usize>>,
    proj_basis: &Option<Mat>,
) -> anyhow::Result<(LinkProfileStore, Option<LinkProfileStore>)> {
    if let Some(ref gp) = *gp_state {
        let raw = build_gene_pair_profiles_for_edges(
            data,
            edge_indices,
            all_edges,
            &gp.gene_adj,
            &gp.gene_means,
            gp.n_gene_pairs,
        )?;
        if let Some(ref assignments) = gp.module_collapse {
            let collapsed = raw.collapse_modules(assignments);
            Ok((collapsed, Some(raw)))
        } else {
            Ok((raw, None))
        }
    } else if let Some(ref g2m) = *gene_to_module {
        let n_modules = g2m.iter().copied().max().unwrap_or(0) + 1;
        let profiles =
            build_module_profiles_for_edges(data, edge_indices, all_edges, g2m, n_modules)?;
        Ok((profiles, None))
    } else {
        let basis = proj_basis.as_ref().expect("projection basis missing");
        let profiles = build_projection_profiles_for_edges(data, edge_indices, all_edges, basis)?;
        Ok((profiles, None))
    }
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
