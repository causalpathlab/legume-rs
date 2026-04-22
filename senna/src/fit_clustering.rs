//! Clustering command for single-cell data
//!
//! Cluster cells based on latent representations (topic proportions, SVD embeddings)

use crate::cluster::{
    hsblock_clustering, kmeans_clustering, leiden_clustering, ClusterMethod, ClusterResult,
};
use crate::cluster_bhc::{run_cluster_bhc, ClusterBhcConfig};
use crate::embed_common::*;
use crate::senna_input::{read_data_on_shared_columns, ReadSharedColumnsArgs};

/// Clustering method CLI enum
#[derive(ValueEnum, Clone, Debug, Default, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ClusterMethodCli {
    /// K-means clustering
    #[default]
    Kmeans,
    /// Leiden clustering (graph-based)
    Leiden,
    /// Hierarchical Stochastic Block Model (graph-based)
    Hsblock,
}

impl From<ClusterMethodCli> for ClusterMethod {
    fn from(cli: ClusterMethodCli) -> Self {
        match cli {
            ClusterMethodCli::Kmeans => ClusterMethod::KMeans,
            ClusterMethodCli::Leiden => ClusterMethod::Leiden,
            ClusterMethodCli::Hsblock => ClusterMethod::Hsblock,
        }
    }
}

#[derive(Args, Debug)]
pub struct ClusteringArgs {
    #[arg(
        long,
        short = 'l',
        required = true,
        help = "Latent representation file (cells × K)",
        long_help = "Latent topic proportions or SVD projection (cells × K matrix).\n\
		     Used as feature space for clustering.\n\n\
		     Expected formats:\n\
		     - From `senna topic`: .latent.parquet (cells × topics)\n\
		     - From `senna svd`: .projection.parquet (cells × components)\n\
		     - First column: cell names"
    )]
    latent: Box<str>,

    #[arg(
        long,
        short = 'k',
        help = "Number of clusters",
        long_help = "Number of clusters for k-means.\n\
		     If not specified, defaults to the number of topics/components in latent.\n\n\
		     Tuning:\n\
		     - Start with number of expected cell types\n\
		     - Use silhouette score or elbow method to optimize\n\
		     - For hierarchical data, start with broader clusters"
    )]
    num_clusters: Option<usize>,

    #[arg(
        long,
        short = 'm',
        default_value = "kmeans",
        help = "Clustering method",
        long_help = "Clustering algorithm:\n\n\
		     - kmeans: K-means clustering (default)\n\
		       Fast, works well for spherical clusters\n\
		       Requires specifying k\n\n\
		     - leiden: Leiden algorithm (graph-based)\n\
		       Finds communities in cell similarity graph\n\
		       Automatically determines number of clusters\n\
		       Use --knn and --resolution to tune\n\n\
		     - hsblock: Hierarchical Stochastic Block Model (graph-based)\n\
		       Collapsed Gibbs sampling + greedy refinement\n\
		       Number of clusters = 2^(tree_depth-1)\n\
		       Use --knn, --tree-depth, and --edge-scale to tune"
    )]
    method: ClusterMethodCli,

    #[arg(long, default_value_t = 100, help = "Maximum iterations for k-means")]
    max_iter: usize,

    #[arg(
        long,
        default_value_t = 15,
        help = "Number of nearest neighbors for graph-based clustering (Leiden/Hsblock)"
    )]
    knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Resolution parameter for Leiden modularity (higher = more clusters, default 1.0)"
    )]
    resolution: f64,

    #[arg(
        long,
        default_value_t = 3,
        help = "Tree depth for HSBM (clusters = 2^(depth-1), default 3 → 4 clusters)"
    )]
    tree_depth: usize,

    #[arg(
        long,
        default_value_t = true,
        help = "Use degree-corrected HSBM (default true)"
    )]
    degree_corrected: bool,

    #[arg(
        long,
        default_value_t = 100.0,
        help = "Edge weight scale for HSBM (default 100.0, scales fuzzy weights to count-like values)"
    )]
    edge_scale: f64,

    #[arg(long, help = "Random seed for graph-based clustering")]
    seed: Option<u64>,

    #[arg(
        long,
        default_value_t = 2,
        help = "Minimum cluster size to report (smaller clusters become unassigned, default 2)"
    )]
    min_cluster_size: usize,

    #[arg(
        long,
        short = 'o',
        required = true,
        help = "Output file prefix",
        long_help = "Output file prefix.\n\n\
		     Generates:\n\
		     - {out}.clusters.parquet: Cluster assignments (cell × cluster)\n\
		     - {out}.bhc.merges.parquet: BHC merge tree (when --data is given)\n\
		     - {out}.bhc.cut.parquet:    BHC consensus cut (when --data is given)"
    )]
    out: Box<str>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Raw count data files (.zarr or .h5) — enables BHC postprocess",
        long_help = "When provided, a Bayesian hierarchical clustering pass runs over\n\
                     the fitted clusters using per-gene sufficient stats\n\
                     T_{k,g}=Σ_{n∈k} y_{n,g} and an empirical-Bayes Dirichlet prior\n\
                     centred on the pooled gene marginal bg. Same recipe pinto uses\n\
                     for link communities. Must match the cell order of --latent.\n\
                     Omit to skip BHC."
    )]
    data_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Per-gene prior strength for the BHC empirical-Bayes Dirichlet prior",
        long_help = "Total Dirichlet concentration γ = bhc_gamma_per_gene × G, where G\n\
                     is the feature dimension. Default 1.0 = Bayes-Laplace (one prior\n\
                     count per gene). Larger values pull every cluster more strongly\n\
                     toward the pooled background, making BHC more eager to merge."
    )]
    bhc_gamma_per_gene: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "log-BF cutoff for the BHC consensus cut (0 = natural Bayes break)"
    )]
    bhc_cut: f64,

    #[arg(
        long,
        default_value_t = 1024,
        help = "Cells per CSC read block when computing BHC sufficient stats"
    )]
    bhc_block_size: usize,
}

pub fn run_clustering(args: &ClusteringArgs) -> anyhow::Result<()> {
    // Read latent representation
    let MatWithNames {
        rows: cell_names,
        cols: _feature_names,
        mat: latent,
    } = read_mat(&args.latent)?;

    info!(
        "Loaded latent representation: {} cells × {} features",
        latent.nrows(),
        latent.ncols()
    );

    // Determine number of clusters
    let k = args.num_clusters.unwrap_or_else(|| {
        let default_k = latent.ncols();
        info!("Number of clusters not specified, using {default_k} (number of features)");
        default_k
    });

    // Run clustering
    let mut result = match args.method {
        ClusterMethodCli::Kmeans => {
            info!(
                "Running k-means clustering with k={}, max_iter={}",
                k, args.max_iter
            );
            kmeans_clustering(&latent, k, args.max_iter)?
        }
        ClusterMethodCli::Leiden => {
            info!(
                "Running Leiden clustering with knn={}, resolution={:.2}, target_k={:?}",
                args.knn, args.resolution, args.num_clusters
            );
            leiden_clustering(
                &latent,
                args.knn,
                args.resolution,
                args.num_clusters,
                args.seed,
            )?
        }
        ClusterMethodCli::Hsblock => {
            info!(
                "Running HSBM clustering with knn={}, tree_depth={}, degree_corrected={}",
                args.knn, args.tree_depth, args.degree_corrected
            );
            hsblock_clustering(
                &latent,
                args.knn,
                args.tree_depth,
                args.degree_corrected,
                args.edge_scale,
                args.seed,
            )?
        }
    };

    // Remove small clusters
    if args.min_cluster_size > 1 {
        result.remove_small_clusters(args.min_cluster_size);
    }

    let n_unassigned = result.labels.iter().filter(|&&l| l == usize::MAX).count();
    info!(
        "Clustering complete: {} cells assigned to {} clusters ({} unassigned)",
        result.labels.len(),
        result.n_clusters,
        n_unassigned,
    );

    // Display cluster statistics when verbose logging is enabled (top 100 biggest)
    if log::log_enabled!(log::Level::Info) {
        eprintln!();
        eprintln!("{}", result.histogram_ascii(50, 100));
        eprintln!();
    }

    // Output cluster assignments as parquet
    let output_file = format!("{}.clusters.parquet", args.out);
    write_cluster_assignments(&result, &cell_names, &output_file)?;

    info!("Wrote cluster assignments to {output_file}");

    if let Some(data_files) = args.data_files.as_ref() {
        run_bhc_postprocess(data_files, &result, &cell_names, args)?;
    }

    Ok(())
}

fn run_bhc_postprocess(
    data_files: &[Box<str>],
    result: &ClusterResult,
    cell_names: &[Box<str>],
    args: &ClusteringArgs,
) -> anyhow::Result<()> {
    info!(
        "BHC: loading raw count data from {} file(s)",
        data_files.len()
    );
    let stack = read_data_on_shared_columns(ReadSharedColumnsArgs {
        data_files: data_files.to_vec(),
        batch_files: None,
        num_types: 1,
        preload: true,
    })?;
    anyhow::ensure!(
        stack.data_stack.stack.len() == 1,
        "BHC: expected a single data stack, got {}",
        stack.data_stack.stack.len()
    );
    let data_vec = &stack.data_stack.stack[0];
    anyhow::ensure!(
        data_vec.num_columns() == cell_names.len(),
        "BHC: data has {} cells but latent has {}",
        data_vec.num_columns(),
        cell_names.len()
    );

    run_cluster_bhc(
        data_vec,
        &result.labels,
        &result.cluster_sizes(),
        &args.out,
        &ClusterBhcConfig {
            gamma_per_gene: args.bhc_gamma_per_gene,
            cutoff: args.bhc_cut,
            block_size: args.bhc_block_size,
        },
    )
}

/// Write cluster assignments to parquet
fn write_cluster_assignments(
    result: &ClusterResult,
    cell_names: &[Box<str>],
    output_path: &str,
) -> anyhow::Result<()> {
    // Create a simple matrix: cells × 1 column (cluster id, NaN if unassigned)
    let mut data = Mat::zeros(cell_names.len(), 1);
    for (i, &cluster_id) in result.labels.iter().enumerate() {
        data[(i, 0)] = if cluster_id == usize::MAX {
            f32::NAN
        } else {
            cluster_id as f32
        };
    }

    let col_names = vec!["cluster".into()];
    data.to_parquet_with_names(
        output_path,
        (Some(cell_names), Some("cell")),
        Some(&col_names),
    )?;

    Ok(())
}
