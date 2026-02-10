//! Clustering command for single-cell data
//!
//! Cluster cells based on latent representations (topic proportions, SVD embeddings)

use crate::cluster::*;
use crate::embed_common::*;
use matrix_util::common_io::*;

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
		       Bayesian inference of hierarchical community structure\n\
		       Number of clusters = 2^(tree_depth-1)\n\
		       Use --knn and --tree-depth to tune"
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
		     - {out}.clusters.parquet: Cluster assignments (cell × cluster)"
    )]
    out: Box<str>,

    #[arg(long, short = 'v', help = "Verbose output with cluster statistics")]
    verbose: bool,
}

pub fn run_clustering(args: &ClusteringArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // Read latent representation
    let MatWithNames {
        rows: cell_names,
        cols: _feature_names,
        mat: latent,
    } = read_matrix(&args.latent)?;

    info!(
        "Loaded latent representation: {} cells × {} features",
        latent.nrows(),
        latent.ncols()
    );

    // Determine number of clusters
    let k = args.num_clusters.unwrap_or_else(|| {
        let default_k = latent.ncols();
        info!(
            "Number of clusters not specified, using {} (number of features)",
            default_k
        );
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

    // Display cluster statistics if verbose (top 100 biggest)
    if args.verbose {
        eprintln!();
        eprintln!("{}", result.histogram_ascii(50, 100));
        eprintln!();
    }

    // Output cluster assignments as parquet
    let output_file = format!("{}.clusters.parquet", args.out);
    write_cluster_assignments(&result, &cell_names, &output_file)?;

    info!("Wrote cluster assignments to {}", output_file);

    Ok(())
}

/// Helper: read matrix file (parquet or text)
fn read_matrix(file_path: &str) -> anyhow::Result<MatWithNames<Mat>> {
    Ok(match file_ext(file_path)?.as_ref() {
        "parquet" => Mat::from_parquet(file_path)?,
        _ => Mat::read_data(file_path, &['\t', ','], None, Some(0), None, None)?,
    })
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
