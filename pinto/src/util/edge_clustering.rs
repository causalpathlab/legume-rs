//! Shared CLI surface for cutting a pair latent into link communities.
//!
//! `cage` and `dsvd` both end in the same place: an `[K_latent × N_pairs]`
//! projection that has to become per-edge community labels. They therefore
//! offer the same flags, with the same meanings and the same defaults, and
//! resolve them through the same [`EdgeClustering`] enum that
//! `compute_propensity_and_gene_community_stat` consumes.

use crate::link_community::profiles::EdgeClustering;
use clap::{Args, ValueEnum};

/// Clusterer for the per-pair latent. `kmeans` fixes the count;
/// `leiden` discovers it from the graph.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum EdgeClusterMethod {
    Kmeans,
    Leiden,
}

impl std::fmt::Display for EdgeClusterMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kmeans => write!(f, "kmeans"),
            Self::Leiden => write!(f, "leiden"),
        }
    }
}

/// The flags every pair-latent subcommand shares. Flatten this rather than
/// re-declaring them, so the two commands cannot drift apart.
#[derive(Args, Debug, Clone)]
pub struct EdgeClusterArgs {
    #[arg(
        long,
        default_value_t = EdgeClusterMethod::Leiden,
        value_enum,
        help = "How to cut the pair latent into link communities",
        long_help = "leiden is the default.\n\
                     It builds a cosine kNN graph over the pair latent,\n\
                     and --leiden-resolution then decides how many communities exist.\n\
                     Under leiden, --n-edge-clusters is only a target:\n\
                     the resolution is steered toward it, not fixed at it.\n\
                     \n\
                     kmeans instead fixes the community count at --n-edge-clusters.\n\
                     Pick it when you need a specific K, or a run comparable to an older one.\n\
                     Nothing else about the run changes:\n\
                     both consume the same pair latent,\n\
                     and both write the same three propensity tables."
    )]
    pub edge_cluster_method: EdgeClusterMethod,

    #[arg(
        long,
        default_value_t = 30,
        help = "Neighbours per pair in the Leiden kNN graph over the pair latent"
    )]
    pub leiden_knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution; higher gives more, finer communities"
    )]
    pub leiden_resolution: f64,

    #[arg(
        long,
        help = "Link communities to cut from the pair latent [default: let leiden decide]",
        long_help = "How many edge clusters to cut from the pair latent.\n\
                     Under the default --edge-cluster-method leiden this is a TARGET:\n\
                     the resolution is steered toward it, and omitting it lets\n\
                     --leiden-resolution alone decide.\n\
                     Under kmeans it is the exact count, defaulting to the latent width.\n\
                     \n\
                     A cell's propensity is its incident-edge fraction, taken per community.\n\
                     This is the definition `pinto lc` and `pinto dsvd` use.\n\
                     Writes {prefix}.propensity.parquet, {prefix}.link_community.parquet,\n\
                     and {prefix}.gene_community.parquet."
    )]
    pub n_edge_clusters: Option<usize>,
}

impl EdgeClusterArgs {
    /// Resolve to the algorithm-side enum. `latent_width` is the k-means
    /// fallback when `--n-edge-clusters` was left unset; Leiden ignores it,
    /// since an unset target is exactly what lets the graph decide.
    pub fn resolve(&self, latent_width: usize, seed: u64) -> EdgeClustering {
        match self.edge_cluster_method {
            EdgeClusterMethod::Kmeans => EdgeClustering::Kmeans {
                n_clusters: self.n_edge_clusters.unwrap_or(latent_width),
            },
            EdgeClusterMethod::Leiden => EdgeClustering::Leiden {
                knn: self.leiden_knn,
                resolution: self.leiden_resolution,
                target: self.n_edge_clusters,
                seed,
            },
        }
    }
}
