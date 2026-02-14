mod fit_srt_delta_svd;
mod fit_srt_gene_network;
mod fit_srt_gene_pair_svd;
mod fit_srt_link_community;
mod fit_srt_propensity;
mod srt_cell_pairs;
mod srt_common;
mod srt_estimate_batch_effects;
mod srt_gene_graph;
mod srt_gene_pairs;
mod srt_input;
mod srt_knn_graph;
mod srt_random_projection;

use clap::{Parser, Subcommand};
use colored::Colorize;
use fit_srt_delta_svd::*;
use fit_srt_gene_network::*;
use fit_srt_gene_pair_svd::*;
use fit_srt_link_community::*;
use fit_srt_propensity::*;

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line.replace('▄', &"▄".truecolor(190, 100, 70).to_string())
        .replace('▓', &"▓".truecolor(217, 119, 87).to_string())
        .replace('█', &"█".truecolor(180, 120, 60).to_string())
        .replace('▀', &"▀".truecolor(190, 100, 70).to_string())
        .replace('━', &"━".truecolor(0, 100, 0).to_string())
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    println!(
        " {}",
        "Proximity-based Interaction Network --> Tissue Organization".bold()
    );
    println!();
}

/// PINTO
#[derive(Parser, Debug)]
#[command(
    version,
    about = "PINTO",
    long_about = "Proximity-based Interaction Network analysis to dissect Tissue Organizations\n\n\
                  PINTO identifies spatial cell-cell interaction patterns from spatially-resolved\n\
                  transcriptomics (SRT) data. It constructs spatial cell pairs from KNN graphs,\n\
                  decomposes pair-level expression into shared/difference channels, and learns\n\
                  latent interaction topics via SVD.\n\n\
                  Data files must be `.zarr` or `.h5` (`data-beans`) format. \n\
		  Or convert `.mtx` files using `data-beans from-mtx`.",
    term_width = 80
)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Gene-level shared/difference analysis by SVD",
        long_about = "Gene-level cell-cell interaction analysis by SVD with shared/difference channels.\n\n\
                      Pipeline stages:\n\
                      1. Load SRT data + spatial coordinates\n\
                      2. Estimate and correct gene-level batch effects\n\
                      3. Build spatial cell-cell KNN graph to define cell pairs\n\
                      4. Random projection + binary sort to assign pairs to pseudobulk samples\n\
                      5. Collapse: accumulate shared (log1p(L)+log1p(R)) and diff (|log1p(L)-log1p(R)|) \
                         statistics per gene per sample\n\
                      6. Fit Poisson-Gamma model on each channel\n\
                      7. Randomized SVD on vertically stacked [shared; diff] posterior log means\n\
                      8. Nystrom extension to project individual cell pairs onto the learned basis\n\n\
                      Outputs:\n\
                      - {out}.delta.parquet: batch effect estimates (when multiple batches)\n\
                      - {out}.coord_pairs.parquet: spatial cell pair coordinates\n\
                      - {out}.dictionary.parquet: SVD dictionary matrix (2*n_genes x n_topics)\n\
                      - {out}.latent.parquet: per-pair latent codes (n_pairs x n_topics)"
    )]
    DeltaSvd(SrtDeltaSvdArgs),

    #[command(
        about = "Gene-gene interaction patterns analysis by SVD",
        long_about = "Gene-gene interaction analysis by randomized SVD.\n\n\
                      Discovers gene-gene co-expression patterns within spatial neighbourhoods. \
                      Builds a gene-gene KNN graph from pseudobulk posterior means, computes \
                      positive interaction deltas (raw count products) for each gene pair, and \
                      applies SVD + Nystrom projection.\n\n\
                      Pipeline stages:\n\
                      1. Load SRT data + spatial coordinates\n\
                      2. Build spatial cell-cell KNN graph\n\
                      3. Assign cells to pseudobulk samples (random projection + binary sort)\n\
                      4. Preliminary collapse: gene x sample sums\n\
                      5. Build gene-gene KNN graph from posterior means\n\
                      6. Compute gene-pair positive deltas per cell\n\
                      7. Fit Poisson-Gamma on gene-pair statistics\n\
                      8. SVD on positive delta posterior log means\n\
                      9. Nystrom projection: per-cell then averaged to per-pair latent codes\n\n\
                      Outputs:\n\
                      - {out}.coord_pairs.parquet: spatial cell pair coordinates\n\
                      - {out}.gene_graph.parquet: gene-gene KNN graph edges\n\
                      - {out}.dictionary.parquet: SVD dictionary (n_edges x n_topics)\n\
                      - {out}.latent.parquet: per-pair latent codes (n_pairs x n_topics)"
    )]
    GenePairDeltaSvd(SrtGenePairSvdArgs),

    #[command(
        about = "Bipartite linked community model (VAE)",
        long_about = "Bipartite Linked Community Model for spatial cell-cell interaction analysis.\n\n\
                      Uses a VAE with bilinear decoder to discover edge communities from spatial\n\
                      KNN graphs. Edges are \"colored\" by communities with vertex propensities\n\
                      θ governing participation.\n\n\
                      Pipeline stages:\n\
                      1. Load SRT data + spatial coordinates\n\
                      2. Estimate and correct gene-level batch effects\n\
                      3. Build spatial cell-cell KNN graph to define cell pairs\n\
                      4. Random projection + binary sort to assign pairs to pseudobulk samples\n\
                      5. Collapse: accumulate left/right expression per gene per sample\n\
                      6. Fit Poisson-Gamma model on each channel\n\
                      7. Initialize encoder (LogSoftmaxEncoder) + decoder (BilinearInteractionDecoder)\n\
                      8. Train with jitter resampling from Poisson-Gamma posterior\n\
                      9. Amortized inference: encoder on individual cells → per-cell propensities\n\
                      10. Save outputs\n\n\
                      Outputs:\n\
                      - {out}.delta.parquet: batch effect estimates (when multiple batches)\n\
                      - {out}.coord_pairs.parquet: spatial cell pair coordinates\n\
                      - {out}.dictionary.parquet: gene dictionary λ (G × K, non-negative)\n\
                      - {out}.propensity.parquet: per-cell community propensities θ (n_cells × K)\n\
                      - {out}.log_likelihood.parquet: training trace (llik, kl)"
    )]
    LinkCommunity(SrtLinkCommunityArgs),

    #[command(
        about = "Estimate vertex propensity from edge clusters",
        long_about = "Estimate vertex (cell) propensity scores from edge (cell-pair) cluster assignments.\n\n\
                      Takes latent edge representations from delta-svd, clusters edges \
                      via K-means, then estimates per-vertex propensity as the distribution of \
                      edge cluster memberships across all edges incident to each vertex.\n\n\
                      Inputs:\n\
                      - Latent edge representations (.latent.parquet from delta-svd)\n\
                      - Coordinate pair file (.coord_pairs.parquet)\n\
                      - Optionally, expression data for additional statistics\n\n\
                      Outputs:\n\
                      - {out}.propensity.parquet: per-vertex propensity scores\n\
                      - {out}.edge_cluster.parquet: edge cluster assignments"
    )]
    Propensity(SrtPropensityArgs),

    #[command(
        about = "Visualize gene network with dictionary loadings",
        long_about = "Visualize gene-gene interaction network from gene-pair pipelines.\n\n\
                      Reads gene_graph.parquet and dictionary.parquet produced by gene-pair-delta-svd\n\
                      or gene-pair-delta-topic, computes a 2D spectral layout of genes using\n\
                      fuzzy-kernel weighted similarity, and K-means clusters gene-pair edges\n\
                      by their dictionary vectors.\n\n\
                      Inputs:\n\
                      - Gene graph file (.gene_graph.parquet)\n\
                      - Dictionary file (.dictionary.parquet)\n\n\
                      Outputs:\n\
                      - {out}.gene_coords.parquet: 2D gene node positions (gene, x, y)\n\
                      - {out}.gene_pair_clusters.parquet: edge clusters + topic loadings"
    )]
    Visualize(SrtGeneNetworkArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Propensity(args) => {
            fit_srt_propensity(args)?;
        }
        Commands::DeltaSvd(args) => {
            fit_srt_delta_svd(args)?;
        }
        Commands::GenePairDeltaSvd(args) => {
            fit_srt_gene_pair_svd(args)?;
        }
        Commands::LinkCommunity(args) => {
            fit_srt_link_community(args)?;
        }
        Commands::Visualize(args) => {
            fit_srt_gene_network(args)?;
        }
    }

    Ok(())
}
