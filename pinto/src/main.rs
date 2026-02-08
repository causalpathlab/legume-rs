mod fit_srt_gene_pair_svd;
mod fit_srt_gene_pair_topic;
mod fit_srt_delta_svd;
mod fit_srt_propensity;
mod fit_srt_topic;
mod srt_cell_pairs;
mod srt_common;
mod srt_estimate_batch_effects;
mod srt_gene_graph;
mod srt_gene_pairs;
mod srt_input;
mod srt_knn_graph;
mod srt_random_projection;

use fit_srt_gene_pair_svd::*;
use fit_srt_gene_pair_topic::*;
use fit_srt_delta_svd::*;
use fit_srt_propensity::*;
use fit_srt_topic::*;

use clap::{Parser, Subcommand};
use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line.replace('▄', &"▄".truecolor(139, 90, 43).to_string())
        .replace('▓', &"▓".truecolor(139, 90, 43).to_string())
        .replace('█', &"█".truecolor(180, 120, 60).to_string())
        .replace('▀', &"▀".truecolor(139, 90, 43).to_string())
        .replace('─', &"─".green().to_string())
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    println!(" {}", "Proximity-based Interaction Network --> Tissue Organization".bold());
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
                  latent interaction topics via SVD or neural topic models.\n\n\
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
        about = "Gene-level shared/difference by topic model",
        long_about = "Gene-level cell-cell interaction analysis by topics with shared/difference channels.\n\n\
                      Pipeline stages:\n\
                      1. Load SRT data + spatial coordinates\n\
                      2. Estimate and correct gene-level batch effects\n\
                      3. Build spatial cell-cell KNN graph to define cell pairs\n\
                      4. Random projection + binary sort to assign pairs to pseudobulk samples\n\
                      5. Collapse: accumulate shared/diff statistics per gene per sample\n\
                      6. Fit Poisson-Gamma model on each channel\n\
                      7. Train encoder-decoder topic model on [shared; diff] posterior samples via SGD\n\
                      8. Encode individual cell pairs into latent topic proportions\n\n\
                      Outputs:\n\
                      - {out}.delta.parquet: batch effect estimates (when multiple batches)\n\
                      - {out}.coord_pairs.parquet: spatial cell pair coordinates\n\
                      - {out}.dictionary.parquet: topic dictionary (2*n_genes x n_topics)\n\
                      - {out}.latent.parquet: per-pair latent topic proportions (n_pairs x n_topics)\n\
                      - {out}.log_likelihood.gz: training log-likelihoods"
    )]
    DeltaTopic(SrtTopicArgs),

    #[command(
        about = "Gene-gene interaction patterns analysis by SVD",
        long_about = "Gene-gene interaction analysis by randomized SVD.\n\n\
                      Discovers gene-gene co-expression patterns within spatial neighbourhoods. \
                      Builds a gene-gene KNN graph from pseudobulk posterior means, computes \
                      directional deltas (delta_pos/delta_neg) for each gene pair, and applies \
                      SVD + Nystrom projection.\n\n\
                      Pipeline stages:\n\
                      1. Load SRT data + spatial coordinates\n\
                      2. Build spatial cell-cell KNN graph\n\
                      3. Assign cells to pseudobulk samples (random projection + binary sort)\n\
                      4. Preliminary collapse: gene x sample sums\n\
                      5. Build gene-gene KNN graph from posterior means\n\
                      6. Compute gene-pair directional deltas per cell\n\
                      7. Fit Poisson-Gamma on gene-pair statistics\n\
                      8. SVD on stacked [delta_pos; delta_neg] posterior log means\n\
                      9. Nystrom projection: per-cell then averaged to per-pair latent codes\n\n\
                      Outputs:\n\
                      - {out}.coord_pairs.parquet: spatial cell pair coordinates\n\
                      - {out}.gene_graph.parquet: gene-gene KNN graph edges\n\
                      - {out}.gene_pairs.parquet: gene-pair delta statistics\n\
                      - {out}.dictionary.parquet: SVD dictionary (2*n_edges x n_topics)\n\
                      - {out}.latent.parquet: per-pair latent codes (n_pairs x n_topics)"
    )]
    GenePairDeltaSvd(SrtGenePairSvdArgs),

    #[command(
        about = "Gene-gene interaction analysis by topic model",
        long_about = "Gene-gene interaction analysis by neural topic model.\n\n\
                      Same gene-pair pipeline as gene-pair-delta-svd, but replaces SVD with \
                      an encoder-decoder topic model trained via SGD.\n\n\
                      Pipeline stages:\n\
                      1. Load SRT data + spatial coordinates\n\
                      2. Build spatial cell-cell KNN graph\n\
                      3. Assign cells to pseudobulk samples\n\
                      4. Preliminary collapse + gene-gene KNN graph\n\
                      5. Compute gene-pair directional deltas\n\
                      6. Fit Poisson-Gamma on gene-pair statistics\n\
                      7. Train encoder-decoder topic model on [delta_pos; delta_neg] posterior samples\n\
                      8. Encode individual cells into latent topics, average to per-pair codes\n\n\
                      Outputs:\n\
                      - {out}.coord_pairs.parquet: spatial cell pair coordinates\n\
                      - {out}.gene_graph.parquet: gene-gene KNN graph edges\n\
                      - {out}.gene_pairs.parquet: gene-pair delta statistics\n\
                      - {out}.dictionary.parquet: topic dictionary (2*n_edges x n_topics)\n\
                      - {out}.latent.parquet: per-pair latent topic proportions\n\
                      - {out}.log_likelihood.gz: training log-likelihoods"
    )]
    GenePairDeltaTopic(SrtGenePairTopicArgs),

    #[command(
        about = "Estimate vertex propensity from edge clusters",
        long_about = "Estimate vertex (cell) propensity scores from edge (cell-pair) cluster assignments.\n\n\
                      Takes latent edge representations from delta-svd/delta-topic, clusters edges \
                      via K-means, then estimates per-vertex propensity as the distribution of \
                      edge cluster memberships across all edges incident to each vertex.\n\n\
                      Inputs:\n\
                      - Latent edge representations (.latent.parquet from delta-svd or delta-topic)\n\
                      - Coordinate pair file (.coord_pairs.parquet)\n\
                      - Optionally, expression data for additional statistics\n\n\
                      Outputs:\n\
                      - {out}.propensity.parquet: per-vertex propensity scores\n\
                      - {out}.edge_cluster.parquet: edge cluster assignments"
    )]
    Propensity(SrtPropensityArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Propensity(args) => {
            fit_srt_propensity(args)?;
        }
        Commands::DeltaTopic(args) => {
            fit_srt_delta_topic(args)?;
        }
        Commands::DeltaSvd(args) => {
            fit_srt_delta_svd(args)?;
        }
        Commands::GenePairDeltaSvd(args) => {
            fit_srt_gene_pair_svd(args)?;
        }
        Commands::GenePairDeltaTopic(args) => {
            fit_srt_gene_pair_topic(args)?;
        }
    }

    Ok(())
}
