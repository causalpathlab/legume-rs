mod fit_srt_gene_pair_svd;
mod fit_srt_gene_pair_topic;
mod fit_srt_delta_svd;
mod fit_srt_propensity;
mod fit_srt_svd;
mod fit_srt_topic;
mod srt_cell_pairs;
mod srt_collapse_pairs;
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
use fit_srt_svd::*;
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

/// Proximity-based Interaction Network analysis to dissect Tissue
/// Organizations
///
/// Data files of either `.zarr` or `.h5` format. All the formats in
/// the given list should be identical. We can convert `.mtx` to
/// `.zarr` or `.h5` using `data-beans from-mtx` or similar commands.
///
#[derive(Parser, Debug)]
#[command(version, about, long_about, term_width = 80)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// by randomized singular value decomposition
    Svd(SrtSvdArgs),
    /// Estimate vertex propensity with clustering
    Propensity(SrtPropensityArgs),
    /// by topic modelling
    DeltaTopic(SrtTopicArgs),
    /// delta SVD with shared/difference channels
    DeltaSvd(SrtDeltaSvdArgs),
    /// gene-gene interaction analysis by SVD
    GenePairDeltaSvd(SrtGenePairSvdArgs),
    /// gene-gene interaction analysis by topic modelling
    GenePairDeltaTopic(SrtGenePairTopicArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Svd(args) => {
            fit_srt_svd(args)?;
        }
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
