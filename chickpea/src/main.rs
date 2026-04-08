mod chickpea_input;
mod cis_mask;
mod coarsening;
mod common;
mod fit_topic;
mod linkage;
mod simulation;
mod topic;

use crate::common::*;
use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");
const ENV_HELP: &str = "Environment variables:\n  RUST_LOG=info  Enable logging to stderr";

fn colorize_logo_line(line: &str) -> String {
    line.replace(':', &":".truecolor(210, 180, 120).to_string())
        .replace('*', &"*".truecolor(240, 210, 140).to_string())
        .replace('o', &"o".truecolor(160, 120, 60).to_string())
        .replace('.', &".".truecolor(180, 160, 110).to_string())
        .replace('^', &"^".truecolor(180, 160, 110).to_string())
        .replace('\'', &"'".truecolor(180, 160, 110).to_string())
        .replace('/', &"/".truecolor(139, 90, 43).to_string())
        .replace('\\', &"\\".truecolor(139, 90, 43).to_string())
        .replace('_', &"_".truecolor(139, 90, 43).to_string())
        .replace('=', &"=".green().to_string())
        .replace('>', &">".bright_green().to_string())
        .replace('-', &"-".green().to_string())
        .replace('|', &"|".green().to_string())
        .replace('▀', &"▀".truecolor(100, 180, 100).to_string())
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    println!(" {}", "Multi-Omic Linkage Analysis".bold());
    println!();
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "chickpea — Multi-Omic Linkage Analysis for paired single-cell RNA + ATAC data",
    long_about = "chickpea — Multi-Omic Linkage Analysis\n\n\
        Discovers cis-regulatory peak-gene links from paired single-cell\n\
        RNA + ATAC data via topic model with SuSiE fine-mapping.\n\n\
        Usage:\n\
          chickpea sim-link -o sim --n-topics 10\n\
          chickpea fit-topic --rna-files sim.rna.zarr --atac-files sim.atac.zarr -o out",
    term_width = 80
)]
struct Cli {
    #[arg(
        short = 'v',
        long,
        global = true,
        help = "Enable verbose logging",
        long_help = "Enable verbose logging to stderr.\n\
                     Equivalent to setting RUST_LOG=info."
    )]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Simulate paired RNA + ATAC data with ground-truth peak-gene links
    #[command(
        long_about = "Simulate paired RNA + ATAC with ground-truth peak-gene links.\n\n\
            Supports nested hierarchical topics: K coarse (ATAC-visible) × K_sub\n\
            subtypes (RNA-only, via linkage). Causal peaks are placed near genes.\n\n\
            Outputs: {out}.rna.zarr, {out}.atac.zarr, {out}.ground_truth.tsv.gz,\n\
            {out}.gene_coords.tsv.gz, {out}.dict.parquet, {out}.prop.parquet",
        after_long_help = ENV_HELP,
	alias = "simulate"
    )]
    SimLink(simulation::SimLinkArgs),

    /// Fit joint topic model to paired RNA + ATAC data
    #[command(
        long_about = "Fit topic model with SuSiE peak-gene linkage.\n\n\
            Uses multi-level pseudobulk collapsing for scalability.\n\
            Indexed encoder with gene-guided ATAC selection.\n\
            Gated RNA decoder interpolates linked and independent dictionaries.\n\n\
            Outputs: {out}.results.bed.gz (linkage PIPs), {out}.prop.parquet,\n\
            {out}.atac_dict.parquet, {out}.rna_dict.parquet, {out}.gate_alpha.parquet",
        after_long_help = ENV_HELP,
        alias = "topic"
    )]
    FitTopic(fit_topic::FitTopicArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    if cli.verbose && std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    match &cli.commands {
        Commands::SimLink(args) => simulation::run_sim_link(args),
        Commands::FitTopic(args) => fit_topic::fit_topic_model(args),
    }
}
