mod common;
mod p2g;
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
          data-beans-sim multiome -o sim --n-topics 10\n\
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
    FitTopic(topic::fit::FitTopicArgs),

    /// Fine-map cis peak→gene links via SuSiE-RSS on pseudobulk summary stats
    #[command(
        long_about = "Link ATAC peaks to RNA genes by summary-statistics fine-mapping.\n\n\
            Pseudobulk the matched RNA + ATAC cells and embed peaks (and the\n\
            projected genes) in a shared ATAC latent space. Score each cis\n\
            peak–gene pair by a log-linear regression z in that space, then\n\
            fine-map per gene with SuSiE-RSS using the peak–peak correlation\n\
            (LD) structure. Lighter and faster than `fit-topic` (no neural model).\n\n\
            Outputs: {out}.results.bed.gz (chr start end peak_id gene_id pip\n\
            effect_mean effect_std z distance).",
        after_long_help = ENV_HELP,
        aliases = ["p2g", "peak2gene"]
    )]
    PeakToGene(p2g::PeakToGeneArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    let default_filter = if cli.verbose {
        matrix_util::common_io::VERBOSE_LOG_FILTER
    } else {
        matrix_util::common_io::QUIET_LOG_FILTER
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
        .init();

    match &cli.commands {
        Commands::FitTopic(args) => topic::fit::fit_topic_model(args),
        Commands::PeakToGene(args) => p2g::run_peak_to_gene(args),
    }
}
