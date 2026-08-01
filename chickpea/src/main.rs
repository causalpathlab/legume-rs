mod common;
mod p2g;

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
    about = "chickpea —\n\
             peak-to-gene cis-regulatory linkage for paired single-cell RNA + ATAC",
    long_about = "chickpea — peak-to-gene cis-regulatory linkage\n\
                  \n\
                  Links ATAC peaks to RNA genes.\n\
                  The input is paired single-cell RNA + ATAC data.\n\
                  Linkage is by summary-statistics fine-mapping, SuSiE-RSS,\n\
                  in a shared pseudobulk embedding.\n\
                  \n\
                  Usage:\n\
                  data-beans-sim multiome -o sim --n-topics 10 chickpea peak-to-gene --rna-files sim.rna.zarr \\\n\
                  --atac-files sim.atac.zarr --gene-coords sim.gene_coords.tsv.gz -o out",
    term_width = 80
)]
struct Cli {
    #[arg(
        short = 'v',
        long,
        global = true,
        help = "Enable verbose logging",
        long_help = "Enable verbose logging to stderr. Equivalent to setting RUST_LOG=info."
    )]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Fine-map cis peak→gene links via SuSiE-RSS on pseudobulk summary stats
    #[command(
        long_about = "Link ATAC peaks to RNA genes by summary-statistics fine-mapping.\n\
                      \n\
                      Pseudobulk the matched RNA + ATAC cells. Embed peaks,\n\
                      and the projected genes, in a shared ATAC latent space.\n\
                      Score each cis peak–gene pair by a log-linear regression z there.\n\
                      Then fine-map per gene with SuSiE-RSS,\n\
                      using the peak–peak correlation (LD) structure.\n\
                      This is lighter and faster than `fit-topic`, with no neural model.\n\
                      \n\
                      Outputs {out}.results.bed.gz. Its columns are chr, start, end, peak_id,\n\
                      gene_id, pip, effect_mean, effect_std, z and distance.",
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
        Commands::PeakToGene(args) => p2g::run_peak_to_gene(args),
    }
}
