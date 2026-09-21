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
                  Links ATAC peaks to RNA genes from paired single-cell RNA + ATAC.\n\
                  Pseudobulk via data-beans multilevel collapse (optional batch adjustment).\n\
                  Train peak/gene embeddings with graph-embedding-util,\n\
                  embed pb samples, cluster, refine peak→gene within each cluster,\n\
                  write E2G-like parquet:\n\
                  peaks.parquet (id, chromosome, start, end, class),\n\
                  clusters.parquet (id, name),\n\
                  peak_gene/chr*.parquet (score, gene fields, enhancer_id, cell_type_id, …).",
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
    /// Peak→gene linkage via graph-embedding-util
    #[command(
        long_about = "Link ATAC peaks to RNA genes.\n\
                      \n\
                      1. Load paired RNA+ATAC; optional RNA-driven cell QC.\n\
                      2. data-beans multilevel pb collapse (+ refine; optional --use-adjusted).\n\
                      3. Rough cis co-occurrence map; train peak/gene embeds with ge-util FNE.\n\
                      4. Embed pb samples → Leiden clusters → within-cluster refine.\n\
                      5. Write E2G-like parquet (peaks, clusters, peak_gene/chr*).\n\
                      \n\
                      See chickpea/todo.md.",
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
        legume_numeric::matrix::common_io::VERBOSE_LOG_FILTER
    } else {
        legume_numeric::matrix::common_io::QUIET_LOG_FILTER
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
        .init();

    match &cli.commands {
        Commands::PeakToGene(args) => p2g::run_peak_to_gene(args),
    }
}
