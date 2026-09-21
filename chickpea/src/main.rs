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
                  The intended path trains peak/gene embeddings with graph-embedding-util,\n\
                  embeds cells, clusters, and refines peak→gene within each cluster,\n\
                  writing E2G-like parquet:\n\
                  peaks.parquet (id, chromosome, start, end, class),\n\
                  clusters.parquet (id, name),\n\
                  peak_gene/chr*.parquet (score, gene fields, enhancer_id, cell_type_id, …).\n\
                  That association path is not fully wired yet — see chickpea/todo.md.",
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
    /// Peak→gene linkage via graph-embedding-util (not wired yet)
    #[command(
        long_about = "Link ATAC peaks to RNA genes.\n\
                      \n\
                      Planned pipeline: rough ABC / co-occurrence map,\n\
                      train peak/gene embeddings with graph-embedding-util (pb-level),\n\
                      embed cells → cluster → refine peak→gene within each cluster,\n\
                      write E2G-like parquet (peaks, clusters, peak_gene).\n\
                      \n\
                      The old rSVD / SuSiE / knockoff / TMLE path has been removed.\n\
                      This subcommand currently exits until the ge-util path is wired.\n\
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
