mod map_sumstat;
mod pseudobulk_cmd;
mod sim_qtl;
mod sim_sumstat;

use map_sumstat::*;
use pseudobulk_cmd::*;
use sim_qtl::*;
use sim_sumstat::*;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line.replace('●', &"●".bright_yellow().to_string())
        .replace('╱', &"╱".truecolor(0, 100, 0).to_string())
        .replace('╲', &"╲".truecolor(0, 100, 0).to_string())
        .replace('(', &"(".truecolor(0, 100, 0).to_string())
        .replace(')', &")".truecolor(0, 100, 0).to_string())
        .replace('\\', &"\\".truecolor(0, 100, 0).to_string())
        .replace('/', &"/".truecolor(0, 100, 0).to_string())
        .replace('~', &"~".truecolor(101, 67, 33).to_string())
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    // Faceted Associations of Genotype Information
    // via Omics-based Locus Identification
    println!("  {}", "fagioli".bold());
    println!();
}

#[derive(Parser)]
#[command(name = "fagioli")]
#[command(about = "Faceted Associations of Genotype Information via Omics-based Locus Identification")]
struct Cli {
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Simulate molecular QTL with cell type heterogeneity and single-cell counts
    SimQtl(SimulationArgs),
    /// Simulate multi-trait GWAS summary statistics with LD structure
    SimSumstat(SimSumstatArgs),
    /// Summary-statistics-based multi-trait fine-mapping with SuSiE
    MapSumstat(MapSumstatArgs),
    /// Collapse single-cell counts into Poisson-Gamma pseudobulk
    Pseudobulk(PseudobulkArgs),
}

fn main() -> Result<()> {
    // Show logo if help is requested
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    if cli.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    match &cli.commands {
        Commands::SimQtl(args) => {
            sim_qtl(args)?;
        }
        Commands::SimSumstat(args) => {
            sim_sumstat(args)?;
        }
        Commands::MapSumstat(args) => {
            map_sumstat(args)?;
        }
        Commands::Pseudobulk(args) => {
            pseudobulk(args)?;
        }
    }

    Ok(())
}
