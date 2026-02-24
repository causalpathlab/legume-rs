mod map_qtl;
mod map_sumstat;
mod pseudobulk_cmd;
mod sim_geno;
mod sim_qtl;
mod sim_sumstat;

use map_qtl::*;
use map_sumstat::*;
use pseudobulk_cmd::*;
use sim_geno::*;
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
#[command(
    about = "Faceted Associations of Genotype Information via Omics-based Locus Identification"
)]
struct Cli {
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Simulate genotype data via Wright-Fisher forward simulation → PLINK BED
    SimGeno(SimGenoArgs),
    /// Simulate single-cell eQTL data with cell-type-specific genetic effects
    SimQtl(SimulationArgs),
    /// Simulate multi-trait GWAS summary statistics with LD blocks and confounders
    SimSumstat(SimSumstatArgs),
    /// Fine-map causal variants from GWAS summary statistics using variational SuSiE
    MapSumstat(MapSumstatArgs),
    /// Fine-map cis-eQTL from single-cell RNA-seq with Poisson-Gamma pseudobulk
    MapQtl(MapQtlArgs),
    /// Collapse single-cell counts into Poisson-Gamma pseudobulk per individual and cell type
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
        Commands::SimGeno(args) => {
            sim_geno(args)?;
        }
        Commands::SimQtl(args) => {
            sim_qtl(args)?;
        }
        Commands::SimSumstat(args) => {
            sim_sumstat(args)?;
        }
        Commands::MapSumstat(args) => {
            map_sumstat(args)?;
        }
        Commands::MapQtl(args) => {
            map_qtl(args)?;
        }
        Commands::Pseudobulk(args) => {
            pseudobulk(args)?;
        }
    }

    Ok(())
}
