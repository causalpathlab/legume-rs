mod fit_prs_susie;
mod fit_qtl_sgvb;
mod fit_sumstat_mcmc;
mod fit_sumstat_sgvb;
mod pseudobulk;
mod sim_geno;
mod sim_mediation;
mod sim_qtl;
mod sim_sumstat;

use fit_prs_susie::*;
use fit_qtl_sgvb::*;
use fit_sumstat_mcmc::*;
use fit_sumstat_sgvb::*;
use pseudobulk::*;
use sim_geno::*;
use sim_mediation::*;
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
    SimQtl(SimQtlArgs),
    /// Simulate multi-trait GWAS summary statistics with LD blocks and confounders
    SimSumstat(SimSumstatArgs),
    /// Simulate mediation analysis data: SNP → Expression → Phenotype with confounders
    SimMediation(SimMediationArgs),
    /// Fine-map causal variants from GWAS summary statistics using variational SuSiE
    FitSumstatSgvb(FitSumstatSgvbArgs),
    /// Fine-map causal variants from GWAS summary statistics using MCMC (ESS)
    FitSumstatMcmc(FitSumstatMcmcArgs),
    /// Fine-map cis-eQTL from single-cell RNA-seq with Poisson-Gamma pseudobulk
    FitQtlSgvb(FitQtlSgvbArgs),
    /// Ridge PRS from z-scores + SuSiE fine-mapping on predicted phenotypes
    FitPrsSusie(FitPrsSusieArgs),
    /// Collapse single-cell counts into Poisson-Gamma pseudobulk per individual and cell type
    Pseudobulk(PseudobulkArgs),
}

fn main() -> Result<()> {
    // Show logo if help is requested
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
        Commands::SimGeno(args) => {
            sim_geno(args)?;
        }
        Commands::SimQtl(args) => {
            sim_qtl(args)?;
        }
        Commands::SimSumstat(args) => {
            sim_sumstat(args)?;
        }
        Commands::SimMediation(args) => {
            sim_mediation(args)?;
        }
        Commands::FitSumstatSgvb(args) => {
            fit_sumstat_sgvb(args)?;
        }
        Commands::FitSumstatMcmc(args) => {
            fit_sumstat_mcmc(args)?;
        }
        Commands::FitQtlSgvb(args) => {
            fit_qtl_sgvb(args)?;
        }
        Commands::FitPrsSusie(args) => {
            fit_prs_susie(args)?;
        }
        Commands::Pseudobulk(args) => {
            pseudobulk(args)?;
        }
    }

    Ok(())
}
