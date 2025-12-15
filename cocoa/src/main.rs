mod collapse_cocoa_data;
mod common;
mod input;
mod randomly_partition_data;
mod run_collapse;
mod run_diff;
mod run_sim;
mod run_sim_collider;
mod stat;

use crate::run_collapse::*;
use crate::run_diff::*;
use crate::run_sim::*;
use crate::run_sim_collider::*;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "CoCoA (Counterfactual Confounder Adjustment)",
    long_about = "Routines in CoCoA will be useful "
)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Differential expression analysis with pseudobulk",
        long_about = "Differential expression analysis on pseudobulk data \n\
		      while adjusting confounding effects by cross-condition \n\
		      or cross-exposure/treatment matching (Park & Kellis, 2020):\n\
		      \n"
    )]
    Diff(DiffArgs),

    #[command(
        about = "Collapse",
        long_about = "\n\
		      \n"
    )]
    Collapse(CollapseArgs),

    #[command(
        about = "Simulate expression data with one cell type",
        long_about = "Simulate expression data with one cell type.\n\
		      \n"
    )]
    Simulate(SimArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.commands {
        Commands::Diff(args) => {
            run_cocoa_diff(args.clone())?;
        }
        Commands::Collapse(args) => {
            run_collapse(args.clone())?;
        }
        Commands::Simulate(args) => {
            run_sim_diff_data(args.clone())?;
        }
    }

    Ok(())
}
