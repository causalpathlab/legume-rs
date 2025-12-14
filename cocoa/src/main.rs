mod collapse_cocoa_data;
mod common;
mod input;
mod randomly_partition_data;
mod run_collapse;
mod run_diff;
mod run_sim;
mod run_sim_collider;
mod stat;

use crate::run_diff::*;
use crate::run_sim::*;
use crate::run_sim_collider::*;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(version, about, long_about)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Differential expression analysis with pseudobulk (Park & Kellis, 2020)
    Diff(DiffArgs),

    /// Simulate differential expression data with one cell type.
    Simulate(SimArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.commands {
        Commands::Diff(args) => {
            run_cocoa_diff(args.clone())?;
        }
        Commands::Simulate(args) => {
            run_sim_diff_data(args.clone())?;
        }
    }

    Ok(())
}
