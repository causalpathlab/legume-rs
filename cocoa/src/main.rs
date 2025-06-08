mod collapse_data;
mod common;
mod run_diff;
mod run_sim;
mod stat;
mod util;

use crate::run_diff::*;
use crate::run_sim::*;

use clap::Subcommand;

#[derive(Parser, Debug)]
#[command(version, about, long_about)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Differential expression analysis
    Diff(DiffArgs),

    /// Simulate differential expression data with one cell type.
    /// ``
    ///
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
