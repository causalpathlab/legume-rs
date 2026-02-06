mod sim_eqtl;

use sim_eqtl::*;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "fagioli")]
#[command(about = "Genetic association analysis toolkit")]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run eQTL simulation with cell type heterogeneity
    SimEqtl(SimulationArgs),
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::SimEqtl(args) => {
            sim_eqtl(args)?;
        }
    }

    Ok(())
}
