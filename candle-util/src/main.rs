use anyhow::Result;
use clap::Parser;
use candle_util::cli::{Cli, Commands, regression};

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Regression(args) => {
            regression::run(args)?;
        }
    }

    Ok(())
}
