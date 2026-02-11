use anyhow::Result;
use candle_util::cli::{regression, Cli, Commands};
use clap::Parser;

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
