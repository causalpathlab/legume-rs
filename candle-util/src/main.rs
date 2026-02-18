use anyhow::Result;
use candle_util::cli::{regression, Cli, Commands};
use clap::Parser;

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    match &cli.command {
        Commands::Regression(args) => {
            regression::run(args)?;
        }
    }

    Ok(())
}
