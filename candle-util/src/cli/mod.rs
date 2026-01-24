pub mod regression;
pub mod regression_likelihood;

use clap::{Parser, Subcommand};

pub use regression::{LikelihoodType, RegressionArgs, VariationalType};

#[derive(Parser)]
#[command(name = "candle-util")]
#[command(about = "Candle utility CLI for variational inference models")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run regression with SGVB
    Regression(RegressionArgs),
}
