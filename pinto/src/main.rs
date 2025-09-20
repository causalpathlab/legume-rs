mod fit_srt_svd;
mod fit_srt_topic;
mod srt_cell_pairs;
mod srt_collapse_pairs;
mod srt_common;
mod srt_input;
mod srt_random_projection;

use fit_srt_svd::*;
use fit_srt_topic::*;

use clap::{Parser, Subcommand};

/// Proximity-based Interaction Network analysis to dissect Tissue
/// Organizations
///
/// Data files of either `.zarr` or `.h5` format. All the formats in
/// the given list should be identical. We can convert `.mtx` to
/// `.zarr` or `.h5` using `data-beans from-mtx` or similar commands.
///
#[derive(Parser, Debug)]
#[command(version, about, long_about, term_width = 80)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// by randomized singular value decomposition
    Svd(SrtSvdArgs),
    /// by fitting a topic model
    Topic(SrtTopicArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Svd(args) => {
            fit_srt_svd(args)?;
        }
        Commands::Topic(args) => {
            fit_srt_topic(args)?;
        }
    }

    Ok(())
}
