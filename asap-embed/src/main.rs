pub mod matrix_param;
pub mod random_projection;

// use asap_data::common_io as io;
// type SData = dyn SparseIo<IndexIter = Vec<usize>>;

use asap_data::sparse_io::*;
use clap::{Args, Parser, Subcommand};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.commands {
        Commands::RP(args) => {
            // do something
        }
    }

    Ok(())
}

#[derive(Parser)]
#[command(version, about, long_about=None)]
#[command(propagate_version = true)]
///
/// Basic utility functions for processing sparse matrices
///
/// - RP: build from .mtx fileset to another faster format
///
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build faster backend data from mtx
    RP(RunRPArgs),
}

#[derive(Args)]
pub struct RunRPArgs {
    /// Data file -- either `.zarr` or `.h5`
    data_file: Box<str>,

    /// Block_size for parallel processing (default: 100)
    #[arg(long, value_enum)]
    block_size: Option<usize>,

    /// backend to use (HDF5 or Zarr), default: Zarr
    #[arg(short, long, value_enum)]
    backend: Option<SparseIoBackend>,
}
