mod common;
mod data;
mod hypothesis_tests;
mod run_dartseq_count;
mod run_read_depth;
mod util;

use crate::common::*;
use run_dartseq_count::*;
use run_read_depth::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about, term_width = 80)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Take DART-seq `C->U` (`C->T`) conversion counts
    CountDartSeq(CountDartSeqArgs),
    ReadDepth(ReadDepthArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::CountDartSeq(args) => {
            run_count_dartseq(args)?;
        }
        Commands::ReadDepth(args) => {
            todo!()
        }
    }

    Ok(())
}
