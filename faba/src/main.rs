mod common;
mod data;
mod run_dartseq_count;
mod hypothesis_tests;
mod util;

// use crate::data::sifter::*;
use crate::common::*;
use run_dartseq_count::*;

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
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::CountDartSeq(args) => {
            run_count_dartseq(args)?;
        }
    }

    Ok(())
}
