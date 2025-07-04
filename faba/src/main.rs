mod common;
mod dartseq_sifter;
mod data;
mod hypothesis_tests;
mod run_dartseq_count;
mod run_gene_count;
mod run_read_depth;

use crate::common::*;
use run_dartseq_count::*;
use run_gene_count::*;
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
    CountDartSeq(DartSeqCountArgs),
    /// Count the number of reads mapped on each gene
    CountGenes(GeneCountArgs),
    /// Genomic coverage of regular intervals
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
            run_read_depth(args)?;
        }
        Commands::CountGenes(args) => {
            run_gene_count(args)?;
        }
    }

    Ok(())
}
