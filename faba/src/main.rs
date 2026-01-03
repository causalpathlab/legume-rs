mod common;
mod dartseq_io;
mod dartseq_sifter;
mod dartseq_stat;
mod data;
mod hypothesis_tests;
mod read_coverage;
mod run_dartseq_count;
mod run_gene_count;
mod run_polya_count;
mod run_read_depth;
mod scan_pwm;

use crate::common::*;
use run_dartseq_count::*;
use run_gene_count::*;
use run_polya_count::*;
use run_read_depth::*;
// use scan_pwm::*; not ready yet

/// Feature statistics Accumulator for Base-pair-level Analysis
#[derive(Parser, Debug)]
#[command(version, about, long_about, term_width = 80)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Take DART-seq `C->U` (`C->T`) conversion counts
    #[command(aliases = ["count-dart", "dart"])]
    CountDartSeq(DartSeqCountArgs),

    /// Count the number of reads mapped on each gene
    CountGenes(GeneCountArgs),

    /// Count poly-A sites at cell level
    #[command(aliases = ["count-polya", "polya"])]
    CountPolyA(PolyACountArgs),

    /// Genomic coverage of regular intervals
    #[command(aliases = ["rd", "read-depth"])]
    ReadDepth(ReadDepthArgs),
}

fn main() -> anyhow::Result<()> {
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
        Commands::CountPolyA(args) => {
            run_count_polya(args)?;
        }
    }

    Ok(())
}
