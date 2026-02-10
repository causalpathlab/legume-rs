mod cell_clustering;
mod common;
mod dartseq_io;
mod dartseq_sifter;
mod dartseq_stat;
mod data;
mod gene_count;
mod hypothesis_tests;
mod read_coverage;
mod run_dartseq_count;
mod run_gene_count;
mod run_polya_count;
mod run_read_depth;
mod scan_pwm;

use crate::common::*;
use colored::Colorize;
use run_dartseq_count::*;
use run_gene_count::*;
use run_polya_count::*;
use run_read_depth::*;
// use scan_pwm::*; not ready yet

const LOGO: &str = include_str!("../logo.txt");

fn colorize_pod(s: &str) -> String {
    s.replace('○', &"○".bright_green().to_string())
        .replace(':', &":".truecolor(100, 160, 80).to_string())
        .replace('.', &".".truecolor(120, 170, 90).to_string())
        .replace('_', &"_".truecolor(60, 120, 40).to_string())
        .replace('~', &"~".truecolor(80, 140, 60).to_string())
        .replace('\'', &"'".truecolor(60, 120, 40).to_string())
        .replace('`', &"`".truecolor(60, 120, 40).to_string())
        .replace('/', &"/".truecolor(60, 120, 40).to_string())
        .replace('\\', &"\\".truecolor(60, 120, 40).to_string())
        .replace('-', &"-".truecolor(80, 140, 60).to_string())
}

fn colorize_funnel(s: &str) -> String {
    s.replace('○', &"○".bright_green().to_string())
        .replace('/', &"/".truecolor(200, 160, 60).to_string())
        .replace('\\', &"\\".truecolor(200, 160, 60).to_string())
        .replace('|', &"|".truecolor(200, 160, 60).to_string())
        .replace('-', &"-".truecolor(200, 160, 60).to_string())
        .replace('>', &">".truecolor(240, 200, 80).to_string())
}

fn colorize_logo_line(line: &str) -> String {
    // Pod ends around column 38, funnel+peas after
    let split = 38.min(line.len());
    let (pod, funnel) = line.split_at(split);
    format!("{}{}", colorize_pod(pod), colorize_funnel(funnel))
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    println!(
        " {}",
        "Feature statistics Accumulator for Base-pair-level Analysis".bold()
    );
    println!();
}

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
    #[command(alias = "rd")]
    ReadDepth(ReadDepthArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

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
