mod apa_mix;
mod cell_clustering;
mod common;
mod dartseq_io;
mod dartseq_sifter;
mod dartseq_stat;
mod data;
mod gene_count;
mod hypothesis_tests;
mod read_coverage;
mod run_count_apa;
mod run_dartseq_count;
mod run_gene_count;
mod run_read_depth;
mod scan_pwm;

use crate::common::*;
use colored::Colorize;
use run_count_apa::*;
use run_dartseq_count::*;
use run_gene_count::*;
use run_read_depth::*;
// use scan_pwm::*; not ready yet

const LOGO: &str = include_str!("../logo.txt");

fn colorize_pod(s: &str) -> String {
    s.replace('○', &"○".truecolor(200, 160, 60).to_string())
        .replace(':', &":".truecolor(200, 160, 80).to_string())
        .replace('.', &".".truecolor(220, 170, 90).to_string())
        .replace('_', &"_".truecolor(60, 120, 40).to_string())
        .replace('~', &"~".truecolor(80, 140, 60).to_string())
        .replace('\'', &"'".truecolor(60, 120, 40).to_string())
        .replace('`', &"`".truecolor(60, 120, 40).to_string())
        .replace('/', &"/".truecolor(60, 120, 40).to_string())
        .replace('\\', &"\\".truecolor(60, 120, 40).to_string())
        .replace('-', &"-".truecolor(80, 140, 60).to_string())
}

fn colorize_funnel(s: &str) -> String {
    s.replace('○', &"○".truecolor(240, 100, 100).to_string())
        .replace('/', &"/".truecolor(200, 160, 60).to_string())
        .replace('\\', &"\\".truecolor(200, 160, 60).to_string())
        .replace('|', &"|".truecolor(200, 160, 60).to_string())
        .replace('=', &"=".truecolor(200, 160, 60).to_string())
        .replace('>', &">".truecolor(200, 170, 60).to_string())
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
#[command(version, about, long_about = None, term_width = 80,
    after_help = "Use `faba <COMMAND> --help` for detailed options on each subcommand.")]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Enable verbose logging")]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Quantify DART-seq m6A sites from C-to-T conversions
    #[command(aliases = ["count-dart", "dart"],
        long_about = "Quantify DART-seq m6A sites from C-to-T conversions\n\n\
            Discovers m6A methylation sites by comparing C->T (forward) or\n\
            G->A (reverse) conversion rates between wild-type and mutant BAM\n\
            files using binomial tests, then quantifies per-cell methylation\n\
            at discovered sites.\n\n\
            Reference:\n  \
            Meyer, \"DART-seq: an antibody-free method for global m6A\n  \
            detection\", Nature Methods, 16(12):1275-1280, 2019.\n  \
            https://doi.org/10.1038/s41592-019-0570-0")]
    CountDartSeq(DartSeqCountArgs),

    /// Count reads per gene for single-cell or bulk RNA-seq
    #[command(
        long_about = "Count reads per gene for single-cell or bulk RNA-seq\n\n\
            Produces a sparse (cells x genes) count matrix from BAM files\n\
            using GFF gene annotations. Supports 10x-style cell barcodes."
    )]
    CountGenes(GeneCountArgs),

    /// Quantify alternative polyadenylation (APA) sites per cell
    #[command(aliases = ["count-polya", "polya", "apa-mix", "apamix", "apa"],
        long_about = "Quantify alternative polyadenylation (APA) sites per cell\n\n\
            Discovers and quantifies poly(A) site usage from 3'-end sequencing\n\
            data. The mixture mode implements the SCAPE model.\n\n\
            Reference:\n  \
            Zhou et al., \"SCAPE: a mixture model revealing single-cell\n  \
            polyadenylation diversity and cellular dynamics during cell\n  \
            differentiation and reprogramming\",\n  \
            Nucleic Acids Research, 50(11):e66, 2022.\n  \
            https://doi.org/10.1093/nar/gkac167")]
    CountApa(CountApaArgs),

    /// Compute read depth over genomic intervals
    #[command(
        alias = "rd",
        long_about = "Compute read depth over genomic intervals\n\n\
            Bins the genome at a given resolution and counts read coverage\n\
            per cell, producing a sparse (cells x bins) matrix."
    )]
    ReadDepth(ReadDepthArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    if cli.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

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
        Commands::CountApa(args) => {
            run_count_apa(args)?;
        }
    }

    Ok(())
}
