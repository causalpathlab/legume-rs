mod apa;
mod cell_clustering;
mod common;
mod data;
mod editing;
mod gene_count;
mod hypothesis_tests;
mod mixture;
mod pipeline_util;
mod read_depth;
mod run_apa;
mod run_atoi;
mod run_gene_count;
mod run_m6a;
mod run_pipeline;
mod run_read_depth;
mod site_analysis;

use crate::common::*;
use colored::Colorize;
use run_apa::*;
use run_atoi::*;
use run_gene_count::*;
use run_m6a::*;
use run_pipeline::*;
use run_read_depth::*;
use site_analysis::metagene::*;
use site_analysis::scan_pwm::*;

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
    after_help = "\
Feature naming convention:\n\
  All sparse matrix row names follow: {gene_key}/{modality}/{detail}\n\
  where gene_key = {gene_id}_{symbol} (e.g. ENSG00001234_BRCA2)\n\n\
  genes:   gene_key/count/spliced, gene_key/count/unspliced\n\
  dartseq: gene_key/m6A/{component} (mixture), gene_key/m6A/{chr}:{pos} (site)\n\
  atoi:    gene_key/A2I/{component} (mixture), gene_key/A2I/{chr}:{pos} (site)\n\
  apa:     gene_key/pA/{component} (mixture), gene_key/pA/{chr}:{pos} (site)\n\n\
  Split on '/' to extract (gene_key, modality, detail) for cross-modal joins.\n\n\
Use `faba <COMMAND> --help` for detailed options on each subcommand.")]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Enable verbose logging")]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(name = "dartseq", aliases = ["dart", "m6a"],
        about = "Quantify DART-seq m6A sites from C-to-T conversions",
        long_about = "Quantify DART-seq m6A sites from C-to-T conversions\n\n\
            Discovers m6A methylation sites by comparing C->T (forward) or\n\
            G->A (reverse) conversion rates between wild-type and mutant BAM\n\
            files using binomial tests, then quantifies per-cell methylation\n\
            at discovered sites.\n\n\
            Reference:\n  \
            Meyer, \"DART-seq: an antibody-free method for global m6A\n  \
            detection\", Nature Methods, 16(12):1275-1280, 2019.\n  \
            https://doi.org/10.1038/s41592-019-0570-0")]
    DartSeq(DartSeqCountArgs),

    #[command(name = "apa", aliases = ["polya"],
        about = "Quantify alternative polyadenylation (APA) sites per cell",
        long_about = "Quantify alternative polyadenylation (APA) sites per cell\n\n\
            Discovers and quantifies poly(A) site usage from 3'-end sequencing\n\
            data. The mixture mode implements the SCAPE model.\n\n\
            Reference:\n  \
            Zhou et al., \"SCAPE: a mixture model revealing single-cell\n  \
            polyadenylation diversity and cellular dynamics during cell\n  \
            differentiation and reprogramming\",\n  \
            Nucleic Acids Research, 50(11):e66, 2022.\n  \
            https://doi.org/10.1093/nar/gkac167")]
    Apa(CountApaArgs),

    #[command(name = "atoi", aliases = ["a2i", "editing"],
        about = "Detect and quantify A-to-I RNA editing sites",
        long_about = "Detect A-to-I (adenosine-to-inosine) RNA editing sites\n\n\
            Discovers editing sites from A->G (forward) or T->C (reverse)\n\
            conversions in BAM files, then quantifies per-cell editing\n\
            at discovered sites.\n\n\
            Output: atoi_sites.parquet (site annotations) + sparse matrix\n\
            (cells x sites). The parquet file can be used as --atoi-mask\n\
            input for `faba dart` or `faba apa`.")]
    AtoI(AtoICountArgs),

    #[command(name = "genes", aliases = ["count-genes"],
        about = "Count reads per gene for single-cell or bulk RNA-seq",
        long_about = "Count reads per gene for single-cell or bulk RNA-seq\n\n\
            Produces a sparse (cells x genes) count matrix from BAM files\n\
            using GFF gene annotations. Supports 10x-style cell barcodes."
    )]
    Genes(GeneCountArgs),

    #[command(name = "depth", aliases = ["read-depth", "rd"],
        about = "Compute read depth over genomic intervals",
        long_about = "Compute read depth over genomic intervals\n\n\
            Bins the genome at a given resolution and counts read coverage\n\
            per cell, producing a sparse (cells x bins) matrix."
    )]
    Depth(ReadDepthArgs),

    #[command(name = "pwm", aliases = ["scan-pwm"],
        about = "Build position weight matrix around genomic sites",
        long_about = "Build position weight matrix around genomic sites\n\n\
            Reads site-level parquet files from dart or apa output, collects\n\
            base frequencies in a +/- window around each site, and outputs\n\
            a position weight matrix as TSV."
    )]
    Pwm(ScanPwmArgs),

    #[command(
        name = "metagene",
        alias = "mg",
        about = "Metagene histogram of site positions across gene features",
        long_about = "Metagene histogram of site positions across gene features\n\n\
            Maps sites from a parquet file onto gene features (5'UTR, CDS,\n\
            3'UTR, non-coding) using GFF annotations, and produces a binned\n\
            histogram showing the distribution of sites across the metagene."
    )]
    Metagene(MetageneArgs),

    #[command(
        name = "all",
        aliases = ["pipeline", "full", "magic"],
        about = "Run all RNA-seq analyses: genes → ATOI → APA → DART",
        long_about = "Run all RNA-seq analyses in a unified pipeline\n\n\
            Orchestrates the complete analysis workflow:\n  \
            1. Gene expression filtering (identify expressed genes)\n  \
            2. ATOI detection (A-to-I editing sites)\n  \
            3. APA quantification (alternative polyadenylation, masked)\n  \
            4. DART analysis (m6A methylation, masked, requires --mut)\n\n\
            Applies gene filtering after step 1 and ATOI masking to steps 3-4."
    )]
    All(PipelineArgs),
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

    match cli.commands {
        Commands::DartSeq(ref args) => run_m6a(args)?,
        Commands::Apa(mut args) => run_apa(&mut args)?,
        Commands::AtoI(ref args) => run_atoi(args)?,
        Commands::Genes(ref args) => run_gene_count(args)?,
        Commands::Depth(ref args) => run_read_depth(args)?,
        Commands::Pwm(ref args) => run_scan_pwm(args)?,
        Commands::Metagene(ref args) => run_metagene(args)?,
        Commands::All(ref args) => run_pipeline(args)?,
    }

    Ok(())
}
