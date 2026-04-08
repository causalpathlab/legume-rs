mod chickpea_input;
mod cis_mask;
mod coarsening;
mod common;
mod fit_topic;
mod linkage;
mod simulation;
mod topic;

use crate::common::*;
use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");
const ENV_HELP: &str = "Environment variables:\n  RUST_LOG=info  Enable logging to stderr";

fn colorize_logo_line(line: &str) -> String {
    line.replace(':', &":".truecolor(210, 180, 120).to_string())
        .replace('*', &"*".truecolor(240, 210, 140).to_string())
        .replace('o', &"o".truecolor(160, 120, 60).to_string())
        .replace('.', &".".truecolor(180, 160, 110).to_string())
        .replace('^', &"^".truecolor(180, 160, 110).to_string())
        .replace('\'', &"'".truecolor(180, 160, 110).to_string())
        .replace('/', &"/".truecolor(139, 90, 43).to_string())
        .replace('\\', &"\\".truecolor(139, 90, 43).to_string())
        .replace('_', &"_".truecolor(139, 90, 43).to_string())
        .replace('=', &"=".green().to_string())
        .replace('>', &">".bright_green().to_string())
        .replace('-', &"-".green().to_string())
        .replace('|', &"|".green().to_string())
        .replace('▀', &"▀".truecolor(100, 180, 100).to_string())
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    println!(" {}", "Multi-Omic Linkage Analysis".bold());
    println!();
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "chickpea — Multi-Omic Linkage Analysis for paired single-cell RNA + ATAC data",
    long_about = "chickpea — Multi-Omic Linkage Analysis\n\n\
        Jointly models paired single-cell RNA-seq and ATAC-seq data to discover\n\
        cis-regulatory peak-gene links via a topic model with SuSiE fine-mapping.\n\n\
        Workflow:\n\
        1. sim-link:  Generate synthetic paired RNA + ATAC data with known ground truth\n\
        2. fit-topic: Fit a joint topic model to learn peak-gene linkage from real or simulated data\n\n\
        Typical usage:\n\
          chickpea sim-link -o sim_out --n-topics 10 --n-genes 2000 --n-peaks 10000\n\
          chickpea fit-topic --rna-files sim_out.rna.zarr --atac-files sim_out.atac.zarr -o fit_out",
    term_width = 80
)]
struct Cli {
    #[arg(
        short = 'v',
        long,
        global = true,
        help = "Enable verbose logging",
        long_help = "Enable verbose logging to stderr.\n\
                     Equivalent to setting RUST_LOG=info."
    )]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Simulate paired RNA + ATAC data with ground-truth peak-gene links
    #[command(
        long_about = "Simulate paired single-cell RNA + ATAC count matrices with ground-truth\n\
            peak-gene linkage for benchmarking.\n\n\
            Generative model (nested hierarchical topics):\n\
            - K coarse topics visible in ATAC, K_sub subtypes per topic visible in RNA\n\
            - K_total = K × K_sub (set --n-sub-topics > 1 for nesting)\n\
            - β_ext[P, K_total]: independent Dirichlet per (topic, subtype) pair\n\
            - β_atac[P, K]: marginalized over subtypes (used for ATAC)\n\
            - θ_coarse[K, N]: coarse cell assignments (used for ATAC)\n\
            - θ_full[K_total, N]: nested assignments (used for RNA)\n\
            - W[G, K_total] = M × β_ext (RNA dictionary via peak-gene linkage)\n\
            - Poisson counts with log-normal per-cell depth noise\n\n\
            Gene coordinates are output as a separate annotation file\n\
            (gene_coords.tsv.gz) for use with fit-topic --gene-coords.\n\n\
            Output files (given --out PREFIX):\n\
            - PREFIX.rna.{zarr,h5}          Sparse RNA count matrix [G × N]\n\
            - PREFIX.atac.{zarr,h5}         Sparse ATAC count matrix [P × N]\n\
            - PREFIX.dict.parquet           ATAC dictionary beta[P,K] (marginalized)\n\
            - PREFIX.derived_dict.parquet   RNA dictionary W[G,K_total] = M × β_ext\n\
            - PREFIX.prop.parquet           Coarse proportions theta[N,K]\n\
            - PREFIX.beta_ext.parquet       Extended dictionary β_ext[P,K_total] (if K_sub>1)\n\
            - PREFIX.theta_full.parquet     Full proportions theta[N,K_total] (if K_sub>1)\n\
            - PREFIX.gamma.parquet          Gene-topic effects (if --gene-topic-sd > 0)\n\
            - PREFIX.ground_truth.tsv.gz    True gene-peak links\n\
            - PREFIX.gene_coords.tsv.gz     Gene annotations (gene, chr, tss)\n\
            - PREFIX.{gene,peak}_names.txt  Feature names\n\
            - PREFIX.barcodes.txt           Cell barcodes\n\n\
            Example:\n\
              chickpea sim-link -o test_sim --n-topics 5 --n-sub-topics 3 --n-genes 500",
        after_long_help = ENV_HELP,
	alias = "simulate"
    )]
    SimLink(simulation::SimLinkArgs),

    /// Fit joint topic model to paired RNA + ATAC data
    #[command(
        long_about = "Fit a joint topic model with SuSiE linkage to paired RNA + ATAC data.\n\n\
            Two-stage training:\n\
            - Stage 1 (ATAC-only): Learn ATAC dictionary beta[P,K] and encoder\n\
            - Stage 2 (Joint): Learn peak-gene linkage M[G,P] via SuSiE SER components\n\n\
            The model uses multi-level pseudobulk collapsing for scalability:\n\
            cells are grouped into super-cells via random projection + binary partitioning,\n\
            then modeled as Gamma-distributed pseudobulk observations.\n\n\
            Output files (given --out PREFIX):\n\
            - PREFIX.atac_dict.parquet   ATAC dictionary exp(log_beta)[P,K]\n\
            - PREFIX.rna_dict.parquet    Derived RNA dictionary M × beta[G,K]\n\
            - PREFIX.log_beta.parquet    Log-scale ATAC dictionary\n\
            - PREFIX.results.bed.gz      SuSiE results: PIP, effect mean/std per gene-peak pair\n\
            - PREFIX.prop.parquet        Inferred topic proportions[N,K]\n\n\
            Example:\n\
              chickpea fit-topic \\\n\
                --rna-files sample.rna.zarr \\\n\
                --atac-files sample.atac.zarr \\\n\
                --n-topics 10 --n-ser-components 3 --max-cis 50 \\\n\
                --epochs 200 -o result",
        after_long_help = ENV_HELP,
        alias = "topic"
    )]
    FitTopic(fit_topic::FitTopicArgs),
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    if cli.verbose && std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    match &cli.commands {
        Commands::SimLink(args) => simulation::run_sim_link(args),
        Commands::FitTopic(args) => fit_topic::fit_topic_model(args),
    }
}
