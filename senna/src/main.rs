mod cluster;
mod cnv_pseudobulk;
mod embed_common;
mod eval_topic;
mod fit_clustering;
mod fit_indexed_topic;
mod fit_joint_topic;
mod fit_topic;
mod geometry;
mod logging;
mod marker_support;
mod postprocess;
mod senna_input;
mod svd;
mod topic;

use embed_common::*;
use eval_topic::*;
use fit_clustering::*;
use fit_indexed_topic::*;
use fit_joint_topic::*;
use fit_topic::*;
use postprocess::*;
use svd::*;

use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line.replace('@', &"@".bright_yellow().to_string())
        .replace('◠', &"◠".bright_yellow().to_string())
        .replace('◡', &"◠".bright_yellow().to_string())
        .replace('_', &"_".bright_yellow().to_string())
        .replace('(', &"(".bright_yellow().to_string())
        .replace(')', &")".bright_yellow().to_string())
        .replace('{', &"{".bright_yellow().to_string())
        .replace('}', &"}".bright_yellow().to_string())
        .replace('\\', &"\\".bright_yellow().to_string())
        .replace('/', &"/".bright_yellow().to_string())
        .replace('|', &"|".green().to_string())
        .replace('‖', &"‖".green().to_string())
        .replace('~', &"~".truecolor(101, 67, 33).to_string())
}

fn print_logo() {
    let intro = [
        "",
        "",
        "SENNA",
        "Stochastic data Embedding with",
        "Nearest Neighbourhood Adjustment",
        "",
    ];

    let logo_lines: Vec<_> = LOGO.lines().collect();
    let max_lines = logo_lines.len().max(intro.len());

    for i in 0..max_lines {
        let logo_part = if i < logo_lines.len() {
            colorize_logo_line(logo_lines[i])
        } else {
            " ".repeat(13) // width of logo box
        };

        let text_part = if i < intro.len() { intro[i] } else { "" };

        println!("{}  {}", logo_part, text_part);
    }
    println!();
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "SENNA — Stochastic data Embedding with Nearest Neighbourhood Adjustment",
    long_about = "SENNA — Stochastic data Embedding with Nearest Neighbourhood Adjustment.\n\n\
                  Input: sparse backends in `.zarr` or `.h5` format.\n\
                  Convert from Matrix Market with `data-beans from-mtx`."
)]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Verbose logging")]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Embed cells by Nyström SVD with batch-aware pseudobulk collapsing",
        long_about = "Nyström SVD embedding in three stages:\n\
                      (1) batch-aware multi-level pseudobulk collapsing,\n\
                      (2) randomized SVD on the collapsed matrix,\n\
                      (3) projection of every cell onto the learned basis."
    )]
    Svd(SvdArgs),

    #[command(
        about = "Embed cells by probabilistic topic modelling (VAE, multi-decoder)",
        long_about = "Topic-model embedding in three stages:\n\
                      (1) batch-aware multi-level pseudobulk collapsing,\n\
                      (2) encoder-decoder VAE training via SGD,\n\
                      (3) encoder inference of per-cell latent topic proportions.\n\n\
                      Supports multinomial, negative-binomial, and vMF decoders\n\
                      (jointly when comma-separated via --decoder)."
    )]
    Topic(TopicArgs),

    #[command(
        about = "Topic model with adaptive top-K feature windows (~4-7× faster decoder)",
        long_about = "Same three-stage pipeline as `topic`, but the encoder and\n\
                      decoder operate on a per-cell top-K feature window instead\n\
                      of the dense D × K dictionary. Useful for very large gene sets.",
        visible_alias = "itopic"
    )]
    IndexedTopic(IndexedTopicArgs),

    #[command(
        about = "Annotate topics using marker genes (vMF cosine similarity)",
        long_about = "Assign cell-type probabilities to topics via vMF softmax on\n\
                      cosine similarity between the topic dictionary and marker sets.\n\n\
                      Modes:\n\
                        direct      — senna annotate -g dict -z latent -m markers -o out\n\
                        interactive — add -I to iteratively augment markers\n\
                        LLM-assist  — --suggest-only out.json, then --apply-suggestions in.json",
        visible_alias = "annotate"
    )]
    AnnotateTopic(AnnotateTopicArgs),

    #[command(
        about = "Apply a trained topic model to new data",
        long_about = "Run encoder inference with a previously trained topic model.\n\
                      Handles gene-set misalignment, re-estimates per-batch delta\n\
                      from the frozen dictionary, and runs on CPU with rayon."
    )]
    EvalTopic(EvalTopicArgs),

    #[command(
        about = "Joint Nyström SVD across multiple modalities",
        long_about = "Nyström SVD on a stack of modalities sharing the same cells.\n\
                      Data files are arranged row-major as a (modality × batch) table;\n\
                      use -m to set the number of modality rows."
    )]
    JointSvd(JointSvdArgs),

    #[command(
        about = "Joint topic model across multiple modalities (independent or delta decoder)",
        long_about = "Joint topic-model embedding across a stack of modalities.\n\n\
                      Data files are arranged row-major as a (modality × batch) table;\n\
                      use -m to set the number of modality rows.\n\n\
                      Decoder types:\n\
                        independent — each modality has its own topic dictionary;\n\
                                      features may differ across modalities.\n\
                        delta       — shared base dictionary + cumulative chain deltas;\n\
                                      modality m = softmax(z @ (W_base + Σ δ_1..m)).\n\
                                      Requires shared features across all modalities."
    )]
    JointTopic(JointTopicArgs),

    #[command(
        about = "2D visualization by pseudobulk spectral / tree / t-SNE layout",
        long_about = "Build 2D coordinates in four stages:\n\
                      (1) partition cells into pseudobulk samples,\n\
                      (2) compute PB-PB similarity from expression profiles,\n\
                      (3) lay out PB samples (spectral, tree, or t-SNE),\n\
                      (4) project cells by soft assignment to their nearest PB samples.",
        visible_alias = "viz"
    )]
    Visualize(VisualizeArgs),

    #[command(
        about = "Cluster cells on latent topic / SVD representations",
        long_about = "Cluster cells using a latent matrix from `senna topic` or `senna svd`.\n\n\
                      Algorithms:\n\
                        kmeans  — k-means (default; requires -k)\n\
                        leiden  — graph-based community detection (auto-k)\n\
                        hsblock — hierarchical stochastic block model (2^(depth-1) clusters)"
    )]
    Clustering(ClusteringArgs),
}

fn main() -> anyhow::Result<()> {
    // Show logo if help is requested
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let cli = Cli::parse();

    logging::init_logger(cli.verbose);

    match &cli.commands {
        Commands::Svd(args) => {
            fit_svd(args)?;
        }
        Commands::Topic(args) => {
            fit_topic_model(args)?;
        }
        Commands::IndexedTopic(args) => {
            fit_indexed_topic_model(args)?;
        }
        Commands::JointTopic(args) => {
            fit_joint_topic_model(args)?;
        }

        Commands::AnnotateTopic(args) => {
            annotate_topics(args)?;
        }
        Commands::EvalTopic(args) => {
            eval_topic_model(args)?;
        }
        Commands::JointSvd(args) => {
            fit_joint_svd(args)?;
        }
        Commands::Visualize(args) => {
            fit_visualize(args)?;
        }
        Commands::Clustering(args) => {
            run_clustering(args)?;
        }
    }

    info!("Done");
    Ok(())
}
