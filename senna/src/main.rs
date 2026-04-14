mod cluster;
mod cnv_pseudobulk;
mod embed_common;
mod eval_topic;
mod fit_clustering;
mod fit_indexed_topic;
mod fit_joint_topic;
mod fit_topic;
mod logging;
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
    about = "SENNA",
    long_about = "Stochastic data Embedding with Nearest Neighbourhood Adjustment\n\
		  Data files of either `.zarr` or `.h5` format. \n\
		  We can convert `.mtx` to `.zarr` or `.h5` using `data-beans from-mtx`"
)]
struct Cli {
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Embedding data by singular value decomposition",
        long_about = "Estimate Nystrom projection (SVD) in the three stages: \n\
		      (1) Collapse sparse data while adjusting batch effects\n\
		      (2) Estimate an orthogonal basis matrix\n\
		      (3) Project the original data onto the basis vectors.\n"
    )]
    Svd(SvdArgs),

    #[command(
        about = "Embedding data by topic modelling",
        long_about = "Estimate a probabilistic topic model in the three stages: \n\
		      (1) Collapse sparse data while adjusting batch effects\n\
		      (2) Estimate encoder-decoder architecture via SGD\n\
		      (3) Estimate latent states on the original data.\n"
    )]
    Topic(TopicArgs),

    #[command(
        about = "Embedding data by indexed topic modelling (adaptive feature windows)",
        long_about = "Estimate a probabilistic topic model with indexed encoder/decoder: \n\
		      (1) Collapse sparse data while adjusting batch effects\n\
		      (2) Estimate indexed encoder-decoder via SGD (top-K feature windows)\n\
		      (3) Estimate latent states on the original data.\n\
		      Uses per-sample adaptive feature selection for ~4-7x decoder speedup.\n",
        visible_alias = "itopic"
    )]
    IndexedTopic(IndexedTopicArgs),

    #[command(
        about = "Annotate topics using marker genes (vMF cosine similarity)",
        long_about = "Assign cell type probabilities to topics via vMF softmax on cosine similarity.\n\n\
              Modes:\n\
              - Direct:      senna annotate -g dict.parquet -z latent.parquet -m markers.tsv -o out\n\
              - Interactive: add -I to iteratively augment markers\n\
              - LLM-assist:  --suggest-only out.json, then --apply-suggestions in.json",
        visible_alias = "annotate"
    )]
    AnnotateTopic(AnnotateTopicArgs),

    #[command(
        about = "Evaluate a trained topic model on new data",
        long_about = "Apply a previously trained topic model to new data files.\n\
                      Handles gene alignment (new data may have different genes),\n\
                      estimates batch effects from the dictionary, and runs\n\
                      encoder inference on CPU with rayon parallelism.",
        visible_alias = "eval-topic"
    )]
    EvalTopic(EvalTopicArgs),

    #[command(
        about = "Embedding data by singular value decomposition on multiple data types",
        long_about = "Estimate Nystrom projection (SVD) in the three stages: \n\
		      (1) Collapse sparse data while adjusting batch effects\n\
		      (2) Estimate an orthogonal basis matrix\n\
		      (3) Project the original data onto the basis vectors.\n"
    )]
    JointSvd(JointSvdArgs),

    #[command(
        about = "Embedding data by topic modelling on multiple data types",
        long_about = "Estimate a joint probabilistic topic model across multiple modalities:\n\
		      (1) Collapse sparse data while adjusting batch effects\n\
		      (2) Estimate encoder-decoder architecture via SGD\n\
		      (3) Estimate latent states on the original data.\n\n\
		      Decoder types:\n\
		      - independent: each modality gets its own topic-to-feature dictionary\n\
		      - delta: shared base dictionary + cumulative chain deltas between \n\
		        consecutive modalities (first=reference, second=base+delta_1, etc.).\n\
		        All modalities must share the same features (genes).\n\n\
		      Data files are organized in a row-major table:\n\
		      files are grouped by modality (rows), sharing cells (columns).\n\
		      Use -m to specify the number of modality rows.\n"
    )]
    JointTopic(JointTopicArgs),

    #[command(
        about = "Visualize topic/SVD results with spectral embedding",
        long_about = "Create 2D visualization coordinates using spectral embedding.\n\
		      (1) Collapse data into pseudobulk samples\n\
		      (2) Compute PB-PB similarity from expression profiles\n\
		      (3) Spectral embedding of PB samples\n\
		      (4) Project cells via soft assignment to PB samples.\n",
        visible_alias = "viz"
    )]
    Visualize(VisualizeArgs),

    #[command(
        about = "Cluster cells based on latent representations",
        long_about = "Cluster cells using latent topic proportions or SVD embeddings.\n\n\
		     Supports multiple clustering algorithms:\n\
		     - K-means (default)\n\
		     - Leiden (graph-based community detection)\n\
		     Output: cluster assignments in parquet format"
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
