mod annotate_topic;
mod cluster;
mod deconv;
mod embed_common;
mod feature_selection;
mod fit_clustering;
mod fit_deconv_reg;
mod fit_joint_svd;
mod fit_joint_topic;
mod fit_knn_regression;
mod fit_svd;
mod fit_topic;
mod fit_visualize;
mod interactive_markers;
mod visualization_alg;
mod pseudobulk_topic;
mod senna_input;

use annotate_topic::*;
use embed_common::*;
use fit_clustering::*;
use fit_deconv_reg::*;
use fit_joint_svd::*;
use fit_joint_topic::*;
use fit_knn_regression::*;
use fit_svd::*;
use fit_topic::*;
use fit_visualize::*;

use colored::Colorize;

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line
        .replace('@', &"@".bright_yellow().to_string())
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
    let intro = vec![
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

        let text_part = if i < intro.len() {
            intro[i]
        } else {
            ""
        };

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
        about = "Annotate the dictionary and latent topics using marker features",
        long_about = "Annotate what each topic would mean using marker features/genes.\n\
              For each topic, we regress a feature vector of the dictionary\n\
              on the marker gene membership matrix (a design matrix)\n\
              to estimate the probability of assigning cell/group types.\n\n\
              LLM-ASSISTED ANNOTATION WORKFLOW:\n\n\
              1. Generate candidates:  --suggest-only candidates.json\n\
              2. Share candidates.json with an LLM (e.g., Claude)\n\
              3. LLM searches each gene and outputs suggestions.json\n\
              4. Apply suggestions:    --apply-suggestions suggestions.json\n\n\
              PROMPT FOR LLM:\n\
              \"Review these candidate genes for cell type annotation.\n\
              For each gene, search what cell types it marks.\n\
              Output JSON: [{\\\"gene\\\": \\\"GENE\\\", \\\"celltype\\\": \\\"TYPE\\\"}]\n\
              Only include genes confidently matching the proposed cell type.\"\n",
        after_help = "EXAMPLE LLM SESSION:\n\n\
              Input (candidates.json excerpt):\n  \
                {\"celltype\": \"Late_Erythroid\", \"topics\": [{\"genes\": [\n    \
                  {\"gene\": \"ENSG00000105205_CLC\", \"weight\": 0.0025}]}]}\n\n\
              LLM analysis:\n  \
                - CLC (Charcot-Leyden crystal galectin): eosinophil marker, NOT erythroid\n  \
                - Action: SKIP (does not match Late_Erythroid)\n\n\
              Input:\n  \
                {\"celltype\": \"Megakaryocyte_Precursor\", \"topics\": [{\"genes\": [\n    \
                  {\"gene\": \"ENSG00000244734_HBB\", \"weight\": 0.0018}]}]}\n\n\
              LLM analysis:\n  \
                - HBB (Hemoglobin subunit beta): erythroid lineage marker\n  \
                - Megakaryocytes share lineage with erythroid (MEP progenitor)\n  \
                - Action: INCLUDE\n\n\
              Output (suggestions.json):\n  \
                [{\"gene\": \"ENSG00000244734_HBB\", \"celltype\": \"Megakaryocyte_Precursor\"}]",
        visible_alias = "annotate"
    )]
    AnnotateTopic(AnnotateTopicArgs),

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
        long_about = "Estimate a probabilistic topic model in the three stages: \n\
		      (1) Collapse sparse data while adjusting batch effects\n\
		      (2) Estimate encoder-decoder architecture via SGD\n\
		      (3) Estimate latent states on the original data.\n"
    )]
    JointTopic(JointTopicArgs),

    #[command(
        about = "Construct matched data modality by kNN regression",
        long_about = "Construct matched data modality by kNN regression.\n\
		      If X data and Y data share no common columns/cells,\n\
		      we can predict Yhat by projecting x's columns onto\n\
		      and imputing values based on shared row/gene features.\n"
    )]
    KnnImputedData(KnnImputeArgs),

    #[command(about = "alias of knn-imputed-data")]
    KnnImpute(KnnImputeArgs),

    /// deconvolve bulk data with single cell reference dictionary
    DeconvReg(DeconvRegArgs),

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
		     - Leiden (graph-based, not yet implemented)\n\
		     - Louvain (graph-based, not yet implemented)\n\n\
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

    match &cli.commands {
        Commands::Svd(args) => {
            fit_svd(args)?;
        }
        Commands::Topic(args) => {
            fit_topic_model(args)?;
        }
        Commands::JointTopic(args) => {
            fit_joint_topic_model(args)?;
        }
        Commands::AnnotateTopic(args) => {
            annotate_topics(args)?;
        }
        Commands::JointSvd(args) => {
            fit_joint_svd(args)?;
        }
        Commands::DeconvReg(args) => {
            fit_deconv_reg(args)?;
        }
        Commands::KnnImputedData(args) => {
            fit_knn_regression(args)?;
        }
        Commands::KnnImpute(args) => {
            fit_knn_regression(args)?;
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
