mod annotate_topic;
mod deconv;
mod embed_common;
mod feature_selection;
mod fit_deconv_reg;
mod fit_joint_svd;
mod fit_joint_topic;
mod fit_knn_regression;
mod fit_svd;
mod fit_topic;
mod senna_input;

use annotate_topic::*;
use embed_common::*;
use fit_deconv_reg::*;
use fit_joint_svd::*;
use fit_joint_topic::*;
use fit_knn_regression::*;
use fit_svd::*;
use fit_topic::*;

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
		      to estimate the probability of assigning cell/group types.\n",
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
}

fn main() -> anyhow::Result<()> {
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
    }

    info!("Done");
    Ok(())
}
