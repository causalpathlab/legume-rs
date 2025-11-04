mod embed_common;
mod fit_deconv_reg;
mod fit_joint_svd;
mod fit_joint_topic;
mod fit_svd;
mod fit_topic;
mod senna_input;

use embed_common::*;
use fit_deconv_reg::*;
use fit_joint_svd::*;
use fit_joint_topic::*;
use fit_svd::*;
use fit_topic::*;

/// Single cell embedding routines with nearest neighbourhood-based
/// adjustment
///
/// Data files of either `.zarr` or `.h5` format. All the formats in
/// the given list should be identical. We can convert `.mtx` to
/// `.zarr` or `.h5` using `data-beans from-mtx` or similar commands.
///
#[derive(Parser, Debug)]
#[command(version, about, long_about, term_width = 80)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// embedding by randomized singular value decomposition
    Svd(SvdArgs),

    /// embedding by fitting a topic model
    Topic(TopicArgs),

    /// embedding by randomized singular value decomposition
    JointSvd(JointSvdArgs),

    /// embedding by randomized singular value decomposition
    JointTopic(JointTopicArgs),

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
        Commands::JointSvd(args) => {
            fit_joint_svd(args)?;
        }
        Commands::DeconvReg(args) => {
            fit_deconv_reg(args)?;
        }
    }

    info!("Done");
    Ok(())
}
