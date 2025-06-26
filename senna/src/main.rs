mod embed_common;
mod fit_deconv;
mod fit_svd;
mod fit_topic;
mod routines_latent_representation;
mod routines_post_process;
mod routines_pre_process;

use embed_common::*;

use fit_deconv::*;
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
    Svd(SvdArgs),
    Topic(TopicArgs),
    Deconv(DeconvArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Svd(args) => {
            fit_svd(args)?;
        }
        Commands::Topic(args) => {
            fit_topic_model(args)?;
        }
        Commands::Deconv(args) => {
            fit_deconv(args)?;
        }
    }

    info!("Done");
    Ok(())
}
