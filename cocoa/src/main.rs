mod alg_collapse;
mod common;
mod run_diff;
mod stat;
mod run_sim;

use crate::run_diff::*;
use crate::run_sim::*;

use clap::Subcommand;

///
#[derive(Parser, Debug)]
#[command(version, about, long_about)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Differential expression analysis
    Diff(DiffArgs),

    /// Simulate differential expression data
    Simulate(SimArgs),
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Diff(args) => {
            run_cocoa_diff(args.clone())?;
        }
        Commands::Simulate(args) => {
	    run_diff_data(args.clone())?;
	}
    }

    Ok(())
}

// // two options: (1) scratch
// // (2) take .delta.log_mean and .delta.log_sd

// fn main() -> anyhow::Result<()> {

//     // let (samples, sample_to_cells): (Vec<_>, Vec<_>) = sample_to_cells.into_iter().unzip();
//     // let sample_to_cells = partition_by_membership(&cell_to_sample, None);

//     // data_vec.collect_stat(cocoa_input);

//     // let n_cells = data_vec.num_columns()?;
//     // let mut cell_to_sample = vec![0; n_cells];

//     // for (s, cells) in sample_to_cells.iter().enumerate() {
//     //     for &j in cells {
//     //         cell_to_sample[j] = s;
//     //     }
//     // }

//     info!("Done");
//     Ok(())
// }
