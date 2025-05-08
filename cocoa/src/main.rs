mod cocoa_common;
mod cocoa_diff;
mod cocoa_collapse;
mod cocoa_stat;
mod simulate;

use crate::cocoa_common::*;
use crate::cocoa_diff::*;

use matrix_param::traits::{Inference, ParamIo, TwoStatParam};
use matrix_util::common_io::{
    extension, read_lines, read_lines_of_words, remove_file, write_types,
};
use matrix_util::traits::*;

use clap::{Subcommand, ValueEnum};

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
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Diff(args) => {
            //
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
