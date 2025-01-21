mod collapse_data;
mod common;
mod normalization;
mod random_projection;

use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;
use clap::{Args, Parser, Subcommand};
use matrix_util::common_io::read_lines;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.commands {
        Commands::CollapseColumns(args) => {
            //
            // 1. push data files and collect batch membership
            //
            let backend = args.backend.clone().unwrap_or(SparseIoBackend::Zarr);

            let mut data_vec = SparseIoVec::new();
            for data_file in args.data_files.iter() {
                let data = open_sparse_matrix(&data_file, &backend)?;
                data_vec.push(Arc::from(data))?;
            }

            let mut batch_name_to_id = HashMap::new();
            let mut batch_membership = vec![];

            if let Some(batch_files) = &args.batch_files {
                if batch_files.len() != args.data_files.len() {
                    return Err(anyhow::anyhow!("# batch files != # of data files"));
                }

                for batch_file in batch_files.iter() {
                    for s in read_lines(&batch_file)? {
                        println!("batch file: {}", s);
                        if let Some(&id) = batch_name_to_id.get(&s) {
                            batch_membership.push(id);
                        } else {
                            let nbatch = batch_name_to_id.len();
                            batch_name_to_id.insert(s.clone(), nbatch);
                        }
                    }
                }
            } else {
                for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
                    batch_membership.extend(vec![id; nn]);
                    batch_name_to_id.insert(id.to_string().into_boxed_str(), id);
                }
            }

            if batch_membership.len() != data_vec.num_columns()? {
                return Err(anyhow::anyhow!("# batch membership != # of columns"));
            }

            //
            // 2. randomly project the columns
            //

            // do something
        }
    }

    Ok(())
}

#[derive(Parser)]
#[command(version, about, long_about=None)]
#[command(propagate_version = true)]
///
/// Basic utility functions for processing sparse matrices
///
/// - RP:
///
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build faster backend data from mtx
    CollapseColumns(RunCollapseArgs),
}

#[derive(Args)]
pub struct RunCollapseArgs {
    /// Data file -- either `.zarr` or `.h5`
    data_files: Vec<Box<str>>,

    #[arg(long)]
    batch_files: Option<Vec<Box<str>>>,

    /// Block_size for parallel processing
    #[arg(long, value_enum, default_value = "100")]
    block_size: Option<usize>,

    /// backend to use (HDF5 or Zarr)
    #[arg(short, long, value_enum, default_value = "Zarr")]
    backend: Option<SparseIoBackend>,
}
