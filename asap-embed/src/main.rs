mod collapse_data;
mod common;
mod normalization;
mod random_projection;

use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;
use clap::{Args, Parser, Subcommand};
use collapse_data::CollapsingOps;
use env_logger;
use log::info;
use matrix_param::traits::*;
use matrix_util::common_io::{extension, read_lines};
use matrix_util::traits::*;
use random_projection::RandProjOps;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::CollapseColumns(args) => {
            // 1. push data files and collect batch membership
            let file = args.data_files[0].as_ref();
            let backend = match extension(file)?.to_string().as_str() {
                "h5" => SparseIoBackend::HDF5,
                "zarr" => SparseIoBackend::Zarr,
                _ => SparseIoBackend::Zarr,
            };

            let mut data_vec = SparseIoVec::new();
            for data_file in args.data_files.iter() {
                info!("Importing data file: {}", data_file);

                match extension(&data_file)?.as_ref() {
                    "zarr" => {
                        assert_eq!(backend, SparseIoBackend::Zarr);
                    }
                    "h5" => {
                        assert_eq!(backend, SparseIoBackend::HDF5);
                    }
                    _ => return Err(anyhow::anyhow!("Unknown file format: {}", data_file)),
                };

                let data = open_sparse_matrix(&data_file, &backend)?;
                data_vec.push(Arc::from(data))?;
            }

            let mut batch_membership = vec![];

            if let Some(batch_files) = &args.batch_files {
                if batch_files.len() != args.data_files.len() {
                    return Err(anyhow::anyhow!("# batch files != # of data files"));
                }

                for batch_file in batch_files.iter() {
                    info!("Reading batch file: {}", batch_file);
                    for s in read_lines(&batch_file)? {
                        batch_membership.push(s.to_string());
                    }
                }
            } else {
                for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
                    batch_membership.extend(vec![id.to_string(); nn]);
                }
            }

            if batch_membership.len() != data_vec.num_columns()? {
                return Err(anyhow::anyhow!(
                    "# batch membership {} != # of columns {}",
                    batch_membership.len(),
                    data_vec.num_columns()?
                ));
            }

            // 2. randomly project the columns
            info!("random projection of data onto {} dims", args.proj_dim);
            let proj_res =
                data_vec.project_columns(args.proj_dim, Some(args.block_size.clone()))?;
            proj_res
                .basis
                .to_tsv(&(args.out.to_string() + ".basis.gz"))?;

            let proj_kn = proj_res.proj;

            proj_kn
                .transpose()
                .to_tsv(&(args.out.to_string() + ".proj.gz"))?;

            info!("assigning {} columns to samples...", proj_kn.ncols());

            let nsamp = data_vec.assign_columns_to_samples(&proj_kn, Some(args.sort_dim))?;
            info!("at most {} samples are assigned", nsamp);

            // 3. register batch membership
            info!("registering batch-specific information");
            data_vec.register_batches(&proj_kn, &batch_membership)?;

            // 4. final collapsing
            info!("collapsing columns... into {} samples", nsamp);
            let ret =
                data_vec.collapse_columns(args.down_sample, Some(args.knn), Some(args.iter_opt))?;

            info!("writing down the results...");

            ret.mu.write_tsv(&(args.out.to_string() + ".mu"))?;

            if let Some(delta) = &ret.delta {
                delta.write_tsv(&(args.out.to_string() + ".delta"))?;
            }

            if let Some(gamma) = &ret.gamma {
                gamma.write_tsv(&(args.out.to_string() + ".gamma"))?;
            }
            info!("done");
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
    /// Data files of either `.zarr` or `.h5` format. All the formats
    /// should be identical. We can convert `.mtx` to `.zarr` or `.h5`
    /// using `asap-data build`
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Projection dimension
    #[arg(long, short, required = true)]
    proj_dim: usize,

    /// Sorting dimension
    #[arg(long, short, default_value = "10")]
    sort_dim: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value = "10")]
    knn: usize,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long)]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value = "100")]
    iter_opt: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    #[arg(long)]
    /// Batch membership files. Each bach file should correspond to
    /// each data file.
    batch_files: Option<Vec<Box<str>>>,

    /// Block_size for parallel processing
    #[arg(long, value_enum, default_value = "100")]
    block_size: usize,
}
