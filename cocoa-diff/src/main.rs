mod cocoa_collapse_data;
mod cocoa_common;

use asap_embed::asap_random_projection as rp;
use rp::RandProjOps;

use cocoa_common::*;
use cocoa_collapse_data::*;
use matrix_param::traits::{Inference, ParamIo, TwoStatParam};
use matrix_util::common_io::{extension, read_lines, remove_file, write_types};
use matrix_util::traits::*;

use clap::{Parser, ValueEnum};
use log::info;
use matrix_util::utils::partition_by_membership;
use std::sync::Arc;

#[derive(Parser, Debug, Clone)]
struct CocoaArgs {
    /// Data files of either `.zarr` or `.h5` format. All the formats
    /// in the given list should be identical. We can convert `.mtx`
    /// to `.zarr` or `.h5` using `asap-data build` command.
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Sample membership files (comma-separated names). Each bach
    /// file should match with each data file.
    #[arg(long, short, value_delimiter(','))]
    sample_files: Vec<Box<str>>,

    /// #k-nearest neighbours within each condition
    #[arg(long, short = 'n', default_value_t = 10)]
    knn: usize,

    /// Random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 30)]
    proj_dim: usize,

    /// Block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

// two options: (1) scratch
// (2) take .delta.log_mean and .delta.log_sd

// tood: should consider celltype

fn main() -> anyhow::Result<()> {
    let args = CocoaArgs::parse();

    info!("Reading data files...");
    let (data_vec, sample_to_cells) = read_data(args.clone())?;

    let proj_kn = data_vec.project_columns(args.proj_dim, Some(args.block_size))?;

    // args.block_size

    //

    // self.visit_column_by_samples(&sample_to_cells, &count_basic, &EmptyArg {}, stat)

    info!("Done");
    Ok(())
}

fn read_data(args: CocoaArgs) -> anyhow::Result<(SparseIoVec, Vec<Vec<usize>>)> {
    // push data files and collect batch membership
    let file = args.data_files[0].as_ref();
    let backend = match extension(file)?.to_string().as_str() {
        "h5" => SparseIoBackend::HDF5,
        "zarr" => SparseIoBackend::Zarr,
        _ => SparseIoBackend::Zarr,
    };

    if args.sample_files.len() != args.data_files.len() {
        return Err(anyhow::anyhow!("# sample files != # of data files"));
    }

    let mut data_vec = SparseIoVec::new();
    let mut cell_to_sample = vec![];

    for (data_file, sample_file) in args.data_files.into_iter().zip(args.sample_files) {
        info!("Importing: {}, {}", data_file, sample_file);

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
        let samples = read_lines(&sample_file)?;

        if samples.len() != data.num_columns().unwrap_or(0) {
            return Err(anyhow::anyhow!(
                "{} and {} don't match",
                sample_file,
                data_file,
            ));
        }

        cell_to_sample.extend(samples);
        data_vec.push(Arc::from(data))?;
    }

    let sample_to_cells: Vec<Vec<usize>> = partition_by_membership(&cell_to_sample, None)
        .into_values()
        .collect();

    Ok((data_vec, sample_to_cells))
}
