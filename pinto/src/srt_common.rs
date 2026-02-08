#![allow(dead_code)]

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;
pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub use data_beans::sparse_data_visitors::*;
pub use data_beans::sparse_io::*;
pub use data_beans::sparse_io_vector::*;

pub use matrix_util::clustering::{Kmeans, KmeansArgs};
pub use matrix_util::common_io::{basename, file_ext, read_lines};
pub use matrix_util::dmatrix_util::*;
pub use matrix_util::traits::*;
pub use matrix_util::utils::partition_by_membership;

pub use indicatif::ParallelProgressIterator;
pub use log::info;
pub use rayon::prelude::*;

pub use std::collections::{HashMap, HashSet};

pub use std::sync::{Arc, Mutex};

pub use candle_util::{candle_core, candle_nn};

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

pub struct Pair {
    pub left: usize,
    pub right: usize,
}

////////////////////////////////////
// miscellaneous helper functions //
////////////////////////////////////

/// a thin wrapper for gzipped tsv out: `{header}.{file_name}.tsv.gz`
pub fn tsv_gz_out(data: &Tensor, header: &str, file_name: &str) -> anyhow::Result<()> {
    let tsv_file = header.to_string() + "." + file_name + ".tsv.gz";
    data.to_device(&candle_core::Device::Cpu)?.to_tsv(&tsv_file)
}

/// a thin wrapper for parquet out: `{header}.{file_name}.parquet`
pub fn named_tensor_parquet_out(
    data: &Tensor,
    row_names: Option<&[Box<str>]>,
    column_names: Option<&[Box<str>]>,
    header: &str,
    file_name: &str,
) -> anyhow::Result<()> {
    let file_path = header.to_string() + "." + file_name + ".parquet";
    data.to_device(&candle_core::Device::Cpu)?
        .to_parquet(row_names, column_names, &file_path)
}

/// a thin wrapper for parquet out: `{header}.{file_name}.parquet`
pub fn tensor_parquet_out(data: &Tensor, header: &str, file_name: &str) -> anyhow::Result<()> {
    named_tensor_parquet_out(data, None, None, header, file_name)
}

/// take names from parquet file
/// * `file_path` - file path
/// * `select_columns` - column names to extract
pub fn names_from_parquet(
    file_path: &str,
    select_columns: &[Box<str>],
) -> anyhow::Result<Vec<Vec<Box<str>>>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::RowAccessor;
    use std::fs::File;

    let file = File::open(file_path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();
    let nrows = metadata.file_metadata().num_rows() as usize;

    let fields = metadata.file_metadata().schema().get_fields();

    let select_columns: HashSet<Box<str>> = select_columns.iter().cloned().collect();

    let select_indices: Vec<usize> = fields
        .iter()
        .enumerate()
        .filter_map(|(j, f)| {
            if select_columns.contains(f.name()) {
                Some(j)
            } else {
                None
            }
        })
        .collect();

    if select_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "no columns matching with {:?}",
            select_columns
        ));
    }

    let row_iter = reader.get_row_iter(None)?;

    let mut pairs = Vec::with_capacity(nrows);

    for record in row_iter {
        let row = record?;
        let pp = select_indices
            .iter()
            .map(|&j| row.get_string(j).unwrap().clone().into_boxed_str())
            .collect::<Vec<_>>();

        pairs.push(pp);
    }

    Ok(pairs)
}
