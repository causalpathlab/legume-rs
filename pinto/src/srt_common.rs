#![allow(dead_code)]

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;
pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub use data_beans::sparse_data_visitors::*;
pub use data_beans::sparse_io::*;
pub use data_beans::sparse_io_vector::*;

pub use matrix_util::common_io::{basename, file_ext, read_lines};
pub use matrix_util::dmatrix_util::*;
pub use matrix_util::knn_match::ColumnDict;
pub use matrix_util::traits::*;
pub use matrix_util::utils::partition_by_membership;

pub use indicatif::ParallelProgressIterator;
pub use log::info;
pub use rayon::prelude::*;

pub use dashmap::DashMap as HashMap;
pub use dashmap::DashSet as HashSet;

pub use std::sync::{Arc, Mutex};

pub use candle_util::{candle_core, candle_nn};

pub struct Pair {
    pub left: usize,
    pub right: usize,
}

pub struct KmeansArgs {
    pub num_clusters: usize,
    pub max_iter: usize,
}

pub trait Kmeans {
    /// do k-means clustering of columns
    fn kmeans_columns(&self, args: KmeansArgs) -> Vec<usize>;
}

impl<T> Kmeans for DMatrix<T>
where
    T: Clone + Sync + Send,
    Vec<T>: clustering::Elem,
{
    fn kmeans_columns(&self, args: KmeansArgs) -> Vec<usize> {
        let data: Vec<Vec<T>> = self
            .column_iter()
            .map(|x| -> Vec<T> { x.iter().cloned().collect() })
            .collect();

        let clust = clustering::kmeans(args.num_clusters, &data, args.max_iter);
        clust.membership
    }
}

/// Impute `y` matrix by its neighbours `y_neigh` Here, we calculate
/// Euclidean distances after log1p transformation.
pub fn impute_with_neighbours(y: &CscMat, y_neigh: &CscMat) -> anyhow::Result<CscMat> {
    let mut log1p_y = y.clone();
    log1p_y.log1p_inplace();
    log1p_y.scale_columns_inplace();

    let mut log1p_y_neigh = y_neigh.clone();
    log1p_y_neigh.log1p_inplace();
    log1p_y_neigh.scale_columns_inplace();

    // columns of neighbours x columns of target y
    let dd = CscMat::from_nonzero_triplets(
        y_neigh.ncols(),
        y.ncols(),
        &log1p_y_neigh.euclidean_distance(&log1p_y)?,
    )?;

    let ww = (-dd).normalize_exp_logits_columns();

    Ok(y_neigh * &ww)
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

    let mut row_iter = reader.get_row_iter(None)?;

    let mut pairs = Vec::with_capacity(nrows);

    while let Some(record) = row_iter.next() {
        let row = record?;
        let pp = select_indices
            .iter()
            .map(|&j| row.get_string(j).unwrap().clone().into_boxed_str())
            .collect::<Vec<_>>();

        pairs.push(pp);
    }

    Ok(pairs)
}
