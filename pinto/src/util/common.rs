pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;

pub use data_beans::sparse_io::*;
pub use data_beans::sparse_io_vector::*;

pub use indicatif::ParallelProgressIterator;
pub use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
pub use log::info;
pub use matrix_util::clustering::{Kmeans, KmeansArgs};
pub use matrix_util::common_io::{basename, file_ext, read_lines};
pub use matrix_util::dmatrix_util::*;
pub use matrix_util::traits::*;
pub use rayon::prelude::*;

pub use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

pub use std::sync::{Arc, LazyLock, Mutex};

/// Global `MultiProgress` instance shared by all progress bars and the log bridge.
pub static MULTI_PROGRESS: LazyLock<MultiProgress> = LazyLock::new(MultiProgress::new);

/// Initialise logging with the `indicatif-log-bridge` so that `log::info!`
/// messages print above progress bars instead of overwriting them.
///
/// * `verbose` – when `true`, sets `RUST_LOG=info` before building the logger.
pub fn init_logger(verbose: bool) {
    if verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    let logger = env_logger::Builder::from_default_env().build();
    let max_level = logger.filter();
    let _ = indicatif_log_bridge::LogWrapper::new(MULTI_PROGRESS.clone(), logger)
        .try_init()
        .map(|()| log::set_max_level(max_level));
}

/// Create a new `ProgressBar` registered with the global `MULTI_PROGRESS`.
pub fn new_progress_bar(len: u64, template: &str) -> ProgressBar {
    let pb = MULTI_PROGRESS.add(ProgressBar::new(len));
    pb.set_style(
        ProgressStyle::with_template(template)
            .unwrap()
            .progress_chars("##-"),
    );
    pb
}

pub struct Pair {
    pub left: usize,
    pub right: usize,
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
            .map(|&j| Ok(row.get_string(j)?.clone().into_boxed_str()))
            .collect::<Result<Vec<_>, parquet::errors::ParquetError>>()?;

        pairs.push(pp);
    }

    Ok(pairs)
}
