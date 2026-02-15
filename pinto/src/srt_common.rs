pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;

pub use data_beans::sparse_data_visitors::*;
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

pub use std::collections::{HashMap, HashSet};

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

/// Filter a parquet file to keep only rows at the given indices.
/// Reads the file, selects rows, and overwrites the file.
pub fn filter_parquet_by_indices(file_path: &str, keep_indices: &[usize]) -> anyhow::Result<()> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::file::writer::SerializedFileWriter;
    use parquet::record::RowAccessor;
    use std::fs::File;

    let file = File::open(file_path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema().clone();
    let field_types: Vec<_> = schema
        .get_fields()
        .iter()
        .map(|f| f.get_physical_type())
        .collect();

    let keep_set: HashSet<usize> = keep_indices.iter().cloned().collect();

    // Collect kept rows
    let row_iter = reader.get_row_iter(None)?;
    let mut kept_rows = Vec::with_capacity(keep_indices.len());

    for (idx, record) in row_iter.enumerate() {
        if keep_set.contains(&idx) {
            kept_rows.push(record?);
        }
    }
    drop(reader);

    // Rewrite file
    let file = File::create(file_path)?;
    let schema_arc = std::sync::Arc::new(schema);
    let mut writer = SerializedFileWriter::new(file, schema_arc, Default::default())?;
    let mut row_group = writer.next_row_group()?;

    // Write column by column
    for (col_idx, &physical_type) in field_types.iter().enumerate() {
        use parquet::basic::Type as PhysicalType;
        use parquet::data_type::*;

        let mut col_writer = row_group
            .next_column()?
            .ok_or_else(|| anyhow::anyhow!("missing column writer at index {}", col_idx))?;

        match physical_type {
            PhysicalType::FLOAT => {
                let vals: Vec<f32> = kept_rows
                    .iter()
                    .map(|r| r.get_float(col_idx))
                    .collect::<Result<_, _>>()?;
                col_writer
                    .typed::<FloatType>()
                    .write_batch(&vals, None, None)?;
            }
            PhysicalType::BYTE_ARRAY => {
                let vals: Vec<ByteArray> = kept_rows
                    .iter()
                    .map(|r| Ok(ByteArray::from(r.get_string(col_idx)?.as_bytes().to_vec())))
                    .collect::<Result<_, parquet::errors::ParquetError>>()?;
                col_writer
                    .typed::<ByteArrayType>()
                    .write_batch(&vals, None, None)?;
            }
            PhysicalType::DOUBLE => {
                let vals: Vec<f64> = kept_rows
                    .iter()
                    .map(|r| r.get_double(col_idx))
                    .collect::<Result<_, _>>()?;
                col_writer
                    .typed::<DoubleType>()
                    .write_batch(&vals, None, None)?;
            }
            PhysicalType::INT32 => {
                let vals: Vec<i32> = kept_rows
                    .iter()
                    .map(|r| r.get_int(col_idx))
                    .collect::<Result<_, _>>()?;
                col_writer
                    .typed::<Int32Type>()
                    .write_batch(&vals, None, None)?;
            }
            PhysicalType::INT64 => {
                let vals: Vec<i64> = kept_rows
                    .iter()
                    .map(|r| r.get_long(col_idx))
                    .collect::<Result<_, _>>()?;
                col_writer
                    .typed::<Int64Type>()
                    .write_batch(&vals, None, None)?;
            }
            other => {
                return Err(anyhow::anyhow!(
                    "unsupported column type {:?} at index {}",
                    other,
                    col_idx
                ));
            }
        }
        col_writer.close()?;
    }

    row_group.close()?;
    writer.close()?;

    Ok(())
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

/// Linearly interpolate sort dimensions from coarsest (min 4) to finest.
pub fn compute_level_sort_dims(finest_sort_dim: usize, num_levels: usize) -> Vec<usize> {
    if num_levels <= 1 {
        return vec![finest_sort_dim];
    }
    let coarsest = 4usize.min(finest_sort_dim);
    let mut dims = Vec::with_capacity(num_levels);
    for level in 0..num_levels {
        let t = level as f32 / (num_levels - 1) as f32;
        let dim = coarsest as f32 + t * (finest_sort_dim - coarsest) as f32;
        dims.push(dim.round() as usize);
    }
    dims
}
