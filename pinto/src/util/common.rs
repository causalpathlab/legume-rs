pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;

pub use data_beans::sparse_io::*;
pub use data_beans::sparse_io_vector::*;

pub use indicatif::ParallelProgressIterator;
pub use legume_numeric::matrix::common_io::{basename, file_ext, read_lines};
pub use legume_numeric::matrix::dmatrix_util::*;
pub use legume_numeric::matrix::traits::*;
pub use log::{info, warn};
pub use rayon::prelude::*;

pub use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

pub use std::sync::{Arc, Mutex};

// Logging and progress bars come from `data_beans::aux::logging`, which draws
// through the workspace-wide `legume_numeric::matrix::progress::MULTI_PROGRESS`. A second
// `MultiProgress` here would leave its bars unbridged from the log output — see
// the `legume_numeric::matrix::progress` module docs. Bar labels are set with
// `new_progress_bar(n).with_message(..)`; the template belongs to the helper.
pub use data_beans::aux::logging::{init_logger, new_progress_bar};

// NB Fisher-info feature weighting helpers moved to
// `data_beans::alg::gene_weighting` so senna / chickpea / pinto share one
// implementation. Re-exported here for backwards compatibility within pinto.
pub use data_beans::alg::gene_weighting::{
    apply_gene_weights as apply_feature_weights, compute_nb_fisher_weights,
};

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
