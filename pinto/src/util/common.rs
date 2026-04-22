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

/// Compute per-gene NB Fisher-info weights `w_g = 1 / (1 + π_g · s̄ · φ(μ_g))`
/// from raw cell-level counts. Streams `data_vec` column-by-column through a
/// `SparseRunningStatistics` accumulator, fits an NB dispersion trend, and
/// emits per-gene weights in row order (same order as `data_vec.row_names()`).
///
/// Bounded in `(0, 1]`, attenuates high-mean / high-dispersion genes,
/// recovers `w_g = 1` in the Poisson limit. The same formula is applied
/// inside DC-Poisson refinement so clustering and reporting stay consistent.
pub fn compute_nb_fisher_weights(
    data_vec: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    use data_beans_alg::nb_dispersion::DispersionTrend;
    use matrix_util::sparse_stat::SparseRunningStatistics;
    use matrix_util::utils::generate_minibatch_intervals;

    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();
    let jobs = generate_minibatch_intervals(n_cells, n_genes, block_size);

    let mut stats = SparseRunningStatistics::<f32>::new(n_genes);
    for &(lb, ub) in &jobs {
        let csc = data_vec.read_columns_csc(lb..ub)?;
        stats.add_csc(&csc);
    }

    let trend = DispersionTrend::from_sparse_stats(&stats);
    let means = stats.mean();
    let sums = stats.sum();
    let total_mass: f64 = sums.iter().map(|&s| s as f64).sum();
    let avg_s = if n_cells > 0 {
        (total_mass / n_cells as f64) as f32
    } else {
        1.0
    };
    let inv_total = if total_mass > 0.0 {
        1.0 / total_mass as f32
    } else {
        0.0
    };

    Ok((0..n_genes)
        .map(|g| trend.fisher_weight(sums[g] * inv_total, avg_s, means[g]))
        .collect())
}

/// Scale each row of `sum_gk` by its corresponding entry of `weights`.
/// `weights.len()` must equal `sum_gk.nrows()`.
pub fn apply_gene_weights(sum_gk: &mut Mat, weights: &[f32]) {
    debug_assert_eq!(weights.len(), sum_gk.nrows());
    for (g, &w) in weights.iter().enumerate() {
        sum_gk.row_mut(g).scale_mut(w);
    }
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
