#![allow(dead_code)]

pub use log::info;
pub use std::sync::{Arc, Mutex};

pub const DEFAULT_KNN: usize = 10;
pub const DEFAULT_OPT_ITER: usize = 100;

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;

pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub use data_beans::sparse_data_visitors::*;
pub use data_beans::sparse_io::*;
pub use data_beans::sparse_io_stack::*;
pub use data_beans::sparse_io_vector::*;

pub use candle_util::{candle_core, candle_nn};

pub use clap::{Args, Parser, Subcommand, ValueEnum};

pub use matrix_param::io::ParamIo;
pub use matrix_param::traits::{Inference, TwoStatParam};
pub use matrix_util::common_io::remove_file;
pub use matrix_util::dmatrix_rsvd::nystrom_basis;
pub use matrix_util::traits::*;

pub use matrix_util::common_io::file_ext;
pub use matrix_util::dmatrix_util::concatenate_horizontal;

pub use data_beans_alg::collapse_data::*;
pub use data_beans_alg::feature_coarsening::*;
pub use data_beans_alg::random_projection::*;

/// Build `{prefix}0..{prefix}{k-1}` axis-id column names — the explicit
/// "this column is topic/cluster N" convention used by every K-dim
/// writer in this crate (and pinto's `C{c}` analogue). A reader can
/// recover the integer ID from the column name alone, surviving column
/// reordering, schema audits, and partial subsetting.
pub fn axis_id_names(prefix: &str, k: usize) -> Vec<Box<str>> {
    (0..k)
        .map(|i| format!("{prefix}{i}").into_boxed_str())
        .collect()
}

/// Inverse of [`axis_id_names`]. Accepts the explicit `{prefix}{c}` form
/// and the legacy bare-integer fallback (matrix-util's default column
/// names) so older parquets still load.
pub fn parse_axis_id(name: &str, prefix: &str) -> Option<i64> {
    if let Some(rest) = name.strip_prefix(prefix) {
        if let Ok(c) = rest.parse::<i64>() {
            return Some(c);
        }
    }
    name.parse::<i64>().ok()
}

/// Map every column to its axis ID via [`parse_axis_id`]. Returns `None`
/// if any column doesn't carry an ID — caller can then fall back to a
/// positional check.
pub fn try_parse_axis_ids(cols: &[Box<str>], prefix: &str) -> Option<Vec<i64>> {
    cols.iter().map(|c| parse_axis_id(c, prefix)).collect()
}

/// Multi-level training schedule
#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum LevelSchedule {
    /// Progressive: coarse→fine, more epochs for coarser levels
    Progressive,
    /// Mixed: all levels trained simultaneously each epoch
    Mixed,
}

/// Shared compute device enum for candle-based models
#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

/// Batch adjustment method
#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum AdjMethod {
    Batch,
    Residual,
}

impl AdjMethod {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            AdjMethod::Batch => "batch",
            AdjMethod::Residual => "residual",
        }
    }
}

/// Shared CNV detection CLI args (used by SVD, topic, indexed-topic).
/// Providing `--gff` or `--cnv-ground-truth` turns on the per-sample HMM CNV
/// model from `cnv::per_sample`.
#[derive(Args, Debug, Clone)]
pub struct CnvArgs {
    #[arg(long, help = "GFF/GTF annotation for CNV detection.")]
    pub gff: Option<Box<str>>,

    #[arg(
        long,
        help = "CNV ground-truth TSV (alternative to --gff; from `data-beans simulate`)."
    )]
    pub cnv_ground_truth: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of CN states (3 = del/neutral/gain; 5/6 = inferCNV i6-style)."
    )]
    pub cnv_states: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "If ≥3, BIC-select K ∈ [3..max] via kmeans on the marginal signal."
    )]
    pub cnv_gmm_k_max: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "(SVD path only) Number of k-means cell clusters used as cell-type proxy."
    )]
    pub cnv_svd_clusters: usize,
}

/// Training score tracker for topic models
pub struct TrainScores {
    pub llik: Vec<f32>,
    pub kl: Vec<f32>,
}

impl TrainScores {
    pub fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
        let mat = Mat::from_columns(&[
            DVec::from_vec(self.llik.clone()),
            DVec::from_vec(self.kl.clone()),
        ]);

        let score_types = vec![
            "log_likelihood".to_string().into_boxed_str(),
            "kl_divergence".to_string().into_boxed_str(),
        ];

        let epochs: Vec<Box<str>> = (0..mat.nrows())
            .map(|x| (x + 1).to_string().into_boxed_str())
            .collect();

        mat.to_parquet_with_names(
            file_path,
            (Some(&epochs), Some("epoch")),
            Some(&score_types),
        )
    }
}

/// Read a matrix from parquet or delimited text file
pub fn read_mat(file_path: &str) -> anyhow::Result<MatWithNames<Mat>> {
    Ok(match file_ext(file_path)?.as_ref() {
        "parquet" => Mat::from_parquet(file_path)?,
        _ => Mat::read_data(file_path, &['\t', ','], None, Some(0), None, None)?,
    })
}

/// Compute per-level sample budgets, weighted inversely by level size.
/// Coarser levels (fewer, higher-quality samples) get more budget.
#[must_use]
pub fn compute_level_budgets(level_sizes: &[usize], target_total: usize) -> Vec<usize> {
    let inv_weights: Vec<f64> = level_sizes.iter().map(|&n| 1.0 / n.max(1) as f64).collect();
    let inv_sum: f64 = inv_weights.iter().sum();
    inv_weights
        .iter()
        .map(|&w| ((w / inv_sum) * target_total as f64).round() as usize)
        .map(|b| b.max(1))
        .collect()
}

/// Subsample rows from a (input, batch, target) tuple to `budget` rows.
/// If the data already has fewer rows than `budget`, returns it unchanged.
pub fn subsample_rows(
    data: (Mat, Option<Mat>, Mat),
    budget: usize,
    rng: &mut impl rand::Rng,
) -> (Mat, Option<Mat>, Mat) {
    let (input, batch, target) = data;
    let n = input.nrows();
    if n <= budget {
        return (input, batch, target);
    }
    let idx: Vec<usize> = rand::seq::index::sample(rng, n, budget).into_vec();
    let sub_input = input.select_rows(idx.iter());
    let sub_batch = batch.map(|b| b.select_rows(idx.iter()));
    let sub_target = target.select_rows(idx.iter());
    (sub_input, sub_batch, sub_target)
}

/// Apply topic smoothing in log-space: exp → mix with uniform → log.
pub fn smooth_topics(
    log_z_nk: candle_core::Tensor,
    alpha: f64,
) -> candle_core::Result<candle_core::Tensor> {
    if alpha > 0.0 {
        let kk = log_z_nk.dim(1)? as f64;
        ((log_z_nk.exp()? * (1.0 - alpha))? + alpha / kk)?.log()
    } else {
        Ok(log_z_nk)
    }
}

/// Output container for bulk data aligned to a reference gene list
pub struct BulkDataOut {
    pub genes: Vec<Box<str>>,
    pub samples: Vec<Box<str>>,
    pub data: Mat,
}

/// Read bulk data files and align rows to the given gene list
pub fn read_bulk_data_aligned(
    bulk_data_files: &[Box<str>],
    genes: &[Box<str>],
) -> anyhow::Result<BulkDataOut> {
    use dashmap::DashMap as HashMap;

    let gene_to_position: HashMap<Box<str>, usize> = genes
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect();

    let ngenes = gene_to_position.len();
    info!("use {ngenes} genes as common features");

    let mut samples = vec![];
    let mut bulk_data_vec = vec![];

    for bulk_file in bulk_data_files {
        let MatWithNames {
            rows: raw_genes,
            cols: raw_samples,
            mat: raw_ds,
        } = read_mat(bulk_file.as_ref())?;

        let ncols = raw_samples.len();

        let mut padded_ds = Mat::zeros(ngenes, ncols);
        for (i, g) in raw_genes.iter().enumerate() {
            if let Some(r) = gene_to_position.get(g) {
                padded_ds.row_mut(*r.value()).copy_from(&raw_ds.row(i));
            }
        }

        samples.extend(raw_samples);
        bulk_data_vec.push(padded_ds);
    }
    let bulk_data = concatenate_horizontal(&bulk_data_vec)?;

    info!(
        "Read bulk data {} genes x {} samples",
        ngenes,
        samples.len()
    );
    Ok(BulkDataOut {
        genes: genes.to_vec(),
        samples,
        data: bulk_data,
    })
}
