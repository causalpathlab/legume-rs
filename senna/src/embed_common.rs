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
    info!("use {} genes as common features", ngenes);

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
