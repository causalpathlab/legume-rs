#![allow(dead_code)]

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;
pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub use asap_data::sparse_data_visitors::*;
pub use asap_data::sparse_io::*;
pub use asap_data::sparse_io_vector::*;

pub use matrix_util::common_io::{extension, read_lines, write_lines, write_types};
pub use matrix_util::dmatrix_util::*;
pub use matrix_util::knn_match::ColumnDict;
pub use matrix_util::traits::*;
pub use matrix_util::utils::partition_by_membership;

pub use indicatif::ParallelProgressIterator;
pub use log::info;
pub use rayon::prelude::*;
pub use std::collections::{HashMap, HashSet};
pub use std::sync::{Arc, Mutex};

pub use candle_util::candle_inference::*;
pub use candle_util::candle_loss_functions as loss_func;
pub use candle_util::candle_matched_decoder_topic::*;
pub use candle_util::candle_matched_encoder_softmax::*;
pub use candle_util::candle_model_traits::*;

/// a quick wrapper for gzipped csv out: `{header}.{file_name}.csv.gz`
pub fn csv_gz_out(data: &Tensor, header: &str, file_name: &str) -> anyhow::Result<()> {
    let csv_file = header.to_string() + "." + file_name + ".csv.gz";
    data.to_device(&candle_core::Device::Cpu)?.to_csv(&csv_file)
}
