#![allow(dead_code)]

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;
pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub use asap_data::sparse_data_visitors::*;
pub use asap_data::sparse_io::*;
pub use asap_data::sparse_io_vector::*;

pub use matrix_util::common_io::{basename, extension, read_lines, write_types};
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
pub use candle_util::{candle_core, candle_nn};

/// a quick wrapper for gzipped tsv out: `{header}.{file_name}.tsv.gz`
pub fn tsv_gz_out(data: &Tensor, header: &str, file_name: &str) -> anyhow::Result<()> {
    let tsv_file = header.to_string() + "." + file_name + ".tsv.gz";
    data.to_device(&candle_core::Device::Cpu)?.to_tsv(&tsv_file)
}

/// a quick wrapper for parquet out: `{header}.{file_name}.parquet`
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

/// a quick wrapper for parquet out: `{header}.{file_name}.parquet`
pub fn tensor_parquet_out(data: &Tensor, header: &str, file_name: &str) -> anyhow::Result<()> {
    named_tensor_parquet_out(data, None, None, header, file_name)
}
