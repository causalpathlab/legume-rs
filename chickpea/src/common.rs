#![allow(dead_code)]
#![allow(unused_imports)]

pub use log::info;
pub use std::sync::Arc;

pub const DEFAULT_KNN: usize = 10;
pub const DEFAULT_OPT_ITER: usize = 100;

pub type Mat = nalgebra::DMatrix<f32>;
pub type DVec = nalgebra::DVector<f32>;
pub type CscMat = nalgebra_sparse::CscMatrix<f32>;

pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub use data_beans::sparse_io::*;
pub use data_beans::sparse_io_stack::*;
pub use data_beans::sparse_io_vector::*;

pub use clap::{Args, Parser, Subcommand};

pub use matrix_param::traits::TwoStatParam;
pub use matrix_util::common_io::remove_file;
pub use matrix_util::traits::*;

pub use data_beans_alg::collapse_data::MultilevelCollapsingOps;
pub use data_beans_alg::feature_coarsening::*;
pub use data_beans_alg::random_projection::*;

pub use matrix_param::io::ParamIo;
pub use matrix_param::traits::Inference;

pub use candle_util::{candle_core, candle_nn};
