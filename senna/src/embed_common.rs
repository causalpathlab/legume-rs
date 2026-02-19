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

pub use data_beans_alg::collapse_data::*;
pub use data_beans_alg::random_projection::*;
