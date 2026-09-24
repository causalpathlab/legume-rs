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

pub use clap::{Args, Parser, Subcommand, ValueEnum};

/// Compute device for candle-based models.
#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

impl std::fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ComputeDevice::Cpu => "cpu",
            ComputeDevice::Cuda => "cuda",
            ComputeDevice::Metal => "metal",
        })
    }
}

impl ComputeDevice {
    pub fn to_device(&self, device_no: usize) -> anyhow::Result<candle_core::Device> {
        Ok(match self {
            ComputeDevice::Cpu => candle_core::Device::Cpu,
            ComputeDevice::Cuda => candle_core::Device::new_cuda(device_no)?,
            ComputeDevice::Metal => candle_core::Device::new_metal(device_no)?,
        })
    }
}

pub use legume_numeric::matrix::common_io::{mkdir_parent, remove_file};
pub use legume_numeric::matrix::traits::*;
pub use legume_numeric::param::traits::TwoStatParam;

pub use data_beans::alg::collapse_data::MultilevelCollapsingOps;
pub use data_beans::alg::feature_coarsening::*;
pub use data_beans::alg::random_projection::*;

pub use legume_numeric::param::io::ParamIo;
pub use legume_numeric::param::traits::Inference;

pub use legume_numeric::candle::{candle_core, candle_nn};
