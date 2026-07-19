//! Shared imports for `faba gem`.
//!
//! Mirrors the senna / pinto convention: re-export `candle_core` and
//! `candle_nn` through `candle_util` so the subcommand modules can do
//! `use crate::gem::common::*;` and then `use candle_core::…`
//! without listing candle-core / candle-nn as direct deps in
//! `faba/Cargo.toml`. That keeps the `cuda` / `metal` feature graph in
//! `Cargo.toml` propagating through a single candle entry point
//! (candle-util), the same way senna and pinto do.

pub use candle_util::{candle_core, candle_nn};

use clap::ValueEnum;

/// Compute device for candle tensors during training. Matches the
/// `ComputeDevice` enum used in senna (`embed_common.rs`) and pinto
/// (`cell_activity_graph_embedding/args.rs`).
#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
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

/// Phase-2 cell-projection method for `faba gem` (maps to
/// [`graph_embedding_util::CellProjection`]). Mirrors senna bge's `--projection`.
#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ProjectionArg {
    /// Exact analytical per-cell Poisson-MAP (IRLS, CPU): θ from spliced edges +
    /// the δ velocity increment from unspliced.
    Analytic,
    /// Stochastic frozen-feature NCE: θ on spliced edges, δ on unspliced (θ+δ),
    /// GPU-batched / CPU-parallel — faster at large H; approximate, seed-dependent.
    Nce,
}

impl ProjectionArg {
    #[must_use]
    pub fn to_ge(&self) -> graph_embedding_util::CellProjection {
        match self {
            ProjectionArg::Analytic => graph_embedding_util::CellProjection::Analytic,
            ProjectionArg::Nce => graph_embedding_util::CellProjection::Nce,
        }
    }
}
