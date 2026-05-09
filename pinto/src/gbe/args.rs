//! `pinto gbe` argument struct. Reuses `SrtInputArgs` for spatial /
//! batch / barcode wiring and adds graph-embedding hyperparameters.

use crate::util::input::SrtInputArgs;
use candle_util::candle_core::Device;
use clap::{Args, ValueEnum};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum GbeDevice {
    Cpu,
    Cuda,
    Metal,
}

impl GbeDevice {
    pub fn to_device(&self, device_no: usize) -> anyhow::Result<Device> {
        Ok(match self {
            GbeDevice::Cpu => Device::Cpu,
            GbeDevice::Cuda => Device::new_cuda(device_no)?,
            GbeDevice::Metal => Device::new_metal(device_no)?,
        })
    }
}

#[derive(Args, Debug)]
pub struct SrtGbeArgs {
    #[command(flatten)]
    pub common: SrtInputArgs,

    #[arg(long, default_value_t = 64, help = "Embedding dimension H")]
    pub embedding_dim: usize,

    #[arg(long, default_value_t = 8, help = "Number of coarsening seeds")]
    pub num_coarsen_seeds: usize,

    #[arg(
        long,
        default_value_t = 200,
        help = "Target super-cell blocks (cell axis)"
    )]
    pub super_cells: usize,

    #[arg(long, default_value_t = 32, help = "Sketch dim for coarsening RP")]
    pub sketch_dim: usize,

    #[arg(long, default_value_t = 200, help = "Training epochs")]
    pub epochs: usize,

    #[arg(long, default_value_t = 100, help = "Batches per epoch")]
    pub batches_per_epoch: usize,

    #[arg(long, default_value_t = 1024, help = "Positive edges per batch")]
    pub batch_size: usize,

    #[arg(long, default_value_t = 16, help = "Negative samples per positive")]
    pub num_negatives: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "AdamW learning rate",
        alias = "lr"
    )]
    pub learning_rate: f64,

    #[arg(
        long,
        help = "Optional feature-feature edge list (TSV/CSV). \
                Activates SGC smoothing of E_feat through the K-hop \
                normalized adjacency."
    )]
    pub feature_network: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Allow prefix matching when resolving feature-network names"
    )]
    pub feature_network_prefix_match: bool,

    #[arg(
        long,
        help = "Optional name-stripping delimiter for feature-network resolution \
                (e.g. '.' to match `TP53.1` → `TP53`)"
    )]
    pub feature_network_delim: Option<char>,

    #[arg(long, default_value_t = 2, help = "SGC propagation hops K")]
    pub feature_network_k: usize,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "SGC neighbor-mix coefficient α ∈ [0, 1]"
    )]
    pub feature_network_alpha: f32,

    #[arg(
        long,
        default_value_t = 5,
        help = "Re-propagate the frozen network residual every N epochs"
    )]
    pub feature_network_refresh: usize,

    #[arg(
        long,
        default_value_t = '_',
        help = "Delimiter for fuzzy gene-name matching across input files. The last \
                token after splitting on this char is the canonical row name, so \
                `ENSG00000000003_TSPAN6` and `TSPAN6` align across files."
    )]
    pub feature_name_delim: char,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable fuzzy gene-name matching (use exact row-name match across files)"
    )]
    pub feature_name_exact: bool,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Cell-cell loss weight λ; final loss = L_bipartite + λ · L_cell_cell. \
                Default 1.0 weights cell-cell equal to cell-feature. Set to 0 to \
                disable. Requires --coord (cell pairs come from the spatial KNN)."
    )]
    pub cell_cell_lambda: f32,

    #[arg(
        long,
        default_value_t = 16,
        help = "Negative cells per positive cell-pair (cell-cell loss only)"
    )]
    pub cell_cell_negatives: usize,

    #[arg(long, default_value_t = GbeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: GbeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,
}
