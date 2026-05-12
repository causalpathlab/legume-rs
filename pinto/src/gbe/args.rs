//! `pinto gbe` argument struct. Reuses `SrtInputArgs` for spatial /
//! batch / barcode wiring and adds graph-embedding hyperparameters.

use crate::util::input::SrtInputArgs;
use candle_util::candle_core::Device;
use clap::{Args, ValueEnum};
use data_beans_alg::dc_poisson::FeatureWeighting;
use data_beans_alg::hvg::HvgCliArgs;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum GbeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum RefineWeightingArg {
    /// Fisher-info weight from the fitted NB mean-variance trend.
    #[default]
    NbFisherInfo,
    /// No per-feature weighting (raw DC-Poisson with entity-level degree correction).
    None,
}

impl From<RefineWeightingArg> for FeatureWeighting {
    fn from(value: RefineWeightingArg) -> Self {
        match value {
            RefineWeightingArg::NbFisherInfo => FeatureWeighting::FisherInfoNb,
            RefineWeightingArg::None => FeatureWeighting::None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum CompositeModeArg {
    /// Per step, sample a coordinated bottom-up chain. All axes share
    /// the same positive feature and negatives per chain. Lowest
    /// variance per step. Default.
    #[default]
    Chain,
    /// Per step, sum NCE losses across every axis (independent draws).
    Sum,
    /// Per step, pick one axis weighted by λ.
    Sample,
}

impl From<CompositeModeArg> for graph_embedding_util::CompositeMode {
    fn from(value: CompositeModeArg) -> Self {
        match value {
            CompositeModeArg::Sum => graph_embedding_util::CompositeMode::Sum,
            CompositeModeArg::Sample => graph_embedding_util::CompositeMode::Sample,
            CompositeModeArg::Chain => graph_embedding_util::CompositeMode::Chain,
        }
    }
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

    #[command(flatten)]
    pub hvg: HvgCliArgs,

    #[arg(
        short = 'e',
        long,
        default_value_t = 16,
        help = "Embedding dimension H"
    )]
    pub embedding_dim: usize,

    #[arg(long, default_value_t = 8, help = "Number of coarsening seeds")]
    pub num_coarsen_seeds: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Cap on the number of genes trained (0 = keep all). When > 0 and \
                less than the feature axis, keeps the top-N genes by NB-Fisher \
                weight before training. Main large-data speed knob."
    )]
    pub max_features: usize,

    #[arg(
        long = "composite-mode",
        value_enum,
        default_value_t = CompositeModeArg::Chain,
        help = "How to mix per-axis NCE losses each step. `chain` (default) samples a \
                coordinated bottom-up chain — all axes share the same positive feature \
                and negatives per chain (lowest variance). `sum` runs every axis with \
                independent minibatches per step. `sample` picks one axis per step \
                weighted by λ (~n_axes× faster epochs, needs more epochs to converge)."
    )]
    pub composite_mode: CompositeModeArg,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable BBKNN + DC-Poisson refinement of the multi-level pseudobulk \
                partition. Default: enabled."
    )]
    pub no_refine: bool,

    #[arg(long, default_value_t = 20, help = "Gibbs sweeps per refinement level")]
    pub refine_gibbs: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Greedy sweeps per refinement level"
    )]
    pub refine_greedy: usize,

    #[arg(
        long = "refine-weighting",
        value_enum,
        default_value_t = RefineWeightingArg::NbFisherInfo,
        help = "DC-Poisson feature weighting: nb-fisher-info (default), none"
    )]
    pub refine_weighting: RefineWeightingArg,

    #[arg(long, default_value_t = 42, help = "Seed for refinement Gibbs sampler")]
    pub refine_seed: u64,

    #[arg(
        long,
        default_value_t = 200,
        help = "Target super-cell blocks (cell axis)"
    )]
    pub super_cells: usize,

    #[arg(long, default_value_t = 32, help = "Sketch dim for coarsening RP")]
    pub sketch_dim: usize,

    #[arg(short = 'i', long, default_value_t = 200, help = "Training epochs")]
    pub epochs: usize,

    #[arg(long, default_value_t = 100, help = "Batches per epoch")]
    pub batches_per_epoch: usize,

    #[arg(long, default_value_t = 1024, help = "Positive edges per batch")]
    pub batch_size: usize,

    #[arg(long, default_value_t = 4, help = "Negative samples per positive")]
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
        default_value_t = false,
        help = "Always recompute NB-Fisher weights and overwrite the cache. By \
                default `{out}.fisher_weights.parquet` is loaded if it exists \
                with matching gene names, otherwise computed and written."
    )]
    pub no_fisher_cache: bool,

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

    #[arg(
        long,
        default_value = "all",
        help = "Pseudobulk levels to chain the cell-cell loss over. \
                `all` (default) uses every level produced by the multilevel \
                collapse; a comma list like `0,2,4` picks specific levels \
                (0 = coarsest, last = finest). Positives are restricted to \
                spatial edges whose endpoints share pb at every selected \
                level; per-level negatives are siblings under the same pb \
                parent, giving the cell embedding a multi-resolution \
                classification signal. To disable the cell-cell term \
                entirely, pass `--cell-cell-lambda 0`."
    )]
    pub cell_cell_pb_levels: String,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Per-level λ for the cell-cell chain (comma list, same length \
                as `--cell-cell-pb-levels` once resolved). Default = uniform \
                1.0 per level. The outer `--cell-cell-lambda` still scales \
                the whole cell-cell term."
    )]
    pub cell_cell_lambda_per_level: Option<Vec<f32>>,

    #[arg(long, default_value_t = GbeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: GbeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,

    #[arg(
        long,
        value_enum,
        help = "Run downstream clustering on the latent after fit. \
                Omit to skip. Writes {out}.clusters.parquet."
    )]
    pub cluster: Option<ClusterMethodArg>,

    #[arg(
        long,
        default_value_t = 15,
        help = "k for Leiden kNN graph on the latent"
    )]
    pub cluster_knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution (higher = more clusters)"
    )]
    pub cluster_resolution: f64,

    #[arg(
        long,
        help = "Target cluster count: when set with --cluster leiden, the \
                resolution is auto-tuned to approximate this k; when set \
                with --cluster kmeans, this is the exact k."
    )]
    pub num_clusters: Option<usize>,

    #[arg(
        long,
        default_value_t = 100,
        help = "Max k-means iterations (--cluster kmeans only)"
    )]
    pub kmeans_max_iter: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "lowercase")]
pub enum ClusterMethodArg {
    Leiden,
    Kmeans,
}

impl ClusterMethodArg {
    pub fn to_method(self) -> crate::gbe::cluster::ClusterMethod {
        match self {
            ClusterMethodArg::Leiden => crate::gbe::cluster::ClusterMethod::Leiden,
            ClusterMethodArg::Kmeans => crate::gbe::cluster::ClusterMethod::Kmeans,
        }
    }
}
