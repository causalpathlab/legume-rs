//! CLI arguments for `pinto cage`.

use crate::cell_activity_graph_embedding::gene_gating::ActivityNorm;
use auxiliary_data::feature_names::FeatureNameKind;
use clap::{Parser, ValueEnum};
use data_beans_alg::hvg::HvgCliArgs;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

impl ComputeDevice {
    pub fn to_device(&self, device_no: usize) -> anyhow::Result<candle_util::candle_core::Device> {
        use candle_util::candle_core::Device;
        Ok(match self {
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::new_cuda(device_no)?,
            ComputeDevice::Metal => Device::new_metal(device_no)?,
        })
    }
}

/// Row-name canonicalization strategy for matching the data's gene
/// names against external resources (PPI networks, marker lists,
/// pretrained gene embeddings). `Auto` sniffs the first data file's
/// row names and dispatches to [`FeatureNameKind::auto_detect`] —
/// gene-symbol-style names (`ENSG..._SYMBOL`) get the `Gene` rule
/// applied automatically.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum GeneNameMode {
    /// Peek the first file's row names and pick `Exact` / `Gene` /
    /// `Locus` / `Mixed` via [`FeatureNameKind::auto_detect`].
    Auto,
    /// Strict string match — no canonicalization. The historical pinto
    /// default for `lc` / `svd`.
    Exact,
    /// `Gene { delim: '_' }`: register every `_`-split component as an
    /// alias of the full row name.
    Gene,
    /// `Locus { merge_overlapping: true }`: normalize chrom-coord names
    /// and collapse overlapping intervals.
    Locus,
    /// Heterogeneous axis: dispatch per row name.
    Mixed,
}

impl GeneNameMode {
    /// Resolve to a concrete [`FeatureNameKind`]. `peek_names` is only
    /// consulted under `Auto`; other modes ignore it.
    pub fn resolve_kind(self, peek_names: &[Box<str>]) -> FeatureNameKind {
        match self {
            GeneNameMode::Auto => FeatureNameKind::auto_detect(peek_names),
            GeneNameMode::Exact => FeatureNameKind::Exact,
            GeneNameMode::Gene => FeatureNameKind::Gene { delim: '_' },
            GeneNameMode::Locus => FeatureNameKind::Locus {
                merge_overlapping: true,
            },
            GeneNameMode::Mixed => FeatureNameKind::Mixed,
        }
    }
}

#[derive(Parser, Debug, Clone)]
pub struct CellActivityGraphEmbeddingArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,

    #[arg(
        long,
        default_value_t = GeneNameMode::Auto,
        value_enum,
        help = "Gene-name canonicalization for matching external resources",
        long_help = "Row-name canonicalization strategy:\n\
                       auto  — peek row names and auto-detect (default)\n\
                       exact — strict string equality (pinto lc / svd default)\n\
                       gene  — split on '_'; both 'ENSG..._TGFB1' and 'TGFB1' alias\n\
                       locus — normalize chrom-coord names; collapse overlaps\n\
                       mixed — per-row dispatch (RNA+ATAC paired axes)"
    )]
    pub gene_name_mode: GeneNameMode,

    #[arg(long, default_value_t = 16, help = "Cell embedding dimensionality")]
    pub embedding_dim: usize,

    #[arg(long, default_value_t = 5, help = "Training epochs over the gene axis")]
    pub epochs: usize,

    #[arg(
        long,
        default_value_t = 64,
        help = "Genes per outer parallel sampling chunk",
        long_help = "Outer training loop samples this many genes in parallel \
                     via rayon, then runs forward / backward serially \
                     (candle Var is not parallel-safe). Default is sized for \
                     a laptop; raise if you have many cores."
    )]
    pub gene_batch_size: usize,

    #[arg(
        long,
        default_value_t = 256,
        help = "Positive edges drawn per (gene, batch) sample"
    )]
    pub per_gene_batch: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "Sibling negatives drawn per positive edge per chain level"
    )]
    pub n_negatives: usize,

    #[arg(
        long,
        default_value_t = 0.75,
        help = "Negative-degree exponent (power-of-degree negative sampling)"
    )]
    pub alpha_neg: f32,

    #[arg(long, default_value_t = 5e-3, help = "AdamW learning rate")]
    pub lr: f32,

    #[arg(
        long,
        default_value_t = ActivityNorm::Log1p,
        value_enum,
        help = "Per-gene activity normalization"
    )]
    pub activity_norm: ActivityNorm,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "L2 penalty on softplus(α) gates (sparsity over levels)"
    )]
    pub gate_l2: f32,

    #[arg(
        long,
        value_delimiter(','),
        default_value = "0,1,2",
        help = "Chain levels (coarsest → finest) drawn from the coarsening hierarchy"
    )]
    pub chain_levels: Vec<usize>,

    /// HVG selection: senna-style shared CLI (`--n-hvg`,
    /// `--feature-list-file`). cage applies it as a **subset** (not a
    /// reweighting like `senna bge`), so the gene-axis training cost
    /// drops linearly with the selected K. `--n-hvg 0` disables.
    #[command(flatten)]
    pub hvg: HvgCliArgs,

    #[arg(
        long,
        default_value_t = 0,
        help = "Window (epochs) for convergence check; 0 disables",
        long_help = "After each epoch, look at the last `convergence-window` \
                     mean losses; if their (max − min) / |mean| is below \
                     --convergence-tol, stop training. 0 runs all --epochs \
                     unconditionally."
    )]
    pub convergence_window: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Relative-range threshold over --convergence-window for stopping"
    )]
    pub convergence_tol: f32,
}
