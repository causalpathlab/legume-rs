//! `senna bge`'s command-line surface: the [`BgeArgs`] clap struct and its
//! `--update` rebase.
//!
//! Split out of the module driver because it is almost entirely help text —
//! several hundred lines of it — and reading `fit_bge` meant scrolling past all
//! of it first. Nothing here computes; the translation to
//! `graph_embedding_util::FitConfig` stays with the driver that performs it.

use crate::embed_common::*;
use data_beans_alg::hvg::HvgCliArgs;
use graph_embedding_util as ge;

/// The label (or, when `None`, the within-group position) namespaces that
/// file's features as `{name}/{modality}`.
pub(crate) type MultiomeFile = (Option<Box<str>>, Box<str>);

#[derive(Args, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "crate::embed_common::clap_defaults")]
pub struct BgeArgs {
    #[arg(
        value_delimiter = ',',
        help = "Sparse count matrices (zarr/h5), comma-separated",
        long_help = "Single-modality input. One or more files share a feature axis.\n\
                     Cells are unified by barcode.\n\
                     \n\
                     Multiome input goes to --multiome instead.\n\
                     There the modalities have distinct feature spaces, glued by barcode.\n\
                     Exactly one of the positional files or --multiome is required."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        short = 'b',
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    /// The parent's carried pseudobulks, when `senna update` chose to reuse
    /// them instead of re-reading its cells. Derived per invocation, so it is
    /// neither a CLI flag nor part of the recorded configuration.
    #[arg(skip)]
    #[serde(skip)]
    pub(crate) pb_reference: Option<crate::pb_reference::ReferenceInput>,

    /// The parent run this one continues, set by `senna update`. Like `svd`,
    /// bge has no weights to warm-start (its ETM is re-derived by archetypal
    /// analysis each run), so this only chains the emitted reference's
    /// generation counter.
    #[arg(skip)]
    #[serde(skip)]
    pub(crate) init_from: Option<Box<str>>,

    #[command(flatten)]
    pub(crate) hvg: HvgCliArgs,

    #[arg(
        long,
        default_value_t = 128,
        help = "Embedding dimension H",
        alias = "dim-embedding"
    )]
    pub(crate) embedding_dim: usize,

    #[command(flatten)]
    pub(crate) collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    pub(crate) qc: QcArgs,

    // Spike-and-slab feature gate — ALWAYS ON for bge (the standard training):
    // Ẽ_{g,h} = σ(S_{g,h}) · E_{g,h}, an INDEPENDENT inclusion probability per
    // (gene, dim). Each dim's inclusion rate π_h is learned, so per-dim mass is
    // controlled by the prior rather than pinned by a normalizer. There is no null
    // column: an unselected gene simply has σ(S) → 0 everywhere. σ(S) is the same
    // estimand `--posterior` reports as feature_pip, so the two are comparable.
    // Temperature is the one knob.
    #[arg(
        long = "feature-gate-temp",
        alias = "feature-softmax-temp",
        default_value_t = 1.0,
        help = "Feature-gate temperature τ (< 1 sharpens each inclusion probability toward 0/1).",
        hide = true
    )]
    pub(crate) feature_gate_temp: f32,

    #[arg(
        long = "gate-ibp-alpha",
        help = "Truncated-IBP concentration for the gate's per-dim inclusion ladder;\n\
                unset = auto",
        long_help = "Concentration alpha of the truncated Indian Buffet Process.\n\
                     Its ladder tilts the feature gate.\n\
                     Dim h carries a fixed logit offset h * ln(alpha/(alpha+1)).\n\
                     So later dims must earn inclusion against a steeper prior.\n\
                     Chosen, never fitted.\n\
                     \n\
                     Unset (the default) derives alpha from --embedding-dim.\n\
                     The ladder then spans 4 logits end to end.\n\
                     That leaves the last dim at the sigmoid's most responsive point,\n\
                     rather than frozen.\n\
                     \n\
                     SMALLER alpha means a steeper ladder and more sparsity.\n\
                     This replaced a KL toward a Beta(1,9) inclusion prior.\n\
                     It had no natural weight under bge's noise-contrastive objective."
    )]
    pub(crate) gate_ibp_alpha: Option<f64>,

    #[arg(
        long = "phase1-cells-per-pb",
        default_value_t = 0,
        help = "Phase-1 cell-axis mode (k); 0 = pure-pb (fastest),\n\
                phase 2 always projects every cell.",
        long_help = "Phase-1 cell-axis mode (k).\n\
                     Controls what shapes the feature dictionary in phase 1;\n\
                     phase 2 ALWAYS analytically projects every cell,\n\
                     so the per-cell embedding output is unaffected.\n\
                     k=0 (default) → suppress the cell axis entirely.\n\
                     This is pure-pb: E_feat from pb aggregates only, and fastest.\n\
                     1≤k<n_cells → keep ≤k cells per pb-sample at each level (union).\n\
                     That cuts the phase-1 step budget, preserving rare-cell coverage.\n\
                     k≥n_cells → all cells (legacy; slowest).",
        hide = true
    )]
    pub(crate) phase1_cells_per_pb: usize,

    #[arg(
        long = "skip-etm",
        default_value_t = false,
        help = "Skip ETM resolution; emit raw bge embeddings (Z and ρ) only.",
        long_help = "Skip the default ETM resolution.\n\
                     Only the raw bge embeddings are then emitted: cell_embedding = Z,\n\
                     dictionary = ρ, and no latent.\n\
                     \n\
                     By default bge resolves ETM topics from the cell embedding,\n\
                     by anchor analysis. It then ALSO writes the topic-model tables:\n\
                     latent = log θ, dictionary = β, topic_embedding = α.\n\
                     \n\
                     Either way, Z lands in {out}.cell_embedding.parquet."
    )]
    pub(crate) skip_etm: bool,

    #[arg(
        long = "no-pip-shrinkage",
        default_value_t = false,
        help = "Do not shrink co-embedded features by posterior confidence",
        long_help = "This applies when --posterior has run.\n\
                     Each feature's co-embedded coordinate is then scaled.\n\
                     The scale factor is its `max_h PIP`.\n\
                     That is the posterior probability of loading ANY dim.\n\
                     \n\
                     Scaling is applied after the softmax.\n\
                     Attention weights and the calibrated temperature are untouched.\n\
                     What it does is compress low-confidence features radially,\n\
                     toward the origin.\n\
                     \n\
                     READ THAT LITERALLY. It is a confidence-weighted radial scaling.\n\
                     It corrects for nothing.\n\
                     An earlier version of this help was wrong about it.\n\
                     It claimed the scaling rescued signal-free genes.\n\
                     Supposedly they piled up on the cell centroid. Measured,\n\
                     there is no such pile-up.\n\
                     0.0% of genes sit within 0.1 cell-radii of the centroid,\n\
                     and the median distance is 0.80.\n\
                     So the shrinkage does not undo a concentration. It CREATES one,\n\
                     at the origin. Whether you want that depends on how you read the plot.\n\
                     \n\
                     The scaling is only as informative as its posterior.\n\
                     When the embedding dimension far exceeds the effective rank,\n\
                     nearly every gene loads something.\n\
                     `max_h PIP` then saturates near 1. The weights degenerate into one constant.\n\
                     The run reports the weight spread.\n\
                     That case is therefore visible rather than silent.\n\
                     \n\
                     Pass this flag to keep the raw co-embedding."
    )]
    pub(crate) no_pip_shrinkage: bool,

    #[arg(
        long = "num-topics",
        help = "ETM topics K (omit to auto-select via SPA-anchor residual-elbow sweep)."
    )]
    pub(crate) num_topics: Option<usize>,

    #[arg(
        long = "bridge-weight",
        default_value_t = 1.0,
        help = "Up-weight matched cells in the cell-axis sampler; 1.0 = off",
        long_help = "Up-weight matched multi-modality cells in the cell-axis sampler.\n\
                     They then anchor the cross-modal alignment.\n\
                     This applies to --multiome only; 1.0 turns it off."
    )]
    pub(crate) bridge_weight: f32,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable BBKNN + DC-Poisson refinement of the multi-level pseudobulk partition.\n\
                Default: enabled."
    )]
    pub(crate) no_refine: bool,

    #[arg(short = 'i', long, default_value_t = 1000, help = "Training epochs")]
    pub(crate) epochs: usize,

    /// Batches per epoch. **Omit for auto** — one weighted pass per
    /// epoch over the largest axis (`ceil(max_axis_units / batch_size)`).
    /// Pass a value to force a fixed step budget per epoch (historical
    /// default: 100).
    #[arg(
        long,
        help = "Batches per epoch (default: auto = one pass over largest axis)",
        hide = true
    )]
    pub(crate) batches_per_epoch: Option<usize>,

    #[arg(
        long,
        help = "Positive edges per batch (unset: 1024, shrunk to fit GPU memory on CUDA)",
        long_help = "Positive edges per SGD batch.\n\
                     Unset, the default is 1024 on CPU.\n\
                     On CUDA the size is chosen automatically:\n\
                     a short probe measures the memory one step retains,\n\
                     and shrinks the batch from 1024\n\
                     when --gpu-mem-fraction of free device memory\n\
                     cannot hold it (it never grows past 1024:\n\
                     batch size is not fit-neutral).\n\
                     Passing a value disables the probe and always wins."
    )]
    pub(crate) batch_size: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.6,
        help = "Fraction of free GPU memory the training batch may target",
        long_help = "Ceiling for the automatic batch sizing on CUDA.\n\
                     The probe grows the batch\n\
                     while one step's retained memory,\n\
                     with half reserved for the backward pass,\n\
                     fits this fraction of the device memory free at start.\n\
                     Fractions outside 0.05 to 0.95 are clamped to that range.\n\
                     Ignored on CPU and when --batch-size is set."
    )]
    pub(crate) gpu_mem_fraction: f32,

    #[arg(long, default_value_t = 4, help = "Negative samples per positive")]
    pub(crate) num_negatives: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "AdamW learning rate",
        alias = "lr"
    )]
    pub(crate) learning_rate: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "L2 penalty λ on E_feat (row-mean of ‖e_d‖²). Default 0 (off).",
        long_help = "L2 penalty λ on the shared feature embedding E_feat ∈ ℝ^{D×H}.\n\
                     It adds λ · mean_d ‖e_d‖² to the per-step composite loss.\n\
                     The norm is summed over the H latent dims.\n\
                     The mean is taken over the D rows.\n\
                     So λ stays scale-invariant across D, but is not diluted by H.\n\
                     \n\
                     E_feat is largely self-bounded under the NCE setup,\n\
                     with its analytical projection, hence the default of 0 (off).\n\
                     Raise it if E_feat drifts on long/deep runs.\n\
                     \n\
                     Note this penalty was previously divided by H as well.\n\
                     That made it ~H× weaker than the same λ buys today."
    )]
    pub(crate) feature_embedding_l2: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay (all params). Default 0.0 = off.",
        long_help = "AdamW decoupled weight decay, applied uniformly to every parameter.\n\
                     That covers E_feat, b_feat, and the per-axis heads.\n\
                     Per-step post-update shrinkage; doesn't enter the backward graph.\n\
                     Default 0.0 (off — plain Adam despite the optimizer name)."
    )]
    pub(crate) weight_decay: f64,

    #[arg(
        long = "max-grad-norm",
        default_value_t = 1.0,
        help = "Global-norm gradient clip per AdamW step (0 = off). When > 0,\n\
                gradients are scaled down if their global L2 norm exceeds this,\n\
                bounding embedding inflation on NCE loss spikes."
    )]
    pub(crate) max_grad_norm: f32,

    #[arg(
        long,
        help = "Cells per block for column I/O / streaming (omit for auto).",
        long_help = "Cells per parallel block for streaming column-block I/O.\n\
                     Omit for auto-scaling, which clamps to 100 for large feature counts.\n\
                     That is slow on rotational disks.\n\
                     Pass 1024+ when you have RAM, especially without --preload-data.",
        hide = true
    )]
    pub(crate) block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse column data into memory. Faster when data fits in RAM;\n\
                required on slow disks.",
        hide = true
    )]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        value_name = "FILE[,FILE...]",
        help = "Multiome modality files (comma-separated); repeat for multiple samples.",
        long_help = "Multiome load. Pass one sample (group) per flag, comma-separated,\n\
                     as in `--multiome rna.zarr,atac.zarr`. Cells are the shared axis.\n\
                     Each modality keeps its own features.\n\
                     Repeat the flag for each additional sample or group:\n\
                     \n\
                     --multiome rna1.zarr,atac1.zarr \\\n\
                     --multiome rna2.zarr,atac2.zarr\n\
                     \n\
                     Cell (barcode) identity. Within a group, equal barcodes are the same cell.\n\
                     Modalities Union-merge. A cell present in only some files is fine.\n\
                     Patchy multiome therefore works. ACROSS groups, barcodes must be disjoint.\n\
                     A shared barcode would merge cells from different samples.\n\
                     This is validated, and a collision is an error.\n\
                     \n\
                     Feature (modality) identity. Features are namespaced `{name}/{modality}`.\n\
                     The SAME modality across samples therefore merges, sharing one gene panel.\n\
                     DIFFERENT modalities stay on separate rows.\n\
                     That holds even when names collide.\n\
                     Spliced versus unspliced `TSPAN6` is the usual case.\n\
                     \n\
                     The modality tag defaults to file position: m0, m1, and so on.\n\
                     Override it with a `label=` prefix:\n\
                     --multiome spliced=spliced.zarr,unspliced=unspliced.zarr\n\
                     File ORDER within a group defines modality order,\n\
                     so the positional default lines up across groups.\n\
                     \n\
                     Batch identity.\n\
                     Each group becomes its own batch when --batch-files is omitted.\n\
                     That is modality-presence auto-batching. Pass a single --batch-files,\n\
                     one label per unified cell, to set batches explicitly.\n\
                     This flag replaces the positional data files.\n\
                     \n\
                     Note: comma-separate files within one group, with no spaces.\n\
                     Use a separate --multiome flag for each additional group."
    )]
    pub(crate) multiome: Vec<Box<str>>,

    #[arg(
        long = "nce-objective",
        default_value_t = NceObjectiveArg::Softmax,
        value_enum,
        help = "NCE objective: softmax or logistic",
        long_help = "NCE objective. softmax is InfoNCE, where negatives compete.\n\
                     It is sharper on dense pseudobulk data, and is the default.\n\
                     logistic is per-pair SGNS."
    )]
    pub(crate) nce_objective: NceObjectiveArg,

    #[arg(
        long,
        default_value_t = 1,
        value_name = "N",
        help = "Seed for training and the posterior (default 1).",
        long_help = "Seed for the fit's sampling RNG and parameter initialization,\n\
                     and for the posterior samplers when --mcmc/--posterior is given.\n\
                     \n\
                     Changing it gives an INDEPENDENT fit.\n\
                     Initialization and minibatch order both differ.\n\
                     That is what an A/B across seeds needs.\n\
                     \n\
                     It does NOT make a run bit-reproducible.\n\
                     The variational gate's noise comes from the device RNG.\n\
                     That sits outside this stream.\n\
                     Two runs at the same seed still differ slightly."
    )]
    pub(crate) seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub(crate) device_no: usize,

    /// The shared `--posterior` / `--mcmc` flag group (see
    /// `ge::posterior::PosteriorArgs`); `senna gem` flattens the same one.
    #[command(flatten)]
    pub(crate) posterior: ge::posterior::PosteriorArgs,

    /// The `--gene-modules` flag group (see `ge::GeneModuleArgs`).
    #[command(flatten)]
    pub(crate) modules: ge::GeneModuleArgs,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix. It produces {out}.cell_embedding.parquet, which is Z,\n\
                     {out}.dictionary.parquet, {out}.feature_embedding.parquet,\n\
                     {out}.feature_bias.parquet, {out}.cell_bias.parquet, and {out}.senna.json.\n\
                     Unless --skip-etm, it adds two more:\n\
                     {out}.latent.parquet and {out}.topic_embedding.parquet."
    )]
    pub(crate) out: Box<str>,
}

impl crate::update::Updatable for BgeArgs {
    fn rebase(&mut self, r: crate::update::Rebase) {
        self.data_files = r.data_files;
        self.batch_files = r.batch_files;
        self.out = r.out;
        self.pb_reference = r.reference;
        // Like `svd`, bge has no weights to warm-start: `update` re-fits on
        // the union with the recorded configuration, and the carried
        // reference (when used) is what keeps that O(new). `init_from` only
        // chains the emitted reference's generation counter.
        self.init_from = Some(r.init_from);
        if let Some(e) = r.epochs {
            self.epochs = e;
        }
    }
}
