//! Shared CLI args for the upstream projection + multilevel pseudobulk
//! collapse pipeline, plus inference-time amortization refinement.
//!
//! - [`CollapseArgs`] bundles every knob of the shared random-projection +
//!   multilevel-collapse pipeline (`--proj-dim`, `--sort-dim`, `--knn-cells`,
//!   `--num-levels`, `--iter-opt`, `--ignore-batch`) and flattens
//!   [`PbRefineArgs`] inside it. Flatten it into any subcommand that runs
//!   that pipeline so the flag surface stays identical across `senna topic`,
//!   `masked-topic`, `joint-topic`, `svd`, `svd joint`, and `gbe`.
//!
//! Two distinct things share the word "refinement", so the CLI keeps them on
//! separate prefixes:
//!
//! - `--pb-refine-*` flags drive [`data_beans_alg::refine_multilevel::RefineParams`]
//!   used during hierarchical pseudobulk collapsing.
//! - `--amort-refine-*` flags drive
//!   [`candle_util::topic_refinement::TopicRefinementConfig`] used at
//!   inference to fine-tune per-cell topic logits against the frozen decoder.

use clap::{Args, ValueEnum};
use data_beans_alg::dc_poisson::FeatureWeighting;
use data_beans_alg::refine_multilevel::RefineParams;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize,
)]
#[value(rename_all = "kebab-case")]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WeightingArg {
    /// Fisher-info weight from fitted NB mean-variance trend. Default.
    #[default]
    NbFisherInfo,
    /// No per-feature weighting (raw DC-Poisson).
    None,
}

impl From<WeightingArg> for FeatureWeighting {
    fn from(value: WeightingArg) -> Self {
        match value {
            WeightingArg::NbFisherInfo => FeatureWeighting::FisherInfoNb,
            WeightingArg::None => FeatureWeighting::None,
        }
    }
}

pub(crate) const WEIGHTING_HELP: &str =
    "DC-Poisson feature weighting: nb-fisher-info (default, NB mean-variance), none (raw)";

/// CLI args for pseudobulk multilevel refinement.
///
/// Flatten into any subcommand args struct with `#[command(flatten)]` to expose
/// `--pb-refine-{gibbs,greedy,weighting,seed}` and call [`PbRefineArgs::to_params`]
/// to build the `RefineParams` passed into `MultilevelParams::refine`.
#[derive(Args, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "crate::embed_common::clap_defaults")]
pub(crate) struct PbRefineArgs {
    #[arg(
        id = "pb_refine_gibbs",
        long = "pb-refine-gibbs",
        default_value_t = 20,
        help = "Gibbs sweeps per PB refinement level"
    )]
    pub(crate) gibbs: usize,

    #[arg(
        id = "pb_refine_greedy",
        long = "pb-refine-greedy",
        default_value_t = 10,
        help = "Greedy sweeps per PB refinement level"
    )]
    pub(crate) greedy: usize,

    #[arg(
        id = "pb_refine_weighting",
        long = "pb-refine-weighting",
        value_enum,
        default_value_t = WeightingArg::NbFisherInfo,
        help = WEIGHTING_HELP,
        hide = true
    )]
    pub(crate) weighting: WeightingArg,

    #[arg(
        id = "pb_refine_seed",
        long = "pb-refine-seed",
        default_value_t = 42,
        help = "Seed for PB refinement Gibbs sampler",
        hide = true
    )]
    pub(crate) seed: u64,
}

impl PbRefineArgs {
    /// Build the algorithm-side [`RefineParams`] from these CLI args.
    pub(crate) fn to_params(&self) -> RefineParams {
        RefineParams {
            num_gibbs: self.gibbs,
            num_greedy: self.greedy,
            feature_weighting: self.weighting.into(),
            seed: self.seed,
            ..RefineParams::default()
        }
    }
}

//////////////////////////////////////////////////////
// Shared projection + multilevel-collapse CLI args //
//////////////////////////////////////////////////////

/// CLI args for the shared random-projection + multilevel pseudobulk
/// collapse pipeline.
///
/// Flatten into any subcommand args struct with `#[command(flatten)]` to
/// expose `--proj-dim`, `--sort-dim`, `--knn-cells`, `--num-levels`,
/// `--iter-opt`, `--ignore-batch`, and (via the nested [`PbRefineArgs`])
/// `--pb-refine-*`. Keeps the upstream flag surface identical across every
/// senna subcommand that collapses cells into pseudobulks.
#[derive(Args, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "crate::embed_common::clap_defaults")]
pub(crate) struct CollapseArgs {
    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension",
        long_help = "Target rank of the initial random sketch,\n\
                     used to seed batch correction and multi-level pseudobulk collapsing."
    )]
    pub(crate) proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Partition depth: ≤ 2^d + 1 pseudobulk groups",
        long_help = "Binary-tree partitioning over the top d projection components.\n\
                     Produces at most 2^d + 1 pseudobulk leaves."
    )]
    pub(crate) sort_dim: usize,

    #[arg(
        long,
        help = "Skip per-batch correction; treat all cells as a single batch",
        long_help = "Collapses batch membership to a single label so the random projection,\n\
                     multilevel collapsing,\n\
                     and δ estimation all run as if there were no batch structure.\n\
                     Useful for homogeneous datasets or as a reference baseline."
    )]
    pub(crate) ignore_batch: bool,

    #[arg(
        long,
        default_value_t = 10,
        help = "In-batch k-NN for pb-sample merging",
        long_help = "Number of within-batch nearest neighbours.\n\
                     They are used when aggregating cells into pseudobulk pb-samples."
    )]
    pub(crate) knn_cells: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Multi-level coarsening levels",
        long_help = "Hierarchical pseudobulk refinement passes.\n\
                     Level sort dims are linearly spaced from 4 to --sort-dim.\n\
                     Set to 1 to disable."
    )]
    pub(crate) num_levels: usize,

    #[arg(
        long,
        default_value_t = 30,
        help = "Batch-correction optimizer iterations",
        long_help = "Coordinate-descent steps when fitting the per-batch delta."
    )]
    pub(crate) iter_opt: usize,

    #[arg(
        long,
        help = "Carry this run's pseudobulks forward for a later `senna update`",
        long_help = "Writes {out}.pb_reference.zarr — one column per pseudobulk, holding\n\
                     its batch-adjusted per-cell rate — plus a sidecar with each\n\
                     column's cell count.\n\
                     \n\
                     `senna update` can then absorb a new sample by re-collapsing\n\
                     only the NEW cells against these, instead of re-reading every\n\
                     cell the model has already seen. Absorbing S samples one at a\n\
                     time goes from quadratic to linear in cell reads.\n\
                     \n\
                     Off by default: it is a second copy of the pseudobulks, useful\n\
                     only if you intend to keep growing this model.\n\
                     \n\
                     The reference is APPEND-ONLY across rounds: carried columns\n\
                     pass through byte-stable and each update adds at most\n\
                     2^sort-dim + 1 new columns for its own cells, so growth is\n\
                     linear in rounds and independent of sample size, with total\n\
                     cell mass conserved exactly. Old rounds are never re-averaged\n\
                     — re-summarizing every round would compound resolution loss.\n\
                     \n\
                     Available on topic, masked-topic, masked-sbp, masked-vae, vae,\n\
                     svd and bge — the families `senna update` can continue."
    )]
    pub(crate) emit_pb_reference: bool,

    #[command(flatten)]
    pub(crate) pb_refine: PbRefineArgs,
}

/// NB-Fisher gene weights for a fit, sourced from whichever population this
/// run actually has.
///
/// **Cells are the default and stay the default.** An A/B on HCA_BM (39k
/// cells, K=20, 300 epochs) could not separate a cell-fitted trend from a
/// pseudobulk-fitted one: the arm differences in ARI (0.0018), purity (0.0097)
/// and held-out likelihood (0.012) all landed *inside* the spread between two
/// runs of the same configuration (0.0057 / 0.0155 / 0.118). There is no
/// measured reason to prefer one, so nothing changes for an ordinary fit.
///
/// **`senna update --use-pb-reference` is not an ordinary fit.** There the
/// backend holds the parent's carried pseudobulks — per-cell **rates** — beside
/// the new cohort's real cell **counts**. A trend fitted across that mixture is
/// fitted on two incompatible units: averaging already removed the Poisson
/// component the trend measures, and no weighting puts it back. So the choice
/// there is not a preference between two estimators; one of them has no
/// coherent population. Hence no flag — and the condition is not even the
/// reference itself. What invalidates the cell-level trend is "this cohort
/// holds columns that are summaries, not cells", which the loader records as
/// [`SparseIoVec::has_column_multiplicity`] when it registers the carried
/// columns' weights. Keying on that keeps the choice correct for any future
/// producer of weighted columns (bulk samples), not just `--use-pb-reference`.
///
/// Measured against the exact re-collapse (900-cell parent absorbing 400
/// cells), on Spearman ρ of the induced gene ranking:
///
/// | trend source | ρ vs exact | mean ratio |
/// | --- | --- | --- |
/// | cells, mixed units (what this replaces) | 0.815 | 1.71 |
/// | pseudobulks, batch-adjusted | 0.469 | 0.53 |
/// | pseudobulks, observed | **0.975** | 0.45 |
///
/// The residual uniform ~2.2× shrink is not explained. It is left alone
/// because the A/B above says model quality is insensitive to this trend at a
/// far larger perturbation than a constant factor.
pub(crate) fn fit_fisher_weights(
    collapsed: &data_beans_alg::collapse_data::CollapsedOut,
    cell_to_pb: Option<&[usize]>,
    coarsening: Option<&data_beans_alg::feature_coarsening::FeatureCoarsening>,
    data_vec: &data_beans::sparse_io_vector::SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    if data_vec.has_column_multiplicity() {
        let cell_to_pb = cell_to_pb.ok_or_else(|| {
            anyhow::anyhow!(
                "weighted columns need this run's cell → pb membership to know how many cells \
                 each pseudobulk stands for; without it the NB-Fisher trend cannot be put back on \
                 the count scale it is defined for."
            )
        })?;
        return crate::pb_reference::fisher_weights_for_weighted_cohort(
            collapsed,
            cell_to_pb,
            data_vec.column_multiplicities(),
            coarsening,
        );
    }
    match coarsening {
        Some(fc) => data_beans_alg::gene_weighting::compute_nb_fisher_weights_coarsened(
            data_vec, fc, block_size,
        ),
        None => data_beans_alg::gene_weighting::compute_nb_fisher_weights(data_vec, block_size),
    }
}

impl CollapseArgs {
    /// Refuse `--emit-pb-reference` on a family that would ignore it.
    ///
    /// The flag rides on this shared struct, so it appears on every command
    /// that flattens `CollapseArgs` — including `joint-topic`, `joint-svd` and
    /// `bge`, which `senna update` cannot continue and which therefore write
    /// nothing. Accepting it there and silently doing nothing is the worst of
    /// the three options: the user believes the reference exists and only finds
    /// out a round later, when the parent turns out to carry nothing.
    pub(crate) fn reject_pb_reference(
        &self,
        kind: crate::run_manifest::RunKind,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.emit_pb_reference,
            "--emit-pb-reference has no effect on `{kind}`: `senna update` cannot continue a \
             '{kind}' run, so the carried pseudobulks would have no consumer. Supported: topic, \
             masked-topic, masked-sbp, masked-vae, vae, svd, bge."
        );
        Ok(())
    }
}

/// CLI args for inference-time amortization refinement on topic models.
///
/// `--amort-refine-steps = 0` disables refinement; in that case
/// [`AmortRefineArgs::to_config`] returns `None`.
#[derive(Args, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default = "crate::embed_common::clap_defaults")]
pub(crate) struct AmortRefineArgs {
    #[arg(
        long = "amort-refine-steps",
        default_value_t = 0,
        help = "Per-cell amortization refinement steps at inference (0 = off)",
        long_help = "Gradient steps that optimize topic logits against the frozen decoder likelihood,\n\
                     anchored to the encoder output by L2."
    )]
    pub(crate) steps: usize,

    #[arg(
        long = "amort-refine-lr",
        default_value_t = 0.01,
        help = "Amortization refinement learning rate"
    )]
    pub(crate) lr: f64,

    #[arg(
        long = "amort-refine-reg",
        default_value_t = 1.0,
        help = "Amortization refinement L2 regularization"
    )]
    pub(crate) reg: f64,
}

impl AmortRefineArgs {
    /// Build the candle-side config from these CLI args. Returns `None` when
    /// `--amort-refine-steps = 0` (refinement disabled).
    pub(crate) fn to_config(&self) -> Option<candle_util::topic_refinement::TopicRefinementConfig> {
        if self.steps == 0 {
            None
        } else {
            Some(candle_util::topic_refinement::TopicRefinementConfig {
                num_steps: self.steps,
                learning_rate: self.lr,
                regularization: self.reg,
            })
        }
    }
}
