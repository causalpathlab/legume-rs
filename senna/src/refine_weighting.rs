//! Shared CLI args for pseudobulk (PB) multilevel refinement and inference-time
//! amortization refinement.
//!
//! Two distinct things share the word "refinement", so the CLI keeps them on
//! separate prefixes:
//!
//! - `--pb-refine-*` flags drive [`data_beans_alg::refine_multilevel::RefineParams`]
//!   used during hierarchical pseudobulk collapsing.
//! - `--amort-refine-*` flags drive
//!   [`candle_util::candle_topic_refinement::TopicRefinementConfig`] used at
//!   inference to fine-tune per-cell topic logits against the frozen decoder.

use clap::{Args, ValueEnum};
use data_beans_alg::dc_poisson::FeatureWeighting;
use data_beans_alg::refine_multilevel::RefineParams;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum WeightingArg {
    /// Fisher-info weight from fitted NB mean-variance trend. Default.
    #[default]
    NbFisherInfo,
    /// No per-feature weighting (raw DC-Poisson with entity-level degree correction).
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
#[derive(Args, Clone, Debug)]
pub(crate) struct PbRefineArgs {
    #[arg(
        long = "pb-refine-gibbs",
        default_value_t = 20,
        help = "Gibbs sweeps per PB refinement level"
    )]
    pub(crate) gibbs: usize,

    #[arg(
        long = "pb-refine-greedy",
        default_value_t = 10,
        help = "Greedy sweeps per PB refinement level"
    )]
    pub(crate) greedy: usize,

    #[arg(
        long = "pb-refine-weighting",
        value_enum,
        default_value_t = WeightingArg::NbFisherInfo,
        help = WEIGHTING_HELP,
    )]
    pub(crate) weighting: WeightingArg,

    #[arg(
        long = "pb-refine-seed",
        default_value_t = 42,
        help = "Seed for PB refinement Gibbs sampler"
    )]
    pub(crate) seed: u64,
}

impl Default for PbRefineArgs {
    fn default() -> Self {
        Self {
            gibbs: 20,
            greedy: 10,
            weighting: WeightingArg::NbFisherInfo,
            seed: 42,
        }
    }
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

/// CLI args for inference-time amortization refinement on topic models.
///
/// `--amort-refine-steps = 0` disables refinement; in that case
/// [`AmortRefineArgs::to_config`] returns `None`.
#[derive(Args, Clone, Debug)]
pub(crate) struct AmortRefineArgs {
    #[arg(
        long = "amort-refine-steps",
        default_value_t = 0,
        help = "Per-cell amortization refinement steps at inference (0 = off)",
        long_help = "Gradient steps that optimize topic logits against the frozen\n\
                     decoder likelihood, anchored to the encoder output by L2."
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

impl Default for AmortRefineArgs {
    fn default() -> Self {
        Self {
            steps: 0,
            lr: 0.01,
            reg: 1.0,
        }
    }
}

impl AmortRefineArgs {
    /// Build the candle-side config from these CLI args. Returns `None` when
    /// `--amort-refine-steps = 0` (refinement disabled).
    pub(crate) fn to_config(
        &self,
    ) -> Option<candle_util::candle_topic_refinement::TopicRefinementConfig> {
        if self.steps == 0 {
            None
        } else {
            Some(candle_util::candle_topic_refinement::TopicRefinementConfig {
                num_steps: self.steps,
                learning_rate: self.lr,
                regularization: self.reg,
            })
        }
    }
}
