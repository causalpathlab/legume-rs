//! CLI-facing enum for the DC-Poisson feature-weighting scheme.
//!
//! Maps one-to-one to [`data_beans_alg::dc_poisson::FeatureWeighting`]. Kept
//! in its own module so every subcommand that exposes the `--weighting`
//! flag imports the same enum and `help` string.

use clap::ValueEnum;
use data_beans_alg::dc_poisson::FeatureWeighting;

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

/// The `help` string reused by every `--weighting` arg across senna
/// subcommands; keeps the CLI self-consistent.
pub(crate) const WEIGHTING_HELP: &str =
    "DC-Poisson feature weighting: nb-fisher-info (default, NB mean-variance), none (raw)";
