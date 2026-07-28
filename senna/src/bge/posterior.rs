//! `--posterior` / `--mcmc` for bge — the exact MCMC posterior over the fitted
//! gate, run after the SGD MAP.
//!
//! Training fits the per-gene softmax gate variationally and reports the point
//! estimate `JointEmbedModel::feature_selection()`. This step replaces it with the
//! exact posterior of the same model: per-dim PIP, a credible set over the dims a
//! gene loads, and the effect posterior.
//!
//! The whole pipeline — flags, calibration, samplers, tables — lives in
//! `ge::posterior::run`, shared with `faba gem` so the two cannot drift on column
//! names or JSON schema. bge is the plainest caller: `feat_factor: None`, so a
//! feature row **is** an anchor and there is exactly one track, with no grouping
//! and nothing held fixed. (gem adds gene-level anchors and the velocity gate.)

use ge::posterior::{FrozenMap, PosteriorPlan, TrackSpec};
use graph_embedding_util as ge;

/// Sample the gate posterior and write its tables. Called only for a complete
/// (non-interrupted) fit, after every normal output is on disk.
pub fn run_posterior(
    args: &super::BgeArgs,
    plan: PosteriorPlan,
    unified: &ge::data::UnifiedData,
    model: &ge::JointEmbedModel,
) -> anyhow::Result<()> {
    let frozen = FrozenMap::from_model(model)?;
    ge::posterior::run_posterior(
        &args.out,
        plan,
        unified,
        &frozen,
        &[TrackSpec {
            tag: "feature",
            label: "identity",
            // One anchor per feature row: bge embeds each row freely.
            row_to_anchor: None,
            names: &unified.feature_names,
            axis: "feature",
            offset: None,
        }],
    )
}
