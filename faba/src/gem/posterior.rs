//! `--posterior` / `--mcmc` for gem — the exact MCMC posterior over gem's **two**
//! fitted gates, run after the SGD MAP.
//!
//! The pipeline itself (flags, calibration, samplers, tables) lives in
//! `ge::posterior::run`, shared with `senna bge`. This file supplies only what is
//! gem-specific, which is the track definitions:
//!
//! **The anchor is a gene, not a feature row.** gem's rows are
//! `{gene}/count/{spliced|unspliced}` and both embed as `β_g` (β-sharing), so a
//! per-row posterior would answer a question the model does not ask. Rows are
//! pooled to their gene by `row_to_gene`, so the output joins directly against
//! `{out}.beta_feature_embedding.parquet`.
//!
//! **Each gate reads a different track**, because that is what defines it:
//!
//! * identity `β_g` — the **spliced** rows. They embed as exactly `β_g`, so this
//!   is the clean estimand with nothing held fixed.
//! * velocity `δ_g` — the **unspliced** rows, which score
//!   `⟨β_g + δ_g, e_c⟩ + b_f + b_c`. Sampling `δ_g` means carrying `β_g` at its MAP
//!   as a frozen per-anchor offset, so the credible set is over *which dims carry
//!   motion*, not over the identity the gene already has.
//!
//! The velocity posterior is the gem-only deliverable — bge has no `δ_g` gate.

use ge::posterior::{FrozenMap, PosteriorPlan, TrackSpec};
use graph_embedding_util as ge;
use log::info;

/// The per-gene inputs the two tracks are built from.
pub struct GemTracks<'a> {
    /// Feature row → gene id, from `build_splice_factor`.
    pub row_to_gene: &'a [u32],
    /// Per-row unspliced flag, same source.
    pub unspliced_rows: &'a [bool],
    /// Gene names, id-ordered — the output tables' row keys.
    pub gene_names: &'a [Box<str>],
    /// Trained `[n_genes × h]` `β_g`, held fixed while `δ_g` is sampled.
    pub beta: &'a [f32],
}

/// Sample gem's gate posteriors and write their tables. Called only for a
/// complete (non-interrupted) fit, after every normal output is on disk.
pub fn run_posterior(
    out_prefix: &str,
    plan: PosteriorPlan,
    unified: &ge::data::UnifiedData,
    model: &ge::JointEmbedModel,
    tracks: &GemTracks,
) -> anyhow::Result<()> {
    let frozen = FrozenMap::from_model(model)?;

    // Rows of the OTHER track map to u32::MAX, which drops them: that row
    // selection is exactly what distinguishes the identity gate from the velocity
    // one, since both read the same gene axis.
    let anchors = |unspliced: bool| -> Vec<u32> {
        tracks
            .row_to_gene
            .iter()
            .zip(tracks.unspliced_rows)
            .map(|(&g, &u)| if u == unspliced { g } else { u32::MAX })
            .collect()
    };
    let spliced = anchors(false);
    let unspliced = anchors(true);

    // gem's dictionaries are keyed by gene under the row axis `feature`
    // (`beta_feature_embedding.parquet`), so the posterior tables match — a
    // name-based join against them has to see the same index column.
    let mut specs = vec![TrackSpec {
        tag: "beta",
        label: "identity β_g",
        row_to_anchor: Some(&spliced),
        names: tracks.gene_names,
        axis: "feature",
        offset: None,
    }];
    if tracks.unspliced_rows.iter().any(|&u| u) {
        specs.push(TrackSpec {
            tag: "delta",
            label: "velocity δ_g",
            row_to_anchor: Some(&unspliced),
            names: tracks.gene_names,
            axis: "feature",
            offset: Some(tracks.beta),
        });
    } else {
        info!("velocity posterior skipped — no unspliced rows in the input");
    }

    ge::posterior::run_posterior(out_prefix, plan, unified, &frozen, &specs)
}
