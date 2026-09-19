//! `senna gem`: joint embedding of gene counts and any co-measured
//! modality, over the shared `graph_embedding_util` engine and `senna bge`'s
//! driver.
//!
//! A gem feature axis is one gene axis carrying several TRACKS: the base
//! gene count (`{gene}/count/spliced`, optionally `{gene}/count/unspliced`),
//! plus, for every `--modality` file, that modality's two channel rows
//! (`{gene}/m6a/{methylated,unmethylated}`, `{gene}/atoi/{edited,unedited}`,
//! `{gene}/apa/{proximal,distal}`). The base track shares a gene's loading
//! outright; every other track adds a low-rank, ridge-shrunk offset to it
//! (`--offset-rank`, `--offset-l2`), so a track moves its genes inside one
//! shared subspace. A table given with `--{freeze,init,lora}-feature-embedding`
//! is read onto the grammar by [`preset::resolve_gem_preset`]: bare names are
//! spliced rows. [`tracks::assign_tracks`] reads this grammar off the row
//! names alone (never a file name or load order) and builds the
//! [`tracks::TrackPlan`] that both `senna gem`'s own per-gene HVG pooling and
//! the engine's per-track training consume.
//!
//! Trained cells carry one encoder per count track (`{out}.cell_encoder.safetensors`
//! for the base track, `{out}.cell_encoder.{modality}.{channel}.safetensors`
//! for any other count track). [`contrast::write_contrast`] reads the
//! finished fit's raw loading and writes one row per gene-and-modality where
//! both of that modality's channels are present
//! (`{out}.feature_contrast.parquet`, `{out}.feature_contrast_bias.parquet`).

pub(crate) mod args;
/// One row per gene-and-modality, contrasting a modality's two channel
/// tracks on the raw loading: `{out}.feature_contrast.parquet` /
/// `{out}.feature_contrast_bias.parquet`.
pub(crate) mod contrast;
/// Pooled per-gene HVG projection weights over a [`tracks::TrackPlan`]
/// (every track of a gene shares one selection decision, weight lands only
/// on the base row). Replaces the deleted `rows` module for the one thing
/// gem's HVG selection still needs.
pub(crate) mod hvg;
/// Multi-file input resolution and loading: matches `--modality` files to
/// gene files by sample id, loads them into one `UnifiedData`, and assigns
/// the [`tracks::TrackPlan`].
pub(crate) mod load;
/// Loading of gem's co-embedded feature table for annotate / lineage lives in
/// [`crate::marker_embedding`] (not β).
/// `--{freeze,init,lora}-feature-embedding` read onto the row grammar: base
/// rows, per-track offsets, carried rows.
pub(crate) mod preset;
/// The `senna gem` run: joint gene-count embedding over the shared
/// `graph_embedding_util` engine (bge, over every feature row). Binary entry: [`run::run_gem_embedding`].
pub mod run;
pub mod sample_id;
/// Row-grammar track assignment: turns a gem feature axis (gene counts plus
/// any `--modality` files) into a [`tracks::TrackPlan`] /
/// `graph_embedding_util::fit::TrackSpec`.
pub(crate) mod tracks;

/// Synthetic fixture builders shared by `gem::run::tests` and
/// `predict::tests`'s gem contract test.
#[cfg(test)]
pub(crate) mod test_fixtures;
