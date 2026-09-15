//! Indexed (top-K) packing for the senna topic models.
//!
//! Packs each row's top-K features into `[N, K]` host buffers, never an
//! `[N, D]` dense matrix. This is a **bounded** view of a row; what a decoder
//! scores is the consumer's choice over the whole feature axis, so a minibatch
//! carries `row_ids` and stops there.
//!
//! Module layout (split 2026-05-15 from a single 1570-line file):
//! - [`types`] — public data types (`IndexedSample`, `IndexedMinibatchData`).
//! - [`top_k`] — weighted top-K selection (`top_k_indices_weighted`,
//!   `csc_columns_to_indexed_samples`, `build_indexed_samples`).
//! - [`pack`] — parallel `[N, K]` pack/gather helpers.
//!
//! The single-track in-memory loader that used to sit here, and the
//! feature-feature graph it fed the encoder's GCN block from, are gone: the
//! masked family reads dense rows ([`crate::data::masked_dense`]) and nothing
//! built either any more. The two-track (spliced / unspliced) gem row/gene
//! map that used to live here (`splice_tracks`) is gone too, with the
//! composite engine that consumed it. What is left are the packing pieces
//! that the old-model windowed evaluation still uses.

use indicatif::ProgressBar;

pub mod full_support;
pub mod pack;
pub mod top_k;
pub mod types;

pub use pack::gather_per_feature_at_indices;
pub use top_k::{csc_columns_to_indexed_samples, top_k_indices_weighted};
pub use types::{IndexedMinibatchData, IndexedSample};

// Crate-public re-exports.
pub use pack::pack_indices_values;
pub use top_k::build_indexed_samples;

///////////////////////////////////////////////////////////////////////////
// Progress bar helper (used here and externally by cell-grouped loader) //
///////////////////////////////////////////////////////////////////////////

/// A bounded progress bar in the **canonical workspace style** (see
/// [`matrix_util::progress::new_progress_bar`]): `[elapsed] bar pos/len (eta) {msg}`,
/// cyan/blue, and — crucially — registered with the shared `MULTI_PROGRESS` so `-v`
/// log output interleaves cleanly above it. `label` is the initial `{msg}` (e.g.
/// "Epochs", "Null rows"); the epoch trainers overwrite it each step with a live metric
/// (`prog_bar.set_message`), matching `senna bge`. Delegating here keeps every
/// candle-util bar on ONE style and ONE bridged `MultiProgress` — a locally-styled
/// `ProgressBar::new` spawns a SECOND, unbridged bar that corrupts log output under
/// `-v` (see the `matrix-util::progress` module doc).
#[must_use]
pub fn labeled_bar(label: &str, len: u64) -> ProgressBar {
    matrix_util::progress::new_progress_bar(len).with_message(label.to_string())
}
