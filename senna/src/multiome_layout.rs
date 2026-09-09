//! Reproduce a multiome run's data load from its run manifest.
//!
//! A multiome fit changes the load in three ways: cells glue by barcode
//! (`ColumnAlignment::Union`), features are namespaced `{name}/{modality}`,
//! and — with several sample groups — barcodes are namespaced
//! `{barcode}@{group}`. None of that was recorded, so every `--from` consumer
//! re-read the files the plain way and matched nothing: the cells came back
//! doubled and unnamespaced, and the feature names no longer matched the
//! dictionary they were supposed to project onto.
//!
//! Two directions, because `--from` consumers do two different things:
//!
//! - **The run's own files** replay [`RunMultiome`] straight out of the
//!   manifest. It is positional against `data.input`, which the trainer wrote
//!   in group order.
//! - **A query file set** is new data, so its layout is *detected* the same
//!   way training detected its own. Detection names modalities after the
//!   query's filenames, which need not match the training run's, so the tags
//!   are then renamed by which trained row block the features actually land
//!   in ([`reconcile_modalities`]).

use auxiliary_data::data_loading::ReadSharedRowsArgs;
use auxiliary_data::feature_names::FeatureNameKind;
use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util as ge;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

/// The resolved multiome layout of a run, positional against `data.input`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunMultiome {
    /// Modality label per input file; features become `{name}/{modality}`.
    pub modality: Vec<String>,
    /// Sample-group label per input file.
    pub group: Vec<String>,
    /// Whether barcodes were tagged `{barcode}@{group}` at load. False for a
    /// single group, where there is no cross-group collision to prevent.
    pub barcode_tagged: bool,
}

impl RunMultiome {
    /// The record for a resolved plan, as the trainer is about to load it.
    #[must_use]
    pub fn from_plan(plan: &ge::MultiomePlan) -> Self {
        Self {
            modality: plan.modality.iter().map(ToString::to_string).collect(),
            group: plan.group.iter().map(ToString::to_string).collect(),
            barcode_tagged: plan.barcode_tagged,
        }
    }
}

/// The loader settings that reproduce a training-time load.
///
/// `Default` is the plain single-modality load, so spreading it into a
/// `ReadSharedRowsArgs` is a no-op for every run that is not multiome. The
/// suffixes are the only degree of freedom: `Union` alignment and `Mixed`
/// canonicalization always travel with them, so they are stated once in
/// [`Self::apply`] rather than stored as fields that a caller could set
/// inconsistently.
#[derive(Debug, Clone, Default)]
pub struct ReloadLayout(Option<Suffixes>);

/// The per-file namespacing a multiome load applied.
#[derive(Debug, Clone)]
struct Suffixes {
    /// Modality per file; rows become `{name}/{modality}`.
    feature: Vec<Box<str>>,
    /// Sample per file, when barcodes were tagged `{barcode}@{group}`.
    barcode: Option<Vec<Option<Box<str>>>>,
}

impl ReloadLayout {
    /// Is this an actual multiome layout, or the plain load?
    #[must_use]
    pub fn is_multiome(&self) -> bool {
        self.0.is_some()
    }

    /// Stamp the layout onto a load, leaving every other field the caller's.
    /// A plain load passes through untouched, so call sites need no branch.
    ///
    /// The caller's `feature_kind` is honoured when it set one — with the one
    /// exception that a multiome axis cannot: its rows are gene names AND
    /// `chrX:s-e` loci together, and only [`FeatureNameKind::Mixed`] dispatches
    /// per name. Silently overriding a caller's choice there would accept a
    /// `--feature-name-kind` and ignore it on exactly the path where the axis
    /// stops matching the dictionary, so that combination is refused instead.
    pub fn apply(&self, mut args: ReadSharedRowsArgs) -> anyhow::Result<ReadSharedRowsArgs> {
        let Some(s) = self.0.as_ref() else {
            return Ok(args);
        };
        if let Some(kind) = args.feature_kind.as_ref() {
            anyhow::ensure!(
                matches!(kind, FeatureNameKind::Mixed),
                "an explicit feature-name kind ({kind:?}) cannot be used with the multiome \
                 layout on this axis ({}). Multiome rows carry gene names and chrX:s-e loci \
                 together, so they need the per-name `Mixed` rule; forcing {kind:?} would \
                 mis-canonicalize every locus row and the axis would stop matching the \
                 dictionary. Drop the flag, or load the modalities separately.",
                s.feature
                    .iter()
                    .map(std::convert::AsRef::as_ref)
                    .collect::<std::collections::BTreeSet<&str>>()
                    .into_iter()
                    .collect::<Vec<&str>>()
                    .join(" + ")
            );
        }
        args.column_alignment = ColumnAlignment::Union;
        args.feature_kind = Some(FeatureNameKind::Mixed);
        args.per_file_feature_suffix = Some(s.feature.clone());
        args.per_file_barcode_suffix = s.barcode.clone();
        Ok(args)
    }
}

/// Replay a recorded layout over the run's own `n_files` inputs.
///
/// The record is positional, so a caller loading a different number of files
/// is loading something else; that is an error rather than a guess.
pub fn recorded_layout(
    record: Option<&RunMultiome>,
    n_files: usize,
) -> anyhow::Result<ReloadLayout> {
    let Some(r) = record else {
        return Ok(ReloadLayout::default());
    };
    anyhow::ensure!(
        r.modality.len() == n_files && r.group.len() == n_files,
        "the run's multiome layout covers {} file(s) but {} were passed — it is \
         positional against the recorded data.input, so a different file set \
         cannot replay it",
        r.modality.len(),
        n_files
    );
    Ok(ReloadLayout(Some(Suffixes {
        feature: r.modality.iter().map(|s| s.as_str().into()).collect(),
        barcode: r
            .barcode_tagged
            .then(|| r.group.iter().map(|s| Some(s.as_str().into())).collect()),
    })))
}

/// Work out how to load a query file set so its feature names land on the axis
/// `trained_features` defines, rewriting `args` in place.
///
/// Takes and returns the whole `ReadSharedRowsArgs` because the file list and
/// the namespacing have to move together: the layout is positional against the
/// re-ordered files, so a caller that set one without the other would load a
/// silently mis-namespaced axis. Falls back to the plain load whenever either
/// side is single-modality, which is every non-multiome model.
pub fn query_load(
    mut args: ReadSharedRowsArgs,
    trained_features: &[Box<str>],
) -> anyhow::Result<ReadSharedRowsArgs> {
    let Some(plan) = query_layout(&args.data_files, trained_features)? else {
        return Ok(args);
    };
    let modalities = plan.modalities();
    log::info!(
        "query multiome: {} group(s) x {} modality(ies) [{}], matched to the \
         model's feature axis",
        plan.n_groups(),
        modalities.len(),
        modalities
            .iter()
            .map(std::convert::AsRef::as_ref)
            .collect::<Vec<&str>>()
            .join(", ")
    );
    let layout = layout_of_plan(&plan);
    args.data_files = plan.files;
    layout.apply(args)
}

/// The `{modality}` tags a trained feature axis carries. Empty for a
/// single-modality model, whose rows are bare names.
#[must_use]
pub fn trained_modalities(trained_features: &[Box<str>]) -> Vec<Box<str>> {
    // A multiome load namespaces every row as `{name}/{modality}` — exactly
    // one separator. Anything else is a different grammar: faba writes, and
    // `senna gem` reads, `{gene}/{modality}/{channel}`, whose trailing field is
    // a splice channel. Reading that as a modality would route a gem model's
    // query down the multiome path and re-scope its whole feature axis, so
    // require the two-field shape on every row.
    if trained_features.is_empty()
        || !trained_features
            .iter()
            .all(|f| f.bytes().filter(|&b| b == b'/').count() == 1)
    {
        return Vec::new();
    }
    let mut seen = FxHashSet::default();
    trained_features
        .iter()
        .filter_map(|f| f.rsplit_once('/').map(|(_, m)| m))
        .filter(|m| seen.insert(*m))
        .map(Into::into)
        .collect()
}

/// Rename a detected plan's modality tags to the trained ones, by which
/// trained row block each detected modality's features actually land in.
///
/// `query_rows[i]` is the raw row-name list of `plan.files[i]`. A detected
/// modality with no trained match keeps its own tag; two claiming the same
/// trained block is an error, since that would stack two assays on one row
/// block.
pub fn reconcile_modalities(
    mut plan: ge::MultiomePlan,
    query_rows: &[Vec<Box<str>>],
    trained_features: &[Box<str>],
) -> anyhow::Result<ge::MultiomePlan> {
    anyhow::ensure!(
        query_rows.len() == plan.files.len(),
        "row lists {} != planned files {}",
        query_rows.len(),
        plan.files.len()
    );
    // Trained rows, grouped by their modality tag.
    let mut trained: FxHashMap<&str, FxHashSet<&str>> = FxHashMap::default();
    for f in trained_features {
        if let Some((name, modality)) = f.rsplit_once('/') {
            trained.entry(modality).or_default().insert(name);
        }
    }
    if trained.is_empty() {
        return Ok(plan);
    }

    // One vote per detected modality: the trained block holding the most of
    // its raw feature names. A tag with no hits never enters, so absence of an
    // entry IS "no match" — there is no second flag to keep consistent.
    let mut best: FxHashMap<&str, (usize, &str)> = FxHashMap::default();
    for (i, rows) in query_rows.iter().enumerate() {
        let tag = plan.modality[i].as_ref();
        for (&trained_tag, names) in &trained {
            let hits = rows.iter().filter(|r| names.contains(r.as_ref())).count();
            if hits > best.get(tag).map_or(0, |&(h, _)| h) {
                best.insert(tag, (hits, trained_tag));
            }
        }
    }

    // No two detected modalities may land on one trained block: that would put
    // two different assays on the same rows.
    let mut claimed: FxHashMap<&str, &str> = FxHashMap::default();
    for (&tag, &(_, want)) in &best {
        if let Some(other) = claimed.insert(want, tag) {
            anyhow::bail!(
                "query modalities {other:?} and {tag:?} both match the trained row \
                 block {want:?}; they cannot share it. Load them separately, or \
                 rename the files so each modality is its own group."
            );
        }
    }

    let renamed: Vec<Box<str>> = plan
        .modality
        .iter()
        .map(|m| {
            best.get(m.as_ref())
                .map_or_else(|| m.clone(), |&(_, w)| w.into())
        })
        .collect();
    plan.modality = renamed;
    Ok(plan)
}

/// Detect a query file set's layout and name its modalities after the trained
/// ones. `Ok(None)` when the query is single-modality, or the model is.
pub fn query_layout(
    files: &[Box<str>],
    trained_features: &[Box<str>],
) -> anyhow::Result<Option<ge::MultiomePlan>> {
    if trained_modalities(trained_features).is_empty() {
        return Ok(None);
    }
    // One read of both name axes, shared by the planner and the reconciliation
    // below: re-opening each backend for its row names would pay a second
    // round of archive opens and name decodes for rows already in hand.
    let axes = ge::read_file_axes(files)?;
    let Some(plan) = ge::plan_from_axes(&axes)? else {
        return Ok(None);
    };
    let by_file: FxHashMap<&str, &ge::FileAxes> =
        axes.iter().map(|a| (a.file.as_ref(), a)).collect();
    let rows: Vec<Vec<Box<str>>> = plan
        .files
        .iter()
        .map(|f| by_file[f.as_ref()].rows.clone())
        .collect();
    Ok(Some(reconcile_modalities(plan, &rows, trained_features)?))
}

/// The loader settings for a detected plan — the same shape `recorded_layout`
/// replays, so the trainer and every replayer agree by construction.
#[must_use]
pub fn layout_of_plan(plan: &ge::MultiomePlan) -> ReloadLayout {
    ReloadLayout(Some(Suffixes {
        feature: plan.modality.clone(),
        barcode: plan.barcode_suffix(),
    }))
}

#[cfg(test)]
mod tests;
