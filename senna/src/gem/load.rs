//! Multi-file input resolution and loading for `senna gem`.
//!
//! `senna gem` takes gene-count files (positional `GENES`) and, optionally,
//! one file per co-measured modality (`--modality`, repeatable). Every file
//! is read once, cheaply (row/column NAMES only, via [`ge::read_file_axes`])
//! to classify it as a gene file (`count` rows only) or a modality file
//! (exactly one non-`count` modality) and to derive its sample id — the
//! matching key that lets `senna gem a_genes.zarr.zip --modality
//! a_m6a.zarr.zip` merge `a`'s two files under one set of barcodes, while
//! keeping `a` and `b` apart.
//!
//! [`resolve_inputs`] does the classification and sample-id matching;
//! [`load_gem_data`] does the actual load (`Union` column alignment, exact
//! row-name matching — the row itself is the join key) and hands the result
//! to [`crate::gem::tracks::assign_tracks`].

use std::collections::BTreeSet;

use auxiliary_data::feature_rows::{parse_feature_row, COUNT};
use graph_embedding_util as ge;
use log::info;
use matrix_util::common_io::basename;

use crate::gem::sample_id::file_sample_id;
use crate::gem::tracks::{assign_tracks, TrackPlan};

/// One resolved gem input: every file to load, its matched sample id, and
/// (for a `--modality` file) which modality it carries. `None` in
/// `modality_of_file` marks a gene file.
#[derive(Debug)]
pub(crate) struct GemInputs {
    pub files: Vec<Box<str>>,
    pub sample_ids: Vec<Box<str>>,
    /// Not yet read by `load_gem_data` (the load itself doesn't need to know
    /// which file is which — `Union` alignment merges them all the same
    /// way); carried for a later task's per-modality bookkeeping.
    #[allow(dead_code)]
    pub modality_of_file: Vec<Option<Box<str>>>,
}

/// Classify and sample-id-match every input file.
///
/// A gene file (from `genes`) must hold only `count` rows. A modality file
/// (from `modality_files`) must hold exactly one non-`count` modality —
/// read from its rows, never its name. Every modality file's sample id must
/// equal some gene file's sample id, so it merges onto the right sample
/// under `Union` column alignment; a mismatch errors, listing both sets.
///
/// Sample id per file: `strip` when non-empty, else `_genes` for a gene file
/// / `_{modality}` for a modality file, stripped from the file's basename
/// ([`strip_sample_id`]). A basename that does not end with that suffix
/// keeps its full name and is warned — see [`sample_id_for`].
pub(crate) fn resolve_inputs(
    genes: &[Box<str>],
    modality_files: &[Box<str>],
    strip: &str,
) -> anyhow::Result<GemInputs> {
    anyhow::ensure!(
        !genes.is_empty(),
        "no gene matrices given: pass them positionally \
         (`senna gem out/*_genes.zarr.zip -o out/gem`)"
    );

    let all_files: Vec<Box<str>> = genes.iter().chain(modality_files.iter()).cloned().collect();
    let axes = ge::read_file_axes(&all_files)?;

    let mut files: Vec<Box<str>> = Vec::with_capacity(all_files.len());
    let mut sample_ids: Vec<Box<str>> = Vec::with_capacity(all_files.len());
    let mut modality_of_file: Vec<Option<Box<str>>> = Vec::with_capacity(all_files.len());

    let mut gene_sample_ids: BTreeSet<Box<str>> = BTreeSet::new();
    // (file, sample id) for every modality file, checked against
    // `gene_sample_ids` once every file has been classified.
    let mut modality_entries: Vec<(Box<str>, Box<str>)> = Vec::new();

    for (i, ax) in axes.iter().enumerate() {
        let mods = distinct_modalities(&ax.rows);
        let non_count: Vec<&str> = mods
            .iter()
            .map(AsRef::as_ref)
            .filter(|&m| m != COUNT)
            .collect();

        if i < genes.len() {
            anyhow::ensure!(
                non_count.is_empty(),
                "{}: a gene file must hold only `count` rows; found modalit{} {:?}",
                ax.file,
                if mods.len() == 1 { "y" } else { "ies" },
                mods
            );
            let sid = sample_id_for(&ax.file, strip, "_genes")?;
            gene_sample_ids.insert(sid.clone());
            files.push(ax.file.clone());
            sample_ids.push(sid);
            modality_of_file.push(None);
        } else {
            anyhow::ensure!(
                non_count.len() == 1,
                "{}: a --modality file must hold exactly one non-count modality; found {:?}",
                ax.file,
                mods
            );
            let modality: Box<str> = non_count[0].into();
            let default_suffix = format!("_{modality}");
            let sid = sample_id_for(&ax.file, strip, &default_suffix)?;
            modality_entries.push((ax.file.clone(), sid.clone()));
            files.push(ax.file.clone());
            sample_ids.push(sid);
            modality_of_file.push(Some(modality));
        }
    }

    let bad: Vec<&(Box<str>, Box<str>)> = modality_entries
        .iter()
        .filter(|(_, sid)| !gene_sample_ids.contains(sid))
        .collect();
    anyhow::ensure!(
        bad.is_empty(),
        "modality file(s) {:?} carry sample id(s) {:?} that match none of the gene sample \
         id(s) {:?}",
        bad.iter().map(|(f, _)| f.as_ref()).collect::<Vec<_>>(),
        bad.iter().map(|(_, s)| s.as_ref()).collect::<Vec<_>>(),
        gene_sample_ids
    );

    Ok(GemInputs {
        files,
        sample_ids,
        modality_of_file,
    })
}

/// Distinct modalities among a file's rows (unparseable rows are skipped
/// here — [`assign_tracks`] is where a bad row is rejected by name, once
/// this file's rows are merged onto the unified axis).
fn distinct_modalities(rows: &[Box<str>]) -> BTreeSet<Box<str>> {
    rows.iter()
        .filter_map(|r| parse_feature_row(r).map(|row| Box::<str>::from(row.modality)))
        .collect()
}

/// A file's sample id: its basename with `strip` (when non-empty) or
/// `default_suffix` removed. A basename that does not end with that suffix
/// keeps its full name — and is warned, since it means this file did not
/// match the naming convention every OTHER file of its kind is assumed to.
fn sample_id_for(file: &str, strip: &str, default_suffix: &str) -> anyhow::Result<Box<str>> {
    let effective = if strip.is_empty() {
        default_suffix
    } else {
        strip
    };
    let sid = file_sample_id(file, effective)?;
    let base = basename(file)?;
    if sid.as_ref() == base.as_ref() {
        log::warn!(
            "{file}: basename {base:?} does not end with {effective:?}; using the full \
             basename as its sample id"
        );
    }
    Ok(sid)
}

/// Load every resolved input into one [`ge::UnifiedData`] and assign its
/// [`TrackPlan`].
///
/// `Union` column alignment (cells merge by barcode) with per-file `@sample`
/// barcode tagging whenever more than one file is given — including with
/// `--batch-files`, which then names the batch of each already-tagged
/// barcode. Row names are matched EXACTLY across files: the row itself
/// (`{gene}/{modality}/{channel}`) is the join key, so no per-file feature
/// suffix is applied.
pub(crate) fn load_gem_data(
    inputs: &GemInputs,
    batch_files: Option<&[Box<str>]>,
    preload: bool,
) -> anyhow::Result<(ge::UnifiedData, TrackPlan)> {
    let per_file_barcode_suffix: Option<Vec<Option<Box<str>>>> =
        (inputs.files.len() > 1).then(|| inputs.sample_ids.iter().cloned().map(Some).collect());

    let unified = ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: inputs.files.clone(),
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        feature_kind: Some(ge::FeatureNameKind::Exact),
        preload,
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::Union,
        per_file_feature_suffix: None,
        per_file_barcode_suffix,
        ..Default::default()
    })?;
    info!(
        "gem: loaded {} feature row(s) x {} cell(s) from {} file(s), {} batch(es)",
        unified.n_features(),
        unified.n_cells(),
        inputs.files.len(),
        unified.n_batches()
    );

    let plan = assign_tracks(&unified.feature_names)?;
    log_cells_per_track(&unified, &plan);

    Ok((unified, plan))
}

/// Log, per track, how many cells carry any mass on that track's rows.
fn log_cells_per_track(unified: &ge::UnifiedData, plan: &TrackPlan) {
    let backend = unified.count_backend();
    for t in &plan.tracks {
        let rows = plan.rows_of(t.id);
        match backend.read_rows_csc(rows.iter().copied()) {
            Ok(csc) => {
                let n_cells = (0..csc.ncols()).filter(|&j| csc.col(j).nnz() > 0).count();
                info!(
                    "gem track {} ({}/{}): {} row(s), {n_cells} cell(s) with mass",
                    t.id,
                    t.modality,
                    t.channel,
                    rows.len()
                );
            }
            Err(e) => {
                log::warn!(
                    "gem track {} ({}/{}): could not count cells with mass: {e}",
                    t.id,
                    t.modality,
                    t.channel
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "load/tests.rs"]
mod tests;
