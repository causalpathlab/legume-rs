//! Resolve bge's multiome layout: which files are which modality, and which
//! files are the same sample.
//!
//! Two ways in, one way out. `--multiome rna.zarr,adt.zarr` declares the
//! layout; with plain positional files the layout is read off the inputs by
//! [`graph_embedding_util::detect_multiome_plan`]. Both produce a
//! [`ge::MultiomePlan`], so everything downstream — feature namespacing,
//! group validation, batch resolution — sees one shape.

use graph_embedding_util as ge;
use log::{info, warn};

/// One parsed `--multiome` file entry: `(optional modality label, file path)`.
/// Kept as an alias rather than inlined at its single use — the bare tuple
/// trips `clippy::type_complexity` two levels deep.
type MultiomeFile = (Option<Box<str>>, Box<str>);

/// Reasons a run declines to look for a multiome layout in its inputs. The
/// declared `--multiome` path is unaffected by all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoAutoDetect {
    /// Union loading gives no guarantee the carried pseudobulks stay
    /// contiguous, and their weights are applied by position.
    CarriedPseudobulks,
    /// Per-file batch labels don't compose when one cell is in two files;
    /// a caller who wrote them out meant the per-file (Disjoint) load.
    PerFileBatchFiles,
}

impl NoAutoDetect {
    fn reason(self) -> &'static str {
        match self {
            Self::CarriedPseudobulks => "a carried pseudobulk reference is in play",
            Self::PerFileBatchFiles => "per-file --batch-files were given",
        }
    }
}

/// Something about this run that makes the Union load unsafe, worth saying out
/// loud. "Fewer than two files" is not one of them — that is the trivial base
/// case, handled by the caller, and logging it on every single-file run would
/// be noise.
pub(crate) fn auto_detect_veto(
    n_batch_files: usize,
    has_pb_reference: bool,
) -> Option<NoAutoDetect> {
    if has_pb_reference {
        return Some(NoAutoDetect::CarriedPseudobulks);
    }
    if n_batch_files > 1 {
        return Some(NoAutoDetect::PerFileBatchFiles);
    }
    None
}

/// Read the layout off the inputs, unless something about this run vetoes it.
pub(crate) fn auto_plan(
    data_files: &[Box<str>],
    n_batch_files: usize,
    has_pb_reference: bool,
) -> anyhow::Result<Option<ge::MultiomePlan>> {
    if data_files.len() < 2 {
        return Ok(None); // nothing to glue against
    }
    if let Some(veto) = auto_detect_veto(n_batch_files, has_pb_reference) {
        info!(
            "Not checking the inputs for a multiome layout: {}. \
             Pass --multiome to declare one.",
            veto.reason()
        );
        return Ok(None);
    }
    ge::detect_multiome_plan(data_files)
}

/// Parse the declared `--multiome` groups into the same plan the detector
/// produces. One flag occurrence is one group; files within it are
/// comma-separated, each optionally prefixed `label=` to name its modality.
/// Without a label the modality is the file's position in its group, so the
/// tag lines up across groups.
pub(crate) fn declared_plan(groups: &[Box<str>]) -> anyhow::Result<Option<ge::MultiomePlan>> {
    let parsed: Vec<Vec<MultiomeFile>> = groups
        .iter()
        .map(|s| {
            s.split(',')
                .map(|tok| match tok.split_once('=') {
                    Some((label, file)) if !label.is_empty() && !file.is_empty() => {
                        (Some(label.into()), file.into())
                    }
                    _ => (None, tok.into()),
                })
                .collect()
        })
        .collect();

    anyhow::ensure!(
        parsed
            .iter()
            .all(|g| !g.is_empty() && g.iter().all(|(_, f)| !f.is_empty())),
        "--multiome: empty group or file name; comma-separate the files of one \
         sample, with no spaces"
    );

    // A one-file layout has no second modality to glue against. Refusing it
    // here rather than downstream is what keeps the plan the single authority:
    // otherwise the HVG resolver clears the multiome flag on file count, the
    // load falls back to Disjoint, and the feature suffixes and the recorded
    // layout still say Union.
    let n_files: usize = parsed.iter().map(Vec::len).sum();
    if n_files < 2 {
        log::warn!("--multiome names a single file, which has nothing to glue against; ignoring.");
        return Ok(None);
    }

    let group_sizes: Vec<usize> = parsed.iter().map(Vec::len).collect();
    let mut files = Vec::new();
    let mut modality = Vec::new();
    let mut group = Vec::new();
    for (g, entries) in parsed.iter().enumerate() {
        for (pos, (label, file)) in entries.iter().enumerate() {
            files.push(file.clone());
            modality.push(
                label
                    .clone()
                    .unwrap_or_else(|| format!("m{pos}").into_boxed_str()),
            );
            group.push(format!("g{g}").into_boxed_str());
        }
    }

    Ok(Some(ge::MultiomePlan {
        files,
        modality,
        group,
        group_sizes,
        // Declared groups keep their documented contract: barcodes must be
        // disjoint across groups, and `validate_multiome_groups` says so with
        // the offending barcode when they are not.
        barcode_tagged: false,
        n_bridge_cells: None,
    }))
}

/// Print the layout a run is about to load under. Auditable by design: this
/// is the only place the derived grouping is visible before training starts.
pub(crate) fn log_plan(plan: &ge::MultiomePlan, declared: bool) {
    let how = if declared { "declared" } else { "auto" };
    let modalities = plan.modalities();
    info!(
        "multiome ({}): {} sample group(s) x {} modality(ies) [{}]",
        how,
        plan.n_groups(),
        modalities.len(),
        modalities
            .iter()
            .map(std::convert::AsRef::as_ref)
            .collect::<Vec<&str>>()
            .join(", ")
    );
    let mut at = 0usize;
    for (g, &size) in plan.group_sizes.iter().enumerate() {
        info!(
            "  group {} [{}]: {}",
            g,
            plan.group[at],
            (at..at + size)
                .map(|i| format!("{}={}", plan.modality[i], plan.files[i]))
                .collect::<Vec<String>>()
                .join(", ")
        );
        at += size;
    }
    info!("  features namespaced as {{name}}/{{modality}}");
    if plan.barcode_tagged {
        info!(
            "  barcodes namespaced as {{barcode}}@{{group}} \
             (several samples on one whitelist would otherwise collide)"
        );
    }
    match plan.n_bridge_cells {
        Some(0) => warn!(
            "multiome: no cell is measured in more than one modality. \
             The modalities share no bridge, so nothing ties their feature \
             blocks together beyond the batch correction."
        ),
        Some(n) => info!("  matched cells (seen in >1 modality): {n}"),
        None => {}
    }
}

#[cfg(test)]
mod tests;
