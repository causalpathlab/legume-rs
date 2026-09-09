//! Derive a multiome load plan from the input files themselves.
//!
//! `--multiome rna1.zarr,adt1.zarr --multiome rna2.zarr,adt2.zarr` asks the
//! caller to hand-partition the file list twice over: once into modalities
//! (which files share a feature axis) and once into samples (which files
//! share cells). Both partitions are already written down in the data — the
//! feature axes say what the modality is, and the barcode lists say which
//! cells are the same cell. This module reads them off.
//!
//! Two axes, two rules:
//!
//! - **Modality** — files whose row names overlap on most of the smaller axis
//!   are the same assay. Their features merge onto one row block.
//! - **Sample (group)** — files of *different* modalities whose barcode lists
//!   overlap on most of the smaller axis are the same cells measured twice.
//!   Only cross-modality links count, so two samples of one assay are never
//!   glued together by a barcode-whitelist collision.
//!
//! The plan is only claimed when at least one cross-modality link exists.
//! Two disjoint feature axes with no shared cells are not evidence of a
//! paired design — they are as likely two assays on different donors, or one
//! assay quantified against two references — so that case falls back to the
//! ordinary single-modality load.

use anyhow::Result;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Row-name overlap (as a fraction of the smaller axis) at which two files
/// are the same modality. Mirrors the loader's multi-modal-shape hint, which
/// calls an axis "mostly disjoint" below the same 50%.
const MODALITY_ROW_OVERLAP: f64 = 0.5;

/// Barcode overlap (as a fraction of the smaller list) at which two files of
/// different modalities are the same sample. Measured on 10x data: a true
/// pair shares 100% of its barcodes, while unrelated backends collide on
/// 0.1–0.3% of the whitelist — so the threshold has three orders of
/// magnitude of room, and is set low enough for patchy pairs whose
/// modalities were QC'd apart.
const PAIR_BARCODE_OVERLAP: f64 = 0.2;

/// One input file's two name axes, as read off its backend.
pub struct FileAxes {
    pub file: Box<str>,
    pub rows: Vec<Box<str>>,
    pub cols: Vec<Box<str>>,
}

/// A resolved multiome load: which file is which modality, which files are
/// the same sample, and the file order the rest of the pipeline expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiomePlan {
    /// Input files reordered so every group's files are contiguous — the
    /// order `validate_multiome_groups` reads `group_sizes` against.
    pub files: Vec<Box<str>>,
    /// Modality label per entry of `files`; namespaces features as
    /// `{name}/{modality}`.
    pub modality: Vec<Box<str>>,
    /// Sample label per entry of `files`.
    pub group: Vec<Box<str>>,
    /// File count per group, in `files` order.
    pub group_sizes: Vec<usize>,
    /// Whether barcodes are tagged `{barcode}@{group}`. True only with more
    /// than one group: with several samples on one whitelist, raw barcodes
    /// collide across samples and Union loading would fold two donors' cells
    /// into one. The per-file vector is [`Self::barcode_suffix`].
    pub barcode_tagged: bool,
    /// Cells observed in more than one modality of their own group — the
    /// matched cells that anchor the cross-modal alignment. `None` when the
    /// layout was declared rather than measured, so nothing counted them.
    pub n_bridge_cells: Option<usize>,
}

impl MultiomePlan {
    #[must_use]
    pub fn n_groups(&self) -> usize {
        self.group_sizes.len()
    }

    /// The per-file barcode tag the loader wants, or `None` when untagged.
    #[must_use]
    pub fn barcode_suffix(&self) -> Option<Vec<Option<Box<str>>>> {
        self.barcode_tagged
            .then(|| self.group.iter().map(|g| Some(g.clone())).collect())
    }

    /// Distinct modality labels, in first-seen order.
    #[must_use]
    pub fn modalities(&self) -> Vec<Box<str>> {
        let mut seen = FxHashSet::default();
        self.modality
            .iter()
            .filter(|m| seen.insert((*m).clone()))
            .cloned()
            .collect()
    }
}

/// Read both name axes of every file and plan from them.
///
/// Backends are opened for their name arrays only — no column data — so this
/// costs milliseconds even on the largest inputs.
pub fn detect_multiome_plan(files: &[Box<str>]) -> Result<Option<MultiomePlan>> {
    if files.len() < 2 {
        return Ok(None);
    }
    plan_from_axes(&read_file_axes(files)?)
}

/// Read both name axes of every file, in parallel.
///
/// Public because a caller that needs the row names *after* planning — the
/// query-side modality reconciliation does — must not pay a second round of
/// archive opens and name decodes for rows the plan was just built from.
pub fn read_file_axes(files: &[Box<str>]) -> Result<Vec<FileAxes>> {
    files
        .par_iter()
        .map(|f| {
            let data = data_beans::convert::try_open_or_convert(f)?;
            Ok(FileAxes {
                file: f.clone(),
                rows: data.row_names()?,
                cols: data.column_names()?,
            })
        })
        .collect()
}

/// The planning rules, over name axes already in hand.
pub fn plan_from_axes(axes: &[FileAxes]) -> Result<Option<MultiomePlan>> {
    let n = axes.len();
    if n < 2 {
        return Ok(None);
    }

    let rows: Vec<FxHashSet<&str>> = axes
        .par_iter()
        .map(|a| a.rows.iter().map(AsRef::as_ref).collect())
        .collect();
    let cols: Vec<FxHashSet<&str>> = axes
        .par_iter()
        .map(|a| a.cols.iter().map(AsRef::as_ref).collect())
        .collect();

    // Modality = connected components of "row axes mostly agree". The pairwise
    // sweep is the only quadratic step here, and each pair is independent.
    let modality_of = components(
        n,
        &linked_pairs(n, |i, j| {
            overlap_of_smaller(&rows[i], &rows[j]) >= MODALITY_ROW_OVERLAP
        }),
    );
    let n_modalities = modality_of.iter().copied().max().map_or(0, |m| m + 1);
    if n_modalities < 2 {
        return Ok(None);
    }

    // Sample = connected components of "different modalities, same cells".
    // Same-modality pairs are never linked, so a whitelist collision between
    // two samples of one assay cannot merge them.
    let links = linked_pairs(n, |i, j| {
        modality_of[i] != modality_of[j]
            && overlap_of_smaller(&cols[i], &cols[j]) >= PAIR_BARCODE_OVERLAP
    });
    if links.is_empty() {
        return Ok(None);
    }
    let group_of = components(n, &links);
    let n_groups = group_of.iter().copied().max().map_or(0, |g| g + 1);

    // One row block per modality per sample: two files of the same modality
    // in one group have no defined place to land.
    let mut seen: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    for i in 0..n {
        if let Some(&prev) = seen.get(&(group_of[i], modality_of[i])) {
            anyhow::bail!(
                "multiome: {:?} and {:?} share cells and a feature axis, so they are \
                 the same modality of the same sample and cannot both be loaded. \
                 Drop one, merge them first, or lay out the groups by hand with \
                 `--multiome`.",
                axes[prev].file,
                axes[i].file,
            );
        }
        seen.insert((group_of[i], modality_of[i]), i);
    }

    let tokens: Vec<Vec<Box<str>>> = axes.iter().map(|a| file_tokens(&a.file)).collect();
    let modality_labels = cluster_labels(&tokens, &modality_of, n_modalities, "m");
    let group_labels = cluster_labels(&tokens, &group_of, n_groups, "g");

    // Flatten groups in first-seen order, files within a group in input order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| group_of[i]); // stable, so input order breaks ties
    let mut group_sizes = vec![0usize; n_groups];
    for &g in &group_of {
        group_sizes[g] += 1;
    }

    // Matched cells: barcodes their own group sees through more than one
    // modality.
    let mut n_bridge_cells = 0usize;
    for (g, &size) in group_sizes.iter().enumerate() {
        if size < 2 {
            continue;
        }
        let mut hits: FxHashMap<&str, usize> = FxHashMap::default();
        for i in (0..n).filter(|&i| group_of[i] == g) {
            for c in &cols[i] {
                *hits.entry(c).or_insert(0) += 1;
            }
        }
        n_bridge_cells += hits.values().filter(|&&h| h > 1).count();
    }

    let files: Vec<Box<str>> = order.iter().map(|&i| axes[i].file.clone()).collect();
    let modality: Vec<Box<str>> = order
        .iter()
        .map(|&i| modality_labels[modality_of[i]].clone())
        .collect();
    let group: Vec<Box<str>> = order
        .iter()
        .map(|&i| group_labels[group_of[i]].clone())
        .collect();
    Ok(Some(MultiomePlan {
        files,
        modality,
        group,
        group_sizes,
        barcode_tagged: n_groups > 1,
        n_bridge_cells: Some(n_bridge_cells),
    }))
}

/// |A ∩ B| / min(|A|, |B|). Normalising by the smaller side is what lets a
/// modality QC'd down to 60% of its partner's barcodes still pair, and what
/// lets two gene panels of different sizes read as one assay.
///
/// On the ROW axis it has a known cost: a small panel whose names are a
/// SUBSET of a larger one scores 1.0 and is absorbed into it. An antibody
/// panel labelled with bare gene symbols (`CD14`, `CD3`) rather than tagged
/// (`ADT-CD14`) therefore reads as part of the gene axis, and the run loads
/// single-modality. Nothing in the names distinguishes that from a genuine
/// gene subset, so it is not decidable here — `--multiome` is the override,
/// and the flag's help says so.
fn overlap_of_smaller(a: &FxHashSet<&str>, b: &FxHashSet<&str>) -> f64 {
    let m = a.len().min(b.len());
    if m == 0 {
        return 0.0;
    }
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let shared = small.iter().filter(|k| large.contains(*k)).count();
    shared as f64 / m as f64
}

/// Split a path's basename into name tokens.
///
/// The basename comes from `matrix_util::common_io::basename`, the same
/// stem+extension rule the loaders use, so adding a backend there cannot leave
/// the detector labelling groups after a file extension.
fn file_tokens(path: &str) -> Vec<Box<str>> {
    let base = matrix_util::common_io::basename(path).unwrap_or_else(|_| path.into());
    base.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(Into::into)
        .collect()
}

/// Name each cluster from the tokens its files all share, minus the tokens
/// *every* cluster shares (those name the cohort, not the cluster). Falls
/// back to `{prefix}{k}` whenever that leaves nothing, or leaves two clusters
/// with the same name.
fn cluster_labels(
    tokens: &[Vec<Box<str>>],
    cluster_of: &[usize],
    n_clusters: usize,
    prefix: &str,
) -> Vec<Box<str>> {
    // Tokens every file of a cluster carries, borrowed from `tokens`.
    let shared: Vec<Vec<&str>> = (0..n_clusters)
        .map(|k| {
            let mut members = (0..tokens.len())
                .filter(|&i| cluster_of[i] == k)
                .map(|i| &tokens[i]);
            let first = members.next().expect("cluster ids are dense");
            let rest: Vec<_> = members.collect();
            first
                .iter()
                .map(AsRef::as_ref)
                .filter(|t| rest.iter().all(|m| m.iter().any(|x| x.as_ref() == *t)))
                .collect()
        })
        .collect();

    // Tokens common to EVERY cluster name the cohort, not the cluster — the
    // same "shared by all" test as above, now spelled once.
    let universal: FxHashSet<&str> = shared[0]
        .iter()
        .copied()
        .filter(|t| shared.iter().all(|s| s.contains(t)))
        .collect();

    let labels: Vec<Box<str>> = shared
        .iter()
        .map(|s| {
            s.iter()
                .copied()
                .filter(|t| !universal.contains(t))
                .collect::<Vec<&str>>()
                .join("_")
                .into_boxed_str()
        })
        .collect();

    let distinct: FxHashSet<&str> = labels.iter().map(AsRef::as_ref).collect();
    if labels.iter().any(|l| l.is_empty()) || distinct.len() != n_clusters {
        return (0..n_clusters)
            .map(|k| format!("{prefix}{k}").into_boxed_str())
            .collect();
    }
    labels
}

/// Every `i < j` the predicate links, evaluated in parallel and returned in a
/// deterministic order so the components below never depend on thread timing.
fn linked_pairs<F>(n: usize, linked: F) -> Vec<(usize, usize)>
where
    F: Fn(usize, usize) -> bool + Sync + Send,
{
    let mut pairs: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| (i + 1..n).map(move |j| (i, j)))
        .filter(|&(i, j)| linked(i, j))
        .collect();
    pairs.sort_unstable();
    pairs
}

/// Component id per file, numbered by the first file index in each — the
/// shared `connected_components` walks roots in ascending node order, which is
/// exactly the input order the group and modality labels are keyed by.
fn components(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    matrix_util::graph::connected_components(
        &matrix_util::graph::AdjListGraph::from_unweighted_edges(n, edges),
    )
}

#[cfg(test)]
mod tests;
