//! Per-cell label loaders for the structure plot's facet axis: batch
//! labels (from `--batch-files` or data-file basenames) and annotation
//! labels (from `senna annotate`'s `argmax.tsv`).

use super::ResolvedInputs;
use crate::embed_common::*;
use rustc_hash::FxHashMap;
use std::fs;
use std::path::Path;

/// Per-cell batch label, length == `n_cells`, in `latent.parquet` row
/// order. Reads the original batch-file paths from the manifest
/// (matching pinto's "paths-in-json" pattern); falls back to a synthetic
/// label per data-file when no batch files were provided.
pub(super) fn load_batch_labels(
    resolved: &ResolvedInputs,
    n_cells: usize,
) -> anyhow::Result<Vec<Box<str>>> {
    use matrix_util::common_io::read_lines;

    if !resolved.batch_files.is_empty() {
        let mut all = Vec::with_capacity(n_cells);
        for bf in &resolved.batch_files {
            info!("Reading batch file: {bf}");
            for s in read_lines(bf)? {
                all.push(s);
            }
        }
        if all.len() != n_cells {
            anyhow::bail!(
                "batch labels total {} != latent rows {} (manifest data.batch may be stale)",
                all.len(),
                n_cells,
            );
        }
        return Ok(all);
    }

    if resolved.data_files.is_empty() {
        // No data file list either — single synthetic batch.
        return Ok(vec!["all".into(); n_cells]);
    }

    // No batch files: derive a label per data file from its basename
    // (with `.zarr.zip`/`.zarr`/`.h5` stripped) — same convention
    // SparseIoVec uses for batch identity. Falls back to the file index
    // only if a basename can't be extracted, and disambiguates duplicate
    // basenames with a `_{i}` suffix. Cheap because num_columns() reads
    // only the on-disk index, not the full matrix.
    use data_beans::convert::try_open_or_convert;
    use data_beans::hdf5_io::strip_backend_suffix;
    let raw_labels: Vec<Box<str>> = resolved
        .data_files
        .iter()
        .enumerate()
        .map(|(idx, df)| {
            Path::new(df.as_str())
                .file_name()
                .and_then(|s| s.to_str())
                .map(strip_backend_suffix)
                .map(Box::<str>::from)
                .unwrap_or_else(|| idx.to_string().into_boxed_str())
        })
        .collect();
    let mut counts: FxHashMap<&str, usize> = FxHashMap::default();
    for n in &raw_labels {
        *counts.entry(n.as_ref()).or_insert(0) += 1;
    }
    let mut seen: FxHashMap<&str, usize> = FxHashMap::default();
    let unique_labels: Vec<Box<str>> = raw_labels
        .iter()
        .map(|n| {
            if counts[n.as_ref()] == 1 {
                n.clone()
            } else {
                let k = seen.entry(n.as_ref()).or_insert(0);
                let s = format!("{}_{}", n, k).into_boxed_str();
                *k += 1;
                s
            }
        })
        .collect();
    let mut all: Vec<Box<str>> = Vec::with_capacity(n_cells);
    for (df, label) in resolved.data_files.iter().zip(unique_labels.iter()) {
        let n = try_open_or_convert(df)?
            .num_columns()
            .ok_or_else(|| anyhow::anyhow!("data file {df} has no column count"))?;
        all.extend(std::iter::repeat_n(label.clone(), n));
    }
    if all.len() != n_cells {
        anyhow::bail!(
            "fallback batch labels {} != latent rows {} (data_files inconsistent)",
            all.len(),
            n_cells,
        );
    }
    Ok(all)
}

/// Per-cell label from `senna annotate`'s `argmax.tsv`. The TSV's
/// `cell` column is matched against `latent.parquet`'s row names —
/// annotate may have run on a subset, so missing cells are tagged
/// `"unannotated"` rather than failing.
pub(super) fn load_annotation_labels(
    resolved: &ResolvedInputs,
    cell_names: &[Box<str>],
) -> anyhow::Result<Vec<Box<str>>> {
    let path = resolved.annotation.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "--group-by annotation requires --annotation PATH or `senna annotate` \
             must have populated manifest.annotate.argmax"
        )
    })?;
    info!("Reading annotation labels from {path}");
    let content = fs::read_to_string(Path::new(path))?;
    let mut by_cell: FxHashMap<Box<str>, Box<str>> = FxHashMap::default();
    for (line_no, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Header: cell\tcell_type\tprobability — skip if first column
        // is "cell" (matches what annotate writes at run.rs:71).
        let mut parts = line.split('\t');
        let cell = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("annotation TSV line {}: missing cell", line_no + 1))?;
        let label = parts.next().ok_or_else(|| {
            anyhow::anyhow!("annotation TSV line {}: missing cell_type", line_no + 1)
        })?;
        if cell == "cell" && label == "cell_type" {
            continue;
        }
        by_cell.insert(cell.into(), label.into());
    }
    let unannotated: Box<str> = "unannotated".into();
    let mut n_missing = 0usize;
    let labels: Vec<Box<str>> = cell_names
        .iter()
        .map(|c| {
            by_cell.get(c).cloned().unwrap_or_else(|| {
                n_missing += 1;
                unannotated.clone()
            })
        })
        .collect();
    if n_missing > 0 {
        info!(
            "annotation: {n_missing}/{} cells absent from {path} → tagged 'unannotated'",
            cell_names.len()
        );
    }
    Ok(labels)
}

/// Cell indices grouped by label, returned in **alphabetical** label
/// order. Matches the canonical fastTopics structure-plot facet order
/// (B cell, CD14+, CD34+, NK cell, T cell …) and is stable across
/// reruns regardless of cell input order.
pub(super) fn cells_by_batch(batch_labels: &[Box<str>]) -> Vec<(Box<str>, Vec<usize>)> {
    let mut buckets: FxHashMap<Box<str>, Vec<usize>> = FxHashMap::default();
    for (i, b) in batch_labels.iter().enumerate() {
        buckets.entry(b.clone()).or_default().push(i);
    }
    let mut keys: Vec<Box<str>> = buckets.keys().cloned().collect();
    keys.sort_unstable();
    keys.into_iter()
        .map(|b| {
            let v = buckets.remove(&b).unwrap_or_default();
            (b, v)
        })
        .collect()
}
