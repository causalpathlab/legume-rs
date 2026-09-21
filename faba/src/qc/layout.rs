//! What a faba output directory contains, by name.
//!
//! Producers write one matrix per (batch, kind) as `{batch}_{kind}.{zarr.zip|zarr|h5}`
//! and one shared site table per editing modality (`{modality}_sites.parquet`).
//! Nothing here opens a file; it only classifies names so `qc` and `qc-report`
//! agree on which file is which.

use std::path::Path;

use data_beans::aux::feature_rows::{APA, ATOI, BAF, COUNT, M6A};
use data_beans::sparse_io::SparseIoBackend;
use rustc_hash::FxHashMap;

/// Matrix kinds, longest suffix first so `_m6a_site` is not read as `_m6a`.
pub const MATRIX_KINDS: &[&str] = &[
    "m6a_site",
    "m6a_mixture",
    "atoi_site",
    "atoi_mixture",
    "apa_mixture",
    M6A,
    ATOI,
    APA,
    COUNT,
    BAF,
    "depth",
];

/// Editing modalities with a shared `{modality}_sites.parquet` and per-batch
/// `_site` matrices.
pub const SITE_MODALITIES: &[&str] = &[M6A, ATOI];

#[derive(Debug, Clone)]
pub struct MatrixFile {
    pub path: Box<str>,
    pub batch: Box<str>,
    /// One of [`MATRIX_KINDS`], or empty when none matched (the file then
    /// keeps its whole stem as its batch name).
    pub kind: Box<str>,
    pub backend: SparseIoBackend,
    /// Whether the input was a `.zarr.zip` archive (the output mirrors it).
    pub zipped: bool,
}

impl MatrixFile {
    /// `{batch}_{kind}`, the name the output file gets under the same layout.
    pub fn stem(&self) -> String {
        if self.kind.is_empty() {
            self.batch.to_string()
        } else {
            format!("{}_{}", self.batch, self.kind)
        }
    }

    /// The editing modality whose sites this `_site` matrix quantifies.
    pub fn site_modality(&self) -> Option<&str> {
        self.kind.strip_suffix("_site")
    }
}

#[derive(Debug, Default)]
pub struct InputLayout {
    pub matrices: Vec<MatrixFile>,
    /// `modality -> path` for every `{modality}_sites.parquet` present.
    pub site_tables: FxHashMap<Box<str>, Box<str>>,
    /// Every other regular file, copied through untouched by `qc`.
    pub other_files: Vec<Box<str>>,
}

impl InputLayout {
    pub fn batches(&self) -> Vec<Box<str>> {
        let mut b: Vec<Box<str>> = self.matrices.iter().map(|m| m.batch.clone()).collect();
        b.sort();
        b.dedup();
        b
    }

    pub fn matrix(&self, batch: &str, kind: &str) -> Option<&MatrixFile> {
        self.matrices
            .iter()
            .find(|m| &*m.batch == batch && &*m.kind == kind)
    }
}

/// Split a matrix file name into `(stem, backend, zipped)`; `None` when it is
/// not a sparse backend at all.
fn classify_matrix_name(name: &str, is_dir: bool) -> Option<(&str, SparseIoBackend, bool)> {
    if let Some(stem) = name.strip_suffix(".zarr.zip") {
        return Some((stem, SparseIoBackend::Zarr, true));
    }
    if let Some(stem) = name.strip_suffix(".zarr") {
        return is_dir.then_some((stem, SparseIoBackend::Zarr, false));
    }
    if let Some(stem) = name.strip_suffix(".h5") {
        return Some((stem, SparseIoBackend::HDF5, false));
    }
    None
}

/// `{batch}_{kind}` → `(batch, kind)`, longest known kind first; an unknown
/// stem becomes `(stem, "")` so it still round-trips under its own name.
fn split_stem(stem: &str) -> (Box<str>, Box<str>) {
    for kind in MATRIX_KINDS {
        if let Some(batch) = stem.strip_suffix(&format!("_{kind}")) {
            if !batch.is_empty() {
                return (batch.into(), (*kind).into());
            }
        }
    }
    (stem.into(), "".into())
}

pub fn scan_input_dir(dir: &str) -> anyhow::Result<InputLayout> {
    let mut layout = InputLayout::default();
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|e| e.path())
        .collect();
    entries.sort();

    for path in entries {
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        let is_dir = path.is_dir();
        let path_str: Box<str> = path.to_string_lossy().as_ref().into();

        if let Some((stem, backend, zipped)) = classify_matrix_name(name, is_dir) {
            let (batch, kind) = split_stem(stem);
            layout.matrices.push(MatrixFile {
                path: path_str,
                batch,
                kind,
                backend,
                zipped,
            });
            continue;
        }
        if is_dir {
            continue;
        }
        if let Some(modality) = name.strip_suffix("_sites.parquet") {
            if SITE_MODALITIES.contains(&modality) {
                layout.site_tables.insert(modality.into(), path_str);
                continue;
            }
        }
        layout.other_files.push(path_str);
    }
    anyhow::ensure!(
        !layout.matrices.is_empty() || !layout.site_tables.is_empty(),
        "{dir}: no faba matrices or site tables found"
    );
    Ok(layout)
}

/// Basename of a path, for output naming.
pub fn file_name(path: &str) -> Box<str> {
    Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().as_ref().into())
        .unwrap_or_else(|| path.into())
}
