//! Shared synthetic fixtures for gem's own tests and for `senna predict`'s
//! gem contract test, so the axis / cell shapes stay identical across both
//! rather than drifting apart as two hand-copied fixtures.

use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use nalgebra::DMatrix;

pub(crate) fn boxes(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&s| s.into()).collect()
}

/// A tiny synthetic zarr backend at `dir/{stem}.zarr`, one small positive
/// integer per cell so nothing is near-empty.
pub(crate) fn synth(dir: &std::path::Path, stem: &str, rows: &[&str], cols: &[&str]) -> Box<str> {
    let path = dir.join(format!("{stem}.zarr"));
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_fn(rows.len(), cols.len(), |r, c| {
        (((r * 7 + c * 11 + 3) % 9) + 1) as f32
    });
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr))
        .expect("create synthetic backend");
    b.register_row_names_vec(&boxes(rows));
    b.register_column_names_vec(&boxes(cols));
    path
}

pub(crate) const CELLS: [&str; 6] = ["C1", "C2", "C3", "C4", "C5", "C6"];

/// The genes file every gem fixture test shares: GENE1 carries both count
/// channels, GENE2 only `spliced` (no `unspliced` anywhere for GENE2).
pub(crate) fn genes_file(dir: &std::path::Path) -> Box<str> {
    synth(
        dir,
        "S1_genes",
        &[
            "GENE1/count/spliced",
            "GENE1/count/unspliced",
            "GENE2/count/spliced",
        ],
        &CELLS,
    )
}

/// GENE1's `m6a` modality file: both channels present, so the contrast
/// table can pair them.
pub(crate) fn m6a_file(dir: &std::path::Path) -> Box<str> {
    synth(
        dir,
        "S1_m6a",
        &["GENE1/m6a/methylated", "GENE1/m6a/unmethylated"],
        &CELLS,
    )
}
