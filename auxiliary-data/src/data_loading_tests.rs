//! `read_data_on_shared_rows` under the default (auto) naming rule, with
//! files that spell their rows differently.

use super::{read_data_on_shared_rows, ReadSharedRowsArgs};
use data_beans::sparse_io::SparseIoBackend;
use nalgebra::DMatrix;

/// A tiny zarr backend whose rows are `rows` and whose single column holds
/// a count for every row, so no row is dropped as empty.
fn backend(dir: &std::path::Path, name: &str, rows: &[&str]) -> Box<str> {
    let path = dir.join(format!("{name}.zarr"));
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_fn(rows.len(), 2, |i, j| (i + j + 1) as f32);
    let mut b = data_beans::sparse_io::create_sparse_from_dmatrix(
        &m,
        Some(&path),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("backend");
    let rows: Vec<Box<str>> = rows.iter().map(|r| (*r).into()).collect();
    b.register_row_names_vec(&rows);
    let cols: Vec<Box<str>> = (0..2).map(|j| format!("{name}_c{j}").into()).collect();
    b.register_column_names_vec(&cols);
    path
}

fn load_rows(files: Vec<Box<str>>) -> usize {
    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: files,
        ..Default::default()
    })
    .expect("load");
    loaded.data.num_rows()
}

#[test]
fn a_raw_cohort_and_its_canonical_reference_share_one_axis() {
    // Pooled: 2 gene-like of 5 names = 40% < 50%, which used to sniff Exact
    // and give 5 rows. Per file: Gene + Exact -> Gene -> 3 rows.
    let dir = tempfile::tempdir().unwrap();
    let raw = backend(dir.path(), "raw", &["ENSG1_A", "ENSG2_B"]);
    let canon = backend(dir.path(), "canon", &["A", "B", "C"]);
    assert_eq!(load_rows(vec![raw, canon]), 3);
}

#[test]
fn overlapping_loci_across_files_still_merge() {
    // The locus-overlap map needs every file's names at once; per-file
    // detection must not take that pool away.
    let dir = tempfile::tempdir().unwrap();
    let a = backend(dir.path(), "a", &["chr1:1-20", "chr2:1-10"]);
    let b = backend(dir.path(), "b", &["chr1:15-30", "chr3:1-10"]);
    assert_eq!(load_rows(vec![a, b]), 3);
}
