//! `--bulk` hands a dense table to a pipeline that only reads sparse backends.
//! The contract is that the temp backend built here is indistinguishable from
//! one built the long way (`data-beans from-mtx` → triplets), so nothing
//! downstream can tell which door the data came in through.

use super::{materialize, BulkBackends};
use crate::embed_common::{BulkTableOpts, HeaderArg, Mat, Orientation};
use data_beans::sparse_io::{create_sparse_from_triplets, open_sparse_matrix, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;
use matrix_util::traits::IoOps;
use std::sync::Arc;

fn labels(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

/// Genes × samples with a zero, a fraction and a large count.
fn fixture() -> (Vec<Box<str>>, Vec<Box<str>>, Mat) {
    let genes = labels(&["TGFB1", "CD8A", "LYZ"]);
    let samples = labels(&["s0", "s1"]);
    let mat = Mat::from_row_slice(3, 2, &[10.0, 0.0, 2.5, 4.0, 123_456.0, 7.0]);
    (genes, samples, mat)
}

fn write_parquet(
    dir: &tempfile::TempDir,
    name: &str,
    rows: &[Box<str>],
    cols: &[Box<str>],
    mat: &Mat,
) -> Box<str> {
    let path = dir.path().join(name);
    mat.to_parquet_with_names(
        path.to_str().unwrap(),
        (Some(rows), Some("gene")),
        Some(cols),
    )
    .expect("write parquet");
    path.to_string_lossy().into_owned().into_boxed_str()
}

fn read_all(v: &SparseIoVec) -> nalgebra_sparse::CscMatrix<f32> {
    v.read_columns_csc(0..v.num_columns()).expect("read")
}

fn open_vec(path: &str) -> SparseIoVec {
    let b = open_sparse_matrix(path, &SparseIoBackend::Zarr).expect("reopen by path");
    let mut v = SparseIoVec::new();
    v.push(Arc::from(b), None).expect("push");
    v
}

/// The backend is reopened BY PATH, after the writer that built it is gone —
/// exactly how the predict paths will see it.
#[test]
fn a_dense_table_round_trips_through_the_temp_backend() {
    let dir = tempfile::tempdir().unwrap();
    let (genes, samples, mat) = fixture();
    let f = write_parquet(&dir, "bulk.parquet", &genes, &samples, &mat);
    let bulk = materialize(&[f], &genes, &BulkTableOpts::default()).expect("materialize");
    assert_eq!(bulk.paths().len(), 1);

    let v = open_vec(&bulk.paths()[0]);
    assert_eq!(v.num_rows(), 3);
    assert_eq!(v.num_columns(), 2);
    assert_eq!(v.row_names().unwrap(), genes);
    assert_eq!(v.column_names().unwrap(), samples);

    let csc = read_all(&v);
    let mut seen = 0;
    for (j, col) in csc.col_iter().enumerate() {
        for (&i, &val) in col.row_indices().iter().zip(col.values()) {
            assert_eq!(val.to_bits(), mat[(i, j)].to_bits(), "gene {i} sample {j}");
            seen += 1;
        }
    }
    assert_eq!(
        seen, 5,
        "five non-zeros; the exact zero is dropped, nothing else"
    );
}

/// The long way: the same matrix as triplets, the route `data-beans from-mtx`
/// takes. Both backends must read back identically.
#[test]
fn the_temp_backend_equals_one_built_from_triplets() {
    let dir = tempfile::tempdir().unwrap();
    let (genes, samples, mat) = fixture();
    let f = write_parquet(&dir, "bulk.parquet", &genes, &samples, &mat);
    let bulk = materialize(&[f], &genes, &BulkTableOpts::default()).expect("materialize");
    let via_bulk = read_all(&open_vec(&bulk.paths()[0]));

    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for i in 0..3 {
        for j in 0..2 {
            if mat[(i, j)] != 0.0 {
                triplets.push((i as u64, j as u64, mat[(i, j)]));
            }
        }
    }
    let mtx_path = dir.path().join("mtx.zarr");
    let mut b = create_sparse_from_triplets(
        &triplets,
        (3, 2, triplets.len()),
        Some(mtx_path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("triplets");
    b.register_row_names_vec(&genes);
    b.register_column_names_vec(&samples);
    let mut v = SparseIoVec::new();
    v.push(Arc::from(b), None).unwrap();
    let via_mtx = read_all(&v);

    assert_eq!(via_bulk.col_offsets(), via_mtx.col_offsets());
    assert_eq!(via_bulk.row_indices(), via_mtx.row_indices());
    assert_eq!(via_bulk.values(), via_mtx.values());
}

#[test]
fn a_samples_by_genes_table_materializes_with_genes_on_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (genes, samples, mat) = fixture();
    let f = write_parquet(&dir, "bulk_t.parquet", &samples, &genes, &mat.transpose());
    let bulk = materialize(&[f], &genes, &BulkTableOpts::default()).expect("materialize");
    let v = open_vec(&bulk.paths()[0]);
    assert_eq!(v.row_names().unwrap(), genes);
    assert_eq!(v.column_names().unwrap(), samples);
}

#[test]
fn a_forced_orientation_is_honoured() {
    let dir = tempfile::tempdir().unwrap();
    let (genes, samples, mat) = fixture();
    let f = write_parquet(&dir, "bulk_t.parquet", &samples, &genes, &mat.transpose());
    let bulk = materialize(
        &[f],
        &genes,
        &BulkTableOpts {
            orientation: Some(Orientation::SamplesByGenes),
            ..Default::default()
        },
    )
    .expect("materialize");
    assert_eq!(open_vec(&bulk.paths()[0]).row_names().unwrap(), genes);
}

#[test]
fn negative_values_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (genes, samples, mut mat) = fixture();
    mat[(0, 0)] = -1.0;
    let f = write_parquet(&dir, "neg.parquet", &genes, &samples, &mat);
    let err = materialize(&[f], &genes, &BulkTableOpts::default())
        .expect_err("negative")
        .to_string();
    assert!(err.contains("negative"), "{err}");
}

#[test]
fn the_temp_backend_is_removed_on_drop() {
    let dir = tempfile::tempdir().unwrap();
    let (genes, samples, mat) = fixture();
    let f = write_parquet(&dir, "bulk.parquet", &genes, &samples, &mat);
    let bulk: BulkBackends =
        materialize(&[f], &genes, &BulkTableOpts::default()).expect("materialize");
    let path = std::path::PathBuf::from(bulk.paths()[0].as_ref());
    assert!(path.exists(), "backend exists while the guard lives");
    drop(bulk);
    assert!(!path.exists(), "backend removed when the guard drops");
}

/// Two files, two batches: names stay per file, nothing is merged here.
#[test]
fn each_bulk_file_becomes_its_own_backend() {
    let dir = tempfile::tempdir().unwrap();
    let (genes, samples, mat) = fixture();
    let a = write_parquet(&dir, "a.parquet", &genes, &samples, &mat);
    let b = write_parquet(&dir, "b.parquet", &genes, &labels(&["t0", "t1"]), &mat);
    let bulk = materialize(&[a, b], &genes, &BulkTableOpts::default()).expect("materialize");
    assert_eq!(bulk.paths().len(), 2);
    assert_eq!(
        open_vec(&bulk.paths()[1]).column_names().unwrap(),
        labels(&["t0", "t1"])
    );
}

/// `--bulk-header yes` is the way through the numeric-sample-ID blind spot,
/// and it has to reach the backend's column names.
#[test]
fn a_forced_header_names_the_samples_in_the_backend() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ids.tsv");
    std::fs::write(&path, "gene\t2019\t2020\nTGFB1\t1\t2\nCD8A\t3\t4\n").unwrap();
    let f: Box<str> = path.to_string_lossy().into_owned().into_boxed_str();
    let (genes, _, _) = fixture();
    let bulk = materialize(
        &[f],
        &genes,
        &BulkTableOpts {
            header: HeaderArg::Yes,
            ..Default::default()
        },
    )
    .expect("materialize");
    let v = open_vec(&bulk.paths()[0]);
    assert_eq!(v.column_names().unwrap(), labels(&["2019", "2020"]));
    assert_eq!(v.row_names().unwrap(), labels(&["TGFB1", "CD8A"]));
}
