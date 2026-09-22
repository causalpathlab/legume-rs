//! Per-cell embedding onto the frozen gene / peak dictionaries.

mod common;

use chickpea::common::Mat;
use chickpea::p2g::cells::{cluster_labels_to_pb, embed_cells, write_cell_parquet, FrozenAxis};
use common::mat;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::candle::candle_core::Device;
use std::path::Path;

fn backend(m: &Mat) -> SparseIoVec {
    let mut b = create_sparse_from_dmatrix(m, None, Some(&SparseIoBackend::Zarr)).unwrap();
    b.register_row_names_vec(
        &(0..m.nrows())
            .map(|r| format!("f{r}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..m.ncols())
            .map(|c| format!("cell_{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).unwrap();
    v
}

/// A dictionary that puts program-A features along +e0 and program-B along +e1.
fn rows(n: usize, split: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|f| {
            let mut r = vec![0f32; dim];
            r[usize::from(f >= split)] = 1.0;
            r[2] = 0.01 * f as f32;
            r
        })
        .collect()
}

/// 12 cells: 0..6 express the first `split` features, 6..12 the rest.
fn counts(n_feat: usize, split: usize) -> Mat {
    mat(n_feat, 12, |f, c| {
        let a = c < 6;
        if (f < split) == a {
            8.0 + (c % 3) as f32
        } else {
            1.0
        }
    })
}

fn barcodes(n: usize) -> Vec<Box<str>> {
    (0..n).map(|c| format!("cell_{c}").into()).collect()
}

#[test]
fn cells_land_on_their_programs_side_and_the_parquet_has_one_row_per_barcode() {
    let dim = 4;
    let rna = counts(4, 2);
    let atac = counts(6, 3);
    let gene_rows = rows(4, 2, dim);
    let peak_rows = rows(6, 3, dim);
    let (gb, pb) = (vec![0f32; 4], vec![0f32; 6]);
    let axes = [
        FrozenAxis {
            label: "gene",
            rows: &gene_rows,
            bias: &gb,
        },
        FrozenAxis {
            label: "peak",
            rows: &peak_rows,
            bias: &pb,
        },
    ];
    let (rb, ab) = (backend(&rna), backend(&atac));
    let out = embed_cells(&axes, &[&rb, &ab], barcodes(12), dim, &Device::Cpu).unwrap();
    assert_eq!((out.theta.nrows(), out.theta.ncols()), (12, dim));
    for c in 0..12 {
        let (own, other) = if c < 6 { (0, 1) } else { (1, 0) };
        assert!(out.theta[(c, own)] > out.theta[(c, other)], "cell {c}");
    }
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    write_cell_parquet(&prefix, &out).unwrap();
    assert!(Path::new(&format!("{prefix}.cell_embedding.parquet")).is_file());
}

#[test]
fn a_single_axis_run_embeds_on_that_axis_alone() {
    let dim = 4;
    let atac = counts(6, 3);
    let peak_rows = rows(6, 3, dim);
    let pb = vec![0f32; 6];
    let axes = [FrozenAxis {
        label: "peak",
        rows: &peak_rows,
        bias: &pb,
    }];
    let ab = backend(&atac);
    let out = embed_cells(&axes, &[&ab], barcodes(12), dim, &Device::Cpu).unwrap();
    for c in 0..12 {
        let (own, other) = if c < 6 { (0, 1) } else { (1, 0) };
        assert!(out.theta[(c, own)] > out.theta[(c, other)], "cell {c}");
    }
}

#[test]
fn a_dictionary_that_does_not_match_the_backend_is_refused() {
    let dim = 4;
    let atac = counts(6, 3);
    let peak_rows = rows(5, 3, dim);
    let pb = vec![0f32; 5];
    let axes = [FrozenAxis {
        label: "peak",
        rows: &peak_rows,
        bias: &pb,
    }];
    let ab = backend(&atac);
    let err = embed_cells(&axes, &[&ab], barcodes(12), dim, &Device::Cpu)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("peak"), "{err}");
}

#[test]
fn pb_labels_are_the_majority_of_their_cells() {
    // pb 0 has cells {0,1,2} labeled 1,1,0; pb 1 has {3,4} labeled None,2; pb 2 has no labeled cell.
    let cell_label = vec![Some(1), Some(1), Some(0), None, Some(2), None];
    let cell_to_pb = vec![0, 0, 0, 1, 1, 2];
    assert_eq!(
        cluster_labels_to_pb(&cell_label, &cell_to_pb, 3),
        vec![Some(1), Some(2), None]
    );
}

#[test]
fn a_tie_goes_to_the_lowest_cluster_id() {
    let cell_label = vec![Some(3), Some(1)];
    let cell_to_pb = vec![0, 0];
    assert_eq!(
        cluster_labels_to_pb(&cell_label, &cell_to_pb, 1),
        vec![Some(1)]
    );
}
