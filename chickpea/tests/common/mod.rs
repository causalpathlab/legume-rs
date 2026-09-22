//! Fixtures shared by the integration tests.
#![allow(dead_code)]

use chickpea::common::Mat;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;
use genomic_data::coordinates::{GeneTss, PeakCoord};

/// A gene TSS on chromosome 1.
pub fn tss(pos: i64) -> Option<GeneTss> {
    Some(GeneTss {
        chr: "1".into(),
        tss: pos,
    })
}

/// A 500-bp peak on chromosome 1.
pub fn peak(start: i64) -> Option<PeakCoord> {
    Some(PeakCoord {
        chr: "1".into(),
        start,
        end: start + 500,
    })
}

/// A smooth positive signal over `s` samples: `max(sin(0.1 j), 0) + 0.05`.
pub fn sine_signal(s: usize) -> Vec<f32> {
    (0..s)
        .map(|j| ((j as f32) * 0.1).sin().max(0.0) + 0.05)
        .collect()
}

/// A `rows × cols` matrix with `f(row, col)` entries.
pub fn mat(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Mat {
    Mat::from_fn(rows, cols, f)
}

/// A zarr-backed `SparseIoVec` of a dense `features × cells` matrix, rows
/// named `f{r}` and columns `cell_{c}`.
pub fn backend(m: &Mat) -> SparseIoVec {
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

/// `cell_0 .. cell_{n-1}`, the column names [`backend`] registers.
pub fn barcodes(n: usize) -> Vec<Box<str>> {
    (0..n).map(|c| format!("cell_{c}").into()).collect()
}
