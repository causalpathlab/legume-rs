//! `{prefix}.cells.parquet` burden loading.

use cnv::clone_bayes::load_burden_from_cells_table;
use legume_numeric::matrix::parquet::{write_named_table, Column};

#[test]
fn missing_cells_take_the_fallback_burden() {
    let path = std::env::temp_dir().join("cnv_cells_table_fallback.parquet");
    let path = path.to_str().unwrap();
    let cells: Vec<Box<str>> = ["a", "b"].map(Into::into).to_vec();
    write_named_table(
        path,
        "cell",
        &cells,
        &[
            ("depth".into(), Column::F32(&[10.0, 20.0])),
            ("cnv_burden".into(), Column::F32(&[0.1, 0.2])),
        ],
    )
    .unwrap();

    let names: Vec<Box<str>> = ["b", "zzz", "a"].map(Into::into).to_vec();
    let fallback = [9.0f32, 0.5, 9.0];
    let got = load_burden_from_cells_table(path, &names, &fallback)
        .unwrap()
        .expect("table exists");
    assert!((got[0] - 0.2).abs() < 1e-6);
    assert!((got[1] - 0.5).abs() < 1e-6, "missing cell uses fallback");
    assert!((got[2] - 0.1).abs() < 1e-6);
    let _ = std::fs::remove_file(path);
}

#[test]
fn absent_table_is_none() {
    let names: Vec<Box<str>> = ["a"].map(Into::into).to_vec();
    let got = load_burden_from_cells_table("/nonexistent/x.cells.parquet", &names, &[0.0]).unwrap();
    assert!(got.is_none());
}
