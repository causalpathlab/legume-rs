use super::*;
use matrix_util::dmatrix_io::DMatrix as IoDMatrix;
use matrix_util::traits::IoOps;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::<str>::from(*s)).collect()
}

fn read_back(path: &str) -> (Vec<String>, Vec<Vec<f32>>) {
    let m = IoDMatrix::<f32>::from_parquet(path).unwrap();
    let rows: Vec<String> = m.rows.iter().map(|s| s.to_string()).collect();
    let vals = (0..m.mat.nrows())
        .map(|i| (0..m.mat.ncols()).map(|j| m.mat[(i, j)]).collect())
        .collect();
    (rows, vals)
}

/// The QC filter must subset ROWS and BARCODES together.
///
/// This is the whole failure mode: if the value rows are filtered but the names
/// are not (or the two use different index sets), every downstream table is
/// silently mislabelled — cell A's embedding carrying cell B's barcode. Nothing
/// errors, and every join afterwards is wrong. A NON-CONTIGUOUS keep set is used
/// deliberately, because a contiguous prefix would pass even under a plain
/// truncation.
#[test]
fn qc_keep_subsets_rows_and_barcodes_together() {
    let dir = std::env::temp_dir().join("faba_gemenc_qc_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("kept.parquet");
    let path = path.to_str().unwrap();

    // 4 cells x 2 columns; each row is identifiable from its values.
    let data: Vec<f32> = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0];
    let cells = names(&["cellA", "cellB", "cellC", "cellD"]);

    // Keep B and D — non-contiguous, and not a prefix.
    write_cell_table(&data, 4, 2, path, &cells, "V", Some(&[1, 3])).unwrap();
    let (rows, vals) = read_back(path);
    assert_eq!(
        rows,
        vec!["cellB", "cellD"],
        "barcodes must follow the keep set"
    );
    assert_eq!(
        vals,
        vec![vec![20.0, 21.0], vec![40.0, 41.0]],
        "rows must follow the same set"
    );

    // No keep set = unchanged, the --no-qc path.
    write_cell_table(&data, 4, 2, path, &cells, "V", None).unwrap();
    let (rows, vals) = read_back(path);
    assert_eq!(rows.len(), 4, "without a keep set every cell is written");
    assert_eq!(vals[0], vec![10.0, 11.0]);
    assert_eq!(vals[3], vec![40.0, 41.0]);

    let _ = std::fs::remove_file(path);
}

/// An out-of-range keep index must fail loudly rather than panic on a slice or,
/// worse, read a neighbouring cell's values.
#[test]
fn qc_keep_index_out_of_range_is_an_error() {
    let dir = std::env::temp_dir().join("faba_gemenc_qc_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.parquet");
    let path = path.to_str().unwrap();

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let cells = names(&["a", "b"]);
    let err = write_cell_table(&data, 2, 2, path, &cells, "V", Some(&[0, 5]))
        .expect_err("an index past the cell count must be rejected");
    assert!(err.to_string().contains("out of range"), "got: {err}");
    let _ = std::fs::remove_file(path);
}
