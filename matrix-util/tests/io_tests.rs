use matrix_util::common_io::create_temp_dir_file;
use matrix_util::traits::{IoOps, SampleOps};

#[test]
fn dmatrix_io_test() -> anyhow::Result<()> {
    let xx = nalgebra::DMatrix::<f32>::runif(50, 50);

    let tsv_file = create_temp_dir_file("txt.gz")?;
    xx.to_tsv(tsv_file.to_str().unwrap())?;

    let yy = nalgebra::DMatrix::<f32>::read_file_delim(tsv_file.to_str().unwrap(), "\t", None)?;

    approx::assert_abs_diff_eq!(xx, yy);

    let parquet_file = create_temp_dir_file("parquet")?;
    xx.to_parquet(parquet_file.to_str().unwrap())?;

    Ok(())
}

#[test]
fn ndarray_io_test() -> anyhow::Result<()> {
    let xx = ndarray::Array2::<f32>::runif(50, 50);

    let tsv_file = create_temp_dir_file("txt.gz")?;
    xx.to_tsv(tsv_file.to_str().unwrap())?;

    let yy = ndarray::Array2::<f32>::read_file_delim(tsv_file.to_str().unwrap(), "\t", None)?;

    assert_eq!(xx, yy);

    let parquet_file = create_temp_dir_file("parquet")?;
    xx.to_parquet(parquet_file.to_str().unwrap())?;

    Ok(())
}

#[test]
fn tensor_io_test() -> anyhow::Result<()> {
    let xx = candle_core::Tensor::runif(50, 50);

    let tsv_file = create_temp_dir_file("txt.gz")?;
    xx.to_tsv(tsv_file.to_str().unwrap())?;

    let yy = candle_core::Tensor::read_file_delim(tsv_file.to_str().unwrap(), "\t", None)?;

    assert_eq!(xx.to_vec2::<f32>()?, yy.to_vec2::<f32>()?);

    let parquet_file = create_temp_dir_file("parquet")?;
    xx.to_parquet(parquet_file.to_str().unwrap())?;

    Ok(())
}

/// The three delimited-reader contracts hardened together: a blank line is
/// ignored rather than capping the width check, explicit indices override a
/// name selection instead of unioning with it, and an R-style header (one
/// field short of the data rows) names the data column to its right.
#[test]
fn delimited_reader_blank_lines_override_and_r_header() -> anyhow::Result<()> {
    use std::io::Write;

    // R write.table shape: header names only the DATA columns; each data row
    // leads with a row label. Plus a stray blank line.
    let f = create_temp_dir_file("csv")?;
    {
        let mut w = std::fs::File::create(&f)?;
        writeln!(w, "a,b,c")?;
        writeln!(w, "r1,1.0,2.0,3.0")?;
        writeln!(w)?;
        writeln!(w, "r2,4.0,5.0,6.0")?;
    }
    let path = f.to_str().unwrap();

    // Name selection under the shifted header: 'b' must yield column values
    // (2, 5) — the data column right of header position 1 — not (1, 4).
    let got = nalgebra::DMatrix::<f32>::read_data(
        path,
        ",",
        Some(0),
        Some(0),
        None,
        Some(&["b".into()]),
    )?;
    assert_eq!(got.rows, vec!["r1".into(), "r2".into()] as Vec<Box<str>>);
    assert_eq!(got.cols, vec!["b".into()] as Vec<Box<str>>);
    assert_eq!(got.mat.as_slice(), &[2.0, 5.0]);

    // Indices override names: asking for index 1 AND name 'c' must read only
    // index 1, one column, not a union of the two selectors.
    let got = nalgebra::DMatrix::<f32>::read_data(
        path,
        ",",
        Some(0),
        Some(0),
        Some(&[1usize]),
        Some(&["c".into()]),
    )?;
    assert_eq!(got.mat.ncols(), 1, "indices must override names");
    assert_eq!(got.mat.as_slice(), &[1.0, 4.0]);

    // The blank line must neither shrink the width check (index 3 is valid)
    // nor produce a phantom row.
    let got =
        nalgebra::DMatrix::<f32>::read_data(path, ",", Some(0), Some(0), Some(&[3usize]), None)?;
    assert_eq!(got.mat.nrows(), 2, "blank line must not become a row");
    assert_eq!(got.mat.as_slice(), &[3.0, 6.0]);
    Ok(())
}
