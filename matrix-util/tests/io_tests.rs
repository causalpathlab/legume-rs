use matrix_util::common_io::create_temp_dir_file;
use matrix_util::traits::{IoOps, SampleOps};

#[test]
fn dmatrix_io_test() -> anyhow::Result<()> {
    let xx = nalgebra::DMatrix::<f32>::runif(50, 50);

    let tsv_file = create_temp_dir_file("txt.gz")?;
    xx.to_tsv(&tsv_file.to_str().unwrap())?;

    let yy = nalgebra::DMatrix::<f32>::read_file_delim(&tsv_file.to_str().unwrap(), "\t", None)?;

    approx::assert_abs_diff_eq!(xx, yy);

    Ok(())
}

#[test]
fn ndarray_io_test() -> anyhow::Result<()> {
    let xx = ndarray::Array2::<f32>::runif(50, 50);

    let tsv_file = create_temp_dir_file("txt.gz")?;
    xx.to_tsv(&tsv_file.to_str().unwrap())?;

    let yy = ndarray::Array2::<f32>::read_file_delim(&tsv_file.to_str().unwrap(), "\t", None)?;

    assert_eq!(xx, yy);

    Ok(())
}

#[test]
fn tensor_io_test() -> anyhow::Result<()> {
    let xx = candle_core::Tensor::runif(50, 50);

    let tsv_file = create_temp_dir_file("txt.gz")?;
    xx.to_tsv(&tsv_file.to_str().unwrap())?;

    let yy = candle_core::Tensor::read_file_delim(&tsv_file.to_str().unwrap(), "\t", None)?;

    assert_eq!(xx.to_vec2::<f32>()?, yy.to_vec2::<f32>()?);

    Ok(())
}
