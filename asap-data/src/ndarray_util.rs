use ndarray::prelude::*;

#[allow(dead_code)]
/// column-wise standardization
/// * `xraw`: (D, N) matrix
pub fn scale_columns(mut xraw: Array2<f32>) -> anyhow::Result<Array2<f32>> {
    let mu = xraw
        .mean_axis(Axis(0))
        .ok_or(anyhow::anyhow!("mean failed"))?;
    let sig = xraw.std_axis(Axis(0), 0.0);
    for j in 0..xraw.ncols() {
        xraw.column_mut(j).mapv_inplace(|x| (x - mu[j]) / sig[j]);
    }
    Ok(xraw)
}
