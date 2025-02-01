use approx::assert_abs_diff_eq;
use matrix_util::traits::{MatOps, SampleOps};

#[test]
fn dmatrix_test() {
    let mut xx = nalgebra::DMatrix::<f32>::runif(100, 10);
    xx.normalize_columns_inplace();

    for j in 0..xx.ncols() {
        let norm = xx.column(j).norm();
        assert_abs_diff_eq!(norm, 1.0);
    }
}
