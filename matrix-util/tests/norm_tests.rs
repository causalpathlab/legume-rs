use approx::assert_abs_diff_eq;
use matrix_util::traits::{MatOps, SampleOps};

#[test]
fn dmatrix_test() {
    let mut xx = nalgebra::DMatrix::<f32>::runif(100, 10);
    xx.normalize_columns_inplace();

    for j in 0..xx.ncols() {
        let norm = xx.column(j).norm();
        // Re-norming a normalized column sums 100 f32 squares and takes a
        // square root, so a couple of ULP of drift is arithmetic, not a bug.
        // The default epsilon is exactly `f32::EPSILON`, which this trips on
        // roughly one seed in ten.
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
    }
}
