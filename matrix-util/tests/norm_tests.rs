use approx::assert_abs_diff_eq;
use matrix_util::traits::MatOps;

#[test]
fn dmatrix_test() {
    use matrix_util::*;

    let mut xx = dmatrix_util::runif(100, 10);
    xx.normalize_columns_inplace();

    for j in 0..xx.ncols() {
        let norm = xx.column(j).norm();
        assert_abs_diff_eq!(norm, 1.0);
    }
}
