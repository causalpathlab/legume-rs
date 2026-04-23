//! Build the FDR-sparse Q matrix from restandardized ES + q-values.

use crate::Mat;

/// Zero-out entries with q ≥ alpha, then row-softmax over survivors with
/// temperature `tau`. Rows with no survivors stay all-zero (the cell
/// projection will produce 0 posterior there).
pub fn build_q_matrix(
    es_restandardized: &Mat,
    q_values: &Mat,
    alpha: f32,
    temperature: f32,
) -> Mat {
    let k = es_restandardized.nrows();
    let c = es_restandardized.ncols();
    let mut q_mat = Mat::zeros(k, c);

    let beta = 1.0 / temperature.max(1e-6);

    for kk in 0..k {
        // Collect surviving entries.
        let mut values: Vec<(usize, f32)> = Vec::new();
        let mut max_val = f32::NEG_INFINITY;
        for cc in 0..c {
            if q_values[(kk, cc)] < alpha && es_restandardized[(kk, cc)] > 0.0 {
                let v = es_restandardized[(kk, cc)] * beta;
                values.push((cc, v));
                if v > max_val {
                    max_val = v;
                }
            }
        }
        if values.is_empty() {
            continue;
        }
        let mut total = 0.0f32;
        for (_, v) in values.iter_mut() {
            *v = (*v - max_val).exp();
            total += *v;
        }
        if total <= 0.0 {
            continue;
        }
        for (cc, v) in values {
            q_mat[(kk, cc)] = v / total;
        }
    }
    q_mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn survivors_sum_to_one_per_row() {
        let es = Mat::from_row_slice(1, 3, &[3.0, 1.0, 2.0]);
        let q = Mat::from_row_slice(1, 3, &[0.01, 0.02, 0.5]); // cc=2 filtered
        let qm = build_q_matrix(&es, &q, 0.1, 1.0);
        let row_sum: f32 = (0..3).map(|c| qm[(0, c)]).sum();
        assert!((row_sum - 1.0).abs() < 1e-5);
        assert_eq!(qm[(0, 2)], 0.0);
    }

    #[test]
    fn no_survivors_yields_zero_row() {
        let es = Mat::from_row_slice(1, 2, &[1.0, 2.0]);
        let q = Mat::from_row_slice(1, 2, &[0.5, 0.9]);
        let qm = build_q_matrix(&es, &q, 0.1, 1.0);
        assert_eq!(qm[(0, 0)], 0.0);
        assert_eq!(qm[(0, 1)], 0.0);
    }

    #[test]
    fn negative_es_is_filtered_even_if_significant() {
        let es = Mat::from_row_slice(1, 2, &[-3.0, 1.0]);
        let q = Mat::from_row_slice(1, 2, &[0.001, 0.001]);
        let qm = build_q_matrix(&es, &q, 0.05, 1.0);
        assert_eq!(qm[(0, 0)], 0.0);
        assert!((qm[(0, 1)] - 1.0).abs() < 1e-5);
    }
}
