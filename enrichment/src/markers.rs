//! Marker-matrix helpers. Consumers do their own parsing (gene-name matching
//! lives in their own crates); this module provides the pure-math transforms
//! applied to a prebuilt G × C membership matrix.

use crate::Mat;

/// Reweight binary membership in place using TF-IDF: `w_g = ln(C / c_g)`
/// where `c_g` is the number of celltypes claiming gene `g`. Genes shared by
/// all celltypes receive weight 0. Returns `ln(C)` — the maximum possible
/// weight — for logging.
pub fn apply_idf_weights(mat: &mut Mat) -> f32 {
    let n_genes = mat.nrows();
    let n_ct = mat.ncols();
    let c_total = n_ct as f32;
    for g in 0..n_genes {
        let c_g = mat.row(g).iter().filter(|&&v| v > 0.0).count() as f32;
        if c_g == 0.0 {
            continue;
        }
        let w = (c_total / c_g).ln();
        for c in 0..n_ct {
            if mat[(g, c)] > 0.0 {
                mat[(g, c)] = w;
            }
        }
    }
    c_total.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idf_zeros_ubiquitous_genes() {
        let mut m = Mat::from_row_slice(
            3,
            3,
            &[
                1.0, 1.0, 1.0, // shared by all 3 celltypes → weight 0
                1.0, 0.0, 0.0, // unique to celltype 0 → max weight
                1.0, 1.0, 0.0, // in 2 of 3 → moderate weight
            ],
        );
        let max = apply_idf_weights(&mut m);
        assert!((max - (3.0f32).ln()).abs() < 1e-6);
        assert_eq!(m[(0, 0)], 0.0);
        assert_eq!(m[(0, 1)], 0.0);
        assert!(m[(1, 0)] > m[(2, 0)]);
    }

    #[test]
    fn idf_preserves_zero_entries() {
        let mut m = Mat::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        apply_idf_weights(&mut m);
        assert_eq!(m[(0, 1)], 0.0);
        assert_eq!(m[(1, 0)], 0.0);
    }
}
