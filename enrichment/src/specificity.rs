//! Specificity transformations on a G × K group profile.

use crate::Mat;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecificityMode {
    /// No transform — rank genes by raw β per topic column. Relies on
    /// training-time NB-Fisher weighting (senna topic) to have suppressed
    /// housekeeping in β. Default for topic kinds because adding a
    /// second housekeeping adjustment can over-suppress genuinely
    /// informative high-mean genes.
    Raw,
    /// Simplex row-normalization: `β_g,k / Σ_k' β_g,k'`. A second
    /// housekeeping adjustment on top of the training-time NB weighting.
    /// Opt-in via `--specificity simplex`.
    Simplex,
    /// Absolute simplex: `|β_g,k| / Σ_k' |β_g,k'|`. For signed profiles
    /// (SVD kinds).
    Abs,
}

/// Compute per-gene per-group specificity. Output shape matches input
/// (G × K). Rows with zero mass are left as zero rows (no NaN).
pub fn compute_specificity(profile_gk: &Mat, mode: SpecificityMode) -> Mat {
    let g = profile_gk.nrows();
    let k = profile_gk.ncols();
    let mut out = Mat::zeros(g, k);

    match mode {
        SpecificityMode::Raw => {
            out.copy_from(profile_gk);
        }
        SpecificityMode::Simplex => {
            for gi in 0..g {
                let s: f32 = (0..k).map(|kj| profile_gk[(gi, kj)].max(0.0)).sum();
                if s <= 0.0 {
                    continue;
                }
                for kj in 0..k {
                    out[(gi, kj)] = profile_gk[(gi, kj)].max(0.0) / s;
                }
            }
        }
        SpecificityMode::Abs => {
            for gi in 0..g {
                let s: f32 = (0..k).map(|kj| profile_gk[(gi, kj)].abs()).sum();
                if s <= 0.0 {
                    continue;
                }
                for kj in 0..k {
                    out[(gi, kj)] = profile_gk[(gi, kj)].abs() / s;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_sums_to_one_per_row() {
        let profile = Mat::from_row_slice(2, 3, &[1.0, 3.0, 6.0, 0.5, 0.25, 0.25]);
        let s = compute_specificity(&profile, SpecificityMode::Simplex);
        for g in 0..profile.nrows() {
            let row_sum: f32 = (0..profile.ncols()).map(|k| s[(g, k)]).sum();
            assert!((row_sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn abs_handles_signed() {
        let profile = Mat::from_row_slice(1, 3, &[-1.0, 2.0, -3.0]);
        let s = compute_specificity(&profile, SpecificityMode::Abs);
        let row_sum: f32 = (0..3).map(|k| s[(0, k)]).sum();
        assert!((row_sum - 1.0).abs() < 1e-5);
        // all positive after abs
        for k in 0..3 {
            assert!(s[(0, k)] >= 0.0);
        }
    }

    #[test]
    fn zero_row_stays_zero() {
        let profile = Mat::zeros(1, 3);
        let s = compute_specificity(&profile, SpecificityMode::Simplex);
        for k in 0..3 {
            assert_eq!(s[(0, k)], 0.0);
        }
    }
}
