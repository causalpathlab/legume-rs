//! Log-scale standardization with robust percentile clipping, plus a
//! grayscale color ramp. Shared by the marker-gene heatmap plot, the
//! per-community propensity heatmap plot, and the size-mapping path
//! for the community-colored marker plot.
//!
//! Every visual mapping (color bin *or* dot radius) runs through this
//! module so a handful of hot cells cannot blow out either the color
//! scale or the dot-size scale.
//!
//! Color mapping: `t = 0` → light gray (near background), `t = 1` →
//! near-black. "Darker = higher expression" — the user's preference.

use plot_utils::palette::Rgb;

const GRAY_HIGH: u8 = 235; // t = 0
const GRAY_LOW: u8 = 25; // t = 1

/// Grayscale color for `t` ∈ [0, 1]. Outside the range is clamped, so
/// callers don't need to pre-bound values. Higher `t` → darker.
#[must_use]
pub fn viridis_rgb(t: f32) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    let v = (GRAY_HIGH as f32 + (GRAY_LOW as f32 - GRAY_HIGH as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8;
    (v, v, v)
}

/// Grayscale color for bin `i` of `bins` (`i` in `0..bins`). Higher
/// bin → darker.
#[must_use]
pub fn viridis_bin(i: usize, bins: usize) -> Rgb {
    let denom = (bins.saturating_sub(1)).max(1) as f32;
    viridis_rgb(i as f32 / denom)
}

/// Robust lower/upper percentile of `values`. `clip` = 0.02 returns
/// (p02, p98). Returns raw min/max when `clip <= 0`. Non-finite inputs
/// are filtered.
#[must_use]
pub fn robust_range(values: &[f32], clip: f32) -> (f32, f32) {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return (0.0, 0.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let clip = clip.clamp(0.0, 0.49);
    let lo_idx = ((v.len() as f32) * clip).floor() as usize;
    let hi_idx = (((v.len() as f32) * (1.0 - clip)).ceil() as usize).saturating_sub(1);
    let hi_idx = hi_idx.min(v.len() - 1);
    (v[lo_idx], v[hi_idx])
}

/// log1p of each input, preserving NaN.
#[must_use]
pub fn log1p_vec(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&x| {
            if x.is_finite() {
                (x as f64).ln_1p() as f32
            } else {
                x
            }
        })
        .collect()
}

/// Percentile-clipped, log-scaled bucket assignment for the marker-gene
/// heatmap plot.
///
/// Returns `bucket[i]` ∈ `0..bins`, where bin 0 is dimmest. Outlier
/// cells (beyond p02 / p98 by default) are clamped to the endpoint bins
/// instead of compressing the rest of the distribution.
///
/// When the clipped range is degenerate (p_lo ≈ p_hi), everything maps
/// to the middle bin — the plot will still render, and the caller can
/// WARN the user.
#[must_use]
pub fn standardize_log_to_bins(values: &[f32], bins: usize, clip: f32) -> Vec<u8> {
    let bins = bins.max(2);
    let logged = log1p_vec(values);
    let (lo, hi) = robust_range(&logged, clip);
    let range = hi - lo;
    if range <= f32::EPSILON {
        return vec![(bins / 2) as u8; values.len()];
    }
    logged
        .into_iter()
        .map(|v| {
            if !v.is_finite() {
                return 0u8;
            }
            let t = ((v - lo) / range).clamp(0.0, 1.0);
            ((t * (bins - 1) as f32).round() as usize).min(bins - 1) as u8
        })
        .collect()
}
