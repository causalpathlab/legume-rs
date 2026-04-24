//! Log-scale standardization with robust percentile clipping, plus a
//! 32-stop viridis LUT. Shared by the marker-gene heatmap plot, the
//! per-community propensity heatmap plot, and the size-mapping path
//! for the community-colored marker plot.
//!
//! Every visual mapping (color bin *or* dot radius) runs through this
//! module so a handful of hot cells cannot blow out either the color
//! scale or the dot-size scale — the user's outlier concern made
//! concrete.

use plot_utils::palette::Rgb;

/// 32-stop viridis-like sequential palette. Hand-picked so callers get
/// reproducible output without pulling in a palette crate. Monotone in
/// perceived lightness; safe for print.
const VIRIDIS32: [Rgb; 32] = [
    (68, 1, 84),
    (71, 13, 96),
    (72, 24, 106),
    (72, 35, 116),
    (71, 46, 124),
    (69, 56, 130),
    (66, 65, 134),
    (62, 74, 137),
    (59, 82, 139),
    (55, 91, 141),
    (52, 99, 141),
    (49, 107, 142),
    (46, 115, 142),
    (43, 123, 142),
    (40, 131, 142),
    (37, 139, 141),
    (34, 147, 140),
    (32, 155, 137),
    (33, 163, 134),
    (42, 171, 129),
    (57, 179, 122),
    (78, 186, 113),
    (103, 193, 102),
    (131, 199, 90),
    (161, 205, 76),
    (194, 209, 66),
    (224, 210, 56),
    (236, 220, 58),
    (244, 232, 65),
    (250, 243, 76),
    (253, 250, 100),
    (253, 253, 130),
];

/// Pick the viridis stop closest to `t` ∈ [0, 1]. Outside the range is
/// clamped, so callers don't need to pre-bound values.
#[must_use]
pub fn viridis_rgb(t: f32) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    let n = VIRIDIS32.len() as f32;
    let idx = ((t * (n - 1.0)).round() as usize).min(VIRIDIS32.len() - 1);
    VIRIDIS32[idx]
}

/// Viridis color for bin `i` of `bins` (`i` in `0..bins`).
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

/// Map expression (log1p + percentile-clipped) to per-point radii for
/// the community-colored marker plot.
///
/// `base_size` maps to expression ≤ p_lo; `base_size * scale` maps to
/// expression ≥ p_hi. Zero-expression cells get `base_size * 0.5`
/// (deliberately smaller than the low bin so they don't look like a
/// marker). The output has the same length as `values`.
#[must_use]
pub fn log_expr_to_radii(values: &[f32], base_size: f32, scale: f32, clip: f32) -> Vec<f32> {
    let logged = log1p_vec(values);
    let (lo, hi) = robust_range(&logged, clip);
    let range = (hi - lo).max(f32::EPSILON);
    let max_size = base_size * scale.max(1.0);
    logged
        .into_iter()
        .map(|v| {
            if !v.is_finite() || v <= 0.0 {
                base_size * 0.5
            } else {
                let t = ((v - lo) / range).clamp(0.0, 1.0);
                base_size + (max_size - base_size) * t
            }
        })
        .collect()
}

/// Map propensity values ∈ [0, 1] to per-point radii. No log step; the
/// raw propensity IS the intensity. `base_size` maps to prop = 0,
/// `base_size * scale` maps to prop = 1.
#[must_use]
pub fn prop_to_radii(values: &[f32], base_size: f32, scale: f32) -> Vec<f32> {
    let max_size = base_size * scale.max(1.0);
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                base_size
            } else {
                let t = v.clamp(0.0, 1.0);
                base_size + (max_size - base_size) * t
            }
        })
        .collect()
}
