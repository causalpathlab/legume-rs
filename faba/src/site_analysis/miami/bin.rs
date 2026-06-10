//! Shared genomic binning for the Miami figure and the ASCII pileup.
//!
//! [`BinEdges`] fixes a `[min_pos, max_pos]` extent over `num_bins` equal
//! columns. Every track (matrix epi signal, BAM read depth, site markers)
//! maps positions through the SAME edges so the stacked tracks line up.
//! The mapping rule is the historical pileup rule (`pos_to_col` /
//! `bin_positions_with_extent` in `pileup.rs`), lifted here so both the
//! ASCII path and the figure share one definition.

/// A fixed binning grid over an inclusive genomic extent.
#[derive(Clone, Copy, Debug)]
pub struct BinEdges {
    pub min_pos: i64,
    pub max_pos: i64,
    pub num_bins: usize,
}

impl BinEdges {
    pub fn new(min_pos: i64, max_pos: i64, num_bins: usize) -> Self {
        Self {
            min_pos,
            max_pos,
            num_bins: num_bins.max(1),
        }
    }

    /// Genomic span in bp (at least 1, so a degenerate single-position
    /// extent doesn't divide by zero).
    #[inline]
    pub fn span(&self) -> u64 {
        (self.max_pos - self.min_pos).max(1) as u64
    }

    /// Map a coordinate to its bin column. Positions at or below `min_pos`
    /// land in column 0; `max_pos` lands in the last column.
    #[inline]
    pub fn col_of(&self, pos: i64) -> usize {
        let rel = (pos - self.min_pos).max(0) as u64;
        (rel * self.num_bins as u64 / self.span()).min(self.num_bins as u64 - 1) as usize
    }

    /// Bin a position/value list into `num_bins` summed columns. Positions
    /// outside `[min_pos, max_pos]` are dropped. When `log_transform`,
    /// each bin becomes `log10(1 + sum)`.
    pub fn bin(&self, positions: &[(i64, f64)], log_transform: bool) -> Vec<f64> {
        let mut bins = vec![0.0f64; self.num_bins];
        for &(pos, val) in positions {
            if pos < self.min_pos || pos > self.max_pos {
                continue;
            }
            bins[self.col_of(pos)] += val;
        }
        if log_transform {
            for v in bins.iter_mut() {
                *v = (1.0 + *v).log10();
            }
        }
        bins
    }

    /// Pixel x of a genomic position within `[x_left, x_left + plot_w]`,
    /// clamped to the plot band. Shared by every track + the gene model so
    /// they align horizontally.
    #[inline]
    pub fn x_px(&self, pos: i64, x_left: f32, plot_w: f32) -> f32 {
        let frac = (pos - self.min_pos) as f32 / self.span() as f32;
        x_left + frac.clamp(0.0, 1.0) * plot_w
    }
}

/// 99th-percentile of positive values — a spike-robust scale so one
/// outlier bin doesn't flatten every panel. Mirrors
/// `senna .../strand/place.rs::robust_max`: quickselect, O(n).
pub fn robust_max(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.filter(|x| *x > 0.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    let idx = (((v.len() - 1) as f64) * 0.99).round() as usize;
    *v.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).unwrap())
        .1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn col_of_endpoints_and_midpoint() {
        let e = BinEdges::new(100, 200, 10);
        // first site -> first column, last site -> last column
        assert_eq!(e.col_of(100), 0);
        assert_eq!(e.col_of(200), 9);
        assert_eq!(e.col_of(150), 5);
        // out-of-range low clamps to col 0
        assert_eq!(e.col_of(50), 0);
        // degenerate single-position extent collapses to column 0
        assert_eq!(BinEdges::new(100, 100, 10).col_of(100), 0);
    }

    #[test]
    fn bin_conserves_mass_without_log() {
        let e = BinEdges::new(0, 99, 10);
        let pts = [(0, 1.0), (5, 2.0), (50, 3.0), (99, 4.0)];
        let bins = e.bin(&pts, false);
        let total: f64 = bins.iter().sum();
        assert!((total - 10.0).abs() < 1e-9, "got {total}");
        // positions outside the extent are dropped
        let bins2 = e.bin(&[(0, 1.0), (1000, 9.0)], false);
        assert!((bins2.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn x_px_spans_plot_band() {
        let e = BinEdges::new(100, 200, 10);
        assert!((e.x_px(100, 10.0, 80.0) - 10.0).abs() < 1e-4);
        assert!((e.x_px(200, 10.0, 80.0) - 90.0).abs() < 1e-4);
        assert!((e.x_px(150, 10.0, 80.0) - 50.0).abs() < 1e-4);
    }

    #[test]
    fn robust_max_ignores_zeros() {
        assert_eq!(robust_max([0.0, 0.0].into_iter()), 0.0);
        let m = robust_max([1.0, 2.0, 3.0, 4.0].into_iter());
        assert!(m >= 3.0);
    }
}
