//! `si.tl.discretize`: log-normalized nonzero values → expression levels.
//!
//! SIMBA bins the nonzero log-normalized values of ALL genes (the step runs
//! before the HVG subset): a [`HIST_BINS`]-bin histogram over `[min, max]`, a
//! weighted k-means on the histogram centroids with the counts as weights,
//! and bin edges at the midpoints between adjacent sorted centres, padded by
//! `(max − min) / (bins · 10)` on the outside. A nonzero value's level is then
//! `np.digitize(v, bin_edges)`, i.e. `1..=n_levels`.
//!
//! The k-means here is solved exactly (dynamic programming over the sorted
//! centroids, `O(k · n²)` on `n ≤ 100` points), which is what sklearn's seeded
//! local search finds on this small 1-D problem — with no RNG to carry.

use super::SCALE_FACTOR;

/// `ln1p(SCALE_FACTOR · x / lib_size)`: `si.pp.normalize` + `si.pp.log_transform`.
pub(crate) fn log_norm(x: f32, lib_size: f64) -> f32 {
    (SCALE_FACTOR * f64::from(x) / lib_size).ln_1p() as f32
}

/// `np.histogram(values, bins=n)` over a fixed `[lo, hi]`: equal-width bins,
/// half-open on the right except the last one, which includes `hi`.
#[derive(Clone, Debug)]
pub struct Histogram {
    lo: f64,
    hi: f64,
    counts: Vec<u64>,
}

impl Histogram {
    pub fn new(lo: f64, hi: f64, n_bins: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(n_bins > 0, "histogram: need at least one bin");
        anyhow::ensure!(
            lo.is_finite() && hi.is_finite() && hi > lo,
            "histogram: degenerate value range [{lo}, {hi}]"
        );
        Ok(Self {
            lo,
            hi,
            counts: vec![0; n_bins],
        })
    }

    pub fn add(&mut self, v: f64) {
        let nb = self.counts.len();
        let f = (v - self.lo) / (self.hi - self.lo) * nb as f64;
        let idx = (f.floor().max(0.0) as usize).min(nb - 1);
        self.counts[idx] += 1;
    }

    #[must_use]
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    #[must_use]
    pub fn range(&self) -> (f64, f64) {
        (self.lo, self.hi)
    }

    /// Bin midpoints, `(edges[i] + edges[i+1]) / 2`.
    #[must_use]
    pub fn centroids(&self) -> Vec<f64> {
        let nb = self.counts.len();
        let width = (self.hi - self.lo) / nb as f64;
        (0..nb)
            .map(|i| self.lo + (i as f64 + 0.5) * width)
            .collect()
    }
}

/// Exact weighted 1-D k-means: the global minimizer of
/// `Σ_i w_i (x_i − c_{a(i)})²` over assignments `a` into `k` clusters,
/// returned as the sorted cluster centres (weighted means). Optimal 1-D
/// clusters are contiguous in sorted order, so a dynamic program over the
/// sorted points is exact. Zero-weight points are ignored; `k` is capped at
/// the number of weighted points.
pub(crate) fn weighted_kmeans_1d(x: &[f64], w: &[f64], k: usize) -> Vec<f64> {
    let mut pts: Vec<(f64, f64)> = x
        .iter()
        .zip(w)
        .filter(|(_, &w)| w > 0.0)
        .map(|(&x, &w)| (x, w))
        .collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = pts.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    // Prefix sums of w, w·x, w·x² so any contiguous block's weighted SSE is O(1).
    let mut sw = vec![0.0; n + 1];
    let mut swx = vec![0.0; n + 1];
    let mut swxx = vec![0.0; n + 1];
    for (i, &(x, w)) in pts.iter().enumerate() {
        sw[i + 1] = sw[i] + w;
        swx[i + 1] = swx[i] + w * x;
        swxx[i + 1] = swxx[i] + w * x * x;
    }
    // Weighted SSE of points i..=j.
    let cost = |i: usize, j: usize| -> f64 {
        let w = sw[j + 1] - sw[i];
        let wx = swx[j + 1] - swx[i];
        let wxx = swxx[j + 1] - swxx[i];
        (wxx - wx * wx / w).max(0.0)
    };
    // d[q][j]: best cost of splitting points 0..=j into q+1 clusters;
    // arg[q][j]: where the last cluster starts.
    let mut d = vec![vec![f64::INFINITY; n]; k];
    let mut arg = vec![vec![0usize; n]; k];
    for (j, dj) in d[0].iter_mut().enumerate() {
        *dj = cost(0, j);
    }
    for q in 1..k {
        for j in q..n {
            for i in q..=j {
                let c = d[q - 1][i - 1] + cost(i, j);
                if c < d[q][j] {
                    d[q][j] = c;
                    arg[q][j] = i;
                }
            }
        }
    }
    // Backtrack the cluster starts, then take each cluster's weighted mean.
    let mut starts = vec![0usize; k];
    let mut j = n - 1;
    for q in (1..k).rev() {
        starts[q] = arg[q][j];
        j = starts[q] - 1;
    }
    (0..k)
        .map(|q| {
            let i = starts[q];
            let j = if q + 1 < k { starts[q + 1] - 1 } else { n - 1 };
            (swx[j + 1] - swx[i]) / (sw[j + 1] - sw[i])
        })
        .collect()
}

/// The fitted discretization: `adata.uns['disc']` plus the k-means centres.
#[derive(Clone, Debug)]
pub struct Discretization {
    /// `[min, max]` of the histogram (`hist_edges[0]`, `hist_edges[-1]`).
    pub hist_range: (f64, f64),
    /// `hist_count`: values per histogram bin.
    pub hist_counts: Vec<u64>,
    /// Sorted k-means centres, one per level.
    pub centers: Vec<f64>,
    /// `bin_edges`: `n_levels + 1` edges, padded outside the histogram range.
    pub bin_edges: Vec<f64>,
}

impl Discretization {
    pub fn fit(hist: &Histogram, n_bins: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(n_bins > 0, "discretize: need at least one level");
        let centroids = hist.centroids();
        let weights: Vec<f64> = hist.counts().iter().map(|&c| c as f64).collect();
        let nonempty = weights.iter().filter(|&&w| w > 0.0).count();
        anyhow::ensure!(nonempty > 0, "discretize: the histogram is empty");
        let centers = weighted_kmeans_1d(&centroids, &weights, n_bins.min(nonempty));
        let (lo, hi) = hist.range();
        let padding = (hi - lo) / (hist.counts().len() as f64 * 10.0);
        let mut bin_edges = Vec::with_capacity(centers.len() + 1);
        bin_edges.push(lo - padding);
        bin_edges.extend(centers.windows(2).map(|c| 0.5 * (c[0] + c[1])));
        bin_edges.push(hi + padding);
        Ok(Self {
            hist_range: (lo, hi),
            hist_counts: hist.counts().to_vec(),
            centers,
            bin_edges,
        })
    }

    /// `np.digitize(v, bin_edges)`: the number of edges `≤ v`, clamped to
    /// `1..=n_levels` (every value seen by the histogram lies strictly inside
    /// the padded edges, so the clamp only guards float round-off).
    #[must_use]
    pub fn level(&self, v: f32) -> u8 {
        let v = f64::from(v);
        let i = self.bin_edges.partition_point(|e| *e <= v);
        i.clamp(1, self.n_levels()) as u8
    }

    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.centers.len()
    }
}

#[cfg(test)]
#[path = "discretize_tests.rs"]
mod discretize_tests;
