//! Agreement between an observed count profile and a predicted one.
//!
//! Lives here rather than in either caller because senna scores cells and pinto
//! scores cell pairs, and a benchmark that puts the two in one table needs the
//! numbers to come from the same arithmetic — down to the tie handling, which is
//! what decides the answer on profiles that are mostly zeros.

/// Per-cell agreement between a cell's observed counts and its prediction.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellAgreement {
    pub spearman: f32,
    pub pearson_log1p: f32,
}

/// Observed counts against a predicted RATE, put on the count scale first.
///
/// The rate is renormalised over the slice it is given and multiplied by the
/// observed total, because [`pearson_log1p`] is not scale-invariant: `log1p(c·p)`
/// is not an affine function of `log1p(p)`, and a held-out profile's many zeros
/// anchor the low end. Feeding it a raw rate understates a good prediction — and
/// understates it by a different amount per engine, since each one's rate carries
/// its own arbitrary scale.
///
/// This lives here, rather than in either caller, for the same reason the
/// correlations do: senna scores cells from a linear reconstruction and pinto
/// scores pairs from a clamped log-rate, and the two numbers are only comparable
/// if the rule that puts them on a common scale is one piece of code. It was
/// written twice before, and one copy was wrong.
#[must_use]
pub fn agreement_from_rate(observed: &[f32], rate: &[f32]) -> CellAgreement {
    let count: f32 = observed.iter().sum();
    let z: f32 = rate.iter().sum::<f32>().max(1e-12);
    let scale = count / z;
    let predicted_counts: Vec<f32> = rate.iter().map(|r| r * scale).collect();
    CellAgreement {
        spearman: spearman(observed, &predicted_counts),
        pearson_log1p: pearson_log1p(observed, &predicted_counts),
    }
}

/// As [`agreement_from_rate`], for an engine that holds its rate in log space.
///
/// Shifted by the maximum before exponentiating, which is exact once the column
/// is normalised and is what keeps an unbounded logit from overflowing.
#[must_use]
pub fn agreement_from_log_rate(observed: &[f32], log_rate: &[f32]) -> CellAgreement {
    let max_logit = log_rate.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let rate: Vec<f32> = log_rate.iter().map(|l| (l - max_logit).exp()).collect();
    agreement_from_rate(observed, &rate)
}

/// Pearson correlation of `log1p(a)` against `log1p(b)`.
///
/// `log1p` because counts span orders of magnitude and a raw Pearson would be
/// decided by the few highest-expressed genes; `NaN` when either side is
/// constant, which is the honest answer rather than 0.
#[must_use]
pub fn pearson_log1p(observed: &[f32], predicted: &[f32]) -> f32 {
    if observed.len() < 2 || observed.len() != predicted.len() {
        return f32::NAN;
    }
    let log1p =
        |v: &[f32]| -> Vec<f64> { v.iter().map(|x| f64::from(x.max(0.0)).ln_1p()).collect() };
    pearson(&log1p(observed), &log1p(predicted))
}

/// Spearman: Pearson on average ranks. Ties get their mean rank, which matters
/// here because a held-out profile is mostly zeros and they are all tied.
#[must_use]
pub fn spearman(observed: &[f32], predicted: &[f32]) -> f32 {
    if observed.len() < 2 || observed.len() != predicted.len() {
        return f32::NAN;
    }
    pearson(&average_ranks(observed), &average_ranks(predicted))
}

/// 1-based ranks with ties averaged.
///
/// Not public: the two correlations above are the API, and they guarantee the
/// length and finiteness this assumes.
fn average_ranks(values: &[f32]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&i, &j| {
        values[i]
            .partial_cmp(&values[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0f64; values.len()];
    let mut run_start = 0;
    while run_start < order.len() {
        let mut run_end = run_start + 1;
        while run_end < order.len() && values[order[run_end]] == values[order[run_start]] {
            run_end += 1;
        }
        // Mean of the 1-based ranks this tied run spans.
        let shared = (run_start + 1 + run_end) as f64 / 2.0;
        for &position in &order[run_start..run_end] {
            ranks[position] = shared;
        }
        run_start = run_end;
    }
    ranks
}

/// Pearson on two equal-length slices; `NaN` when either is constant.
fn pearson(a: &[f64], b: &[f64]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "pearson: length mismatch");
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    // `denom`, not `d` — `d` is the feature-axis size everywhere else in this
    // workspace, and a reader should not have to check which one this is.
    let denom = (saa * sbb).sqrt();
    if denom > 0.0 {
        (sab / denom) as f32
    } else {
        f32::NAN
    }
}

#[cfg(test)]
mod tests;
