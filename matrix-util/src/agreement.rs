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

/// Pearson correlation of `log1p(a)` against `log1p(b)`.
///
/// `log1p` because counts span orders of magnitude and a raw Pearson would be
/// decided by the few highest-expressed genes; `NaN` when either side is
/// constant, which is the honest answer rather than 0.
#[must_use]
pub fn pearson_log1p(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < 2 || a.len() != b.len() {
        return f32::NAN;
    }
    let la: Vec<f64> = a.iter().map(|v| f64::from(v.max(0.0)).ln_1p()).collect();
    let lb: Vec<f64> = b.iter().map(|v| f64::from(v.max(0.0)).ln_1p()).collect();
    pearson(&la, &lb)
}

/// Spearman: Pearson on average ranks. Ties get their mean rank, which matters
/// here because a held-out profile is mostly zeros and they are all tied.
#[must_use]
pub fn spearman(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < 2 || a.len() != b.len() {
        return f32::NAN;
    }
    pearson(&average_ranks(a), &average_ranks(b))
}

/// 1-based ranks with ties averaged.
///
/// Not public: the two correlations above are the API, and they guarantee the
/// length and finiteness this assumes.
fn average_ranks(v: &[f32]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0f64; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // Mean of the 1-based ranks this tied run spans.
        let r = (i + 1 + j) as f64 / 2.0;
        for &k in &idx[i..j] {
            out[k] = r;
        }
        i = j;
    }
    out
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
    let d = (saa * sbb).sqrt();
    if d > 0.0 {
        (sab / d) as f32
    } else {
        f32::NAN
    }
}

#[cfg(test)]
mod tests;
