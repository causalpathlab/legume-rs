//! Sample-level inference shared across the workspace: FDR adjustment, a
//! nonparametric bootstrap interval, and a permutation p-value.
//!
//! These operate on a plain `&[f32]` of per-unit statistics — one p-value per
//! test, one observation per resample — and are deliberately separate from the
//! matrix-shaped statistics modules ([`crate::ndarray_stat`],
//! [`crate::sparse_stat`]), which accumulate sufficient statistics over an axis
//! rather than testing a hypothesis about a sample.
//!
//! They live here because their callers span crates that do not otherwise know
//! about each other: the trajectory and association steps, the enrichment
//! pipeline, and per-droplet quality control all need the same BH adjustment,
//! and it existed as two independent copies before this module.

/// Benjamini-Hochberg FDR adjustment. Returns q-values in the input order:
/// `q_i = min_{j >= i_sorted} (m * p_j / j)`, a cumulative minimum walked from
/// the largest p downwards and clamped to `[0, 1]`.
///
/// BH controls the FDR only under independence or positive regression
/// dependence (PRDS), so check that the tests are of that kind before reaching
/// for it. Units that do not share observations — branches, genes, droplets —
/// qualify.
///
/// Tests over overlapping evidence do NOT. Where neighbouring units are covered
/// by the same reads, and a read supporting one is evidence against its
/// neighbour, the dependence is not even reliably positive. The valid procedure
/// under arbitrary dependence is Benjamini-Yekutieli, whose `sum(1/i) ~ ln m`
/// penalty is an order of magnitude at tens of thousands of units; a caller in
/// that position should select on a marginal p-value and claim no FDR
/// guarantee, rather than one whose assumption fails.
///
/// `NaN` p-values sort as equal rather than panicking, so a caller that lets one
/// through gets a usable vector instead of an abort.
#[must_use]
pub fn benjamini_hochberg(pvalues: &[f32]) -> Vec<f32> {
    let m = pvalues.len();
    if m == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        pvalues[a]
            .partial_cmp(&pvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut q = vec![0.0f32; m];
    let mut running_min = 1.0f32;
    // Walk from largest p to smallest, enforcing monotonic non-decreasing q.
    for rank in (0..m).rev() {
        let idx = order[rank];
        let adj = (pvalues[idx] * (m as f32) / ((rank + 1) as f32)).clamp(0.0, 1.0);
        running_min = running_min.min(adj);
        q[idx] = running_min;
    }
    q
}

/// Mean of a slice (`NaN` if empty).
#[must_use]
pub fn mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        f32::NAN
    } else {
        x.iter().sum::<f32>() / x.len() as f32
    }
}

/// Nonparametric bootstrap of the sample **mean**: resample `x` with replacement `n_boot`
/// times and return `(standard_error, ci_lo, ci_hi)` at confidence level `1 − alpha`
/// (percentile interval). `NaN`s when `x` or `n_boot` is empty.
pub fn bootstrap_mean_ci(
    x: &[f32],
    n_boot: usize,
    alpha: f64,
    rng: &mut impl rand::RngExt,
) -> (f32, f32, f32) {
    let n = x.len();
    if n == 0 || n_boot == 0 {
        return (f32::NAN, f32::NAN, f32::NAN);
    }
    let mut means: Vec<f32> = (0..n_boot)
        .map(|_| {
            let mut s = 0f32;
            for _ in 0..n {
                s += x[rng.random_range(0..n)];
            }
            s / n as f32
        })
        .collect();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = mean(&means);
    let var = if n_boot < 2 {
        0.0
    } else {
        means.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / (n_boot - 1) as f32
    };
    let se = var.max(0.0).sqrt();
    let lo = means[(((alpha / 2.0) * n_boot as f64) as usize).min(n_boot - 1)];
    let hi = means[(((1.0 - alpha / 2.0) * n_boot as f64) as usize).min(n_boot - 1)];
    (se, lo, hi)
}

/// Two-sided **sign-flip** permutation p-value for H0: `E[x] = 0`. Each draw flips the
/// sign of every element independently (fixing magnitudes, testing only the mean's
/// direction), add-one corrected. `NaN` when `x` is empty.
pub fn sign_flip_pvalue(x: &[f32], n_perm: usize, rng: &mut impl rand::RngExt) -> f32 {
    let n = x.len();
    if n == 0 {
        return f32::NAN;
    }
    let obs = mean(x).abs();
    let mut ge = 0usize;
    for _ in 0..n_perm {
        let mut s = 0f32;
        for &xi in x {
            if rng.random::<bool>() {
                s += xi;
            } else {
                s -= xi;
            }
        }
        if (s / n as f32).abs() >= obs {
            ge += 1;
        }
    }
    (1.0 + ge as f32) / (1.0 + n_perm as f32)
}

#[cfg(test)]
#[path = "hypothesis/tests.rs"]
mod tests;
