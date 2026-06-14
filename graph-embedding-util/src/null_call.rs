//! Empirical-Bayes "null vs signal" call for embedding-based routines.
//!
//! A model that fits embeddings (gem β_g / e_cell, bge feature/cell vectors,
//! topic loadings, …) leaves *null* items at their random init: an item the
//! model never moved has `v ~ N(0, σ²I)`, so `‖v‖²` follows a scaled χ². The
//! old per-item χ² QC asserted σ at the init value *and* the nominal dof `h`;
//! both are wrong in practice. AdamW + weight decay shrink the null items
//! below init (so σ must be **estimated**), and the `h` embedding coordinates
//! are **correlated**, so a null item's `‖v‖²` is over-dispersed relative to
//! `χ²_h` — it behaves like `σ²·χ²_ν` with an *effective* dof `ν ≪ h`
//! (Satterthwaite: `ν = (Σλ_i)² / Σλ_i²` for the coordinate-covariance
//! eigenvalues `λ_i`). Forcing `ν = h` makes the upper-tail p-values
//! anti-conservative and over-calls "live".
//!
//! So we fit the null `σ²·χ²_ν` — **both** the scale `σ̂²` and the effective
//! dof `ν̂` — from the *lower* quantiles of the statistic (signal only inflates
//! the upper tail, so the lower tail is null-pure), decoupled from the
//! significance call to avoid a feedback loop. Then each item gets a `χ²_ν`
//! upper-tail p-value, a Storey π̂₀, a BH q-value, and is called **live** (kept)
//! when `q ≤ fdr` — demonstrable signal above the estimated null — else
//! **null** (dropped). This is the ashr spirit (flexible null, point mass at
//! 0, π̂₀ from the data) made cheap for the collapsed-norm statistic, where the
//! per-coordinate standard errors ashr would use are folded into `ν̂`.
//!
//! Two entry points: [`chi2_null_call`] on a precomputed scaled-χ² statistic
//! vector (the reusable core), and [`embedding_null_call`] for the common case
//! where the statistic is the squared norm of each row of a flat `[n × h]`
//! embedding. Per-item and deterministic (no clustering, no RNG).

use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

/// Result of a null call: a live/null flag per item plus the fitted null.
pub struct NullCall {
    /// Per-item: `true` = live (signal above null, keep), `false` = null (drop).
    pub live: Vec<bool>,
    /// Estimated null per-coordinate variance σ̂².
    pub sigma2: f64,
    /// Estimated null effective dof ν̂ (≤ the nominal dof; smaller ⇒ more
    /// correlated coordinates ⇒ more over-dispersed null).
    pub eff_dof: f64,
    /// Estimated null proportion π̂₀ (Storey).
    pub pi0: f64,
    /// Number of live items.
    pub n_live: usize,
}

/// Squared-norm null call on a flat row-major embedding `rows` (`[n × h]`):
/// computes `s_i = ‖row_i‖²` and defers to [`chi2_null_call`] with `dof = h`.
pub fn embedding_null_call(rows: &[f32], n: usize, h: usize, fdr: f32) -> NullCall {
    let s: Vec<f64> = (0..n)
        .map(|i| {
            rows[i * h..(i + 1) * h]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum()
        })
        .collect();
    chi2_null_call(&s, h, fdr)
}

/// Empirical-Bayes null call on scaled-χ² statistics `s` (each `s_i` assumed
/// `~ σ²·χ²_ν` under the null, with `ν` an *effective* dof `≤ dof`) at target
/// false-discovery rate `fdr`. `dof` is the nominal dof (the upper bound /
/// independent-coordinate count). Fits the null `(σ̂², ν̂)` from the lower
/// quantiles of `s` — null-pure because signal only inflates the upper tail,
/// and decoupled from the significance call so there is no σ̂²-shrinks-the-call
/// feedback loop — then keeps items significant above that null (Storey π̂₀ +
/// BH q ≤ fdr).
pub fn chi2_null_call(s: &[f64], dof: usize, fdr: f32) -> NullCall {
    let n = s.len();
    if n == 0 || dof == 0 {
        return NullCall {
            live: vec![false; n],
            sigma2: 0.0,
            eff_dof: dof as f64,
            pi0: 1.0,
            n_live: 0,
        };
    }
    let dof_max = dof as f64;

    let mut sorted = s.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = |p: f64| -> f64 {
        let idx = ((p * n as f64) as usize).min(n - 1);
        sorted[idx]
    };

    // Pick two *low* probabilities that sit inside the null bulk. Signal only
    // inflates `s`, so the lower tail is null-dominated whenever π₀ exceeds the
    // upper probability. A rough Storey π̂₀ (nominal `χ²_dof`, lower-quartile
    // scale) places the fit quantiles below the null mass; cap at 0.40 so they
    // stay clear of the signal even when π₀ is moderate.
    let nominal = ChiSquared::new(dof_max).expect("chi-squared dof");
    let s2_rough = (quantile(0.25) / nominal.inverse_cdf(0.25).max(1e-9)).max(1e-12);
    let lambda = 0.5;
    let m = n as f64;
    let storey_pi0 = |s2: f64, chi: &ChiSquared| -> f64 {
        let n_above = s
            .iter()
            .filter(|&&si| 1.0 - chi.cdf(si / s2) > lambda)
            .count() as f64;
        (n_above / ((1.0 - lambda) * m)).clamp(1e-3, 1.0)
    };
    let mut pi0 = storey_pi0(s2_rough, &nominal);
    let p_hi = (0.5 * pi0).clamp(0.15, 0.40);
    let p_lo = 0.5 * p_hi;
    let (q_lo, q_hi) = (quantile(p_lo).max(1e-12), quantile(p_hi).max(1e-12));
    let r = q_hi / q_lo;

    // Fit the null `(σ̂², ν̂)` from the two null-bulk quantiles. The p-th *mixture*
    // quantile is the `(p/π₀)`-th *null* quantile, so map both probabilities
    // into null-space before matching: ν̂ from the (scale-free) quantile ratio,
    // σ̂² from the lower quantile. π₀ feeds the map and is itself refined from
    // the fitted null — a few iterations on the global proportion (not the
    // per-item call) converge without the σ̂²-shrinks-the-call feedback loop.
    let mut eff_dof = dof_max;
    let mut sigma2 = s2_rough;
    let mut chi = ChiSquared::new(eff_dof).expect("effective dof");
    for _ in 0..4 {
        let a_lo = (p_lo / pi0).min(0.95);
        let a_hi = (p_hi / pi0).min(0.98).max(a_lo + 1e-3);
        eff_dof = solve_eff_dof(r, a_lo, a_hi, dof_max);
        chi = ChiSquared::new(eff_dof).expect("effective dof");
        sigma2 = (q_lo / chi.inverse_cdf(a_lo).max(1e-9)).max(1e-12);
        pi0 = storey_pi0(sigma2, &chi);
    }

    let _ = pi0; // the fit's π₀ fed the quantile map; finish_call re-derives it
    finish_call(s, &chi, sigma2, eff_dof, fdr)
}

/// Result of an [`embedding_lower_tail_call`]: which items are the lower-tail
/// "empty / collapsed" minority (to drop) plus the fitted dominant-mode null.
pub struct LowerTailCall {
    /// Per-item: `true` = lower-tail outlier (empty/ambient — drop).
    pub drop: Vec<bool>,
    /// Null (dominant-mode) location on the `log` statistic.
    pub mu: f64,
    /// Null (dominant-mode) scale on the `log` statistic (robust MAD).
    pub sigma: f64,
    /// Number of dropped (empty) items.
    pub n_drop: usize,
}

/// EB "is this an empty / collapsed item?" call on a per-item embedding **norm**
/// (e.g. faba gem's pre-L2 `cell_nrms`). Mirror-image of [`chi2_null_call`]:
/// there the null is the dominant *low* bulk and signal is the *upper* tail;
/// here the **dominant population is the real/kept mode** and the *empties are
/// the lower tail* — their MAP projection collapsed to ≈0, so `log(norm)` sits
/// far below the real-cell mode.
///
/// The null is the real mode on `x = log(norm)`, fit with **median μ̂ +
/// 1.4826·MAD σ̂** — both 50%-breakdown, so the empty minority *and* any
/// heavy upper (doublet/large) tail cannot bias the null scale (the failure
/// mode of a peak-curvature or upper-half fit, which underestimate the real
/// mode's true spread and over-drop). Each item gets a **lower-tail** p-value
/// `Φ((x−μ̂)/σ̂)`, conservative BH q-values (`π₀ = 1` — Storey is degenerate for
/// a lower-tail test against a symmetric null, and conservative suits the
/// asymmetric cost of dropping a real item), and is **dropped** when `q ≤ fdr`
/// *and* it lies below the mode. Robust, automatic (no bounds), deterministic.
pub fn embedding_lower_tail_call(nrm: &[f32], fdr: f32) -> LowerTailCall {
    let n = nrm.len();
    if n == 0 {
        return LowerTailCall {
            drop: vec![],
            mu: 0.0,
            sigma: 0.0,
            n_drop: 0,
        };
    }
    let median = |v: &[f64]| -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = s.len();
        if m % 2 == 1 {
            s[m / 2]
        } else {
            0.5 * (s[m / 2 - 1] + s[m / 2])
        }
    };
    // log statistic (floor tiny/zero norms so the log is finite).
    let x: Vec<f64> = nrm.iter().map(|&v| (v.max(1e-6) as f64).ln()).collect();
    let mu = median(&x);
    let dev: Vec<f64> = x.iter().map(|&v| (v - mu).abs()).collect();
    let sigma = (1.4826 * median(&dev)).max(1e-9);
    let normal = Normal::new(mu, sigma).expect("normal");
    // Lower-tail p under the real-mode null: small for empties (far below μ̂).
    let p: Vec<f64> = x.iter().map(|&xi| normal.cdf(xi).clamp(0.0, 1.0)).collect();
    // Conservative BH (π₀ = 1) on the lower-tail p-values.
    let m = n as f64;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut q = vec![1.0_f64; n];
    let mut running = 1.0_f64;
    for rank in (0..n).rev() {
        let gi = order[rank];
        let raw = (m * p[gi] / (rank as f64 + 1.0)).min(1.0);
        running = running.min(raw);
        q[gi] = running;
    }
    // Drop = empty: significant on the lower side only.
    let drop: Vec<bool> = (0..n).map(|i| q[i] <= fdr as f64 && x[i] < mu).collect();
    let n_drop = drop.iter().filter(|&&v| v).count();
    LowerTailCall {
        drop,
        mu,
        sigma,
        n_drop,
    }
}

/// Given a fixed null `σ²·χ²_ν` (via `chi`), compute upper-tail p-values on
/// `s`, a Storey π̂₀, BH step-up q-values, and the live/null flags at `fdr`.
/// Used by [`chi2_null_call`] (null fitted from `s`).
fn finish_call(s: &[f64], chi: &ChiSquared, sigma2: f64, eff_dof: f64, fdr: f32) -> NullCall {
    let n = s.len();
    let m = n as f64;
    let lambda = 0.5;
    // Upper-tail χ²_ν p-value per item (large stat ⇒ small p ⇒ non-null).
    let p: Vec<f64> = s
        .iter()
        .map(|&si| (1.0 - chi.cdf(si / sigma2)).clamp(0.0, 1.0))
        .collect();
    // Storey π̂₀ from the null-flat upper tail of the p-distribution.
    let n_above = p.iter().filter(|&&x| x > lambda).count() as f64;
    let pi0 = (n_above / ((1.0 - lambda) * m)).clamp(1e-3, 1.0);
    // BH step-up q-values, scaled by π̂₀.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut q = vec![1.0_f64; n];
    let mut running = 1.0_f64;
    for rank in (0..n).rev() {
        let gi = order[rank];
        let raw = (pi0 * m * p[gi] / (rank as f64 + 1.0)).min(1.0);
        running = running.min(raw);
        q[gi] = running;
    }
    let live: Vec<bool> = q.iter().map(|&qg| qg <= fdr as f64).collect();
    let n_live = live.iter().filter(|&&v| v).count();
    NullCall {
        live,
        sigma2,
        eff_dof,
        pi0,
        n_live,
    }
}

/// Solve `χ²_ν⁻¹(a_hi)/χ²_ν⁻¹(a_lo) = r` for the effective dof `ν` by bisection,
/// where `a_lo < a_hi` are the (null-space) probabilities of the two matched
/// quantiles. The ratio is monotone *decreasing* in `ν` (more dof ⇒ tighter
/// quantiles), so a larger observed ratio `r` (more over-dispersion) maps to a
/// smaller `ν`. Clamped to `[0.2, dof_max]`: `r` at/below the `dof_max` ratio ⇒
/// no detectable over-dispersion, return `dof_max`.
fn solve_eff_dof(r: f64, a_lo: f64, a_hi: f64, dof_max: f64) -> f64 {
    let ratio = |nu: f64| -> f64 {
        let c = ChiSquared::new(nu).expect("chi-squared nu");
        c.inverse_cdf(a_hi) / c.inverse_cdf(a_lo).max(1e-12)
    };
    let (mut lo, mut hi) = (0.2_f64, dof_max.max(0.3));
    // Decreasing in ν: ratio(hi) is the min, ratio(lo) the max achievable.
    if r <= ratio(hi) {
        return hi;
    }
    if r >= ratio(lo) {
        return lo;
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if ratio(mid) > r {
            lo = mid; // need more dof to tighten the ratio
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}
