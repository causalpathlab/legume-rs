//! Empirical-Bayes "null vs signal" call for embedding-based routines.
//!
//! A model that fits embeddings (gem `β_g` / `e_cell`, bge feature/cell vectors,
//! topic loadings, …) leaves *null* items at their random init: an item the
//! model never moved has `v ~ N(0, σ²I)`, so `‖v‖²` follows a scaled χ². The
//! old per-item χ² QC asserted σ at the init value *and* the nominal dof `h`;
//! both are wrong in practice. `AdamW` + weight decay shrink the null items
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
//! The **feature-null QC** (senna bge / faba gem) is [`ash_null_call`]: it works
//! per-coordinate rather than on the collapsed norm, so it reads legitimate
//! non-dominant structure (e.g. cross-donor variance) that the single-`(σ²,ν)`
//! norm test masks — it calibrates a per-axis null scale from the n-hvg
//! presumed-null and calls each feature via ash lfsr. The scaled-χ² machinery
//! remains as [`chi2_null_call`] — an empirical-Bayes call on a precomputed
//! scaled-χ² vector (the reusable core), used on the per-gene LRT by the held-out
//! feature-projection gate in [`crate::fit::feature_projection`] and by the
//! data-driven feature selection. The χ² call is per-item and deterministic; the
//! ash call runs a seeded collapsed-Gibbs sampler per axis.

use crate::ash::{ash_normal, AshOpts};
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};

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

/// Row `i` of a flat row-major embedding (`[· × h]`) **when it carries a usable
/// embedding** — in range and not all-zero. `None` for a dead row.
///
/// An exactly-all-zero row is this crate's in-band "no usable embedding" signal: a
/// held-out feature that fails its null call is *zeroed* rather than handed a
/// fabricated direction (see [`crate::fit::feature_projection`]), and `faba gem`
/// records which in `gene_qc.parquet`. Consumers must read it as **missing data, not
/// an observation of zero** — averaging it in would drag the mean toward the origin.
///
/// This is an invariant, not a heuristic: a row the model actually trained is never
/// exactly zero (SGD from a random init in `f32`, and there is no sparsity penalty on
/// β anywhere), so the only all-zero rows are the deliberately-zeroed ones.
#[must_use]
pub fn live_row(rows: &[f32], i: usize, h: usize) -> Option<&[f32]> {
    let row = rows.get(i * h..(i + 1) * h)?;
    row.iter().any(|&x| x != 0.0).then_some(row)
}

/// Empirical-Bayes null call on scaled-χ² statistics `s` (each `s_i` assumed
/// `~ σ²·χ²_ν` under the null, with `ν` an *effective* dof `≤ dof`) at target
/// false-discovery rate `fdr`. `dof` is the nominal dof (the upper bound /
/// independent-coordinate count). Fits the null `(σ̂², ν̂)` from the lower
/// quantiles of `s` — null-pure because signal only inflates the upper tail,
/// and decoupled from the significance call so there is no σ̂²-shrinks-the-call
/// feedback loop — then keeps items significant above that null (Storey π̂₀ +
/// BH q ≤ fdr).
#[must_use]
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
    let live: Vec<bool> = q.iter().map(|&qg| qg <= f64::from(fdr)).collect();
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

// ────────────────────────────────────────────────────────────────────────────
// Adaptive-shrinkage (ash) feature-null call with an n-hvg-guided null
// ────────────────────────────────────────────────────────────────────────────
//
// [`chi2_null_call`] fits its null to the *distribution of the collapsed norm*
// — a scalar that has already mixed the shared-mode variance and the
// per-feature variance — so under a dominant legitimate-variance axis it
// calibrates to that axis and calls ~everything null. This call instead runs an
// **empirical-null ash per embedding dimension** (see [`crate::ash`]):
//
//   1. treats the signed loadings `v_{i,d}` on each axis `d` as the observations
//      (ash is a normal-means model, so signed effects are its natural input);
//   2. fits ash `g = π₀δ₀ + Σπ_k N(0,σ_k²)` with an EMPIRICAL null — the common
//      per-axis noise variance `s_d²` is estimated jointly by the collapsed
//      Gibbs sampler, seeded from the presumed-null (bottom `n − n_hvg` by norm),
//      so the anisotropy is absorbed axis-by-axis with no external scale rule;
//   3. reads each axis's local false-sign rate `lfsr_{i,d}` (Rao-Blackwellised);
//   4. calls a feature live when it is confidently non-null on ANY axis —
//      `lfsr_i = min(1, h·min_d lfsr_{i,d})` (Bonferroni over the h looks) ≤ fdr.
//
// The h axes are independent samplers, run in parallel. No eigenbasis, no
// drop-k, no parametric norm null: anisotropy is handled by the per-axis `s_d`,
// the decision by ash's lfsr.

/// Adaptive-shrinkage feature-null call on a flat `[n × h]` embedding: one
/// empirical-null ash ([`ash_normal`], collapsed Gibbs) per embedding dimension,
/// each seeded from the presumed-null (bottom `n − n_hvg` features by norm; `0`
/// or `≥ n` ⇒ all features). Calls a feature live when it is confidently
/// non-null on any axis (ash lfsr, Bonferroni over the `h` coordinates) at FDR
/// `fdr`. See the module note above.
#[must_use]
pub fn ash_null_call(rows: &[f32], n: usize, h: usize, fdr: f32, n_hvg: usize) -> NullCall {
    if n == 0 || h == 0 {
        return NullCall {
            live: vec![false; n],
            sigma2: 0.0,
            eff_dof: 0.0,
            pi0: 1.0,
            n_live: 0,
        };
    }

    // Presumed-null set = the bottom `n − n_hvg` features by norm (the empirical
    // left mode of the bimodal norm distribution — the features the model never
    // moved). Seeds each axis's null-SD init; the Gibbs sampler refines it.
    let presumed_null: Vec<usize> = {
        let norm2: Vec<f64> = (0..n)
            .map(|i| {
                rows[i * h..(i + 1) * h]
                    .iter()
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum()
            })
            .collect();
        let n_pn = if n_hvg == 0 || n_hvg >= n {
            n
        } else {
            n - n_hvg
        };
        let mut order: Vec<usize> = (0..n).collect();
        if n_pn < n {
            // Partition (O(n)) so `order[..n_pn]` are the smallest-norm features.
            order.select_nth_unstable_by(n_pn, |&a, &b| {
                norm2[a]
                    .partial_cmp(&norm2[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        order.truncate(n_pn);
        order
    };

    // One empirical-null ash per axis, in parallel; each returns per-feature lfsr
    // on that axis plus the posterior-mean null variance `s_d²` and π̂₀.
    let opts = AshOpts::default();
    let per_axis: Vec<(Vec<f64>, f64, f64)> = (0..h)
        .into_par_iter()
        .map(|d| {
            let x_d: Vec<f64> = (0..n).map(|i| f64::from(rows[i * h + d])).collect();
            // Seed the null SD from the presumed-null RMS on this axis.
            let ss: f64 = presumed_null.iter().map(|&i| x_d[i] * x_d[i]).sum::<f64>()
                / presumed_null.len().max(1) as f64;
            let se_init = ss.sqrt().max(1e-6);
            let axis_opts = AshOpts {
                seed: opts.seed ^ (d as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                ..opts
            };
            let res = ash_normal(&x_d, se_init, &axis_opts);
            (res.lfdr, res.null_var, res.pi0)
        })
        .collect();

    // Per-feature: live if confidently non-null on any axis (Bonferroni lfdr).
    let hf = h as f64;
    let live: Vec<bool> = (0..n)
        .map(|i| {
            let min_lfdr = per_axis
                .iter()
                .map(|(lfdr, ..)| lfdr[i])
                .fold(1.0f64, f64::min);
            (hf * min_lfdr).min(1.0) <= f64::from(fdr)
        })
        .collect();
    let n_live = live.iter().filter(|&&v| v).count();
    let sigma2 = per_axis.iter().map(|(_, s2, _)| s2).sum::<f64>() / hf;
    let pi0 = per_axis.iter().map(|(_, _, p)| p).sum::<f64>() / hf;
    NullCall {
        live,
        sigma2,
        eff_dof: hf,
        pi0,
        n_live,
    }
}
