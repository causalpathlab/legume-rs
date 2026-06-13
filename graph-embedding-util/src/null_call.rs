//! Empirical-Bayes "null vs signal" call for embedding-based routines.
//!
//! A model that fits embeddings (gem β_g / e_cell, bge feature/cell vectors,
//! topic loadings, …) leaves *null* items at their random init: an item the
//! model never moved has `v ~ N(0, σ²I_dof)`, so `‖v‖²/σ² ~ χ²_dof`. The old
//! per-item χ² QC asserted σ at the init value; here we **estimate** the null
//! scale σ̂² and null proportion π̂₀ from the data (ashr-style), give each item a
//! χ²_dof upper-tail p-value and a Storey/BH q-value, and call an item **live**
//! (kept) when its q ≤ target FDR — demonstrable signal above the estimated
//! null — and **null** (dropped) otherwise.
//!
//! Two entry points: [`chi2_null_call`] on a precomputed `χ²_dof` statistic
//! vector (the reusable core), and [`embedding_null_call`] for the common case
//! where the statistic is the squared norm of each row of a flat `[n × h]`
//! embedding. Per-item and deterministic (no clustering).

use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of a null call: a live/null flag per item plus the fitted null.
pub struct NullCall {
    /// Per-item: `true` = live (signal above null, keep), `false` = null (drop).
    pub live: Vec<bool>,
    /// Estimated null per-coordinate variance σ̂².
    pub sigma2: f64,
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

/// Empirical-Bayes null call on χ²-distributed statistics `s` (each `s_i`
/// assumed `~ σ²·χ²_dof` under the null) at target false-discovery rate `fdr`.
/// Estimates σ̂² (lower-quartile init, refined on the called-null set) and π̂₀
/// (Storey), then keeps items significant above that null (BH q ≤ fdr).
pub fn chi2_null_call(s: &[f64], dof: usize, fdr: f32) -> NullCall {
    let n = s.len();
    if n == 0 || dof == 0 {
        return NullCall { live: vec![false; n], sigma2: 0.0, pi0: 1.0, n_live: 0 };
    }
    let chi = ChiSquared::new(dof as f64).expect("chi-squared dof");

    // Init σ̂² from the lower quartile (null-dominated unless π₀ < 0.25):
    // q25(s) ≈ σ² · q25(χ²_dof).
    let mut sorted = s.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q25 = sorted[n / 4];
    let mut sigma2 = (q25 / chi.inverse_cdf(0.25).max(1e-9)).max(1e-12);

    let lambda = 0.5; // Storey tuning constant
    let m = n as f64;
    let mut live = vec![true; n];
    let mut pi0 = 1.0_f64;

    for _ in 0..5 {
        // Upper-tail χ²_dof p-value per item (large stat ⇒ small p ⇒ non-null).
        let p: Vec<f64> = s
            .iter()
            .map(|&si| (1.0 - chi.cdf(si / sigma2)).clamp(0.0, 1.0))
            .collect();

        // Storey π̂₀ from the null-flat tail of the p-distribution.
        let n_above = p.iter().filter(|&&x| x > lambda).count() as f64;
        pi0 = (n_above / ((1.0 - lambda) * m)).clamp(0.0, 1.0);

        // BH step-up q-values, scaled by π̂₀ (Storey).
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

        let new_live: Vec<bool> = q.iter().map(|&qg| qg <= fdr as f64).collect();
        let converged = new_live == live;

        // Refine σ̂² on the called-null set (s ≈ σ²·dof there), if it's big
        // enough to estimate; otherwise keep the quartile-based estimate.
        let (mut sum, mut cnt) = (0.0_f64, 0usize);
        for i in 0..n {
            if !new_live[i] {
                sum += s[i];
                cnt += 1;
            }
        }
        if cnt >= 50 {
            sigma2 = (sum / (cnt as f64 * dof as f64)).max(1e-12);
        }

        live = new_live;
        if converged {
            break;
        }
    }

    let n_live = live.iter().filter(|&&v| v).count();
    NullCall { live, sigma2, pi0, n_live }
}
