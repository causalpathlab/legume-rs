use statrs::function::gamma::ln_gamma;

/// Upper-tail beta-binomial p-value for single-sample editing detection.
///
/// Models the alt-allele (edited) read count `k` out of `n` ref+alt reads at a
/// site as `k ~ BetaBinomial(n, α, β)` under the null that alt reads are
/// sequencing noise: mean error `eps = α/(α+β)`, overdispersion
/// `rho = 1/(α+β+1)` (the beta-binomial intra-site correlation). Returns
/// `P(K >= k)` — small when observed editing exceeds what noise at rate `eps`
/// (allowing for overdispersion `rho`) would produce. This is the JACUSA2
/// call-1 / SAILOR-style single-condition test: no control sample required.
///
/// `rho <= 0` reduces to the plain `Binomial(n, eps)` tail. The upper tail is
/// summed directly via an online (streaming) log-sum-exp — no allocation, and
/// no `1 - lower_tail` complement (which underflows to 0 for very significant
/// sites).
pub fn betabinom_pvalue_greater(k: u64, n: u64, eps: f64, rho: f64) -> f32 {
    if n == 0 || k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    let eps = eps.clamp(1e-9, 1.0 - 1e-9);
    let lgn1 = ln_gamma((n + 1) as f64);
    let ln_choose = |i: u64| lgn1 - ln_gamma((i + 1) as f64) - ln_gamma((n - i + 1) as f64);

    // Sum the upper tail P(K>=k) directly. Each emission log-pmf is built once
    // per mode (binomial limit vs beta-binomial) with all i-independent terms
    // hoisted, then folded by `upper_tail_p` with no intermediate buffer.
    let p = if rho <= 0.0 {
        let (le, l1) = (eps.ln(), (1.0 - eps).ln());
        upper_tail_p(k, n, |i| {
            ln_choose(i) + (i as f64) * le + ((n - i) as f64) * l1
        })
    } else {
        let rho = rho.min(1.0 - 1e-9);
        let s = (1.0 - rho) / rho; // α + β (precision)
        let alpha = eps * s;
        let beta = (1.0 - eps) * s;
        let lnb_ab = ln_gamma(alpha) + ln_gamma(beta) - ln_gamma(alpha + beta);
        let lg_denom = ln_gamma(n as f64 + alpha + beta);
        upper_tail_p(k, n, |i| {
            ln_choose(i) + ln_gamma(i as f64 + alpha) + ln_gamma((n - i) as f64 + beta)
                - lg_denom
                - lnb_ab
        })
    };
    p.clamp(0.0, 1.0) as f32
}

/// Sum `exp(log_pmf(i))` over `i in k..=n` via an online log-sum-exp, returning
/// the probability (not log). No heap allocation.
fn upper_tail_p(k: u64, n: u64, log_pmf: impl Fn(u64) -> f64) -> f64 {
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    for i in k..=n {
        let x = log_pmf(i);
        if x > max {
            sum = sum * (max - x).exp() + 1.0;
            max = x;
        } else {
            sum += (x - max).exp();
        }
    }
    if max == f64::NEG_INFINITY {
        0.0
    } else {
        (max + sum.ln()).exp()
    }
}

/// Benjamini-Hochberg FDR adjustment. Returns q-values in the input order.
pub fn benjamini_hochberg(pvalues: &[f32]) -> Vec<f32> {
    let m = pvalues.len();
    if m == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| pvalues[a].partial_cmp(&pvalues[b]).unwrap());
    let mut q = vec![0.0f32; m];
    let mut running_min = 1.0f32;
    // Walk from largest p to smallest, enforcing monotonic non-decreasing q.
    for rank in (0..m).rev() {
        let idx = order[rank];
        let adj = pvalues[idx] * (m as f32) / ((rank + 1) as f32);
        running_min = running_min.min(adj);
        q[idx] = running_min.min(1.0);
    }
    q
}

/// `ln C(n, k)` via log-gamma.
fn ln_choose(n: u64, k: u64) -> f64 {
    ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64)
}

/// One-sided (upper-tail) Fisher exact test on the 2×2 conversion table
///
/// ```text
///           converted   unconverted
///   WT      a_w         u_w
///   MUT     a_m         u_m
/// ```
///
/// Returns `P(WT converted count ≥ a_w)` conditioning on all four margins
/// (the hypergeometric right tail). Small p ⇒ WT editing exceeds the MUT
/// control. A genomic C/T variant has equal rates in both arms ⇒ p ≈ 1. This
/// is the exact small-count branch of the DART m6A two-sample call; it makes
/// no distributional assumption, so it is the right tool when any cell is tiny.
pub fn fisher_exact_greater(a_w: u64, u_w: u64, a_m: u64, u_m: u64) -> f32 {
    let r1 = a_w + u_w; // WT row total
    let r2 = a_m + u_m; // MUT row total
    let c1 = a_w + a_m; // converted column total
    let n = r1 + r2;
    if r1 == 0 || r2 == 0 || c1 == 0 || c1 == n {
        return 1.0; // degenerate margin: the table carries no information
    }
    let c2 = n - c1; // unconverted column total
    let ln_denom = ln_choose(n, r1);
    let hi = r1.min(c1); // largest WT-converted count compatible with the margins

    // Streaming log-sum-exp over a' = a_w ..= hi (tables at least as WT-extreme).
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    for a in a_w..=hi {
        let b = r1 - a; // WT unconverted under this table
        if b > c2 {
            continue; // infeasible (not enough unconverted reads to fill the row)
        }
        let lp = ln_choose(c1, a) + ln_choose(c2, b) - ln_denom;
        if lp > max {
            sum = sum * (max - lp).exp() + 1.0;
            max = lp;
        } else {
            sum += (lp - max).exp();
        }
    }
    if max == f64::NEG_INFINITY {
        return 1.0;
    }
    ((max + sum.ln()).exp() as f32).clamp(0.0, 1.0)
}

/// Beta-binomial log-likelihood of `a` edited of `n` at rate `p` and dispersion
/// `s = (1-ρ)/ρ`, **dropping** the `ln C(n, a)` term (it cancels in the LRT
/// since `(a, n)` are shared between the full and null fits).
fn betabinom_loglik_kernel(a: u64, n: u64, p: f64, s: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    let alpha = p * s;
    let beta = (1.0 - p) * s;
    let lbeta = |x: f64, y: f64| ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y);
    lbeta(a as f64 + alpha, (n - a) as f64 + beta) - lbeta(alpha, beta)
}

/// One-sided beta-binomial likelihood-ratio test that the WT conversion rate
/// exceeds the MUT (control) rate, sharing a global overdispersion `rho`.
///
/// `a_W ~ BetaBinom(n_W, p_W, ρ)`, `a_M ~ BetaBinom(n_M, p_M, ρ)`; H₀: `p_W = p_M`.
/// `D = 2[ℓ(p̂_W) + ℓ(p̂_M) − ℓ(p̂₀) − ℓ(p̂₀)]`, `p̂₀` the pooled rate; the one-sided
/// p-value is `½·P(χ²₁ ≥ D)` when `p̂_W > p̂_M`, else `1`. Unlike Fisher, the
/// overdispersion `ρ` prevents a high-coverage variant with a stable allelic
/// ratio from being called significant by sheer read count — this is the
/// large-count branch of the contrast.
pub fn betabinom_lrt_greater(a_w: u64, n_w: u64, a_m: u64, n_m: u64, rho: f64) -> f32 {
    if n_w == 0 || n_m == 0 {
        return 1.0;
    }
    let pw = a_w as f64 / n_w as f64;
    let pm = a_m as f64 / n_m as f64;
    if pw <= pm {
        return 1.0; // one-sided: nothing to call when WT is not above MUT
    }
    let rho = rho.clamp(1e-6, 1.0 - 1e-6);
    let s = (1.0 - rho) / rho;
    let p0 = (a_w + a_m) as f64 / (n_w + n_m) as f64;
    let d = 2.0
        * (betabinom_loglik_kernel(a_w, n_w, pw, s) + betabinom_loglik_kernel(a_m, n_m, pm, s)
            - betabinom_loglik_kernel(a_w, n_w, p0, s)
            - betabinom_loglik_kernel(a_m, n_m, p0, s));
    let d = d.max(0.0);
    // Half-χ²₁ one-sided tail. For 1 dof the survival function is closed form:
    // P(χ²₁ ≥ d) = erfc(√(d/2)), avoiding a ChiSquared object + incomplete-gamma.
    let p = 0.5 * statrs::function::erf::erfc((d / 2.0).sqrt());
    (p as f32).clamp(0.0, 1.0)
}

/// Two-sample WT-vs-MUT conversion p-value for DART m6A. Dispatches to the
/// exact [`fisher_exact_greater`] when any cell is small (or total coverage is
/// low), and to the overdispersed [`betabinom_lrt_greater`] otherwise (where an
/// exact test would over-call on high-coverage variants). `rho` is the
/// beta-binomial overdispersion of the LRT null.
pub fn contrast_pvalue(a_w: u64, u_w: u64, a_m: u64, u_m: u64, rho: f64) -> f32 {
    /// Per-cell count below which the χ²-asymptotic LRT is unreliable → use the
    /// exact (Fisher) branch. The textbook expected-cell-count threshold.
    const FISHER_MIN_CELL: u64 = 5;
    /// Total coverage below which the asymptotic LRT is unreliable → exact branch.
    const LRT_MIN_TOTAL_COVERAGE: u64 = 100;

    let n_w = a_w + u_w;
    let n_m = a_m + u_m;
    let small = a_w < FISHER_MIN_CELL
        || u_w < FISHER_MIN_CELL
        || a_m < FISHER_MIN_CELL
        || u_m < FISHER_MIN_CELL
        || (n_w + n_m) < LRT_MIN_TOTAL_COVERAGE;
    if small {
        fisher_exact_greater(a_w, u_w, a_m, u_m)
    } else {
        betabinom_lrt_greater(a_w, n_w, a_m, n_m, rho)
    }
}

#[cfg(test)]
mod tests;
