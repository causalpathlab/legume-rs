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
