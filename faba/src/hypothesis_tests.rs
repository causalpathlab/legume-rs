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

/// Two-sample WT-vs-MUT conversion p-value for DART m6A: the one-sided
/// [`fisher_exact_greater`] on the 2x2 of (WT, MUT) x (converted, unconverted).
///
/// This used to dispatch to an overdispersed beta-binomial LRT once every cell
/// reached 5 and total coverage reached 100. Both tests were individually
/// correct; the DISPATCH was not. Two different nulls met at a count threshold,
/// so the p-value jumped discontinuously across it. Measured: one extra
/// converted read in the control moved it 7.6e6-fold (3.7e-9 -> 2.8e-2), and
/// doubling coverage at a fixed effect made a site LESS significant (5.6e-4 ->
/// 5.3e-2). A statistic that is not monotone in its own evidence cannot rank
/// sites, and BH then ranks them by which side of a count threshold they fell.
///
/// Dropping the asymptotic branch rather than the exact one was not close:
/// measured on rep1, 94.6% of called sites already took Fisher, because DART
/// control background is 0.1-1% so the control converted count is 0-2 at 88% of
/// sites. The LRT was both the minority path and the only consumer of `rho`.
///
/// The cost, stated plainly: overdispersion (fitted at 0.022-0.045) is now
/// unmodelled, rather than applied to the 5.4% of sites that reached the LRT.
/// That is the configuration Phase A validated -- 32.1% replicate
/// reproducibility, 270x over chance -- because that validation ran on a rule
/// which was already 94.6% Fisher.
pub fn contrast_pvalue(a_w: u64, u_w: u64, a_m: u64, u_m: u64) -> f32 {
    fisher_exact_greater(a_w, u_w, a_m, u_m)
}

/// Mean of a slice (`NaN` if empty).
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
mod tests;
