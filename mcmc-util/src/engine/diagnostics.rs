//! Monte-Carlo accuracy diagnostics for a single chain.
//!
//! A posterior summary reported without one of these is a point estimate of unknown
//! precision. The two that matter for a scalar summary of one chain:
//!
//! - [`ess`] — the **effective sample size**: how many *independent* draws the chain's
//!   `n` correlated draws are worth. An elliptical-slice chain that rejects and re-emits
//!   its current state (the bracket-exhaustion path) produces duplicate draws, so `n` is
//!   an upper bound that can be far from the truth.
//! - [`mcse_proportion`] — the standard error of a **tail probability** estimated from
//!   that chain (an `lfsr` is exactly such a probability). This is what says whether a
//!   site sitting near a reporting threshold is genuinely near it, or just under-sampled.
//!
//! Note the name collision this crate lives with: throughout `engine::ess`, "ESS" means
//! *elliptical slice sampling*. Here — and only here — it means *effective sample size*.

/// Effective sample size of a scalar chain, by Geyer's initial monotone positive sequence.
///
/// `ess = n / (1 + 2 Σ ρ_t)`, with the autocorrelation sum truncated where Geyer's adjacent
/// pair sums `Γ_k = ρ_{2k+1} + ρ_{2k+2}` first go non-positive, and each pair sum capped by
/// its predecessor so the sequence stays monotone. Truncating at the first non-positive pair
/// sum is what makes this cheap: the sum runs over the few dozen lags that actually carry
/// autocorrelation rather than all `n`, so the cost is `O(n · lag*)`, not `O(n²)`.
///
/// Returns `n` for a chain too short to estimate from (`n < 4`) or a constant one — a chain
/// that never moves has no autocorrelation to discount, and callers still need a finite
/// divisor. That is the right answer for [`mcse_proportion`]'s purposes (its smoothing keeps
/// a constant indicator chain from reporting zero error), but it does mean `ess == n` is not
/// on its own evidence of good mixing.
///
/// Accumulates in `f64`: the lag products are sums of `n` like-signed terms, and the whole
/// estimate is a ratio of small differences. The chain is centered **once** into an `f64`
/// buffer, so each lag is a plain dot product over that buffer rather than an `n`-long
/// re-centering — this runs per `(site, group)` and there are thousands of those.
#[must_use]
pub fn ess(x: &[f32]) -> f32 {
    let n = x.len();
    if n < 4 {
        return n as f32;
    }
    let nf = n as f64;
    let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / nf;
    let c: Vec<f64> = x.iter().map(|&v| f64::from(v) - mean).collect();

    // γ_t = (1/n) Σ_i c_i · c_{i+t}; ρ_t = γ_t / γ_0.
    let autocov = |t: usize| -> f64 { c.iter().zip(&c[t..]).map(|(a, b)| a * b).sum::<f64>() / nf };

    let var0 = autocov(0);
    if var0 <= 0.0 {
        return n as f32; // constant chain — no autocorrelation to discount
    }

    // Geyer's initial monotone positive sequence over the adjacent pair sums. The `n / 2` cap
    // is the usual convention: past halfway the lag products are averaging over so few terms
    // that they are noise, and without it a chain whose pair sums never turn negative would
    // walk the whole series and make this O(n²).
    let mut sum_rho = 0.0f64;
    let mut prev = f64::INFINITY;
    let mut t = 1usize;
    while t + 1 < n / 2 {
        let gamma = (autocov(t) + autocov(t + 1)) / var0;
        if gamma <= 0.0 {
            break; // past here the estimate is noise, not signal
        }
        let gamma = gamma.min(prev); // enforce monotonicity
        sum_rho += gamma;
        prev = gamma;
        t += 2;
    }

    let tau = (1.0 + 2.0 * sum_rho).max(1.0); // integrated autocorrelation time; τ ≥ 1 ⇒ ess ≤ n
    (nf / tau).max(1.0) as f32
}

/// **Split-R̂** for one chain: the Gelman–Rubin statistic computed by cutting the chain
/// into consecutive segments and treating them as separate chains.
///
/// Why split rather than multi-chain. R̂ compares between-segment variance against
/// within-segment variance, so it is sensitive to exactly the failure a single long run
/// hides: a chain still drifting, where the first half and the second half are sampling
/// different regions. `1.0` means the segments are indistinguishable; the conventional
/// threshold is `1.01`, and anything above `1.1` should be read as "this has not
/// converged" rather than as a mild warning.
///
/// What it CANNOT see, and this bound matters: a mode both halves are stuck in. Split-R̂
/// certifies stationarity, not that the chain found the right place — for that there is
/// no substitute for independent chains from dispersed starts.
///
/// Returns `1.0` for a chain too short to split (nothing to compare), and `1.0` for a
/// perfectly constant chain, since a zero within-segment variance means the segments
/// agree exactly rather than that they disagree infinitely.
#[must_use]
pub fn split_rhat(x: &[f32]) -> f32 {
    let t = x.len();
    // Four segments where there is length for it — more segments give more power to
    // detect drift — falling back to two, and to "no answer" below that.
    let m = if t >= 40 {
        4
    } else if t >= 8 {
        2
    } else {
        return 1.0;
    };
    let n = t / m;
    if n < 2 {
        return 1.0;
    }

    let mut means = Vec::with_capacity(m);
    let mut within = 0.0f64;
    for s in 0..m {
        let seg = &x[s * n..(s + 1) * n];
        let mean = seg.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let var = seg
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1) as f64;
        means.push(mean);
        within += var;
    }
    let w = within / m as f64;
    if w <= 0.0 {
        // Every segment is internally constant. If they also agree with each other the
        // chain is simply pinned; if they do not, `w = 0` makes the ratio meaningless,
        // so report the disagreement as non-convergence rather than dividing by zero.
        let lo = means.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        return if (hi - lo).abs() <= f64::EPSILON {
            1.0
        } else {
            f32::INFINITY
        };
    }

    let grand = means.iter().sum::<f64>() / m as f64;
    let b = means
        .iter()
        .map(|mu| {
            let d = mu - grand;
            d * d
        })
        .sum::<f64>()
        / (m - 1) as f64
        * n as f64;

    let var_hat = (n - 1) as f64 / n as f64 * w + b / n as f64;
    (var_hat / w).sqrt() as f32
}

/// Monte-Carlo standard error of a probability `p` estimated from a chain with effective
/// sample size `ess` — e.g. the `lfsr`, which is a posterior tail probability.
///
/// Uses the Jeffreys (+½) smoothed proportion rather than the plug-in `√(p(1−p)/ess)`.
/// The plug-in reports an error of **exactly zero** when no draw fell on the minority side
/// (`p = 0`) — i.e. it claims infinite precision precisely at the most significant sites,
/// which are the ones whose ranking a reader is most likely to trust. Smoothing reports the
/// resolution the chain can actually support there (about `0.7 / ess`), which is the honest
/// statement: *not observed in `ess` effective draws*, not *impossible*.
#[must_use]
pub fn mcse_proportion(p: f32, ess: f32) -> f32 {
    if ess <= 0.0 {
        return f32::NAN;
    }
    let (p, n) = (f64::from(p).clamp(0.0, 1.0), f64::from(ess));
    let smoothed = (p * n + 0.5) / (n + 1.0);
    ((smoothed * (1.0 - smoothed) / (n + 1.0)).sqrt()) as f32
}

#[cfg(test)]
mod tests;
