//! Shared primitives for the two Bayesian binomial-logit assoc tests
//! ([`super::contrast_bayes`] and [`super::trend_bayes`]).
//!
//! Both fit a binomial GLM `logit(p_i) = ηᵢ(β)` with vague intercept + Gaussian shrinkage
//! prior on the effect coefficients, via [`mcmc_util::engine::EssSampler`], and summarize the
//! posterior of a **linear contrast** of `β` (start→end for the trend, branch indicator for
//! the contrast) as a mean + 90% credible interval + local false sign rate. The scaffolding
//! is orthogonal to the specific design matrix, so it lives here to prevent drift between
//! the two files.

/// Vague prior standard deviation for the intercept, on the logit scale (weakly informative;
/// the per-bin baselines / spline intercept sit here). The effect coefficients use a
/// caller-supplied `τ`.
pub(super) const INTERCEPT_SD: f32 = 10.0;

/// Numerically-stable `log(1 + exp(e))` with the standard ±20 clamp — the softplus in the
/// binomial-logit log-likelihood `Σ_i (k_i·ηᵢ − n_i·log(1 + e^ηᵢ))`.
#[inline]
pub(super) fn softplus(e: f32) -> f32 {
    if e > 20.0 {
        e
    } else if e < -20.0 {
        0.0
    } else {
        (1.0 + e.exp()).ln()
    }
}

/// Posterior summary of the linear contrast: mean, sd, 5% / 95% quantiles,
/// `lfsr = min(P(effect > 0), P(effect < 0))` — the local false sign rate — and the two
/// numbers that say how much to trust it (`ess`, `mcse_lfsr`). Consumes `effects` for the
/// O(n) `select_nth_unstable`-based quantiles (no full sort). Returns `None` when the chain
/// is empty (a degenerate fit).
///
/// The `lfsr` is a plain Monte-Carlo tail proportion, so it carries sampling error of its
/// own: a site whose `lfsr` sits near the reporting threshold can cross it from one `--seed`
/// to the next, and the shrinkage prior stabilises `beta`, **not** the sampler. `mcse_lfsr`
/// is that error, per site, so a borderline call is visible in the row rather than inferred
/// globally: an `lfsr` within a couple of `mcse_lfsr` of the cutoff is not resolved by this
/// many draws, and the fix is `--posterior-samples`, not a re-read of the effect size.
///
/// The diagnostics must be computed **first**: the quantile step below reorders `effects`
/// in place, and an autocorrelation estimate on a reordered chain is meaningless.
pub(super) fn summarize_posterior(mut effects: Vec<f32>) -> Option<Posterior> {
    let ns = effects.len();
    if ns == 0 {
        return None;
    }
    let mean = effects.iter().sum::<f32>() / ns as f32;
    let var = effects.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / ns as f32;

    // One pass for both tail counts and the sign-indicator chain (rather than three): this
    // runs once per (site, group), and there are thousands of those.
    let mut signs = Vec::with_capacity(ns);
    let (mut n_pos, mut n_neg) = (0usize, 0usize);
    for &e in &effects {
        signs.push(if e > 0.0 {
            n_pos += 1;
            1.0
        } else {
            if e < 0.0 {
                n_neg += 1;
            }
            0.0
        });
    }
    let lfsr = (n_pos.min(n_neg) as f32) / ns as f32;

    // While the chain is still in draw order. `ess` is the effect chain's own effective size
    // (the headline "how many independent draws is this worth"); the lfsr's error is governed
    // by the effective size of the SIGN INDICATOR chain — the sequence the proportion is
    // actually a mean of — which mixes differently from the effect itself.
    let ess = mcmc_util::engine::ess(&effects);
    let mcse_lfsr = mcmc_util::engine::mcse_proportion(lfsr, mcmc_util::engine::ess(&signs));

    // `total_cmp`, not `partial_cmp().unwrap()`: a diverged chain must not panic the run.
    let cmp = |a: &f32, b: &f32| a.total_cmp(b);
    let idx = |pr: f32| ((pr * (ns as f32 - 1.0)).round() as usize).min(ns - 1);
    let (lo_i, hi_i) = (idx(0.05), idx(0.95));
    effects.select_nth_unstable_by(hi_i, cmp);
    let effect_hi = effects[hi_i];
    let effect_lo = if lo_i < hi_i {
        *effects[..hi_i].select_nth_unstable_by(lo_i, cmp).1
    } else {
        effect_hi
    };
    Some(Posterior {
        effect: mean,
        effect_sd: var.sqrt(),
        effect_lo,
        effect_hi,
        lfsr,
        ess,
        mcse_lfsr,
    })
}

/// Distinct per-(site, branch) seed so parallel ESS chains stay reproducible regardless of
/// thread order. `1009` is prime, keeping the site strides mutually offset from the branch
/// index.
#[inline]
pub(super) fn derive_seed(base: u64, si: usize, l: usize) -> u64 {
    base.wrapping_add((si as u64).wrapping_mul(1009))
        .wrapping_add(l as u64)
}

/// One QC-passing `(site, group)` row of a Bayesian test: which cell it is, how much data
/// backed it, and the posterior summary of the test's linear contrast.
///
/// **Both** Bayesian tests report exactly this — they differ only in *what the contrast is*,
/// not in what they say about it. For the between-group contrast, `effect` is `β`, the
/// pseudotime-adjusted log-odds excess of the group vs the pooled rest; for the within-group
/// trend, it is the net log-odds change from the group's start to its end. Everything else
/// (the sd, the 90% credible interval, the lfsr and its two sampling diagnostics) means the
/// same thing in both. Sharing the row is what keeps their two output schemas identical,
/// which the `run_assoc` module doc promises and which a second struct could only ever
/// promise by hand.
pub struct BayesResult {
    pub site: usize,
    /// The group id — a branch at the branch level, a cell type at the cell-type level.
    pub branch: usize,
    pub n_cells: usize,
    pub total_cov: u64,
    /// Posterior mean of the test's linear contrast (see the struct doc for which).
    pub effect: f32,
    pub effect_sd: f32,
    /// 5% / 95% posterior quantiles of the effect (90% credible interval).
    pub effect_lo: f32,
    pub effect_hi: f32,
    /// Local false sign rate `min(P(effect > 0), P(effect < 0))`.
    pub lfsr: f32,
    /// Effective sample size of the effect chain.
    pub ess: f32,
    /// Monte-Carlo standard error of `lfsr` — how sharply this row's call is resolved.
    pub mcse_lfsr: f32,
}

impl BayesResult {
    /// Assemble a row from the test-specific metadata plus the shared [`Posterior`].
    pub(super) fn new(
        site: usize,
        branch: usize,
        n_cells: usize,
        total_cov: u64,
        p: &Posterior,
    ) -> Self {
        Self {
            site,
            branch,
            n_cells,
            total_cov,
            effect: p.effect,
            effect_sd: p.effect_sd,
            effect_lo: p.effect_lo,
            effect_hi: p.effect_hi,
            lfsr: p.lfsr,
            ess: p.ess,
            mcse_lfsr: p.mcse_lfsr,
        }
    }
}

/// Posterior summary shared by both Bayesian tests. Consumers wrap it in a [`BayesResult`]
/// with their test-specific metadata (`n_cells`, `total_cov`).
pub(super) struct Posterior {
    pub effect: f32,
    pub effect_sd: f32,
    pub effect_lo: f32,
    pub effect_hi: f32,
    pub lfsr: f32,
    /// Effective sample size of the effect chain — how many independent draws its
    /// `--posterior-samples` correlated ones are worth.
    pub ess: f32,
    /// Monte-Carlo standard error of `lfsr`. A site whose `lfsr` is within a couple of these
    /// of the reporting cutoff is not resolved at this sample count.
    pub mcse_lfsr: f32,
}
