//! Bootstrap consensus: turning a tally of resampled votes into a call, a credible set, and an
//! honest refusal.
//!
//! This is the part of a marker bootstrap that knows nothing about *how* the votes were earned.
//! One resample of the marker panel produces one winner per item — a cell, on the embedding path
//! (nearest marker centroid); a cluster, on the raw-count enrichment path (top over-represented
//! celltype). Tally those winners across `B` resamples and you have a distribution per item, and
//! everything downstream — the label, whether to abstain, the credible set, the entropy — is a
//! function of that distribution alone.
//!
//! So it lives here, in the crate *both* annotation paths already depend on, rather than in
//! either one of them. `graph-embedding-util` depends on `enrichment` (for `bh_fdr`), so the
//! embedding path re-exports these; the enrichment path uses them directly. There is exactly one
//! implementation of "what counts as agreement", which is the point — the support null and the
//! observed run had already drifted apart once (see [`label_support`]).
//!
//! **The support is not a posterior.** It is the fraction of replicates that agreed: the sampling
//! variability of the pipeline's own output. It sees variance, not bias. A panel whose markers are
//! simply *wrong* comes back perfectly stable — catching that needs a permutation null, not this.

use rand::rngs::SmallRng;
use rand::SeedableRng;

#[cfg(test)]
mod tests;

/// The "this item has no call" sentinel. `usize::MAX`, so it is never a valid label index and any
/// `t < c` test excludes it.
pub const UNASSIGNED: usize = usize::MAX;

/// Live markers a celltype needs before it is allowed to compete.
///
/// **You cannot bootstrap a single point.** Resampling a one-element panel with replacement always
/// returns that same element, so such a type's evidence never moves and it comes out looking
/// *perfectly* stable — the exact opposite of the truth, and it would then win with full
/// confidence. The sampling variance of a mean of one is not zero, it is *unknown*. A type located
/// by a single surviving marker is not located, so it does not compete.
pub const MIN_LIVE_MARKERS: usize = 2;

/// When an item's top label is allowed to stand.
///
/// These ask genuinely different questions, and the difference matters:
///
/// * [`Self::Support`] asks **"is the top label probable enough to act on?"** — a fixed bar on its
///   share of the replicates. Simple, but the bar is not scale-free: with `C` types, chance
///   agreement is `1/C`, so `0.5` sits at 3× chance on a 6-type panel and 12× chance on a 24-type
///   one. The *same* flag is a different test on different panels, which makes their abstention
///   rates incomparable.
/// * [`Self::Separable`] asks **"can the top label even be told apart from the runner-up?"** — an
///   exact sign test, with no magic number and no dependence on `C`. Among the `m` replicates that
///   picked one of the two leading labels, each is a coin flip if the two are equally probable, so
///   `n₁ ~ Binomial(m, ½)` and its upper tail is the p-value.
///
/// Note what `Separable` does *not* do: it says nothing about whether the winner is *likely*, only
/// that it is distinguishable from second place. Its power grows with `n_boot`, so more replicates
/// resolve more items — correct (more evidence, finer resolution), but it means a 51/49 item will
/// eventually be called given enough draws. Whichever you pick, the honest output for an item whose
/// leaders are inseparable is not a shrug but a **set** — see [`Consensus::label_set`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Abstain {
    /// Call only if the top label won at least this fraction of the replicates.
    Support(f32),
    /// Call only if the top label beat the runner-up by more than resampling noise, at this α.
    Separable(f64),
}

impl Abstain {
    /// Does the top label (share `p1`, runner-up `p2`) survive, out of `b` replicates?
    #[must_use]
    pub fn allows(self, p1: f32, p2: f32, b: usize) -> bool {
        match self {
            Self::Support(min) => p1 >= min,
            Self::Separable(alpha) => {
                let (n1, n2) = (
                    (p1 * b as f32).round() as usize,
                    (p2 * b as f32).round() as usize,
                );
                binom_half_upper_tail(n1 + n2, n1) < alpha
            }
        }
    }
}

/// The three knobs that turn a vote distribution into a call: when to abstain, and how large a
/// credible set is still an annotation rather than a shrug.
#[derive(Copy, Clone, Debug)]
pub struct AbstainConfig {
    /// When a call is allowed to stand.
    pub abstain: Abstain,
    /// The **credible set**: the smallest set of labels whose replicate shares sum to at least
    /// this much. `0.8` ⇒ "the labels that account for 80% of what the resampling said". An item
    /// that cannot be given one label can still be given two, and that is information, not a
    /// failure — the alternative is to compute the whole distribution and then throw it away.
    pub set_coverage: f32,
    /// Largest set worth printing. `HSPC/LMPP` is an annotation; a four-way tie is not — past a
    /// point a set stops narrowing anything down and starts laundering "we don't know" as though
    /// it were a finding.
    pub max_set_size: usize,
}

/// The consensus over `B` resamples, for `n` items.
#[derive(Debug)]
pub struct Consensus {
    /// `[n × k]` row-major, normalized to shares; the last column (`k - 1`) is `unassigned`.
    pub post: Vec<f32>,
    /// Consensus label, or [`UNASSIGNED`] when the [`Abstain`] rule rejects it.
    pub label: Vec<usize>,
    /// **The mixed annotation**: the smallest set of labels covering `set_coverage` of the
    /// replicates. An item whose leaders are inseparable gets a two-element set, and `HSPC/LMPP` is
    /// a far better answer than `unassigned` — this is populated for items `label` gives up on,
    /// which is the point of it.
    ///
    /// **Empty** when even `max_set_size` labels cannot reach the coverage: that item has no call
    /// in any form, and pretending a 5-way tie is an annotation would be worse than saying so.
    pub label_set: Vec<Vec<usize>>,
    /// Fraction of replicates that agreed on the consensus label.
    pub support: Vec<f32>,
    /// Share of the replicates covered by `label_set`.
    pub set_support: Vec<f32>,
    /// Normalized entropy of the per-item distribution, in `[0, 1]`.
    pub entropy: Vec<f32>,
}

/// Reduce a raw vote tally to a consensus.
///
/// `post` is `[n × k]` row-major **counts** (not shares); it is normalized in place by `n_drawn`
/// and returned inside the [`Consensus`]. Column `k - 1` is `unassigned`. `n_drawn` is the number
/// of replicates that actually completed — not the number requested, which differ after an
/// interrupt.
#[must_use]
pub fn summarize_consensus(
    mut post: Vec<f32>,
    n: usize,
    k: usize,
    n_drawn: usize,
    cfg: &AbstainConfig,
) -> Consensus {
    let drawn = n_drawn.max(1);
    for v in &mut post {
        *v /= drawn as f32;
    }

    let c = k - 1; // the type columns; the last one is `unassigned`
    let unassigned_col = c;
    let ln_k = (k as f32).ln();

    let (mut label, mut support, mut entropy) = (vec![UNASSIGNED; n], vec![0f32; n], vec![0f32; n]);
    let mut label_set: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut set_support = vec![0f32; n];

    for i in 0..n {
        let row = &post[i * k..(i + 1) * k];
        let (best, p1, p2) = top_two(row);
        support[i] = p1; // == `label_support(row)`; kept inline, `best`/`p2` are needed too
                         // `.max(0.0)` is not a clamp on a real value: entropy is non-negative by definition, and a
                         // unanimous item gives `-1 * (1 * ln 1) = -0.0`, which would otherwise print as `-0.0000`
                         // in the QC files and read as a bug.
        entropy[i] = (-row
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f32>()
            / ln_k)
            .max(0.0);
        // A label survives only if the replicates actually agreed on it.
        if best != unassigned_col && cfg.abstain.allows(p1, p2, drawn) {
            label[i] = best;
        }
        // …and whether it survived or not, say what the replicates *did* say — unless even that is
        // more hedge than answer, in which case the item has no call at all.
        //
        // **The set is over TYPES, not over the `unassigned` column** — but the shares are still
        // shares of *all* the replicates, so the `unassigned` mass stays in the denominator and
        // pushes the set toward not reaching coverage. An item whose replicates mostly declined to
        // call it therefore gets no call, rather than the nonsense set `Erythroid/unassigned`.
        let set = credible_set(&row[..c], cfg.set_coverage, cfg.max_set_size).unwrap_or_default();
        set_support[i] = set.iter().map(|&t| row[t]).sum::<f32>().max(0.0);
        label_set.push(set);
    }

    Consensus {
        post,
        label,
        label_set,
        support,
        set_support,
        entropy,
    }
}

/// **The support statistic**, in one place.
///
/// `row` is an item's distribution over the `c` types **and** the `unassigned` column, and the
/// support is the largest entry of it — including `unassigned`, which is a legitimate outcome an
/// item's replicates can agree on.
///
/// This exists so an observed run and its permutation null cannot drift apart. They did: the null
/// was taking the max over the *types only* while the shipped number took it over everything, so a
/// p-value was being formed by comparing two different statistics.
#[must_use]
pub fn label_support(row: &[f32]) -> f32 {
    top_two(row).1
}

/// `(argmax, top share, runner-up share)` of a distribution.
#[must_use]
pub fn top_two(row: &[f32]) -> (usize, f32, f32) {
    let (mut best, mut p1, mut p2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (t, &p) in row.iter().enumerate() {
        if p > p1 {
            p2 = p1;
            p1 = p;
            best = t;
        } else if p > p2 {
            p2 = p;
        }
    }
    (best, p1.max(0.0), p2.max(0.0))
}

/// The smallest set of labels whose shares sum to `coverage`, largest first — a credible set.
///
/// `None` when it would take more than `max_len` labels to get there: at that point the set has
/// stopped narrowing anything down, and printing it would launder "we don't know" as a finding.
///
/// `row` need not sum to 1: the caller passes only the *type* columns while the shares remain
/// shares of every replicate, so any mass the replicates spent on `unassigned` is simply absent and
/// makes `coverage` correspondingly harder to reach. That is the intended behaviour — an item its
/// replicates mostly declined to call should come out uncalled.
#[must_use]
pub fn credible_set(row: &[f32], coverage: f32, max_len: usize) -> Option<Vec<usize>> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
    let mut acc = 0f32;
    let mut out = Vec::new();
    for t in order.into_iter().take(max_len.max(1)) {
        out.push(t);
        acc += row[t];
        if acc >= coverage {
            return Some(out);
        }
    }
    None
}

/// `P(X ≥ k)` for `X ~ Binomial(m, ½)` — the exact sign test. Computed in log space so `m` in the
/// hundreds is safe, and summed from the far tail inward so the small terms land first.
#[must_use]
pub fn binom_half_upper_tail(m: usize, k: usize) -> f64 {
    use statrs::function::gamma::ln_gamma;
    if m == 0 {
        return 1.0; // nobody voted for either: nothing to separate
    }
    let ln_fact = |x: usize| ln_gamma(x as f64 + 1.0);
    let ln_denom = m as f64 * std::f64::consts::LN_2;
    let mut acc = 0f64;
    for x in (k.min(m)..=m).rev() {
        acc += (ln_fact(m) - ln_fact(x) - ln_fact(m - x) - ln_denom).exp();
    }
    acc.min(1.0)
}

/// An RNG keyed by `(seed, draw, item)` — so a resample depends only on *which* one it is, never on
/// how rayon scheduled it. That is what makes `--seed` reproduce.
///
/// `item` is the domain separator: pass a celltype index, and a distinct large constant per null,
/// so two resampling schemes driven by the same `--seed` never walk the same stream.
#[must_use]
pub fn keyed_rng(seed: u64, draw: usize, item: u64) -> SmallRng {
    // Fold the `item` domain separator into the base, then run it through the
    // shared SplitMix64 mixer. Byte-identical to the former inline finalizer:
    // `mix_seed(base, draw)` = `finalize(base ^ draw·GOLDEN)`, and XOR commutes,
    // so `base = seed ^ item·C2B2` reproduces the old `seed ^ draw·GOLDEN ^ item·C2B2`.
    let base = seed ^ item.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    SmallRng::seed_from_u64(matrix_util::rand_util::mix_seed(base, draw as u64))
}
