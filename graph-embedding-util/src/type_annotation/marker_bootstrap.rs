//! Non-parametric **marker bootstrap** — how stable is a cell's call under resampling of
//! the marker panel that produced it?
//!
//! The firm path ([`super::term_ora`]) builds `e_T = Σ w_g e_g / Σ w_g` and hard-assigns
//! every cell to its nearest centroid. `argmin` always returns *something*, and it returns
//! it with no error bar, so two real error sources are silently treated as zero: the
//! **marker list can be wrong** (a listed gene isn't actually specific to the type), and the
//! **feature embedding can disagree with it** (a type's markers don't sit together in
//! β-space, so its centroid is a fiction).
//!
//! Rather than posit a generative density over the embedding, resample the evidence:
//!
//! ```text
//! for b in 1..B:
//!     for each type t:  draw |live(t)| of t's live markers WITH REPLACEMENT
//!                       e_T^(b) = the IDF-weighted mean of that multiset
//!     assign every cell to its nearest e_·^(b)
//! P(z_c = t) = (1/B) · #{ b : cell c was assigned to t }
//! ```
//!
//! This is Efron's bootstrap over the marker panel, and it is deliberately *non-parametric*:
//! it perturbs **exactly the quantity that enters the decision**, so the jitter it induces is
//! automatically on the scale of the decision. There is nothing to calibrate — no variance
//! to set against the (incommensurable) spread of the cells, and no free per-type variance a
//! component could shrink-wrap a blob with. Whatever geometry `assign_nearest` uses, this
//! reports how reproducible its answer is.
//!
//! What it buys, concretely:
//!
//! * **It attacks the winner's curse.** `argmax` over types favours the *noisiest* type, not
//!   the best-supported one: a type with 8 live markers has a high-variance centroid, and the
//!   maximum of a noisy score wins more often than it should. Under the bootstrap that same
//!   type's centroid flies around, it loses its cells on half the draws, and the dilution
//!   *is* its confidence. A type with 40 mutually-consistent markers barely moves and keeps
//!   its cells.
//! * **Abstention is instability**, not a threshold on a test statistic: a cell whose call
//!   flips across resamples has no stable call, and saying so is the honest answer.
//! * The per-cell numbers finally *vary within a cluster* — the firm path's `confidence` is a
//!   cluster-level softmaxed statistic, identical for every cell in it.
//!
//! It does **not** manufacture information. If the marker centroids are crammed together
//! relative to the cell cloud — as they are whenever the panel was never trained into the
//! embedding — the bootstrap will report near-total instability, and it should. Read
//! `{out}.type_qc.tsv`: when a type's `centroid_jitter` exceeds the `decision_gap` its cells
//! are being assigned by noise, and no amount of downstream statistics will fix that. The
//! cure is upstream (`faba gem --must-train-features <panel>`), not here.

use super::UNASSIGNED;
use crate::null_call::live_row;
use anyhow::Result;
use rand::RngExt;
use rayon::prelude::*;

pub(super) use enrichment::consensus::label_support;
/// The vote-tally half of the bootstrap — how a distribution over resampled winners becomes a
/// label, a credible set, or a refusal. It knows nothing about embeddings, so it lives in
/// `enrichment`, where the raw-count marker bootstrap (`enrichment::marker_bootstrap`) shares it.
/// One definition of "what counts as agreement", used by both paths.
pub use enrichment::consensus::Abstain;
use enrichment::consensus::{
    keyed_rng, summarize_consensus, top_two, AbstainConfig, Consensus, MIN_LIVE_MARKERS,
};

#[cfg(test)]
mod tests;

/// Tunables for [`run_marker_bootstrap`].
#[derive(Clone, Debug)]
pub struct MarkerBootstrapConfig {
    /// Bootstrap resamples of the marker panel.
    pub n_boot: usize,
    /// When a call is allowed to stand ([`Abstain`]).
    pub abstain: Abstain,
    /// The **credible set** reported for every cell: the smallest set of labels whose replicate
    /// shares sum to at least this much. `0.8` ⇒ "the labels that account for 80% of what the
    /// resampling said". A cell that cannot be given one label can still be given two, and that
    /// is information, not a failure — the alternative is to compute the whole distribution and
    /// then throw it away.
    pub set_coverage: f32,
    /// Largest set worth printing. `HSPC/LMPP` is an annotation; a four-way tie is not — past a
    /// point a set stops narrowing anything down and starts laundering "we don't know" as though
    /// it were a finding. A cell that needs more than this many labels to reach `set_coverage`
    /// has no call, and says so.
    pub max_set_size: usize,
    /// Re-derive the cell clustering inside every resample, so the **pipeline's own
    /// stochasticity** is absorbed into the support alongside the marker panel's.
    ///
    /// The kNN graph is deterministic now (matrix-util's seeded instant-distance backend), so the
    /// pipeline reproduces run-to-run. What reclustering absorbs is the clustering's *within-run*
    /// arbitrariness: Leiden picks among near-equal modularity optima, so re-partitioning under a
    /// fresh seed each replicate moves the cells that only ever sat on a partition boundary. A
    /// label that flips when nothing but the partition seed moved is not a robust label. Rather
    /// than trust one partition, resample it and let the consensus reject whatever it held up.
    pub recluster: bool,
}

impl Default for MarkerBootstrapConfig {
    fn default() -> Self {
        Self {
            n_boot: 200,
            abstain: Abstain::Support(0.5),
            set_coverage: 0.8,
            max_set_size: 3,
            recluster: true,
        }
    }
}

/// Per-type diagnostics — *how well determined is the type itself?*, the question the firm
/// path never asks.
pub struct TypeQc {
    /// Markers carrying a live β row: the only ones that can contribute evidence.
    pub n_live: usize,
    /// RMS displacement of the type's centroid across the resamples. **This is the noise in
    /// the decision variable**, in the same units as the cell→centroid distances below.
    pub centroid_jitter: f32,
    /// Median over this type's cells of the gap between the nearest and the second-nearest
    /// centroid — the margin the assignment is actually decided by.
    pub decision_gap: f32,
    /// Mean bootstrap support of the cells assigned to this type.
    pub mean_support: f32,
    /// Fraction of cells whose consensus call is this type.
    pub occupancy: f32,
}

/// What the resampling found. Deliberately **not** called a posterior: `support` is the
/// fraction of replicates that agreed on a call — the sampling variability of *this pipeline's
/// output* — not `P(type | data)`. It measures **variance, not bias**: a systematically wrong
/// call that every replicate agrees on comes out with support 1.0. Read it as "re-run this with
/// a differently-drawn panel and a different RNG and you'd get the same answer this often".
pub struct BootstrapResult {
    /// Number of types.
    pub c: usize,
    /// Replicates that actually ran. **Not** `MarkerBootstrapConfig::n_boot`, which is only what
    /// was *asked* for: a Ctrl+C leaves this smaller, and then every `support` below is a
    /// fraction of *this* number. A support of 0.51 earned over 3 draws and one earned over 200
    /// are not the same claim, and the log line that said so has long since scrolled away by the
    /// time anyone opens the parquet — so it travels with the output.
    pub n_draws: usize,
    /// `[n × c]` row-major: the fraction of replicates that assigned each cell to each type by
    /// **nearest centroid** — i.e. stability of the raw geometric call under panel resampling
    /// alone, before any clustering.
    pub post: Vec<f32>,
    /// Nearest-centroid consensus, or [`UNASSIGNED`] below `min_support`.
    pub assign: Vec<usize>,
    /// Distance to the mean bootstrap centroid of the assigned type (`NaN` when unassigned).
    pub dist: Vec<f32>,
    /// The winning type's share of the replicates — the support of the nearest-centroid call.
    pub support: Vec<f32>,
    /// Normalized entropy of the per-cell nearest-centroid distribution, in `[0, 1]`.
    pub entropy: Vec<f32>,
    /// The **shipped** label's consensus (panel *and* clustering resampled). `None` when no
    /// [`CoarseStep`] was supplied. This — not `support` above — is what `membership.tsv`
    /// carries.
    pub coarse: Option<CoarseConsensus>,
    /// Per type, per listed marker: robust z of the marker's distance to its type's centroid.
    /// **Large ⇒ the embedding puts this gene nowhere near the type it is listed under.**
    /// `NaN` for a dead marker. Aligned one-to-one with `type_markers[t]`.
    pub marker_dev: Vec<Vec<f32>>,
    /// Whether each listed marker has a live β row. Aligned with `type_markers[t]`.
    pub marker_live: Vec<Vec<bool>>,
    /// Per-type diagnostics.
    pub type_qc: Vec<TypeQc>,
}

/// The consensus over the **shipped**, cluster-driven label — the one that lands in
/// `membership.tsv`. Distinct from the nearest-centroid numbers on [`BootstrapResult`]: this
/// one additionally absorbs the clustering's own stochasticity, so it is the stricter (and the
/// operationally relevant) statement.
pub struct CoarseConsensus {
    /// `[n × (c+1)]` row-major; the last column is `unassigned`.
    pub post: Vec<f32>,
    /// Consensus label, or [`UNASSIGNED`] when the [`Abstain`] rule rejects it.
    pub label: Vec<usize>,
    /// **The mixed annotation**: the smallest set of labels covering `set_coverage` of the
    /// replicates (`c` is the `unassigned` column). A cell whose leaders are inseparable gets a
    /// two-element set, and `HSPC/LMPP` is a far better answer than `unassigned` — this is
    /// populated for cells `label` gives up on, which is the point of it.
    ///
    /// **Empty** when even `max_set_size` labels cannot reach the coverage: that cell has no
    /// call in any form, and pretending a 5-way tie is an annotation would be worse than saying
    /// so.
    pub label_set: Vec<Vec<usize>>,
    /// Fraction of replicates that agreed on the consensus label.
    pub support: Vec<f32>,
    /// Share of the replicates covered by `label_set` (≥ `set_coverage`, unless one label has
    /// everything).
    pub set_support: Vec<f32>,
    /// Normalized entropy of the per-cell distribution, in `[0, 1]`.
    pub entropy: Vec<f32>,
    /// Community count seen in each replicate — its spread *is* the clustering's instability.
    pub n_comm: Vec<usize>,
}

/// The abstention triple, lifted out of the embedding-specific [`MarkerBootstrapConfig`].
fn abstain_cfg(cfg: &MarkerBootstrapConfig) -> AbstainConfig {
    AbstainConfig {
        abstain: cfg.abstain,
        set_coverage: cfg.set_coverage,
        max_set_size: cfg.max_set_size,
    }
}

/// One type's live markers: `(row index into feature_emb, IDF weight, index into
/// type_markers[t])`.
type LiveMarkers = Vec<(usize, f32, usize)>;

/// The resampleable panel: each type's live markers, and whether it has enough of them to
/// compete at all. Built once, then drawn from repeatedly.
pub(super) struct LivePanel {
    live: Vec<LiveMarkers>,
    /// Types with at least [`MIN_LIVE_MARKERS`] live markers — the only ones allowed to win
    /// cells (see the constant for why a 1-marker type must not).
    pub(super) usable: Vec<bool>,
}

impl LivePanel {
    /// A dead β row is a gene the embedding never learned; it carries no evidence about where
    /// its type sits, so it is not part of the panel we resample (the same rule
    /// `term_ora::term_centroids` applies).
    ///
    /// **Silent by design.** [`super::support_null`] builds one of these per permutation, inside
    /// the rayon fan-out, and a `warn!` there would fire `P` times *and* funnel every worker
    /// through the logger's mutex. The observed run reports the dropped types once, via
    /// [`Self::warn_unusable`].
    pub(super) fn new(feature_emb: &[f32], type_markers: &[Vec<(u32, f32)>], h: usize) -> Self {
        let live: Vec<LiveMarkers> = type_markers
            .iter()
            .map(|markers| {
                markers
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(gi, w))| {
                        w > 0.0 && live_row(feature_emb, gi as usize, h).is_some()
                    })
                    .map(|(j, &(gi, w))| (gi as usize, w, j))
                    .collect()
            })
            .collect();
        let usable: Vec<bool> = live.iter().map(|m| m.len() >= MIN_LIVE_MARKERS).collect();
        Self { live, usable }
    }

    /// Name the types that cannot be bootstrapped, and refuse to run if *none* can.
    ///
    /// **Zero usable types is the "magnet" failure, and it must not be survivable.** With every
    /// centroid a zero row and every `usable[t]` false, `nearest` never enters its update branch
    /// and falls back on its `best = 0` initializer — so every cell in the dataset is called as
    /// whichever type happens to be listed first in the marker file, at a perfectly stable
    /// `support` of 1.0, and the run exits 0 with an annotation that looks confident and is
    /// entirely fictional. It is reachable: a panel matched by gene *name* but never trained into
    /// the embedding has no live β rows at all.
    fn warn_unusable(&self) -> Result<()> {
        anyhow::ensure!(
            self.usable.iter().any(|&u| u),
            "not one of the {} marker type(s) has {MIN_LIVE_MARKERS} live markers in this \
             embedding, so there is nothing to assign cells to. The marker genes were matched by \
             name but never trained: re-run `faba gem --must-train-features <panel>` with THIS \
             panel.",
            self.live.len()
        );
        let dropped = self.usable.iter().filter(|&&u| !u).count();
        if dropped > 0 {
            log::warn!(
                "{dropped} of {} type(s) have fewer than {MIN_LIVE_MARKERS} live markers and \
                 cannot be bootstrapped — they are excluded from the assignment rather than \
                 allowed to win cells on evidence whose variance is unmeasurable. Train the \
                 panel into the embedding (`faba gem --must-train-features <panel>`) to bring \
                 them back.",
                self.live.len()
            );
        }
        Ok(())
    }

    /// Draw resample `b`: for each type, pick `|live(t)|` of its live markers **with
    /// replacement** and write the IDF-weighted mean of that multiset into `out` (`[c × h]`,
    /// row-major). An unusable type gets a zero row and is never selected against.
    pub(super) fn resample_into(
        &self,
        feature_emb: &[f32],
        h: usize,
        seed: u64,
        draw: usize,
        out: &mut [f32],
    ) {
        out.par_chunks_mut(h)
            .zip(self.live.par_iter())
            .enumerate()
            .for_each(|(t, (row, lm))| {
                row.fill(0.0);
                if lm.len() < MIN_LIVE_MARKERS {
                    return;
                }
                let mut rng = keyed_rng(seed, draw, t as u64);
                let mut wsum = 0f32;
                for _ in 0..lm.len() {
                    let (gi, w, _) = lm[rng.random_range(0..lm.len())];
                    wsum += w;
                    for (r, &e) in row.iter_mut().zip(&feature_emb[gi * h..(gi + 1) * h]) {
                        *r += w * e;
                    }
                }
                if wsum > 0.0 {
                    for v in row.iter_mut() {
                        *v /= wsum;
                    }
                }
            });
    }
}

/// What a replicate does with its resampled panel once the cells have been assigned to it:
/// re-derive the clustering, run the cluster × term over-representation test, and return the
/// **shipped** per-cell label — an index into `0..=c`, where `c` itself means `unassigned`
/// — together with the community count that replicate produced.
///
/// `None` means the draw was degenerate (too few cells left assigned to test anything); it
/// contributes to nothing rather than poisoning the tally.
///
/// This is a callback because the clustering and the ORA belong to [`super::term_ora`] — it
/// owns the pipeline — while the resampling belongs here. Passing it in keeps **one** loop
/// over the replicates: the panel is drawn once per `b` and both the fine and the shipped
/// label are read off the same draw, instead of two loops re-deriving the same centroids.
pub(super) type CoarseStep<'a> =
    dyn Fn(usize, &[usize], &[f32]) -> Result<Option<(Vec<usize>, usize)>> + Sync + 'a;

/// Bootstrap the marker panel and return the induced per-cell assignment distribution.
///
/// `feature_emb` is `[g × h]` and `cell_flat` `[n × h]`, both row-major; `type_markers[t]` is
/// the type's `(gene index, IDF weight)` list from
/// [`parse_and_match_markers`](super::markers::parse_and_match_markers). `coarse` is the
/// per-replicate pipeline step ([`CoarseStep`]); pass `None` to bootstrap the nearest-centroid
/// call alone.
///
/// Deterministic given `seed`: each resample is keyed by `(seed, draw, type)`, never by how
/// rayon scheduled it.
pub fn run_marker_bootstrap(
    feature_emb: &[f32],
    cell_flat: &[f32],
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
    cfg: &MarkerBootstrapConfig,
    seed: u64,
    coarse: Option<&CoarseStep<'_>>,
) -> Result<BootstrapResult> {
    let c = type_markers.len();
    let n = cell_flat.len() / h;
    let panel = LivePanel::new(feature_emb, type_markers, h);
    panel.warn_unusable()?;

    // Replicates are independent. Each is dominated by the *serial* Leiden pass inside
    // `coarse`, which left most cores idle when they ran one at a time — fanning out here
    // overlaps those stretches (the inner per-cell parallelism nests safely under rayon's
    // work-stealing). Ctrl+C keeps whatever finished; see `crate::stop::par_replicates`.
    let draws: Vec<Draw> = crate::stop::par_replicates(cfg.n_boot, "marker bootstrap", |b| {
        let mut centroids = vec![0f32; c * h];
        panel.resample_into(feature_emb, h, seed, b, &mut centroids);
        let (fine, gap) = assign_with_margin(cell_flat, n, &centroids, c, h, &panel.usable);
        let shipped = match coarse {
            Some(f) => f(b, &fine, &centroids)?,
            None => None,
        };
        Ok(Draw {
            fine,
            gap,
            centroids,
            shipped,
        })
    })?;

    // Tally. The fine call is read off every draw; the shipped label only off those the
    // pipeline could actually complete.
    let mut fine_post = vec![0f32; n * c];
    let mut coarse_post = vec![0f32; n * (c + 1)];
    let mut centroid_sum = vec![0f64; c * h];
    let mut centroid_sq = vec![0f64; c * h];
    let mut gap_sum = vec![0f64; n];
    let mut n_comm = Vec::new();
    for d in &draws {
        for (i, (&t, &g)) in d.fine.iter().zip(&d.gap).enumerate() {
            fine_post[i * c + t] += 1.0;
            gap_sum[i] += f64::from(g);
        }
        for (k, &v) in d.centroids.iter().enumerate() {
            centroid_sum[k] += f64::from(v);
            centroid_sq[k] += f64::from(v) * f64::from(v);
        }
        if let Some((per_cell, m)) = &d.shipped {
            n_comm.push(*m);
            for (i, &col) in per_cell.iter().enumerate() {
                coarse_post[i * (c + 1) + col] += 1.0;
            }
        }
    }

    Ok(summarize(
        Summary {
            n,
            c,
            h,
            n_draws: draws.len(),
            fine_post,
            coarse_post: coarse.is_some().then_some(coarse_post),
            n_comm,
            centroid_sum,
            centroid_sq,
            gap_sum,
            live: &panel.live,
            type_markers,
            feature_emb,
            cell_flat,
        },
        cfg,
    ))
}

/// One replicate's raw output, before it is tallied.
struct Draw {
    /// Nearest-centroid call per cell, against this draw's resampled centroids.
    fine: Vec<usize>,
    /// Per cell, the gap between the nearest and second-nearest centroid — the margin the
    /// assignment actually turned on.
    gap: Vec<f32>,
    /// This draw's `[c × h]` resampled centroids (kept for the jitter moments).
    centroids: Vec<f32>,
    /// The shipped label per cell + the community count, when a [`CoarseStep`] ran.
    shipped: Option<(Vec<usize>, usize)>,
}

/// Cell `i`'s nearest usable centroid, and how close the runner-up came.
#[inline]
fn nearest(cell: &[f32], centroids: &[f32], c: usize, h: usize, usable: &[bool]) -> (usize, f32) {
    let (mut best, mut d1, mut d2) = (0usize, f32::INFINITY, f32::INFINITY);
    for t in 0..c {
        if !usable[t] {
            continue;
        }
        let d = sq_dist(cell, &centroids[t * h..(t + 1) * h]);
        if d < d1 {
            d2 = d1;
            d1 = d;
            best = t;
        } else if d < d2 {
            d2 = d;
        }
    }
    let gap = if d2.is_finite() {
        d2.max(0.0).sqrt() - d1.max(0.0).sqrt()
    } else {
        f32::NAN
    };
    (best, gap)
}

/// Assign every cell to its nearest usable centroid, keeping how close the runner-up came.
pub(super) fn assign_with_margin(
    cell_flat: &[f32],
    n: usize,
    centroids: &[f32],
    c: usize,
    h: usize,
    usable: &[bool],
) -> (Vec<usize>, Vec<f32>) {
    (0..n)
        .into_par_iter()
        .map(|i| nearest(&cell_flat[i * h..(i + 1) * h], centroids, c, h, usable))
        .unzip()
}

/// [`assign_with_margin`] without the margin — for callers that discard it.
///
/// The null passes run this `P × B` times (two *million* calls at `P = 10 000`, `B = 200`) and
/// never look at the gaps, so materializing the `n`-cell gap vector each time is pure allocator
/// churn. Same loop, one output.
pub(super) fn assign_nearest(
    cell_flat: &[f32],
    n: usize,
    centroids: &[f32],
    c: usize,
    h: usize,
    usable: &[bool],
) -> Vec<usize> {
    (0..n)
        .into_par_iter()
        .map(|i| nearest(&cell_flat[i * h..(i + 1) * h], centroids, c, h, usable).0)
        .collect()
}

struct Summary<'a> {
    n: usize,
    c: usize,
    h: usize,
    /// Replicates that actually ran (a degenerate draw is dropped, not counted).
    n_draws: usize,
    fine_post: Vec<f32>,
    /// `[n × (c+1)]` shipped-label tallies — `None` when no [`CoarseStep`] ran.
    coarse_post: Option<Vec<f32>>,
    n_comm: Vec<usize>,
    centroid_sum: Vec<f64>,
    centroid_sq: Vec<f64>,
    gap_sum: Vec<f64>,
    live: &'a [LiveMarkers],
    type_markers: &'a [Vec<(u32, f32)>],
    feature_emb: &'a [f32],
    cell_flat: &'a [f32],
}

fn summarize(s: Summary<'_>, cfg: &MarkerBootstrapConfig) -> BootstrapResult {
    let Summary {
        n,
        c,
        h,
        n_draws,
        mut fine_post,
        coarse_post,
        n_comm,
        centroid_sum,
        centroid_sq,
        gap_sum,
        live,
        type_markers,
        feature_emb,
        cell_flat,
    } = s;
    let b = n_draws.max(1) as f64;
    for p in &mut fine_post {
        *p /= b as f32;
    }
    let post = fine_post;

    // The shipped label's consensus, over the replicates the pipeline could complete. The tally →
    // (label, set, support, entropy) reduction is `enrichment::consensus`, shared with the
    // raw-count bootstrap so there is one definition of what counts as agreement.
    let coarse = coarse_post.map(|cp| {
        let n_drawn = n_comm.len().max(1);
        let Consensus {
            post,
            label,
            label_set,
            support,
            set_support,
            entropy,
        } = summarize_consensus(cp, n, c + 1, n_drawn, &abstain_cfg(cfg));
        CoarseConsensus {
            post,
            label_set,
            set_support,
            label,
            support,
            entropy,
            n_comm,
        }
    });
    // Mean centroid, and the RMS distance the centroid travelled across resamples — the
    // noise in the decision variable, in cell-distance units.
    let mean: Vec<f32> = centroid_sum.iter().map(|&v| (v / b) as f32).collect();
    let jitter: Vec<f32> = (0..c)
        .map(|t| {
            let var: f64 = (0..h)
                .map(|j| {
                    let k = t * h + j;
                    (centroid_sq[k] / b - (centroid_sum[k] / b).powi(2)).max(0.0)
                })
                .sum();
            var.sqrt() as f32
        })
        .collect();

    let ln_c = (c as f32).ln();
    let mut assign = vec![UNASSIGNED; n];
    let mut dist = vec![f32::NAN; n];
    let mut support = vec![0f32; n];
    let mut entropy = vec![0f32; n];
    for i in 0..n {
        let row = &post[i * c..(i + 1) * c];
        let (best, top, second) = top_two(row);
        support[i] = top;
        entropy[i] = if ln_c > 0.0 {
            -row.iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f32>()
                / ln_c
        } else {
            0.0
        };
        // The call stands only if it is reproducible under resampling of its own evidence.
        if cfg.abstain.allows(top, second, n_draws.max(1)) {
            assign[i] = best;
            dist[i] = sq_dist(
                &cell_flat[i * h..(i + 1) * h],
                &mean[best * h..(best + 1) * h],
            )
            .max(0.0)
            .sqrt();
        }
    }

    // Which listed markers does the embedding actually corroborate? A marker far from its own
    // type's centroid, relative to the type's own spread, is one the embedding places
    // somewhere else entirely.
    let mut marker_dev: Vec<Vec<f32>> = type_markers
        .iter()
        .map(|m| vec![f32::NAN; m.len()])
        .collect();
    let mut marker_live: Vec<Vec<bool>> =
        type_markers.iter().map(|m| vec![false; m.len()]).collect();
    for t in 0..c {
        let ct = &mean[t * h..(t + 1) * h];
        let d: Vec<f32> = live[t]
            .iter()
            .map(|&(gi, _, _)| {
                sq_dist(&feature_emb[gi * h..(gi + 1) * h], ct)
                    .max(0.0)
                    .sqrt()
            })
            .collect();
        let (med, mad) = median_mad(&d);
        for (k, &(_, _, j)) in live[t].iter().enumerate() {
            marker_live[t][j] = true;
            marker_dev[t][j] = if mad > 0.0 { (d[k] - med) / mad } else { 0.0 };
        }
    }

    let type_qc = (0..c)
        .map(|t| {
            let mine: Vec<usize> = (0..n).filter(|&i| assign[i] == t).collect();
            let mean_of = |f: &dyn Fn(usize) -> f64| -> f32 {
                if mine.is_empty() {
                    f32::NAN
                } else {
                    (mine.iter().map(|&i| f(i)).sum::<f64>() / mine.len() as f64) as f32
                }
            };
            TypeQc {
                n_live: live[t].len(),
                centroid_jitter: jitter[t],
                decision_gap: mean_of(&|i| gap_sum[i] / b),
                mean_support: mean_of(&|i| f64::from(support[i])),
                occupancy: mine.len() as f32 / n as f32,
            }
        })
        .collect();

    BootstrapResult {
        c,
        n_draws,
        post,
        assign,
        dist,
        support,
        entropy,
        coarse,
        marker_dev,
        marker_live,
        type_qc,
    }
}

/// Median and median-absolute-deviation (scaled to a normal-consistent σ).
fn median_mad(x: &[f32]) -> (f32, f32) {
    if x.is_empty() {
        return (f32::NAN, 0.0);
    }
    let med = median(x);
    let dev: Vec<f32> = x.iter().map(|&v| (v - med).abs()).collect();
    (med, 1.4826 * median(&dev))
}

fn median(x: &[f32]) -> f32 {
    let mut v = x.to_vec();
    v.sort_by(f32::total_cmp);
    let m = v.len() / 2;
    if v.len().is_multiple_of(2) {
        0.5 * (v[m - 1] + v[m])
    } else {
        v[m]
    }
}

/// Squared Euclidean distance, kept **in `f32` end to end**.
///
/// This is the hot loop — it runs `n × c` times per replicate (≈73 M calls over a 200-draw
/// bootstrap of 15 k cells × 24 types). Widening each element to `f64` inside it, as this used
/// to, forces a per-element convert and leaves LLVM emitting 4-wide `f64` lanes at best;
/// staying in `f32` lets it emit 8-wide AVX. The precision is not close to binding: with
/// `h ≈ 64` and coordinates of order 10, the accumulated relative error is ~1e-5, four orders
/// below the ~1% margin these distances are actually compared at.
#[inline]
pub(super) fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}
