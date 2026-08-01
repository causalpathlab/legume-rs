//! Null-cell calling: which WT cells actually carry APOBEC1-YTH editing.
//!
//! dartseq's existing cell QC — droplet calling ([`crate::cell_qc::call_cells`]),
//! the `cell_min_genes` complexity floor, the mitochondrial-fraction filter — is
//! entirely **expression**-based. None of it looks at editing, so a perfectly
//! healthy cell that expresses no functional fusion protein passes every check and
//! then contributes coverage-without-signal to every site's contrast. Measured on
//! DART data at MYC: 90.9% of cells covering the gene contribute *zero* converted
//! reads while carrying 74.5% of its coverage.
//!
//! This module adds the missing stage. It is **QC, not a hypothesis test** — the
//! same two-populations-one-cutoff shape as EmptyDrops, and deliberately not
//! multiple testing:
//!
//! * a false-discovery rate over *cells* is not a quantity anyone wants controlled
//!   — a stray null cell costs a little dilution, an excluded competent cell costs
//!   coverage, and neither is what an FDR bounds;
//! * a BH threshold moves with the number of cells, which is set by the depth
//!   floor — a QC knob. Measured: ~2% of verdicts flip on a fixed set of cells
//!   purely by changing how many *other* cells are in the pool;
//! * the per-cell tail probabilities need an empirical 1.4–1.6× overdispersion
//!   inflation before they are even calibrated, so a nominal `q` promises nothing.
//!
//! Instead the cut is placed by a QC criterion with no `α`: **cut as aggressively
//! as possible while the discarded population remains indistinguishable from the
//! catalytically-dead control**. That is well posed here in a way it is not for
//! EmptyDrops — EmptyDrops must *estimate* an ambient profile from low-count
//! droplets, whereas the MUT library is a *measured* null at matched depth.
//!
//! Measured on rep1/2/3: the criterion keeps 43.2 / 39.8 / 40.9% of cells at
//! rejected-over-control ratios of 1.17 / 1.57 / 1.03, and correctly refuses a
//! top-15% cut, where the discarded pool still edits 2.0–2.7× above control.

use crate::common::*;
use genomic_data::gff::GffRecordMap;
use genomic_data::sam::CellBarcode;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

pub mod scan;

///////////////////////
// Per-cell activity //
///////////////////////

/// A cell's editing tally: converted reads out of covered reads, aggregated over
/// every candidate position genome-wide. Both arms use the same struct; MUT
/// supplies the null, WT is what gets called.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CellActivity {
    /// Reads showing the conversion (C→T forward, G→A reverse) at motif candidates.
    pub edited: u64,
    /// Reads covering a motif candidate with either the reference or converted base.
    pub covered: u64,
    /// Same, at reference bases of the same channel that are **not** motif
    /// candidates and sit well away from any — the promiscuous, m6A-independent
    /// deamination. **Reported, never used in the cut.**
    ///
    /// This is the only per-cell readout of APOBEC1-YTH activity available: the
    /// construct itself is invisible to gene quantification, because the fusion
    /// uses *rat* APOBEC1 (does not map to the human reference) and 10x 3'
    /// chemistry captures the vector 3'UTR, which is absent from the assembly.
    /// Endogenous YTHDF2 expression is no substitute — measured, it correlates
    /// +0.14 with actual activity, with competent and null cells at medians 7 vs 6.
    ///
    /// Not used as a per-cell denominator: normalising the motif rate by a cell's
    /// own background needed 3.4–4.7x variance inflation to calibrate, because
    /// the per-cell background is itself a noisy estimate.
    pub bg_edited: u64,
    pub bg_covered: u64,
}

impl CellActivity {
    pub fn add(&mut self, edited: u64, covered: u64) {
        self.edited += edited;
        self.covered += covered;
    }

    pub fn add_background(&mut self, edited: u64, covered: u64) {
        self.bg_edited += edited;
        self.bg_covered += covered;
    }

    /// Promiscuous (non-motif) conversion rate — the APOBEC1-YTH activity proxy.
    pub fn background_rate(&self) -> f64 {
        if self.bg_covered == 0 {
            0.0
        } else {
            self.bg_edited as f64 / self.bg_covered as f64
        }
    }

    /// Observed conversion rate; `0.0` when the cell has no coverage (never NaN).
    pub fn rate(&self) -> f64 {
        if self.covered == 0 {
            0.0
        } else {
            self.edited as f64 / self.covered as f64
        }
    }
}

impl std::ops::AddAssign<&CellActivity> for CellActivity {
    fn add_assign(&mut self, other: &Self) {
        self.edited += other.edited;
        self.covered += other.covered;
        self.bg_edited += other.bg_edited;
        self.bg_covered += other.bg_covered;
    }
}

/// Per-cell tallies for one library. Built by [`scan`], consumed by
/// [`call_competent_cells`].
pub type ActivityTally = FxHashMap<CellBarcode, CellActivity>;

/// The editing-competent cells of each signal BAM, keyed by that BAM's path.
///
/// Per library rather than one flat set: barcodes are only unique within a
/// single 10x run, so a shared set would let a competent cell in one library
/// rescue the null cell that happens to share its barcode in another.
///
/// The value is an `Arc` because `restrict_to_competent_cells` re-applies it
/// once per gene per BAM inside the rayon loop: an owned set there would deep
/// clone thousands of barcodes per gene and, worse, have every worker bump the
/// same `Arc<str>` refcounts concurrently — true sharing that caps thread scaling.
pub type CompetentCells = FxHashMap<Box<str>, Arc<FxHashSet<CellBarcode>>>;

/// The control cells a null cell is compared against — counts only, no
/// barcodes, since the null needs no identity and keying by barcode would merge
/// unrelated cells that different libraries happen to share.
///
/// DART supplies this from the YTHmut arm: a *measured* null at matched depth.
/// There is deliberately no fixed-rate alternative for A-to-I. DART's
/// de-dilution works because the reporter is **transfected** — APOBEC1-YTH
/// either got expressed or did not, giving two populations (measured: 0.88% vs
/// 0.196%, the latter sitting on the dead-enzyme control). ADAR is endogenous
/// and graded, so a "null cell" may not exist as a population at all, and
/// cutting a continuum against an error floor would look like it worked while
/// meaning something else. Establish bimodality first, then add it back.
pub type ControlCells = [CellActivity];

/// Pooled rate over a set of cells — the coverage-weighted mean, which is what
/// the QC criterion compares (not the mean of per-cell rates, which would let a
/// shallow cell outvote a deep one).
pub fn pooled_rate<'a>(cells: impl Iterator<Item = &'a CellActivity>) -> f64 {
    let (mut e, mut n) = (0u64, 0u64);
    for c in cells {
        e += c.edited;
        n += c.covered;
    }
    if n == 0 {
        0.0
    } else {
        e as f64 / n as f64
    }
}

////////////////////////////////
// Method-of-moments beta-bin //
////////////////////////////////

/// Method-of-moments fit of `y_i ~ BetaBinomial(n_i, mean, rho)`, after
/// Kleinman, J.C. (1973), "Proportions with extraneous variance: single and
/// independent samples", *JASA* 68(341):46–54.
///
/// `rho` is the *between-cell* overdispersion: how much more the per-cell rates
/// scatter than binomial sampling alone allows. Under a plain binomial the
/// weighted scatter `S = Σ nᵢ(p̂ᵢ − m)²` has expectation `m(1−m)(k−1)` — Pearson's
/// chi-square — so the *excess* over that is what identifies ρ:
///
/// ```text
/// E[S] = m(1−m)·[(k−1) + ρ·(N − Σnᵢ²/N)]
///   ⇒  ρ̂ = [ S/(m(1−m)) − (k−1) ] / (N − Σnᵢ²/N)
/// ```
///
/// Chosen over a beta-binomial MLE because it is **closed form** — no iteration,
/// so the call is deterministic — and because it handles the wildly unequal
/// `nᵢ` that cells of different coverage produce. Assuming `ρ = 0` is not safe:
/// measured on DART control cells it is ~3e-4, and the variance inflation is
/// `1+(n−1)ρ`, so at `n = 10,000` that is ≈4× — deep cells would score ~2× too
/// extreme, which is exactly the "deep null cell mistaken for a competent one"
/// failure.
///
/// Returns `(mean, rho)` with ρ clamped to `[0, 1)`; with fewer than 3 covered
/// cells there is no dispersion to estimate, so `rho = 0`.
pub fn fit_betabinom_mom(counts: &[(u64, u64)]) -> (f64, f64) {
    let live: Vec<(f64, f64)> = counts
        .iter()
        .filter(|(_, n)| *n > 0)
        .map(|(y, n)| (*y as f64, *n as f64))
        .collect();
    let k = live.len();
    let total_n: f64 = live.iter().map(|(_, n)| n).sum();
    if total_n <= 0.0 {
        return (0.0, 0.0);
    }
    let mean = live.iter().map(|(y, _)| y).sum::<f64>() / total_n;
    if k < 3 || mean <= 0.0 || mean >= 1.0 {
        return (mean, 0.0);
    }
    let s: f64 = live.iter().map(|(y, n)| n * (y / n - mean).powi(2)).sum();
    let sum_sq: f64 = live.iter().map(|(_, n)| n * n).sum();
    let denom = total_n - sum_sq / total_n;
    if denom <= 0.0 {
        return (mean, 0.0);
    }
    let rho = (s / (mean * (1.0 - mean)) - (k as f64 - 1.0)) / denom;
    (mean, rho.clamp(0.0, 0.99))
}

//////////////////////
// Depth strata     //
//////////////////////

/// Equal-count strata over `values`, `0..n_strata`. Deterministic: ties break by
/// index, so the same input always yields the same labels.
///
/// Cells are stratified by coverage before scoring because the null distribution
/// of a conversion rate depends on its denominator. Ranking a 100-read cell
/// against a 10,000-read cell on raw rate ranks them by sampling noise.
pub fn quantile_strata(values: &[f64], n_strata: usize, min_per: usize) -> Vec<usize> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let k = n_strata.min(n / min_per.max(1)).max(1);
    if k <= 1 {
        return vec![0; n];
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut label = vec![0usize; n];
    for (rank, &i) in order.iter().enumerate() {
        label[i] = rank * k / n;
    }
    label
}

////////////////////////
// The null-cell call //
////////////////////////

/// How much the discarded pool may still edit, as a multiple of the control.
/// Read by both [`NullCallOpts::default`] and the `--cell-scan-tolerance` clap
/// default, so the shipped binary and the tests cannot disagree.
///
/// **1.0 is the data-driven, parameter-free point**: cut exactly where the
/// discarded pool's conversion rate equals the control's, i.e. where "what I am
/// throwing away is indistinguishable from the dead enzyme" is literally true
/// rather than approximately so. Nothing is tuned — the data picks the cut.
///
/// Measured on chr19+MYC it lands at 53.8% of cells kept, with dropped 0.0942%
/// against control 0.0942%, equivalent to the control's p95. It is deliberately
/// the conservative choice: a sweep on rep1 found ACTB recovery still climbing
/// past it (8 -> 13 sites at tolerance 2.0, with GAPDH still correctly rejected),
/// so a user optimising recall has measured reason to raise it — and the logged
/// `dropped/control` tells them exactly what they are spending to do so.
pub const DEFAULT_REJECT_TOLERANCE: f64 = 1.0;

/// Minimum coverage for a cell to be scored at all.
pub const DEFAULT_SCAN_MIN_COVERAGE: u64 = 50;

/// Knobs for [`call_competent_cells`].
#[derive(Clone, Copy, Debug)]
pub struct NullCallOpts {
    /// Minimum coverage for a cell to be scored at all. Below this the rate is
    /// too noisy to rank; such cells are rejected without a score.
    pub min_coverage: u64,
    /// Number of coverage strata (equal-count).
    pub n_strata: usize,
    /// Minimum cells per stratum; fewer ⇒ fewer strata.
    pub min_per_stratum: usize,
    /// The QC tolerance: keep cutting while the *discarded* pool edits at no more
    /// than this multiple of the control. `1.0` (the default) is the
    /// parameter-free point — cut where the discarded pool *equals* the control.
    /// Raising it trades that guarantee for a more concentrated kept pool.
    /// Physical meaning throughout, unlike an FDR level.
    pub reject_tolerance: f64,
    /// Never reject more than this fraction, however the ratio behaves — a guard
    /// against a pathological control making everything look competent.
    pub max_reject_frac: f64,
    /// Place the cut directly on the control's own scale: keep WT cells scoring
    /// above `1 - control_tail` of depth-matched control cells. `0` disables it
    /// and the [`Self::reject_tolerance`] sweep decides instead.
    ///
    /// The two agree closely in practice — measured on rep1, a `0.05` tail keeps
    /// 44.6% of cells where the `1.2` tolerance keeps 40.4% — so this is mainly
    /// a matter of which knob is easier to reason about. The tail says what it
    /// means ("edits more than 95% of control cells at the same depth"); the
    /// tolerance is self-calibrating but abstract.
    ///
    /// It is a quantile on an empirical reference, **not** a test: no p-value, no
    /// multiplicity, and the tail is not an error rate. Held-out control cells
    /// exceed their own `p95` about 6% of the time rather than 5%, because
    /// control cells are themselves heterogeneous in fusion expression.
    pub control_tail: f64,
}

impl Default for NullCallOpts {
    fn default() -> Self {
        Self {
            min_coverage: DEFAULT_SCAN_MIN_COVERAGE,
            n_strata: 12,
            min_per_stratum: 50,
            reject_tolerance: DEFAULT_REJECT_TOLERANCE,
            max_reject_frac: 0.95,
            control_tail: 0.0,
        }
    }
}

/// The outcome, including everything worth logging as QC.
#[derive(Clone, Debug)]
pub struct NullCellCall {
    /// Cells that edit — the set handed to discovery as `CellMembership`.
    pub selected: FxHashSet<CellBarcode>,
    pub n_scored: usize,
    /// Pooled conversion rate of the kept cells.
    pub selected_rate: f64,
    /// Pooled conversion rate of the discarded cells.
    pub rejected_rate: f64,
    /// Pooled conversion rate of the control arm.
    pub control_rate: f64,
    /// Percentile of depth-matched CONTROL cells that the weakest kept cell
    /// exceeds — i.e. where the cut sits on the control's own scale.
    pub control_percentile: f64,
}

impl NullCellCall {
    pub fn n_selected(&self) -> usize {
        self.selected.len()
    }

    pub fn kept_frac(&self) -> f64 {
        if self.n_scored == 0 {
            0.0
        } else {
            self.n_selected() as f64 / self.n_scored as f64
        }
    }

    /// The QC number: ≈1 means the discarded cells are indistinguishable from
    /// the dead enzyme, i.e. nothing real was thrown away. Derived rather than
    /// stored — as a field the two construction sites disagreed on the
    /// no-control case (`NaN` vs a computed ratio).
    pub fn rejected_over_control(&self) -> f64 {
        self.rejected_rate / self.control_rate
    }
}

/// Depth-adjusted QC score: how far a cell's rate sits above its stratum's
/// control mean, in units of the beta-binomial standard deviation at that cell's
/// coverage.
///
/// Deliberately a standardized deviate rather than a tail probability. A p-value
/// underflows to exactly 0 for strongly-editing cells, which collapses the top of
/// the ranking into an arbitrary tie — fatal for a threshold sweep. The deviate
/// stays continuous and monotone, and it never invites reading a `q` off it.
fn stratum_score(a: &CellActivity, mean: f64, rho: f64) -> f64 {
    let n = a.covered as f64;
    if n <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let m = mean.clamp(1e-9, 1.0 - 1e-9);
    let var = m * (1.0 - m) / n * (1.0 + (n - 1.0).max(0.0) * rho);
    if var <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (a.rate() - m) / var.sqrt()
}

/// Call the editing-competent WT cells against a depth-matched control.
///
/// 1. score every WT cell against its coverage stratum's control null,
/// 2. rank by that score,
/// 3. sweep the cut and stop where the discarded pool stops looking like the
///    control.
///
/// The sweep is monotone in the obvious direction — keeping more cells leaves a
/// purer null in the rejected pool — so it walks the ranking from "reject
/// everything" toward "reject nothing" and takes the first cut whose rejected
/// pool is within tolerance. That is the most aggressive defensible cut.
pub fn call_competent_cells(
    wt: &ActivityTally,
    control: &ControlCells,
    opts: &NullCallOpts,
) -> NullCellCall {
    // The scoreable control cells, resolved ONCE: this same predicate decides
    // the pooled control rate, the stratum fits and the control scores, and the
    // three must see identical cells in identical order for the `n_wt + i`
    // stratum offset below to address the right cell.
    let ctrl: Vec<&CellActivity> = control
        .iter()
        .filter(|c| c.covered >= opts.min_coverage)
        .collect();
    let control_rate = pooled_rate(ctrl.iter().copied());

    // Scoreable WT cells, in a deterministic order.
    let mut cells: Vec<(&CellBarcode, &CellActivity)> = wt
        .iter()
        .filter(|(_, a)| a.covered >= opts.min_coverage)
        .collect();
    cells.sort_by(|a, b| a.0.cmp(b.0));
    if cells.is_empty() || control_rate <= 0.0 {
        // No null to calibrate against: refuse to cut rather than silently
        // discard the arm.
        return NullCellCall {
            selected: wt.keys().cloned().collect(),
            n_scored: 0,
            control_percentile: f64::NAN,
            selected_rate: pooled_rate(wt.values()),
            rejected_rate: 0.0,
            control_rate,
        };
    }
    let n_wt = cells.len();

    // The scoreable control cells, resolved ONCE: this same predicate decides
    // the pooled control rate, the stratum fits and the control scores, and the
    // three must see identical cells in identical order for the `n_wt + i`
    // stratum offset below to address the right cell.
    let ctrl: Vec<&CellActivity> = control
        .iter()
        .filter(|c| c.covered >= opts.min_coverage)
        .collect();

    // Stratify by coverage and fit each stratum's null. Both arms go on one
    // coverage scale and the fit uses the CONTROL only — never WT, which would
    // absorb the very signal being called.
    let mut depths: Vec<f64> = cells.iter().map(|(_, a)| a.covered as f64).collect();
    depths.extend(ctrl.iter().map(|c| c.covered as f64));
    let strata = quantile_strata(&depths, opts.n_strata, opts.min_per_stratum);
    let n_strata = strata.iter().copied().max().unwrap_or(0) + 1;
    // Bucket the control cells once rather than rescanning all of them
    // per stratum, and write the `n_wt + i` offset in exactly one place.
    let mut by_stratum: Vec<Vec<(u64, u64)>> = vec![Vec::new(); n_strata];
    for (i, c) in ctrl.iter().enumerate() {
        by_stratum[strata[n_wt + i]].push((c.edited, c.covered));
    }
    let null_by_stratum: Vec<(f64, f64)> = by_stratum
        .iter()
        .map(|counts| {
            // Too few cells to estimate a dispersion: fall back to the
            // pooled control rate rather than a noisy per-stratum fit.
            if counts.len() >= 3 {
                fit_betabinom_mom(counts)
            } else {
                (control_rate, 0.0)
            }
        })
        .collect();

    // Score the CONTROL cells the same way, so the cut can be reported on a scale
    // a reader can interpret. Purely descriptive — no test, no threshold.
    let control_scores: Vec<f64> = {
        let mut v: Vec<f64> = ctrl
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let k = strata.get(n_wt + i).copied().unwrap_or(0);
                let (m, rho) = null_by_stratum
                    .get(k)
                    .copied()
                    .unwrap_or((control_rate, 0.0));
                stratum_score(c, m, rho)
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v
    };

    // Rank by the depth-adjusted deviate, best first; ties by barcode.
    let mut ranked: Vec<(f64, &CellBarcode, &CellActivity)> = cells
        .iter()
        .enumerate()
        .map(|(i, (cb, a))| {
            let (m, rho) = null_by_stratum[strata[i]];
            (stratum_score(a, m, rho), *cb, *a)
        })
        .collect();
    ranked.sort_by(|x, y| {
        y.0.partial_cmp(&x.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.1.cmp(y.1))
    });

    // Suffix sums: the rejected pool for a cut at `k` is ranks `k..n`.
    let n = ranked.len();
    let mut suf_e = vec![0u64; n + 1];
    let mut suf_n = vec![0u64; n + 1];
    for i in (0..n).rev() {
        suf_e[i] = suf_e[i + 1] + ranked[i].2.edited;
        suf_n[i] = suf_n[i + 1] + ranked[i].2.covered;
    }
    let ratio_at = |k: usize| -> f64 {
        if suf_n[k] == 0 {
            0.0
        } else {
            (suf_e[k] as f64 / suf_n[k] as f64) / control_rate
        }
    };

    // Walk from the most aggressive cut toward the least and take the first that
    // is within tolerance. `min_keep` guards against rejecting everything.
    let min_keep = ((1.0 - opts.max_reject_frac) * n as f64).ceil() as usize;
    let mut cut = n;
    if opts.control_tail > 0.0 && !control_scores.is_empty() {
        // Cut on the control's own scale: keep cells scoring above the
        // (1 - tail) quantile of depth-matched control cells.
        let idx = ((1.0 - opts.control_tail) * control_scores.len() as f64).floor() as usize;
        let threshold = control_scores[idx.min(control_scores.len() - 1)];
        cut = ranked
            .partition_point(|(s, _, _)| *s > threshold)
            .max(min_keep.max(1));
    } else {
        for k in min_keep.max(1)..=n {
            if ratio_at(k) <= opts.reject_tolerance {
                cut = k;
                break;
            }
        }
    }

    // Where the cut lands among control cells: "kept everything above the Nth
    // percentile of depth-matched controls". Far more legible than a tolerance,
    // and it is a description of the chosen cut, not a criterion for choosing it.
    let control_percentile = match ranked.get(cut.saturating_sub(1)) {
        Some((boundary, _, _)) if !control_scores.is_empty() => {
            let below = control_scores.partition_point(|v| v < boundary);
            100.0 * below as f64 / control_scores.len() as f64
        }
        _ => f64::NAN,
    };

    let selected: FxHashSet<CellBarcode> = ranked[..cut]
        .iter()
        .map(|(_, cb, _)| (*cb).clone())
        .collect();
    let sel_rate = pooled_rate(ranked[..cut].iter().map(|(_, _, a)| *a));
    let rej_rate = pooled_rate(ranked[cut..].iter().map(|(_, _, a)| *a));

    NullCellCall {
        selected,
        n_scored: n,
        control_percentile,
        selected_rate: sel_rate,
        rejected_rate: rej_rate,
        control_rate,
    }
}

#[cfg(test)]
mod tests;

///////////////////////////
// CLI + pipeline driver //
///////////////////////////

/// Shared knobs, `#[command(flatten)]`-ed into both `faba dartseq` and
/// `faba all` so the two paths cannot drift apart.
///
/// There is deliberately no off switch. Discovering on cells in which the
/// reporter never worked is not an alternative analysis, it is a diluted one:
/// those cells contribute coverage without signal and bury real sites (measured
/// on DART data, ~91% of the cells covering MYC convert nothing while carrying
/// 74% of its coverage, and MYC is called only once they are dropped). The scan
/// no-ops on its own when there is nothing to calibrate against — see
/// [`call_and_report`].
#[derive(Args, Debug, Clone, serde::Serialize)]
pub struct CellScanArgs {
    /// Keep WT cells above this upper-tail fraction of depth-matched CONTROL
    /// cells. `0` (default) uses `--cell-scan-tolerance` instead.
    #[arg(long = "cell-scan-control-tail", default_value_t = 0.0)]
    pub cell_scan_control_tail: f64,

    /// Diagnostic only: m6A reader genes to summarise per cell, by symbol
    #[arg(long = "reader-genes", value_delimiter = ',')]
    pub reader_genes: Vec<String>,

    /// Diagnostic only: m6A writer/eraser genes to summarise per cell, by symbol
    #[arg(long = "writer-genes", value_delimiter = ',')]
    pub writer_genes: Vec<String>,

    /// Override the data-driven cut: how much the DISCARDED cells may still edit,
    /// as a multiple of the control
    #[arg(
        long = "cell-scan-tolerance",
        help = "Override the data-driven cut (multiple of the control rate the discarded pool may reach)",
        long_help = "Override the data-driven cut.\n\
                     \n\
                     Unset (the default), the cut is decided from the data:\n\
                     cells are dropped up to the point where the discarded pool's conversion rate EQUALS the control's,\n\
                     so nothing demonstrably real is thrown away. Nothing is tuned.\n\
                     \n\
                     Set it to cut deeper and concentrate the kept pool.\n\
                     e.g. `1.5` permits the discarded pool to edit 50% above the control.\n\
                     Every run logs `dropped/control` and the equivalent control percentile,\n\
                     so the cost of raising it is visible. Measured on rep1,\n\
                     ACTB recovery was still climbing at 2.0 (8 -> 13 sites, GAPDH still rejected),\n\
                     so a recall-oriented user has reason to;\n\
                     the default is the defensible choice, not the maximal one.\n\
                     \n\
                     `--cell-scan-control-tail` takes precedence over this if both are set."
    )]
    pub cell_scan_tolerance: Option<f64>,

    /// Minimum conversion-site coverage for a cell to be scored
    #[arg(long = "cell-scan-min-coverage", default_value_t = DEFAULT_SCAN_MIN_COVERAGE)]
    pub cell_scan_min_coverage: u64,

    /// Drop null cells from the OUTPUT matrices too, not just from site calling
    #[arg(
        long = "quantify-competent-only",
        help = "Restrict the m6A matrices to editing-competent cells as well"
    )]
    pub quantify_competent_only: bool,
}

impl CellScanArgs {
    fn opts(&self) -> NullCallOpts {
        NullCallOpts {
            min_coverage: self.cell_scan_min_coverage,
            reject_tolerance: self.cell_scan_tolerance.unwrap_or(DEFAULT_REJECT_TOLERANCE),
            control_tail: self.cell_scan_control_tail,
            ..NullCallOpts::default()
        }
    }
}

/// Tally per-cell editing, call the competent cells against the control, write
/// the audit, and return the set to restrict the signal arm to.
///
/// `None` means "keep every cell" — returned whenever there is nothing to
/// calibrate against (no control arm, an empty scan, or a cut that would keep
/// nothing). Failing open matters: a silent empty set would delete the signal
/// arm entirely.
pub fn call_and_report(
    gff_map: &GffRecordMap,
    params: &crate::editing::pipeline::ConversionParams,
    args: &CellScanArgs,
    // Takes the whole QC result rather than two projections of it: both call
    // sites used to shred it into a cell map and a matrix-path vector with
    // identical hand-copied code, which is drift waiting to happen.
    gene_qc: Option<&crate::quant::GeneCountQc>,
    output_dir: &str,
    label: &str,
) -> anyhow::Result<Option<Arc<CompetentCells>>> {
    let qc_cells = gene_qc.map(|q| &q.cells_by_batch);
    let matrix_paths: Vec<Box<str>> = gene_qc
        .map(|q| q.matrix_by_batch.values().cloned().collect())
        .unwrap_or_default();
    let matrix_paths = matrix_paths.as_slice();
    // No control arm ⇒ nothing to calibrate a null against (A-to-I, or m6A run
    // without `--control-bam`). Not a disable switch: the call is impossible,
    // not unwanted.
    if params.mut_bam_files.is_empty() {
        info!("cell scan: no control arm; keeping every cell");
        return Ok(None);
    }

    // The scan is now the first thing in the m6A path to touch the reference, so
    // it owns the up-front check that `find_all_conversion_sites` does for
    // discovery. Without it a bad `--genome` surfaces as a panic inside a rayon
    // worker, after the whole gene-QC pass has already run.
    crate::data::util_htslib::load_fasta_index(&params.genome_file)?;

    // The null pools every control library's cells as counts. Safe to pool —
    // the null needs no identity — whereas the signal arms must stay separate.
    // Pooled across every control library, deliberately. Pairing a signal library
    // with "its" control cannot be expressed in general: file names need not encode
    // it, and the counts need not even match — one control serving three signal
    // libraries (`--wt a,b,c --control-bam x`) is a normal, economical design.
    //
    // The cost is that a control library which runs hot exports that to everyone.
    // Measured across three replicates, the pooled control read 0.2197% where two
    // of the libraries alone read 0.1634% and 0.1510% — a 34-45% inflation of the
    // null they were judged against. So report each library's own rate: pooling is
    // the right default, but an outlier control should not be invisible.
    let mut control = Vec::new();
    let mut per_library = Vec::new();
    for bam in &params.mut_bam_files {
        let mut tally = scan::scan_cell_activity(gff_map, params, bam, "control")?;
        retain_qc_cells(&mut tally, qc_cells, bam);
        let rate = pooled_rate(
            tally
                .values()
                .filter(|c| c.covered >= args.cell_scan_min_coverage),
        );
        per_library.push((bam.clone(), rate, tally.len()));
        control.extend(tally.into_values());
    }
    if per_library.len() > 1 {
        for (bam, rate, n) in &per_library {
            info!(
                "control library {bam}: {n} cells, C->U {:.4}%",
                100.0 * rate
            );
        }
        let rates: Vec<f64> = per_library.iter().map(|(_, r, _)| *r).collect();
        let (lo, hi) = rates
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        if lo > 0.0 && hi / lo > 1.25 {
            log::warn!(
                "control libraries differ {:.2}x in background editing ({:.4}% to {:.4}%); \
                 they are POOLED into one null, so the hotter library raises the bar for \
                 every signal library",
                hi / lo,
                100.0 * lo,
                100.0 * hi
            );
        }
    }
    if control.is_empty() {
        info!("cell scan: control tallied no cells; keeping every cell");
        return Ok(None);
    }
    let opts = args.opts();

    // One call per signal library, because barcodes are only unique within a
    // library and each library has its own transfection efficiency — so the
    // competent fraction, and the cut, legitimately differ between them.
    // Batch names must come from the same list the matrices use, or an audit can
    // be named for a batch that has no matrices: `uniq_batch_names` only appends
    // `_{i}` when the list *it is given* has colliding basenames, and the standard
    // `wt/possorted_genome_bam.bam` + `mut/possorted_genome_bam.bam` layout
    // collides only once the control arm is included.
    let quant = params.quant_bam_files();
    let quant_batches = uniq_batch_names(&quant)?;
    let batch_of: FxHashMap<&str, &str> = quant
        .iter()
        .zip(quant_batches.iter())
        .map(|(f, b)| (f.as_ref(), b.as_ref()))
        .collect();
    let mut by_batch = CompetentCells::default();
    for bam in params.wt_bam_files.iter() {
        let batch = batch_of.get(bam.as_ref()).copied().unwrap_or(bam.as_ref());
        let mut signal = scan::scan_cell_activity(gff_map, params, bam, "signal")?;
        let observed = signal.len();
        retain_qc_cells(&mut signal, qc_cells, bam);
        if signal.is_empty() {
            // Left out of the map ⇒ this library keeps every cell while the others
            // are de-diluted, and the pooled WT marginal silently mixes the two.
            log::warn!(
                "cell scan ({batch}): no cells tallied (missing {} tag?); this library \
                 stays UNDILUTED while the others are filtered",
                params.cell_barcode_tag
            );
            continue;
        }
        let scored = signal
            .values()
            .filter(|a| a.covered >= opts.min_coverage)
            .count();
        // The whole cell-QC cascade in one line. Reported per stage because the
        // stages answer different questions and can fail independently: droplet
        // calling asks "is this a real cell?", the coverage floor asks "is there
        // enough signal to rank it?", and the null call asks "does it edit?".
        info!(
            "cell QC cascade ({batch}): {observed} barcodes observed -> {qc} passed \
             droplet/complexity/mito QC ({qc_pct:.1}%) -> {scored} had coverage >= {floor} \
             ({scored_pct:.1}% of QC) -> see the competence call below",
            observed = observed,
            qc = signal.len(),
            qc_pct = 100.0 * signal.len() as f64 / observed.max(1) as f64,
            scored = scored,
            floor = opts.min_coverage,
            scored_pct = 100.0 * scored as f64 / signal.len().max(1) as f64,
        );
        if log::log_enabled!(log::Level::Info) {
            log_rate_histogram(&signal, &control, &opts);
        }
        let call = call_competent_cells(&signal, &control, &opts);
        info!(
            "cell QC ({batch}): competent {}/{} scored cells ({:.1}%); editing {:.4}% \
             kept vs {:.4}% dropped vs {:.4}% control; dropped/control {:.2}; \
             cut sits at control p{:.0}",
            call.n_selected(),
            call.n_scored,
            100.0 * call.kept_frac(),
            100.0 * call.selected_rate,
            100.0 * call.rejected_rate,
            100.0 * call.control_rate,
            call.rejected_over_control(),
            call.control_percentile
        );
        let rule = if args.cell_scan_control_tail > 0.0 {
            format!(
                "control p{:.0}",
                100.0 * (1.0 - args.cell_scan_control_tail)
            )
        } else if let Some(t) = args.cell_scan_tolerance {
            format!("tolerance {t} (overriding the data-driven cut)")
        } else {
            "data-driven (discarded pool == control)".to_string()
        };
        info!("cell QC ({batch}): cut placed by {rule}");
        if call.rejected_over_control() > opts.reject_tolerance {
            info!(
                "cell QC ({batch}): dropped cells still edit {:.2}x the control -- the \
                 cut is limited by --cell-scan-tolerance, not by the data",
                call.rejected_over_control()
            );
        }

        let path = format!("{output_dir}/{batch}_{label}_cell_qc.tsv.gz");
        write_audit(&path, &signal, &call, opts.min_coverage)?;
        info!("cell QC audit -> {path}");

        // Diagnostic only. Deliberately after the cut, and never consulted by it.
        for (lbl, genes) in [
            ("m6A readers", &args.reader_genes),
            ("m6A writers/erasers", &args.writer_genes),
        ] {
            if let Err(e) = log_family_expression(lbl, genes, matrix_paths, &signal, &call.selected)
            {
                log::warn!("{lbl}: could not summarise expression ({e})");
            }
        }

        if call.n_selected() == 0 {
            info!("cell QC ({batch}): selected no cells; keeping every cell instead");
            continue;
        }
        by_batch.insert(bam.clone(), Arc::new(call.selected));
    }

    Ok((!by_batch.is_empty()).then(|| Arc::new(by_batch)))
}

/// Intersect a per-BAM QC cell set with the competent cells, for the **signal**
/// BAMs only.
///
/// By default the null-cell call governs which SITES are found, while the output
/// matrices keep every QC-passing cell — that is what makes the m6A matrices
/// column-compatible with the gene/apa/atoi matrices, since faba's rule is that
/// every modality inherits one cell set. Opting in trades that away: measured on
/// whole-genome runs, ~60% of the cells in the matrix are null cells, so a
/// per-cell methylation rate computed from the default matrix reads low.
///
/// Control BAMs are never touched — they carry no competent set, and filtering
/// the control on activity is the bias this whole stage exists to avoid.
/// The cell set the m6A matrices are quantified over.
///
/// Returns `Some` only when `--quantify-competent-only` is set AND there is a
/// competence call to apply; the caller falls back to the full QC set otherwise.
///
/// Exists so `faba dartseq` and `faba all` cannot answer this differently. They
/// did: the flag was wired into the standalone path only, so `faba all
/// --quantify-competent-only` exited 0 and wrote undiluted matrices.
pub fn cells_for_quantification(
    competent: Option<&Arc<CompetentCells>>,
    qc_cells: Option<&FxHashMap<Box<str>, FxHashSet<CellBarcode>>>,
    competent_only: bool,
) -> Option<FxHashMap<Box<str>, FxHashSet<CellBarcode>>> {
    match (competent, qc_cells) {
        (Some(comp), Some(qc)) if competent_only => Some(intersect_for_quantification(qc, comp)),
        _ => None,
    }
}

pub fn intersect_for_quantification(
    qc_cells: &FxHashMap<Box<str>, FxHashSet<CellBarcode>>,
    competent: &CompetentCells,
) -> FxHashMap<Box<str>, FxHashSet<CellBarcode>> {
    qc_cells
        .iter()
        .map(|(bam, cells)| match competent.get(bam) {
            Some(keep) => {
                let kept: FxHashSet<CellBarcode> = cells
                    .iter()
                    .filter(|c| keep.contains(*c))
                    .cloned()
                    .collect();
                info!(
                    "quantification ({bam}): restricted to {} of {} QC cells",
                    kept.len(),
                    cells.len()
                );
                (bam.clone(), kept)
            }
            // A control BAM (or a signal BAM with no call) keeps every QC cell.
            None => (bam.clone(), cells.clone()),
        })
        .collect()
}

/// Keep only cells that already passed droplet calling / complexity / mito QC.
///
/// Null-cell calling is the LAST stage of the cell-QC cascade, not a parallel
/// one: it asks "does this real cell edit?", which is only meaningful once
/// "is this a real cell?" has been answered. Scanning every barcode instead
/// scores competence on ambient droplets — measured on a whole-genome run,
/// 12,259 of 12,730 "competent" barcodes were ones EmptyDrops had rejected —
/// and then lets their reads into discovery.
///
/// A BAM with no QC entry is left unfiltered rather than emptied, so a run
/// with `--skip-gene-qc` (no cell set at all) still works.
fn retain_qc_cells(
    tally: &mut ActivityTally,
    qc_cells: Option<&FxHashMap<Box<str>, FxHashSet<CellBarcode>>>,
    bam_file: &str,
) {
    let Some(keep) = qc_cells.and_then(|m| m.get(bam_file)) else {
        return;
    };
    tally.retain(|cb, _| keep.contains(cb));
}

/// Side-by-side ASCII histogram of the per-cell conversion rate, signal vs
/// control, on a **logit** axis.
///
/// Logit rather than linear because the rates are small (~1e-4 to 1e-2) and a
/// linear axis piles every cell into the leftmost bins; for small `p`,
/// `logit(p) = log(p/(1-p)) ~ log(p)`, which spreads the low tail where the
/// separation actually lives. The control column is what the cut is referenced
/// against, so showing them on one shared axis makes the comparison the method
/// relies on directly visible — including when it ISN'T there.
fn log_rate_histogram(signal: &ActivityTally, control: &ControlCells, opts: &NullCallOpts) {
    const BINS: usize = 24;
    const WIDTH: usize = 28;
    let logit = |p: f64| -> f64 {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    };
    let grab = |it: &mut dyn Iterator<Item = &CellActivity>| -> Vec<f64> {
        it.filter(|c| c.covered >= opts.min_coverage && c.edited > 0)
            .map(|c| logit(c.rate()))
            .collect()
    };
    let sig = grab(&mut signal.values());
    let ctl = grab(&mut control.iter());
    info!("per-cell C->U rate, logit axis (signal | control), cells with >=1 conversion:");
    info!("{:>10}  {:>28} | {:<28}", "rate", "SIGNAL", "CONTROL");
    two_series_histogram(&sig, &ctl, BINS, WIDTH, |centre| {
        format!("{:>9.4}%", 100.0 / (1.0 + (-centre).exp()))
    });
}

/// Does this feature row belong to gene symbol `g`?
///
/// Two name shapes reach here. Raw matrices carry `{ensembl}_{symbol}/count/...`,
/// but `load_unified_data` canonicalises with `rsplit('_')`, which strips the
/// Ensembl prefix and leaves `{symbol}/count/...`. Matching only the raw shape
/// silently matches nothing after canonicalisation — which is exactly what it did
/// on the first genome-wide run, warning "none matched" for genes that were
/// certainly present.
///
/// Field-anchored via [`faba::feature_name::parse_feature_row`], the crate's
/// single source of truth for row layout: a bare `contains(symbol)` would let
/// `METTL3` match `METTL3L` and `RBM15` match `RBM15B`.
fn feature_is_gene(feature: &str, symbol: &str) -> bool {
    faba::feature_name::parse_feature_row(feature)
        .is_some_and(|r| r.gene == symbol || r.gene.rsplit('_').next() == Some(symbol))
}

/// Per-cell expression of a named gene family, competent cells vs null cells.
///
/// **Diagnostic only — nothing here feeds the cut.** It exists to answer the
/// question users ask on seeing most of their cells dropped: "do the null cells
/// simply not express the machinery?" Measured on rep1 the answer is no — the
/// two distributions sit on top of each other — and showing that is worth more
/// than asserting it.
///
/// Counts are **normalised per 10k UMI**, which is not optional: on raw counts
/// every family trends the same direction merely because competent cells have
/// slightly lower totals (41,618 vs 43,125 median), which reads as a signal and
/// is not one.
fn log_family_expression(
    label: &str,
    symbols: &[String],
    matrix_paths: &[Box<str>],
    scanned: &ActivityTally,
    selected: &FxHashSet<CellBarcode>,
) -> anyhow::Result<()> {
    use data_beans::qc::collect_column_stat_across_vec;
    use data_beans::sparse_io_vector::{ColumnAlignment, SparseIoVec};
    use graph_embedding_util::{load_unified_data, LoadUnifiedArgs};
    if symbols.is_empty() {
        return Ok(());
    }
    // The gene-count matrices come from dartseq's own QC stage, which
    // `--valid-cells` skips entirely. Asking for the diagnostic and silently
    // getting nothing back is worse than being told why.
    if matrix_paths.is_empty() {
        log::warn!(
            "{label}: {symbols:?} requested, but there is no gene-count matrix to read \
             (--valid-cells skips the gene-count QC stage that builds it); skipping"
        );
        return Ok(());
    }
    let unified = load_unified_data(LoadUnifiedArgs {
        data_files: matrix_paths.to_vec(),
        column_alignment: ColumnAlignment::Disjoint,
        ..Default::default()
    })?;
    let features = unified.feature_names.clone();
    let rows: Vec<usize> = features
        .iter()
        .enumerate()
        .filter(|(_, f)| symbols.iter().any(|g| feature_is_gene(f, g)))
        .map(|(i, _)| i)
        .collect();
    if rows.is_empty() {
        log::warn!("{label}: none of {symbols:?} matched a feature row; skipping");
        return Ok(());
    }
    let backend: &SparseIoVec = unified.count_backend();
    let (_, fam, _, _) = collect_column_stat_across_vec(backend, Some(&rows), None)?.to_f32_vecs();
    let (_, tot, _, _) = collect_column_stat_across_vec(backend, None, None)?.to_f32_vecs();
    let mut kept: Vec<f32> = Vec::new();
    let mut dropped: Vec<f32> = Vec::new();
    for (i, cb) in unified.barcodes.iter().enumerate() {
        // `load_unified_data` suffixes each column with `@{batch}` to keep
        // libraries apart; the scan keys on the bare 10x barcode.
        let bare = cb.split_once('@').map_or(cb.as_ref(), |(b, _)| b);
        let key = CellBarcode::Barcode(bare.into());
        // Only the SCANNED library can answer this. The control arm is a
        // different transfection (catalytically dead YTH), so letting its cells
        // fall into the "null" bucket would contrast conditions rather than
        // competent-vs-null within one library — a different question, and one
        // this diagnostic would silently appear to answer.
        if !scanned.contains_key(&key) {
            continue;
        }
        let t = *tot.get(i).unwrap_or(&0.0);
        if t <= 0.0 {
            continue;
        }
        let v = *fam.get(i).unwrap_or(&0.0) / t * 1e4;
        if selected.contains(&key) {
            kept.push(v)
        } else {
            dropped.push(v)
        }
    }
    // A side-by-side needs both sides. Empty here means the matrix columns and the
    // scan's barcodes do not name the same cells, which is invisible unless said
    // out loud — the counts and an example from each side make it obvious which.
    if kept.is_empty() || dropped.is_empty() {
        log::warn!(
            "{label}: cannot compare — {} competent / {} null cells matched the matrix. \
             Matrix column e.g. {:?}; scanned barcode e.g. {:?}",
            kept.len(),
            dropped.len(),
            unified.barcodes.first().map(|b| b.as_ref()),
            scanned.keys().next()
        );
        return Ok(());
    }
    let (mk, md) = (
        matrix_util::utils::median(&kept),
        matrix_util::utils::median(&dropped),
    );
    info!(
        "{label} ({} genes, {} rows): per-10k median {:.1} in competent cells vs {:.1} in \
         null cells{}",
        symbols.len(),
        rows.len(),
        mk,
        md,
        if (mk - md).abs() <= 0.15 * mk.max(1e-9) {
            "  -- indistinguishable, so competence is not explained by this family"
        } else {
            ""
        }
    );
    side_by_side(&kept, &dropped, "competent", "null");
    Ok(())
}

/// Two distributions on one shared log axis, per-10k units.
///
/// Cells with **zero** family expression are counted out and reported as a line
/// of their own rather than binned. On a log axis a zero can only be clamped to
/// some arbitrary floor, and that floor then sets the axis minimum: with the
/// floor at 1e-6 and real values near 4, ten of sixteen bins rendered empty and
/// every real difference collapsed into two. The count is also the more useful
/// form — "no machinery expressed at all" is a categorical statement about a
/// cell, not a point on a continuum.
fn side_by_side(a: &[f32], b: &[f32], la: &str, lb: &str) {
    let split = |v: &[f32]| -> (usize, Vec<f64>) {
        let pos: Vec<f64> = v
            .iter()
            .filter(|x| **x > 0.0)
            .map(|x| (*x as f64).ln())
            .collect();
        (v.len() - pos.len(), pos)
    };
    let ((za, x), (zb, y)) = (split(a), split(b));
    info!(
        "  zero expression: {}/{} {la} vs {}/{} {lb}",
        za,
        a.len(),
        zb,
        b.len()
    );
    info!("{:>10}  {:>22} | {:<22}", "per-10k", la, lb);
    two_series_histogram(&x, &y, 16, 22, |c| format!("{:>10.1}", c.exp()));
}

/// Two series binned on one SHARED axis and printed side by side.
///
/// Values arrive already transformed (logit, log, ...); `axis` renders a bin
/// centre back into the units the reader thinks in. The shared axis is the
/// whole point — it is what makes "the signal sits exactly where the control
/// sits" legible, which for this module is the answer as often as not.
fn two_series_histogram(
    a: &[f64],
    b: &[f64],
    bins: usize,
    width: usize,
    axis: impl Fn(f64) -> String,
) {
    if a.is_empty() || b.is_empty() {
        return;
    }
    let (lo, hi) = a
        .iter()
        .chain(b.iter())
        .fold((f64::MAX, f64::MIN), |(x, y), &v| (x.min(v), y.max(v)));
    // Also catches NaN, where every comparison is false and the binning below
    // would divide by a non-positive span.
    if hi <= lo || !hi.is_finite() || !lo.is_finite() {
        return;
    }
    let bin = |v: f64| -> usize { (((v - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1) };
    let (mut ha, mut hb) = (vec![0usize; bins], vec![0usize; bins]);
    for v in a {
        ha[bin(*v)] += 1;
    }
    for v in b {
        hb[bin(*v)] += 1;
    }
    let peak = ha
        .iter()
        .chain(hb.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);
    let bar = |n: usize| {
        let k = (n * width).div_ceil(peak).min(width);
        format!("{}{}", "#".repeat(k), " ".repeat(width - k))
    };
    for i in (0..bins).rev() {
        let centre = lo + (i as f64 + 0.5) * (hi - lo) / bins as f64;
        info!(
            "{}  {} | {}  {:>5} {:>5}",
            axis(centre),
            bar(ha[i]),
            bar(hb[i]),
            ha[i],
            hb[i]
        );
    }
}

/// Per-cell audit: every scored cell, its tally, and whether it was kept.
fn write_audit(
    path: &str,
    signal: &ActivityTally,
    call: &NullCellCall,
    min_coverage: u64,
) -> anyhow::Result<()> {
    let mut rows: Vec<_> = signal.iter().collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));
    // `scored` separates "assessed and rejected as null" from "never assessed"
    // (too little coverage to rank). Without it the kept fraction reads against
    // every tallied barcode, most of which are ambient, and looks far harsher
    // than the call actually was.
    let mut lines: Vec<Box<str>> = Vec::with_capacity(rows.len() + 1);
    lines.push(
        "barcode\tedited\tcovered\trate\tbg_edited\tbg_covered\tbg_rate\tscored\tkept".into(),
    );
    for (cb, act) in rows {
        lines.push(
            format!(
                "{}\t{}\t{}\t{:.6}\t{}\t{}\t{:.6}\t{}\t{}",
                cb,
                act.edited,
                act.covered,
                act.rate(),
                act.bg_edited,
                act.bg_covered,
                act.background_rate(),
                u8::from(act.covered >= min_coverage),
                u8::from(call.selected.contains(cb))
            )
            .into(),
        );
    }
    // `write_lines` flushes and propagates the error; hand-rolling a `GzEncoder`
    // finishes it in `Drop`, which swallows a short write into a truncated file.
    write_lines(&lines, path)
}
