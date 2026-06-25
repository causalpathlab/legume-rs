//! Cell calling (barcode QC), CellRanger-style: OrdMag knee ∪ EmptyDrops.
//!
//! `faba genes` (and the shared gene-count QC that feeds apa/atoi/m6a) counts
//! every observed cell barcode, so the raw output is a superset of real cells
//! that includes empty/ambient droplets. This module separates real cells from
//! ambient using two stages, run **independently per batch** (each BAM is a
//! separate library with its own depth and ambient profile):
//!
//! 1. **OrdMag knee** — keep barcodes whose total count exceeds `fraction ×`
//!    the (`1 - quantile`)-quantile count of the top `expected_cells` barcodes.
//! 2. **EmptyDrops** — for barcodes below the knee (but above `min_umis`), test
//!    their expression profile against an ambient multinomial estimated by
//!    Simple-Good-Turing from the low-count droplets; keep those that deviate
//!    (Benjamini-Hochberg FDR `< ed_fdr`).
//!
//! The final cell set is the union of the two stages.

use crate::common::*;
use crate::pipeline_util::extract_gene_key;

use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustc_hash::{FxHashMap, FxHashSet};

////////////////////////////////////////////////////////////////////////////////
//                              Policy + parameters                           //
////////////////////////////////////////////////////////////////////////////////

/// Which barcode-QC criterion to apply when selecting cells.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CellFilter {
    /// OrdMag knee ∪ EmptyDrops (CellRanger-style; the default).
    #[default]
    #[value(name = "empty-drops")]
    EmptyDrops,
    /// OrdMag knee only.
    #[value(name = "ord-mag")]
    OrdMag,
    /// Keep barcodes with total count ≥ `min_counts`.
    #[value(name = "min-counts")]
    MinCounts,
    /// No cell calling — keep every observed barcode (today's raw superset).
    #[value(name = "nnz")]
    Nnz,
}

/// Tunables for [`call_cells`]. Defaults track CellRanger.
#[derive(Clone, Copy, Debug)]
pub struct CellCallParams {
    pub filter: CellFilter,
    /// `MinCounts` threshold; also the EmptyDrops candidate floor.
    pub min_counts: usize,
    /// OrdMag recovered-cell estimate (top-N for the knee).
    pub expected_cells: usize,
    /// OrdMag quantile of the top cells used as the baseline (CellRanger 0.99).
    pub ordmag_quantile: f64,
    /// OrdMag knee fraction of the baseline (CellRanger 0.10).
    pub ordmag_fraction: f64,
    /// EmptyDrops BH FDR cutoff.
    pub ed_fdr: f64,
    /// EmptyDrops Monte-Carlo iterations.
    pub ed_n_sims: usize,
    /// EmptyDrops ambient window: lower rank bound (by total, descending).
    pub ed_ambient_lo: usize,
    /// EmptyDrops ambient window: upper rank bound.
    pub ed_ambient_hi: usize,
    /// EmptyDrops minimum total UMIs for a candidate barcode.
    pub ed_min_umis: usize,
    /// RNG seed (deterministic Monte-Carlo).
    pub seed: u64,
}

impl Default for CellCallParams {
    fn default() -> Self {
        Self {
            filter: CellFilter::EmptyDrops,
            min_counts: 500,
            expected_cells: 3000,
            ordmag_quantile: 0.99,
            ordmag_fraction: 0.10,
            ed_fdr: 0.001,
            ed_n_sims: 10_000,
            ed_ambient_lo: 45_000,
            ed_ambient_hi: 90_000,
            ed_min_umis: 500,
            seed: 42,
        }
    }
}

/// Shared CLI knobs for cell calling, flattened into each subcommand that does
/// gene-count QC (`genes`, `apa`, `atoi`, `dartseq`, `all`).
#[derive(clap::Args, Debug, Clone)]
pub struct CellQcArgs {
    /// Cell-calling method (barcode QC)
    #[arg(
        long = "cell-filter",
        value_enum,
        default_value = "empty-drops",
        help = "Cell-calling method (barcode QC)",
        long_help = "How to separate real cells from empty/ambient droplets:\n\
                     - empty-drops: OrdMag knee ∪ EmptyDrops (CellRanger-style; default)\n\
                     - ord-mag:     OrdMag knee only\n\
                     - min-counts:  keep barcodes with total ≥ --cell-min-umis\n\
                     - nnz:         no cell calling (raw superset; nnz cutoffs only)"
    )]
    pub cell_filter: CellFilter,

    /// OrdMag expected/recovered cell count (top-N for the knee)
    #[arg(long = "expected-cells", default_value_t = 3000)]
    pub expected_cells: usize,

    /// Minimum total UMIs for an EmptyDrops candidate / the min-counts threshold
    #[arg(long = "cell-min-umis", default_value_t = 500)]
    pub cell_min_umis: usize,

    /// EmptyDrops Benjamini-Hochberg FDR cutoff
    #[arg(long = "cell-fdr", default_value_t = 0.001)]
    pub cell_fdr: f64,

    /// EmptyDrops Monte-Carlo iterations
    #[arg(long = "cell-sims", default_value_t = 10_000)]
    pub cell_sims: usize,

    /// EmptyDrops ambient window lower rank bound (by total, descending)
    #[arg(long = "ambient-lo", default_value_t = 45_000)]
    pub ambient_lo: usize,

    /// EmptyDrops ambient window upper rank bound
    #[arg(long = "ambient-hi", default_value_t = 90_000)]
    pub ambient_hi: usize,
}

impl Default for CellQcArgs {
    fn default() -> Self {
        let d = CellCallParams::default();
        Self {
            cell_filter: d.filter,
            expected_cells: d.expected_cells,
            cell_min_umis: d.ed_min_umis,
            cell_fdr: d.ed_fdr,
            cell_sims: d.ed_n_sims,
            ambient_lo: d.ed_ambient_lo,
            ambient_hi: d.ed_ambient_hi,
        }
    }
}

impl CellQcArgs {
    /// Resolve to [`CellCallParams`].
    pub fn params(&self) -> CellCallParams {
        CellCallParams {
            filter: self.cell_filter,
            min_counts: self.cell_min_umis,
            expected_cells: self.expected_cells,
            ed_fdr: self.cell_fdr,
            ed_n_sims: self.cell_sims,
            ed_ambient_lo: self.ambient_lo,
            ed_ambient_hi: self.ambient_hi,
            ed_min_umis: self.cell_min_umis,
            ..CellCallParams::default()
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//                              Per-cell count columns                        //
////////////////////////////////////////////////////////////////////////////////

/// One cell's sparse gene-count column.
pub struct CellColumn {
    pub barcode: CellBarcode,
    pub total: f64,
    /// `(gene_index, summed count)` over a shared per-batch gene vocabulary.
    pub entries: Vec<(u32, f32)>,
}

/// Per-cell sparse columns over a batch's gene vocabulary, the input to
/// [`call_cells`]. OrdMag needs only `total`; EmptyDrops needs the columns.
pub struct CellCounts {
    pub n_genes: usize,
    pub cells: Vec<CellColumn>,
}

impl CellCounts {
    /// Build from a batch's spliced + unspliced triplets: sum spliced+unspliced
    /// per `(cell, gene_key)`, index the gene_key vocabulary, and produce per-
    /// cell sparse columns with totals. `Missing` barcodes are dropped.
    pub fn from_triplets(
        spliced: &[(CellBarcode, Box<str>, f32)],
        unspliced: &[(CellBarcode, Box<str>, f32)],
    ) -> Self {
        // Gene vocabulary over both modalities, sorted so indices are
        // deterministic regardless of parallel iteration order.
        let mut gene_keys: Vec<&str> = spliced
            .par_iter()
            .chain(unspliced.par_iter())
            .map(|(_, feat, _)| extract_gene_key(feat))
            .collect::<FxHashSet<&str>>()
            .into_iter()
            .collect();
        gene_keys.sort_unstable();
        let gene_index: FxHashMap<&str, u32> = gene_keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, i as u32))
            .collect();
        let n_genes = gene_index.len();

        // Per-cell sparse aggregation via rayon fold/reduce (mirrors the
        // `totals` shard in gene_count::pipeline). Read counts are integers held
        // in f32, so the sums are exact and order-independent. `Missing`
        // barcodes are dropped.
        type Shard = FxHashMap<CellBarcode, FxHashMap<u32, f32>>;
        let cell_map: Shard = spliced
            .par_iter()
            .chain(unspliced.par_iter())
            .filter(|(cb, _, _)| *cb != CellBarcode::Missing)
            .fold(Shard::default, |mut acc, (cb, feat, v)| {
                let gi = gene_index[extract_gene_key(feat)];
                *acc.entry(cb.clone()).or_default().entry(gi).or_default() += *v;
                acc
            })
            .reduce(Shard::default, |mut a, mut b| {
                if a.len() < b.len() {
                    std::mem::swap(&mut a, &mut b);
                }
                for (cb, gmap) in b {
                    let e = a.entry(cb).or_default();
                    for (gi, v) in gmap {
                        *e.entry(gi).or_default() += v;
                    }
                }
                a
            });

        let cells: Vec<CellColumn> = cell_map
            .into_par_iter()
            .map(|(barcode, m)| {
                let total: f64 = m.values().map(|&x| x as f64).sum();
                let entries: Vec<(u32, f32)> = m.into_iter().collect();
                CellColumn {
                    barcode,
                    total,
                    entries,
                }
            })
            .collect();

        CellCounts { n_genes, cells }
    }
}

////////////////////////////////////////////////////////////////////////////////
//                              Entry point                                   //
////////////////////////////////////////////////////////////////////////////////

/// Decide which barcodes are real cells, per [`CellCallParams::filter`].
pub fn call_cells(counts: &CellCounts, p: &CellCallParams) -> FxHashSet<CellBarcode> {
    match p.filter {
        CellFilter::Nnz => counts.cells.iter().map(|c| c.barcode.clone()).collect(),
        CellFilter::MinCounts => counts
            .cells
            .iter()
            .filter(|c| c.total >= p.min_counts as f64)
            .map(|c| c.barcode.clone())
            .collect(),
        CellFilter::OrdMag => {
            let cutoff = ordmag_cutoff(counts, p);
            info!("OrdMag cutoff: total count ≥ {:.0}", cutoff);
            counts
                .cells
                .iter()
                .filter(|c| c.total >= cutoff)
                .map(|c| c.barcode.clone())
                .collect()
        }
        CellFilter::EmptyDrops => {
            let cutoff = ordmag_cutoff(counts, p);
            let mut set: FxHashSet<CellBarcode> = counts
                .cells
                .iter()
                .filter(|c| c.total >= cutoff)
                .map(|c| c.barcode.clone())
                .collect();
            info!(
                "OrdMag cutoff: total count ≥ {:.0} ({} cells); running EmptyDrops below it",
                cutoff,
                set.len()
            );
            empty_drops_extend(counts, p, cutoff, &mut set);
            set
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//                              Stage A — OrdMag                              //
////////////////////////////////////////////////////////////////////////////////

/// OrdMag knee: baseline = total at rank `floor(expected_cells*(1-quantile))`
/// within the top `expected_cells` barcodes (≈ quantile of the top cells);
/// cutoff = `round(fraction * baseline)`.
fn ordmag_cutoff(counts: &CellCounts, p: &CellCallParams) -> f64 {
    if counts.cells.is_empty() {
        return f64::INFINITY;
    }
    let mut totals: Vec<f64> = counts.cells.iter().map(|c| c.total).collect();
    totals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap()); // descending

    let top = p.expected_cells.min(totals.len());
    let idx = ((p.expected_cells as f64) * (1.0 - p.ordmag_quantile)).floor() as usize;
    let idx = idx.min(top.saturating_sub(1));
    let baseline = totals[idx];
    (p.ordmag_fraction * baseline).round().max(1.0)
}

////////////////////////////////////////////////////////////////////////////////
//                              Stage B — EmptyDrops                          //
////////////////////////////////////////////////////////////////////////////////

/// Test candidate barcodes (`[ed_min_umis, ord_cutoff)`) against the ambient
/// multinomial; add those with BH q-value `< ed_fdr` to `set`.
fn empty_drops_extend(
    counts: &CellCounts,
    p: &CellCallParams,
    ord_cutoff: f64,
    set: &mut FxHashSet<CellBarcode>,
) {
    if counts.n_genes == 0 {
        return;
    }

    // Ambient profile from the low-count window (by total, descending rank).
    let mut order: Vec<usize> = (0..counts.cells.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        counts.cells[b]
            .total
            .partial_cmp(&counts.cells[a].total)
            .unwrap()
    });
    let lo = p.ed_ambient_lo.min(order.len());
    let hi = p.ed_ambient_hi.min(order.len());
    if hi <= lo {
        info!(
            "EmptyDrops: too few barcodes ({}) for ambient window [{}, {}); skipping",
            order.len(),
            p.ed_ambient_lo,
            p.ed_ambient_hi
        );
        return;
    }
    let mut ambient = vec![0u64; counts.n_genes];
    for &ci in &order[lo..hi] {
        for &(g, v) in &counts.cells[ci].entries {
            ambient[g as usize] += v.round() as u64;
        }
    }
    if ambient.iter().all(|&x| x == 0) {
        info!("EmptyDrops: empty ambient profile; skipping");
        return;
    }

    let probs = simple_good_turing(&ambient);
    let log_p: Vec<f64> = probs.iter().map(|&x| x.ln()).collect();

    // Candidates: below the knee, above the floor, not already a cell.
    let candidates: Vec<usize> = counts
        .cells
        .iter()
        .enumerate()
        .filter(|(_, c)| c.total >= p.ed_min_umis as f64 && c.total < ord_cutoff)
        .map(|(i, _)| i)
        .collect();
    if candidates.is_empty() {
        info!(
            "EmptyDrops: no candidate barcodes in [{}, {:.0})",
            p.ed_min_umis, ord_cutoff
        );
        return;
    }

    // Observed ambient log-likelihood per candidate, and the sorted unique
    // candidate totals (the snapshot points for the nested Monte-Carlo path).
    let obs_ll: Vec<f64> = candidates
        .par_iter()
        .map(|&ci| {
            counts.cells[ci]
                .entries
                .iter()
                .map(|&(g, v)| v as f64 * log_p[g as usize])
                .sum::<f64>()
        })
        .collect();

    let mut uniq_totals: Vec<usize> = candidates
        .iter()
        .map(|&ci| counts.cells[ci].total.round() as usize)
        .collect();
    uniq_totals.sort_unstable();
    uniq_totals.dedup();
    let total_to_slot: FxHashMap<usize, usize> = uniq_totals
        .iter()
        .enumerate()
        .map(|(slot, &t)| (t, slot))
        .collect();
    let max_total = *uniq_totals.last().unwrap();

    info!(
        "EmptyDrops: {} candidates, {} distinct totals (max {}), {} sims",
        candidates.len(),
        uniq_totals.len(),
        max_total,
        p.ed_n_sims
    );

    // Nested cumulative Monte-Carlo: each iteration walks one categorical path
    // of length `max_total`, snapshotting the running log-likelihood whenever
    // the running count reaches a candidate total. One pass yields a null draw
    // for every candidate total simultaneously (DropletUtils-style).
    let dist = match WeightedIndex::new(probs.iter().copied()) {
        Ok(d) => d,
        Err(_) => {
            info!("EmptyDrops: degenerate ambient profile; skipping");
            return;
        }
    };
    let seed = p.seed;
    let log_p_ref = &log_p;
    let uniq_ref = &uniq_totals;

    // null_by_iter[iter][slot] = running LL at uniq_totals[slot] in that iter.
    let null_by_iter: Vec<Vec<f64>> = (0..p.ed_n_sims)
        .into_par_iter()
        .map(|iter| {
            let mut rng = StdRng::seed_from_u64(seed ^ (iter as u64).wrapping_mul(0x9E37_79B9));
            let mut snap = vec![0.0f64; uniq_ref.len()];
            let mut running = 0.0f64;
            let mut slot = 0usize;
            for step in 1..=max_total {
                let g = dist.sample(&mut rng);
                running += log_p_ref[g];
                if slot < uniq_ref.len() && uniq_ref[slot] == step {
                    snap[slot] = running;
                    slot += 1;
                }
            }
            snap
        })
        .collect();

    // Transpose to per-total sorted null distributions.
    let mut null_by_total: Vec<Vec<f64>> = vec![Vec::with_capacity(p.ed_n_sims); uniq_totals.len()];
    for snap in &null_by_iter {
        for (slot, &v) in snap.iter().enumerate() {
            null_by_total[slot].push(v);
        }
    }
    null_by_total
        .par_iter_mut()
        .for_each(|v| v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()));

    // p-value = (1 + #{null_LL <= obs_LL}) / (n_sims + 1): a real cell's profile
    // is unlike ambient, so its observed LL falls in the lower tail.
    let n_sims = p.ed_n_sims as f64;
    let pvals: Vec<f32> = candidates
        .par_iter()
        .zip(obs_ll.par_iter())
        .map(|(&ci, &obs)| {
            let t = counts.cells[ci].total.round() as usize;
            let slot = total_to_slot[&t];
            let nd = &null_by_total[slot];
            // count of null samples <= obs (partition point of obs)
            let le = nd.partition_point(|&x| x <= obs);
            ((1.0 + le as f64) / (n_sims + 1.0)) as f32
        })
        .collect();

    let qvals = faba::hypothesis_tests::benjamini_hochberg(&pvals);
    let mut added = 0usize;
    for (k, &ci) in candidates.iter().enumerate() {
        if (qvals[k] as f64) < p.ed_fdr && set.insert(counts.cells[ci].barcode.clone()) {
            added += 1;
        }
    }
    info!(
        "EmptyDrops: rescued {} additional cells (FDR < {})",
        added, p.ed_fdr
    );
}

////////////////////////////////////////////////////////////////////////////////
//                          Simple Good-Turing smoothing                      //
////////////////////////////////////////////////////////////////////////////////

/// Simple Good-Turing (Gale & Sampson 1995) ambient profile. Returns a strictly
/// positive probability vector over genes (length = `counts.len()`): observed
/// genes get Good-Turing-smoothed mass, unseen (zero-count) genes share the
/// `N_1/N` reserve. Falls back to add-one smoothing if the log-log fit degenerates.
fn simple_good_turing(counts: &[u64]) -> Vec<f64> {
    let n_total: u64 = counts.iter().sum();
    let n_genes = counts.len();
    if n_total == 0 {
        return vec![1.0 / n_genes.max(1) as f64; n_genes];
    }

    // Frequency of frequencies: r -> N_r (r >= 1).
    let mut nr: FxHashMap<u64, u64> = FxHashMap::default();
    for &c in counts {
        if c > 0 {
            *nr.entry(c).or_default() += 1;
        }
    }
    let n_zero = counts.iter().filter(|&&c| c == 0).count();

    let mut r: Vec<u64> = nr.keys().copied().collect();
    r.sort_unstable();
    let big_n = n_total as f64;
    let n1 = *nr.get(&1).unwrap_or(&0) as f64;
    let p0 = if n_zero > 0 { n1 / big_n } else { 0.0 };

    // Averaging transform Z_r = N_r / (0.5 * (r_next - r_prev)), then a log-log
    // linear fit log Z_r ~ a + b log r for the "linear Good-Turing" estimate.
    let k = r.len();
    let (mut sx, mut sy, mut sxx, mut sxy, mut m) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut logr = Vec::with_capacity(k);
    let mut logz = Vec::with_capacity(k);
    for i in 0..k {
        let ri = r[i] as f64;
        let prev = if i == 0 { 0.0 } else { r[i - 1] as f64 };
        let next = if i + 1 < k {
            r[i + 1] as f64
        } else {
            2.0 * ri - prev
        };
        let z = (*nr.get(&r[i]).unwrap() as f64) / (0.5 * (next - prev));
        let (lx, ly) = (ri.ln(), z.ln());
        logr.push(lx);
        logz.push(ly);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
        m += 1.0;
    }
    let denom = m * sxx - sx * sx;
    let (slope, intercept) = if denom.abs() < 1e-12 {
        (-1.5, 0.0) // degenerate; defer to add-one below
    } else {
        let b = (m * sxy - sx * sy) / denom;
        let a = (sy - b * sx) / m;
        (b, a)
    };
    // Smoothed S(r) = exp(intercept + slope * ln r).
    let s_smooth = |rv: f64| -> f64 { (intercept + slope * rv.ln()).exp() };

    // r*: Turing estimate for small r, switching to Linear Good-Turing once the
    // two disagree by less than ~1.96 standard errors (Gale & Sampson switch).
    let nr_get = |rv: u64| -> f64 { *nr.get(&rv).unwrap_or(&0) as f64 };
    let mut rstar: FxHashMap<u64, f64> = FxHashMap::default();
    let mut use_lgt = false;
    for &ri in &r {
        let lgt = (ri as f64 + 1.0) * s_smooth(ri as f64 + 1.0) / s_smooth(ri as f64);
        if !use_lgt {
            let nr1 = nr_get(ri + 1);
            let nri = nr_get(ri);
            if nr1 == 0.0 || nri == 0.0 {
                use_lgt = true;
            } else {
                let turing = (ri as f64 + 1.0) * nr1 / nri;
                let se = ((ri as f64 + 1.0).powi(2) * nr1 / (nri * nri) * (1.0 + nr1 / nri)).sqrt();
                if (turing - lgt).abs() <= 1.96 * se {
                    use_lgt = true;
                }
                if !use_lgt {
                    rstar.insert(ri, turing);
                    continue;
                }
            }
        }
        rstar.insert(ri, lgt);
    }

    // Normalize observed mass to (1 - p0); fall back to add-one if anything is
    // non-finite or non-positive.
    let mass: f64 = r.iter().map(|&ri| nr_get(ri) * rstar[&ri]).sum();
    let valid = mass.is_finite() && mass > 0.0 && rstar.values().all(|v| v.is_finite() && *v > 0.0);

    let mut probs = vec![0.0f64; n_genes];
    if valid {
        let zero_share = if n_zero > 0 { p0 / n_zero as f64 } else { 0.0 };
        let scale = (1.0 - p0) / mass;
        for (i, &c) in counts.iter().enumerate() {
            probs[i] = if c == 0 {
                zero_share
            } else {
                scale * rstar[&c]
            };
        }
    } else {
        // Add-one (Laplace) smoothing.
        let denom = big_n + n_genes as f64;
        for (i, &c) in counts.iter().enumerate() {
            probs[i] = (c as f64 + 1.0) / denom;
        }
    }

    // Guard: strictly positive, renormalized.
    let floor = 1e-12 / n_genes as f64;
    for x in probs.iter_mut() {
        if !x.is_finite() || *x <= 0.0 {
            *x = floor;
        }
    }
    let s: f64 = probs.iter().sum();
    for x in probs.iter_mut() {
        *x /= s;
    }
    probs
}

////////////////////////////////////////////////////////////////////////////////
//                              Tests                                         //
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests;
