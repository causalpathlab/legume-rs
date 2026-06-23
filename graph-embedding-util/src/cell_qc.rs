//! Per-batch cell QC: drop the low-complexity DEBRIS tail of each batch.
//!
//! Shared by `faba gem` and `senna bge`. Replaces the 1-D mixture empty-call
//! (the retired `null_call::embedding_mixture_empty_call`), which could not fire
//! on already-cell-called data: the cell-embedding norm is single-peaked there,
//! so there is no empty *mode* to split, and the call dropped 0 cells. Instead,
//! for each batch independently, set a low cutoff on BOTH nnz and depth (exact
//! 2-means + BIC guard, MAD-lower fallback) and drop cells below *both* — the
//! low-complexity debris tail.
//!
//! Every depth decision is **per batch**: pooling replicates of different depth
//! confounds "shallow replicate" with "empty" — a low-depth replicate's real
//! cells look empty against a deep one, so a global threshold over-drops the
//! shallow replicate and under-drops the deep one. Single-batch data degenerates
//! to one group automatically.

use data_beans::qc::{print_nnz_summary, suggest_nnz_cutoff};
use data_beans::qc_lib::{robust_outlier_keep, Tail};
use log::{info, warn};
use nalgebra::DMatrix;

/// Tuning knobs for [`per_batch_cell_qc`]. Plain struct (no clap) so each tool
/// builds it from its own CLI surface.
#[derive(Clone, Debug)]
pub struct CellQcConfig {
    /// Master switch. `false` ⇒ keep every cell (the `--skip-cell-qc` path).
    pub enabled: bool,
    /// Per-batch lower-band MAD multiplier for the debris fallback (used only
    /// when the 2-means BIC guard finds no split).
    pub debris_mads: f32,
}

impl Default for CellQcConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            debris_mads: 5.0,
        }
    }
}

/// Why a cell was dropped (or kept).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropReason {
    Kept,
    Debris,
}

impl DropReason {
    /// Numeric code for the `drop_reason` column of the QC parquet.
    #[must_use]
    pub fn code(self) -> f32 {
        match self {
            DropReason::Kept => 0.0,
            DropReason::Debris => 1.0,
        }
    }
}

/// Per-cell QC verdict over the full (pre-subset) cell axis.
pub struct CellQcResult {
    /// Final keep decision, one per cell.
    pub keep: Vec<bool>,
    /// Per-cell drop reason.
    pub reason: Vec<DropReason>,
    /// The per-batch debris nnz cutoff that applied to each cell (`NaN` when none).
    pub batch_nnz_cut: Vec<f32>,
    /// The per-batch debris depth cutoff that applied to each cell (`NaN` when none).
    pub batch_sum_cut: Vec<f32>,
}

impl CellQcResult {
    /// A "keep everything" verdict for `n` cells (QC disabled).
    #[must_use]
    pub fn all_kept(n: usize) -> Self {
        Self {
            keep: vec![true; n],
            reason: vec![DropReason::Kept; n],
            batch_nnz_cut: vec![f32::NAN; n],
            batch_sum_cut: vec![f32::NAN; n],
        }
    }

    /// Number of cells dropped.
    #[must_use]
    pub fn n_dropped(&self) -> usize {
        self.keep.iter().filter(|&&k| !k).count()
    }
}

/// Inputs for [`per_batch_cell_qc`]. All per-cell slices share the same
/// (pre-subset) cell order. `cell_nnz`/`cell_sum` are whatever per-cell
/// detected-feature count and total depth the caller measured on its own feature
/// axis (gem: spliced rows pre-refine; bge: all features post-refine).
pub struct CellQcInputs<'a> {
    /// Per-cell detected-feature count (nnz).
    pub cell_nnz: &'a [f32],
    /// Per-cell total count (depth / library size).
    pub cell_sum: &'a [f32],
    /// Per-cell batch id (indexes `batch_names`).
    pub batch_membership: &'a [u32],
    /// Batch labels (length = number of batches).
    pub batch_names: &'a [Box<str>],
    pub cfg: CellQcConfig,
}

/// Run the per-batch debris cell QC: for each batch, drop cells below BOTH a
/// per-batch nnz and depth cut. Pure function of `cell_nnz`/`cell_sum`/batches —
/// never errors; returns a per-cell keep mask + the applied cuts.
#[must_use]
pub fn per_batch_cell_qc(inputs: CellQcInputs<'_>) -> CellQcResult {
    let CellQcInputs {
        cell_nnz,
        cell_sum,
        batch_membership,
        batch_names,
        cfg,
    } = inputs;
    let n = cell_nnz.len();
    if !cfg.enabled || n < 2 || cell_sum.len() != n || batch_membership.len() != n {
        return CellQcResult::all_kept(n);
    }

    // Group cell indices by batch (single in-memory pass).
    let n_batches = batch_names.len().max(1);
    let mut batch_cells: Vec<Vec<usize>> = vec![Vec::new(); n_batches];
    for (c, &b) in batch_membership.iter().enumerate() {
        batch_cells[(b as usize).min(n_batches - 1)].push(c);
    }

    let mut reason = vec![DropReason::Kept; n];
    let mut batch_nnz_cut = vec![f32::NAN; n];
    let mut batch_sum_cut = vec![f32::NAN; n];

    for (bi, cells) in batch_cells.iter().enumerate() {
        if cells.is_empty() {
            continue;
        }
        let bname: &str = batch_names.get(bi).map_or("batch", |s| s.as_ref());
        let bnnz: Vec<f32> = cells.iter().map(|&c| cell_nnz[c]).collect();
        let bsum: Vec<f32> = cells.iter().map(|&c| cell_sum[c]).collect();

        // Debris call: a low cutoff on BOTH nnz and depth. A cell is debris only
        // when it falls below both cuts.
        let nnz_cut = batch_low_cut(&bnnz, cfg.debris_mads);
        let sum_cut = batch_low_cut(&bsum, cfg.debris_mads);
        print_nnz_summary(
            &format!("cell-qc {bname}"),
            &bnnz,
            if nnz_cut.is_finite() {
                nnz_cut as usize
            } else {
                0
            },
            nnz_cut.is_finite().then_some(nnz_cut as usize),
        );

        // Safety: never gut a whole batch. If ≥95% would go, keep the batch.
        let n_drop = (0..cells.len())
            .filter(|&j| bnnz[j] < nnz_cut && bsum[j] < sum_cut)
            .count();
        if n_drop * 100 >= cells.len() * 95 {
            warn!(
                "cell-qc {bname}: debris cut would drop {}/{} (≥95%) — keeping the whole batch (inspect cell_qc parquet / --cell-qc-debris-mads)",
                n_drop,
                cells.len()
            );
            continue;
        }

        for (j, &c) in cells.iter().enumerate() {
            batch_nnz_cut[c] = nnz_cut;
            batch_sum_cut[c] = sum_cut;
            if bnnz[j] < nnz_cut && bsum[j] < sum_cut {
                reason[c] = DropReason::Debris;
            }
        }
        info!(
            "cell-qc {bname}: {} cells, {} debris cells (cut: nnz<{:.0} & sum<{:.0})",
            cells.len(),
            n_drop,
            nnz_cut,
            sum_cut,
        );
    }

    let keep: Vec<bool> = reason.iter().map(|&r| r == DropReason::Kept).collect();
    CellQcResult {
        keep,
        reason,
        batch_nnz_cut,
        batch_sum_cut,
    }
}

/// Per-batch low cutoff on a depth-like vector: cells with value `< cut` are
/// debris. Returns `NEG_INFINITY` (⇒ no cut) when the batch shows no genuine
/// low-end debris tail.
///
/// Primary: exact 1-D 2-means with a BIC "is it bimodal at all?" guard
/// (`suggest_nnz_cutoff`, deterministic). The split is accepted only when it
/// carves a **low-end minority** — the cut must sit at/below the batch median and
/// drop ≤ half the cells. That rejects two failure modes: a *uniformly shallow*
/// batch (no debris — per-batch relativity can't call a whole shallow replicate
/// empty) and a *bulk-vs-high* split (the cut would land above the bulk).
///
/// Fallback when the BIC guard finds no split: a robust MAD lower band, honoured
/// only if it removes a minority (else no cut — firmness).
fn batch_low_cut(vals: &[f32], mads: f32) -> f32 {
    let n = vals.len();
    if n < 4 {
        return f32::NEG_INFINITY;
    }
    let median = {
        let mut v = vals.to_vec();
        let (_, m, _) = v.select_nth_unstable_by(n / 2, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        *m
    };

    if let Some(cut) = suggest_nnz_cutoff(vals) {
        let cutf = cut as f32;
        let n_below = vals.iter().filter(|&&v| v < cutf).count();
        if cutf <= median && n_below * 2 <= n {
            return cutf;
        }
    }
    // Fallback: robust lower band; keep[i]==false ⇒ low outlier.
    let keep = robust_outlier_keep(vals, mads, Tail::Lower, true, None);
    let n_drop = keep.iter().filter(|&&k| !k).count();
    if n_drop == 0 || n_drop * 2 > n {
        return f32::NEG_INFINITY;
    }
    // Threshold = smallest KEPT value, so `value < cut` ⇔ low outlier.
    vals.iter()
        .zip(&keep)
        .filter(|(_, &k)| k)
        .map(|(&v, _)| v)
        .fold(f32::INFINITY, f32::min)
}

/// Write the shared cell-QC parquet (`{out}.cell_qc.parquet`), one row per cell
/// over the full pre-subset axis. `cell_nrms` is the pre-L2 embedding norm
/// (`None`/`NaN` when phase-2 was skipped). Columns:
/// `nnz, sum, pre_l2_norm, batch_nnz_cut, batch_sum_cut, drop_reason, kept`.
pub fn write_cell_qc_parquet(
    path: &str,
    barcodes: &[Box<str>],
    cell_nrms: Option<&[f32]>,
    cell_nnz: &[f32],
    cell_sum: &[f32],
    res: &CellQcResult,
) -> anyhow::Result<()> {
    use matrix_util::traits::IoOps;
    let n = barcodes.len();
    let col_names = [
        "nnz",
        "sum",
        "pre_l2_norm",
        "batch_nnz_cut",
        "batch_sum_cut",
        "drop_reason",
        "kept",
    ];
    let mut m = DMatrix::<f32>::zeros(n, col_names.len());
    for c in 0..n {
        m[(c, 0)] = cell_nnz[c];
        m[(c, 1)] = cell_sum[c];
        m[(c, 2)] = cell_nrms.map_or(f32::NAN, |v| v[c]);
        m[(c, 3)] = res.batch_nnz_cut[c];
        m[(c, 4)] = res.batch_sum_cut[c];
        m[(c, 5)] = res.reason[c].code();
        m[(c, 6)] = u8::from(res.keep[c]) as f32;
    }
    let cols: Vec<Box<str>> = col_names.iter().map(|s| Box::from(*s)).collect();
    m.to_parquet_with_names(path, (Some(barcodes), Some("cell")), Some(&cols))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG → standard normal (Box–Muller). No RNG crate, fully
    /// reproducible; gives smooth log-normal depth so the BIC guard behaves as it
    /// does on real data (a unimodal cluster is *not* split).
    struct Lcg(u64);
    impl Lcg {
        fn u01(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.u01().max(1e-12), self.u01());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }
    fn lognormal(n: usize, seed: u64, log_mean: f64, log_sd: f64) -> Vec<f32> {
        let mut r = Lcg(seed);
        (0..n)
            .map(|_| (log_mean + log_sd * r.normal()).exp() as f32)
            .collect()
    }

    #[test]
    fn debris_cut_is_per_batch_relative() {
        // Batch A deep (nnz~1000), batch B shallow (nnz~100) with a debris
        // sub-population (nnz~10). A *global* cut would call all of B empty;
        // per-batch must keep B's bulk and drop only B's debris, A untouched.
        let a_nnz = lognormal(200, 1, 1000f64.ln(), 0.15);
        let b_bulk = lognormal(200, 2, 100f64.ln(), 0.15);
        let b_deb = lognormal(60, 3, 10f64.ln(), 0.25);
        let mut nnz = vec![];
        let mut sum = vec![];
        let mut batch = vec![];
        for &v in &a_nnz {
            nnz.push(v);
            sum.push(v * 5.0);
            batch.push(0u32);
        }
        for &v in b_bulk.iter().chain(&b_deb) {
            nnz.push(v);
            sum.push(v * 5.0);
            batch.push(1u32);
        }
        let names: Vec<Box<str>> = vec![Box::from("A"), Box::from("B")];
        let res = per_batch_cell_qc(CellQcInputs {
            cell_nnz: &nnz,
            cell_sum: &sum,
            batch_membership: &batch,
            batch_names: &names,
            cfg: CellQcConfig::default(),
        });
        for i in 0..200 {
            assert!(res.keep[i], "batch A cell {i} wrongly dropped");
        }
        let b_bulk_kept = (200..400).filter(|&i| res.keep[i]).count();
        let b_debris_dropped = (400..460)
            .filter(|&i| res.reason[i] == DropReason::Debris)
            .count();
        assert!(
            b_bulk_kept > 190,
            "batch B bulk over-dropped: {b_bulk_kept}/200"
        );
        assert!(
            b_debris_dropped > 45,
            "batch B debris under-dropped: {b_debris_dropped}/60"
        );
    }

    #[test]
    fn unimodal_batch_drops_nothing() {
        // One smooth unimodal batch → BIC guard finds no split → keep all.
        let nnz = lognormal(300, 7, 500f64.ln(), 0.15);
        let sum: Vec<f32> = nnz.iter().map(|&v| v * 5.0).collect();
        let batch = vec![0u32; nnz.len()];
        let names: Vec<Box<str>> = vec![Box::from("solo")];
        let res = per_batch_cell_qc(CellQcInputs {
            cell_nnz: &nnz,
            cell_sum: &sum,
            batch_membership: &batch,
            batch_names: &names,
            cfg: CellQcConfig::default(),
        });
        assert_eq!(res.n_dropped(), 0, "unimodal batch should drop nothing");
    }

    #[test]
    fn disabled_keeps_all() {
        let n = 50;
        let nnz: Vec<f32> = (0..n).map(|i| if i < 5 { 2.0 } else { 500.0 }).collect();
        let sum: Vec<f32> = (0..n).map(|i| if i < 5 { 4.0 } else { 2000.0 }).collect();
        let batch = vec![0u32; n];
        let names: Vec<Box<str>> = vec![Box::from("solo")];
        let res = per_batch_cell_qc(CellQcInputs {
            cell_nnz: &nnz,
            cell_sum: &sum,
            batch_membership: &batch,
            batch_names: &names,
            cfg: CellQcConfig {
                enabled: false,
                ..Default::default()
            },
        });
        assert_eq!(res.n_dropped(), 0);
    }
}
