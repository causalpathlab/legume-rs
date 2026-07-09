use crate::sparse_data_visitors::*;
use crate::sparse_io::*;
use crate::sparse_io_vector::*;

use indicatif::ParallelProgressIterator;
use log::warn;
use matrix_util::sparse_stat::{SparseColumnRunningStatistics, SparseRunningStatistics};
use matrix_util::traits::RunningStatOps;
use matrix_util::utils::partition_by_membership;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use rustc_hash::FxHashMap as HashMap;

#[derive(Clone)]
pub struct SqueezeCutoffs {
    pub row: usize,
    pub column: usize,
}

/// squeeze out rows and columns with excessive zero values
pub fn squeeze_by_nnz(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    cutoffs: SqueezeCutoffs,
    block_size: Option<usize>,
    preload: bool,
) -> anyhow::Result<()> {
    let col_stat = collect_column_stat(data, block_size)?;
    let row_stat = collect_row_stat(data, block_size)?;

    let file = data.get_backend_file_name();
    let backend = data.backend_type();

    let mut data = open_sparse_matrix(file, &backend)?;
    if preload {
        data.preload_columns()?;
    }

    fn nnz_index(nnz: &[f32], cutoff: usize) -> Option<Vec<usize>> {
        let ret: Vec<usize> = nnz
            .iter()
            .enumerate()
            .filter(|&(_, &x)| (x as usize) >= cutoff)
            .map(|(i, _)| i)
            .collect();

        (!ret.is_empty()).then_some(ret)
    }

    let row_nnz_vec = row_stat.count_positives();
    let col_nnz_vec = col_stat.count_positives();
    let row_idx = nnz_index(&row_nnz_vec, cutoffs.row);
    let col_idx = nnz_index(&col_nnz_vec, cutoffs.column);

    if row_idx.is_none() {
        warn!(
            "No rows can be kept with this cutoff {}!\n\
	     \n\
	     We will stop squeezing on the rows.\n\
	     \n",
            cutoffs.row
        );
    }

    if col_idx.is_none() {
        warn!(
            "No columns can be kept with this cutoff {}!\n\
	     \n\
	     We will stop squeezing on the columns.\n\
	     \n",
            cutoffs.column
        );
    }

    data.subset_columns_rows(col_idx.as_ref(), row_idx.as_ref())
}

/// collect row-wise sufficient statistics for Q/C
/// * `data` - `SparseIoVec` across many data matrices
/// * `block_size` - a block size for each parallelized job
pub fn collect_row_stat_across_vec(
    data: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<SparseRunningStatistics<f32>> {
    let mut row_stat = SparseRunningStatistics::new(data.num_rows());
    data.visit_columns_by_block(
        &row_stat_vec_visitor,
        &EmptyArgs {},
        &mut row_stat,
        block_size,
    )?;
    Ok(row_stat)
}

/// collect row statistics for each group of columns
/// * `data` - `SparseIo`
/// * `column_membership` - a hashmap assign columns to groups
/// * `block_size` - a block size for each parallelized job
#[allow(clippy::type_complexity)]
pub fn collect_stratified_row_stat_across_vec(
    data: &SparseIoVec,
    column_membership: &HashMap<Box<str>, Box<str>>,
    block_size: Option<usize>,
) -> anyhow::Result<(Vec<Box<str>>, Vec<SparseRunningStatistics<f32>>)> {
    let column_names = data.column_names()?;
    let default = "".to_string().into_boxed_str();
    let membership = column_names
        .into_iter()
        .map(|k| column_membership.get(&k).unwrap_or(&default).clone())
        .collect::<Vec<_>>();

    let partitions = partition_by_membership(&membership, None);
    let mut group_names = Vec::with_capacity(partitions.len());
    let mut group_stats = Vec::with_capacity(partitions.len());
    let num_features = data.num_rows();

    for (k, cols) in partitions {
        let jobs = create_jobs(cols.len(), num_features, block_size);
        let mut row_stat = SparseRunningStatistics::new(data.num_rows());
        let arc_stat = Arc::new(Mutex::new(&mut row_stat));

        jobs.par_iter()
            .progress_with(styled_progress_bar(jobs.len() as u64, "blocks"))
            .for_each(|&(lb, ub)| {
                let cols_sub = cols[lb..ub].iter().cloned();
                let csc = data
                    .read_columns_csc(cols_sub)
                    .expect("failed to read data");
                let mut stat = arc_stat.lock().expect("failed to lock row_stat");
                stat.add_csc(&csc);
            });

        group_names.push(k);
        group_stats.push(row_stat);
    }

    Ok((group_names, group_stats))
}

/// collect row-wise sufficient statistics for Q/C
/// * `data` - `SparseIo`
/// * `block_size` - a block size for each parallelized job
pub fn collect_row_stat(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    block_size: Option<usize>,
) -> anyhow::Result<SparseRunningStatistics<f32>> {
    let nrows = data.num_rows().unwrap_or(0);
    let mut row_stat = SparseRunningStatistics::new(nrows);
    let arc_stat = Arc::new(Mutex::new(&mut row_stat));

    let jobs = create_jobs(data.num_columns().unwrap_or(0), nrows, block_size);

    jobs.par_iter()
        .progress_with(styled_progress_bar(jobs.len() as u64, "blocks"))
        .for_each(|&(lb, ub)| {
            let csc = data
                .read_columns_csc((lb..ub).collect())
                .expect("failed to read data");
            let mut stat = arc_stat.lock().expect("failed to lock row_stat");
            stat.add_csc(&csc);
        });

    Ok(row_stat)
}

/// collect column-wise sufficient statistics for Q/C
/// * `data` - `SparseIoVec` across many data matrices
/// * `select_rows` - selected row indices
/// * `block_size` - a block size for each parallelized job
pub fn collect_column_stat_across_vec(
    data: &SparseIoVec,
    select_rows: Option<&[usize]>,
    block_size: Option<usize>,
) -> anyhow::Result<SparseColumnRunningStatistics<f32>> {
    let ncols = data.num_columns();
    let nrows_total = data.num_rows();

    let row_mask: Option<Vec<bool>> = select_rows.map(|sel| {
        let mut m = vec![false; nrows_total];
        for &r in sel {
            if r < nrows_total {
                m[r] = true;
            }
        }
        m
    });
    let nrows_denom = row_mask
        .as_ref()
        .map(|m| m.iter().filter(|x| **x).count())
        .unwrap_or(nrows_total);

    let mut col_stat = SparseColumnRunningStatistics::<f32>::new(ncols, nrows_denom);
    data.visit_columns_by_block(&col_stat_visitor, &row_mask, &mut col_stat, block_size)?;
    Ok(col_stat)
}

/// collect column-wise sufficient statistics for Q/C
/// * `data` - `SparseIo`
/// * `block_size` - a block size for each parallelized job
pub fn collect_column_stat(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    block_size: Option<usize>,
) -> anyhow::Result<SparseColumnRunningStatistics<f32>> {
    let ncols = data.num_columns().unwrap_or(0);
    let nrows = data.num_rows().unwrap_or(0);
    let mut col_stat = SparseColumnRunningStatistics::<f32>::new(ncols, nrows);
    let arc_stat = Arc::new(Mutex::new(&mut col_stat));

    let jobs = create_jobs(ncols, nrows, block_size);

    jobs.par_iter()
        .progress_with(styled_progress_bar(jobs.len() as u64, "blocks"))
        .for_each(|&(lb, ub)| {
            let csc = data
                .read_columns_csc((lb..ub).collect())
                .expect("failed to read data");
            let mut stat = arc_stat.lock().expect("failed to lock col_stat");
            stat.add_csc(&csc, lb);
        });

    Ok(col_stat)
}

struct EmptyArgs {}

fn row_stat_vec_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    _: &EmptyArgs,
    arc_stat: Arc<Mutex<&mut SparseRunningStatistics<f32>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let csc = data.read_columns_csc(lb..ub)?;

    let mut stat = arc_stat.lock().expect("failed to lock row_stat");
    stat.add_csc(&csc);
    Ok(())
}

fn col_stat_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    row_mask: &Option<Vec<bool>>,
    arc_stat: Arc<Mutex<&mut SparseColumnRunningStatistics<f32>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let csc = data.read_columns_csc(lb..ub)?;
    let mut stat = arc_stat.lock().expect("failed to lock col_stat");
    match row_mask {
        Some(mask) => stat.add_csc_masked(&csc, lb, mask),
        None => stat.add_csc(&csc, lb),
    }
    Ok(())
}

//////////////////////////////////////////////////////////////////////////////////
// Automatic nnz-cutoff selection + ASCII histogram (shared by `squeeze` and by //
// callers that want cell-calling on a per-column nnz vector, e.g. `faba gem`). //
//////////////////////////////////////////////////////////////////////////////////

/// Suggest a reasonable nnz cutoff by an **exact 1-D 2-means** split of
/// `log(1+nnz)`, **guarded by a BIC model-selection test** so a cutoff is only
/// proposed when the data is genuinely bimodal (ambient + real cells).
///
/// k-means on a line is always a *contiguous* split, so the global optimum is a
/// single threshold — found exactly here by a sorted prefix-sum sweep
/// (O(n log n), **deterministic, no RNG or restarts**). The two clusters' means
/// then feed a BIC comparison against a single Gaussian. To stay robust on
/// discrete low-count data, the mixture is **homoscedastic** (both components
/// share the pooled within-cluster variance): a degenerate near-zero-variance
/// cluster therefore cannot masquerade as a delta spike and force a split (the
/// failure mode of a per-component variance floor). The cutoff (smallest nnz in
/// the higher cluster) is returned only when BIC favors two components. `None`
/// for degenerate input (n < 4 or all-identical nnz) or a unimodal distribution.
pub fn suggest_nnz_cutoff(nnz: &[f32]) -> Option<usize> {
    let n = nnz.len();
    if n < 4 {
        return None;
    }

    // Sort by log(1+nnz) (monotonic in nnz), carrying nnz for the reported cutoff.
    let mut pairs: Vec<(f64, f32)> = nnz.iter().map(|&x| ((1.0 + x as f64).ln(), x)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if pairs[n - 1].0 <= pairs[0].0 {
        return None; // all-identical → nothing to split
    }

    // Prefix sums of x and x² (x = log1p) for O(1) range mean / variance / SSE.
    let mut psum = vec![0.0_f64; n + 1];
    let mut psq = vec![0.0_f64; n + 1];
    for (i, &(x, _)) in pairs.iter().enumerate() {
        psum[i + 1] = psum[i] + x;
        psq[i + 1] = psq[i] + x * x;
    }
    // Within-cluster SSE over the sorted half-open range [a, b): Σx² − (Σx)²/cnt.
    let sse = |a: usize, b: usize| -> f64 {
        let cnt = (b - a) as f64;
        if cnt <= 0.0 {
            return 0.0;
        }
        let s = psum[b] - psum[a];
        (psq[b] - psq[a] - s * s / cnt).max(0.0)
    };

    // Exact 2-means: the split k minimizing SSE(0,k) + SSE(k,n).
    let mut best_k = 1;
    let mut best_sse = f64::INFINITY;
    for k in 1..n {
        let s = sse(0, k) + sse(k, n);
        if s < best_sse {
            best_sse = s;
            best_k = k;
        }
    }

    let (n0, n1) = (best_k, n - best_k);
    let nf = n as f64;
    let mean0 = psum[best_k] / n0 as f64;
    let mean1 = (psum[n] - psum[best_k]) / n1 as f64;
    // Shared (pooled) within-cluster variance — bounded below by the real spread,
    // so a degenerate single-value cluster can't claim near-infinite density.
    let pooled_var = (best_sse / nf).max(1e-9);
    let cutoff = pairs[best_k].1 as usize; // smallest nnz in the higher cluster

    // BIC: homoscedastic 2-Gaussian vs single Gaussian on log(1+nnz).
    // var_all = SSE(0,n)/n is the variance about the global mean (closed-form ll1).
    let var_all = (sse(0, n) / nf).max(1e-9);
    let ll1 = -0.5 * nf * ((2.0 * std::f64::consts::PI * var_all).ln() + 1.0);
    let bic1 = -2.0 * ll1 + 2.0 * nf.ln(); // params: μ, σ²

    let lpi = [(n0 as f64 / nf).ln(), (n1 as f64 / nf).ln()];
    let ll2: f64 = pairs
        .iter()
        .map(|&(x, _)| {
            logsumexp2(
                lpi[0] + gaussian_logpdf(x, mean0, pooled_var),
                lpi[1] + gaussian_logpdf(x, mean1, pooled_var),
            )
        })
        .sum();
    let bic2 = -2.0 * ll2 + 4.0 * nf.ln(); // params: μ0, μ1, σ² (shared), π

    let favors_two = bic2 < bic1;
    log::info!(
        "nnz cell-calling: BIC 1-cluster={:.0}, 2-cluster={:.0} → {}",
        bic1,
        bic2,
        if favors_two {
            format!("bimodal, cutoff at nnz {cutoff}")
        } else {
            "unimodal, no cutoff".to_string()
        }
    );
    favors_two.then_some(cutoff)
}

/// Gaussian log-density at `x` for `N(mu, var)` (variance floored for stability).
fn gaussian_logpdf(x: f64, mu: f64, var: f64) -> f64 {
    use std::f64::consts::PI;
    let var = var.max(1e-9);
    -0.5 * ((2.0 * PI * var).ln() + (x - mu) * (x - mu) / var)
}

/// `log(exp(a) + exp(b))`, numerically stable.
fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m.is_infinite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// One log10(nnz+1) histogram bin, carrying the real nnz range that fell into it
struct HistBin {
    nnz_min: usize,
    nnz_max: usize,
    log_nnz: f64,
    count: usize,
    is_cutoff: bool,
}

/// Create histogram with log10(nnz+1) binning, tracking the real nnz range per bin
fn create_log_histogram(nnz: &[f32], cutoff: usize) -> Vec<HistBin> {
    let cutoff_log = ((cutoff as f64 + 1.0).log10() * 10.0).round() as i32;

    // Bin key represents log10(nnz+1)*10 as integer; value is (count, min nnz, max nnz)
    let mut bins: std::collections::BTreeMap<i32, (usize, usize, usize)> =
        std::collections::BTreeMap::new();

    for &val in nnz {
        let v = val as usize;
        let log_val = ((val as f64 + 1.0).log10() * 10.0).round() as i32;
        let entry = bins.entry(log_val).or_insert((0, usize::MAX, 0));
        entry.0 += 1;
        entry.1 = entry.1.min(v);
        entry.2 = entry.2.max(v);
    }

    // Mark the first bin at or above the cutoff so the arrow always renders,
    // even when no value's log bucket exactly matches cutoff_log.
    let cutoff_bin = bins.keys().copied().find(|&b| b >= cutoff_log);

    bins.into_iter()
        .map(|(bin, (count, nnz_min, nnz_max))| HistBin {
            nnz_min,
            nnz_max,
            log_nnz: bin as f64 / 10.0,
            count,
            is_cutoff: Some(bin) == cutoff_bin,
        })
        .collect()
}

/// Print summary statistics + an ASCII log-histogram of a per-unit nnz vector,
/// marking the `cutoff` bin and (optionally) the 2-means `suggested` cutoff.
/// Used by `data-beans squeeze --show-histogram` and `faba gem --auto-cell-cutoff`.
pub fn print_nnz_summary(label: &str, nnz: &[f32], cutoff: usize, suggested: Option<usize>) {
    const MAX_BAR_WIDTH: usize = 50; // Maximum width for histogram bars

    let total = nnz.len();
    let below_cutoff = nnz.iter().filter(|&&x| (x as usize) < cutoff).count();
    let pct_removed = if total > 0 {
        100.0 * below_cutoff as f64 / total as f64
    } else {
        0.0
    };

    // Calculate basic statistics
    let min = nnz.iter().copied().fold(f32::INFINITY, f32::min) as usize;
    let max = nnz.iter().copied().fold(f32::NEG_INFINITY, f32::max) as usize;
    let sum: f32 = nnz.iter().sum();
    let mean = if total > 0 { sum / total as f32 } else { 0.0 };

    // Calculate median
    let mut sorted = nnz.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if total > 0 {
        if total.is_multiple_of(2) {
            (sorted[total / 2 - 1] + sorted[total / 2]) / 2.0
        } else {
            sorted[total / 2]
        }
    } else {
        0.0
    };

    println!("{} NNZ Distribution:", label);
    println!("  Total: {}", total);
    println!(
        "  Min: {}, Max: {}, Mean: {:.2}, Median: {:.2}",
        min, max, mean, median
    );
    println!(
        "  Cutoff: {} (removes {} / {} = {:.2}%)",
        cutoff, below_cutoff, total, pct_removed
    );
    if let Some(s) = suggested {
        let below_s = nnz.iter().filter(|&&x| (x as usize) < s).count();
        let pct_s = if total > 0 {
            100.0 * below_s as f64 / total as f64
        } else {
            0.0
        };
        println!(
            "  Suggested cutoff (2-means on log1p nnz): {} (would remove {} / {} = {:.2}%)",
            s, below_s, total, pct_s
        );
    }

    // Create histogram with log10(nnz+1) bins, tracking the real nnz range per bin
    let hist = create_log_histogram(nnz, cutoff);

    // Scale bar width on log10(count+1) so a few outlier bins don't flatten the rest
    let max_log_count = hist
        .iter()
        .map(|b| ((b.count as f64) + 1.0).log10())
        .fold(0.0_f64, f64::max)
        .max(1e-9);

    println!("  Histogram (x: actual nnz range [log10(nnz+1)], bar: log10(count+1)):");
    for b in hist {
        let marker = if b.is_cutoff { " <-- CUTOFF" } else { "" };
        let log_count = ((b.count as f64) + 1.0).log10();
        let bar_width = ((log_count / max_log_count) * MAX_BAR_WIDTH as f64).round() as usize;
        let bar_width = if b.count > 0 { bar_width.max(1) } else { 0 };
        let bar = "█".repeat(bar_width);
        let range = if b.nnz_min == b.nnz_max {
            b.nnz_min.to_string()
        } else {
            format!("{}-{}", b.nnz_min, b.nnz_max)
        };
        println!(
            "    {:>9} [{:>4.2}]: {:>6} {}{}",
            range, b.log_nnz, b.count, bar, marker
        );
    }
}

#[cfg(test)]
mod cutoff_tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    #[test]
    fn bic_accepts_clear_bimodal() {
        // A large ambient peak (~5) + a real-cell mode (~300), well separated.
        let mut rng = StdRng::seed_from_u64(7);
        let amb = Normal::new(5.0_f64, 1.5).unwrap();
        let real = Normal::new(300.0_f64, 40.0).unwrap();
        let mut nnz: Vec<f32> = (0..2000)
            .map(|_| amb.sample(&mut rng).max(1.0) as f32)
            .collect();
        nnz.extend((0..400).map(|_| real.sample(&mut rng).max(1.0) as f32));
        let c = suggest_nnz_cutoff(&nnz).expect("clear bimodal → Some(cutoff)");
        assert!(
            c > 15 && c < 300,
            "cutoff {c} should sit in the ambient↔real valley"
        );
    }

    #[test]
    fn bic_rejects_unimodal() {
        // A single Gaussian nnz cloud: the exact split still partitions it, but
        // the BIC guard must reject (the hard split fits the centre worse).
        let mut rng = StdRng::seed_from_u64(11);
        let one = Normal::new(60.0_f64, 12.0).unwrap();
        let nnz: Vec<f32> = (0..3000)
            .map(|_| one.sample(&mut rng).max(1.0) as f32)
            .collect();
        assert!(
            suggest_nnz_cutoff(&nnz).is_none(),
            "single-mode nnz → no cutoff (BIC favors 1 cluster)"
        );
    }

    #[test]
    fn bic_rejects_discrete_lowcount_unimodal() {
        // Regression for the variance-floor defeat: a single unimodal LOW-COUNT
        // discrete cloud must NOT be split. The old per-component variance floor
        // (1e-9) let a near-constant k-means cluster earn a delta-spike density;
        // the homoscedastic (shared-variance) BIC bounds every component's
        // variance by the real within-cluster spread, so this is rejected.
        let mut rng = StdRng::seed_from_u64(3);
        let p = rand_distr::Poisson::new(6.0_f64).unwrap();
        let nnz: Vec<f32> = (0..5000)
            .map(|_| p.sample(&mut rng).max(1.0) as f32)
            .collect();
        assert!(
            suggest_nnz_cutoff(&nnz).is_none(),
            "unimodal discrete low-count nnz → no cutoff"
        );
    }

    #[test]
    fn bic_accepts_lowcount_bimodal() {
        // Genuine bimodality is still found even when the ambient mode is a tight
        // low-count spike (the case the variance fix must not over-correct).
        let mut rng = StdRng::seed_from_u64(5);
        let amb = rand_distr::Poisson::new(2.0_f64).unwrap();
        let real = Normal::new(500.0_f64, 60.0).unwrap();
        let mut nnz: Vec<f32> = (0..6000)
            .map(|_| amb.sample(&mut rng).max(1.0) as f32)
            .collect();
        nnz.extend((0..800).map(|_| real.sample(&mut rng).max(1.0) as f32));
        let c = suggest_nnz_cutoff(&nnz).expect("low-count ambient + real mode → Some(cutoff)");
        assert!(c > 5 && c < 500, "cutoff {c} should land in the valley");
    }

    #[test]
    fn cutoff_is_deterministic() {
        // The exact sweep has no RNG, so repeated calls on the same data are
        // bit-identical (no seeding / restarts needed).
        let mut rng = StdRng::seed_from_u64(9);
        let amb = Normal::new(4.0_f64, 1.0).unwrap();
        let real = Normal::new(200.0_f64, 30.0).unwrap();
        let mut nnz: Vec<f32> = (0..3000)
            .map(|_| amb.sample(&mut rng).max(1.0) as f32)
            .collect();
        nnz.extend((0..500).map(|_| real.sample(&mut rng).max(1.0) as f32));
        let a = suggest_nnz_cutoff(&nnz);
        let b = suggest_nnz_cutoff(&nnz);
        assert_eq!(a, b, "exact 1-D 2-means must be deterministic");
        assert!(a.is_some());
    }
}
