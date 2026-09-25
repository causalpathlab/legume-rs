//! Individual-level confounder factors from negative-control genes (RUV-g),
//! or from all genes as top principal components.
//!
//! Control genes respond to individual-level confounders V but not to the
//! exposure X. Factors learned from them therefore never look at X: they keep
//! the part of V that predicts X (so the propensity on them is informative)
//! without absorbing the exposure's own program (which any ordinary gene set
//! would carry, since X shifts genes genome-wide).
//!
//! Features are per topic, so a composition shift that the exposure induces
//! does not leak in: feature `(k, g)` is individual `i`'s log rate of control
//! gene `g` within topic `k`, from the stage-1 sums of raw counts, which use
//! no exposure labels (so no separate pass over the data is needed).
//!
//! With all genes instead of controls, the factors are the top principal
//! components of individual expression, under the assumption that they
//! carry no exposure signal. A single gene's share of them is about one over
//! the number of genes, so it hardly adjusts for itself.
//!
//! Reference: Risso, Ngai, Speed & Dudoit (2014) Nat Biotechnol, "Normalization
//! of RNA-seq data using factor analysis of control genes or samples";
//! Gagnon-Bartsch & Speed (2012) Biostatistics, RUV-2.

use crate::common::*;
use crate::stat::CocoaStat;

#[cfg(test)]
mod tests;

/// Individuals need at least this much topic weight (cells) to contribute
/// a feature value in that topic.
const MIN_TOPIC_CELLS: f32 = 5.0;
/// A (topic, gene) feature needs this mean count per observed individual.
const MIN_MEAN_COUNT: f32 = 1.0;
/// Upper bound on the number of factors kept.
const MAX_FACTORS: usize = 10;

/// Per-topic individual totals of the control genes and the library size.
struct ControlTotals {
    /// topic -> control gene x individual
    y: Vec<Mat>,
    /// topic -> individual library size (all genes)
    lib: Vec<DVec>,
    /// topic -> individual topic weight (cells)
    cells: Vec<DVec>,
}

/// Control totals from the stage-1 statistics: the control rows of each
/// topic's gene x individual sums, the library size as their column sum over
/// all genes, and the cells as the row sums of the individual x pseudobulk
/// weights.
fn totals_from_stat(stat: &CocoaStat, rows: &[usize]) -> ControlTotals {
    let n_topics = stat.n_topics();
    let mut totals = ControlTotals {
        y: Vec::with_capacity(n_topics),
        lib: Vec::with_capacity(n_topics),
        cells: Vec::with_capacity(n_topics),
    };
    for k in 0..n_topics {
        let y1 = stat.indv_y1_stat(k);
        totals.y.push(y1.select_rows(rows));
        totals.lib.push(y1.row_sum_tr());
        totals.cells.push(stat.indv_size_stat(k).column_sum());
    }
    totals
}

/// Learned factors with the diagnostics worth logging.
pub struct ControlFactors {
    /// individual x factor
    pub factors: Mat,
    /// features that entered the SVD
    pub n_features: usize,
    /// singular values, largest first
    pub singular_values: Vec<f32>,
}

/// Learn individual-level confounder factors from control genes.
///
/// * `stat` - the stage-1 statistics (label-free sums per topic)
/// * `control_rows` - gene rows of the control genes (all genes for PCs)
/// * `n_factors` - how many to keep; `None` picks by the eigenvalue ratio
pub fn learn_control_factors(
    stat: &CocoaStat,
    control_rows: &[usize],
    n_factors: Option<usize>,
) -> anyhow::Result<ControlFactors> {
    let totals = totals_from_stat(stat, control_rows);
    let features = log_rate_features(&totals);
    anyhow::ensure!(
        features.ncols() >= 2,
        "only {} usable control-gene features; need more expressed control genes",
        features.ncols()
    );
    let (factors, singular_values) = leading_factors(&features, n_factors);
    Ok(ControlFactors {
        n_features: features.ncols(),
        factors,
        singular_values,
    })
}

/// Individual x feature matrix of standardized log rates; individuals
/// without enough cells in a topic sit at the feature mean (zero).
fn log_rate_features(t: &ControlTotals) -> Mat {
    let n_indv = t.lib.first().map_or(0, |l| l.len());
    let mut cols: Vec<DVec> = Vec::new();
    for k in 0..t.y.len() {
        let observed: Vec<bool> = t.cells[k]
            .iter()
            .zip(t.lib[k].iter())
            .map(|(&c, &l)| c >= MIN_TOPIC_CELLS && l > 0.0)
            .collect();
        let n_obs = observed.iter().filter(|&&o| o).count();
        if n_obs < 3 {
            continue;
        }
        for row in t.y[k].row_iter() {
            let mean_count = row
                .iter()
                .zip(&observed)
                .filter(|(_, &o)| o)
                .map(|(y, _)| y)
                .sum::<f32>()
                / n_obs as f32;
            if mean_count < MIN_MEAN_COUNT {
                continue;
            }
            let mut f = DVec::zeros(n_indv);
            for i in 0..n_indv {
                if observed[i] {
                    f[i] = ((row[i] + 0.5) / (t.lib[k][i] + 1.0) * 1e6).ln();
                }
            }
            let mean = (0..n_indv)
                .filter(|&i| observed[i])
                .map(|i| f[i])
                .sum::<f32>()
                / n_obs as f32;
            let var = (0..n_indv)
                .filter(|&i| observed[i])
                .map(|i| (f[i] - mean).powi(2))
                .sum::<f32>()
                / n_obs as f32;
            if var < 1e-10 {
                continue;
            }
            let sd = var.sqrt();
            for i in 0..n_indv {
                f[i] = if observed[i] { (f[i] - mean) / sd } else { 0.0 };
            }
            cols.push(f);
        }
    }
    if cols.is_empty() {
        Mat::zeros(n_indv, 0)
    } else {
        Mat::from_columns(&cols)
    }
}

/// Leading left singular vectors of the individual x feature matrix, with
/// the rank from the eigenvalue-ratio rule (Ahn & Horenstein 2013, the
/// largest `s_k^2 / s_{k+1}^2` for `k <= k_max`), `k_max` capped at a
/// quarter of the individuals and `MAX_FACTORS`; or exactly `fixed`.
fn leading_factors(features: &Mat, fixed: Option<usize>) -> (Mat, Vec<f32>) {
    let n = features.nrows();
    // nalgebra returns the singular values in descending order
    let svd = features.clone().svd(true, false);
    let s: Vec<f32> = svd.singular_values.iter().cloned().collect();

    let k_max = (n / 4)
        .clamp(1, MAX_FACTORS)
        .min(s.len().saturating_sub(1))
        .max(1);
    let mut r = 1;
    let mut best = f32::NEG_INFINITY;
    for k in 1..=k_max {
        if k >= s.len() || s[k] <= 0.0 {
            break;
        }
        let ratio = (s[k - 1] / s[k]).powi(2);
        if ratio > best {
            best = ratio;
            r = k;
        }
    }

    if let Some(k) = fixed {
        r = k.clamp(1, s.len());
    }

    let u = svd.u.expect("svd u");
    (u.columns(0, r).into_owned(), s)
}
