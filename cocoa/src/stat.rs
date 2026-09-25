use crate::common::*;
use rayon::prelude::*;
use special::Error;

mod baseline;
pub use baseline::*;
pub mod control_factors;
pub mod estimability;
pub mod propensity;
pub mod propensity_effect;
#[cfg(test)]
pub(crate) mod test_util;

#[cfg(test)]
mod tests;

pub struct CocoaStat {
    y1_sum_dp_vec: Vec<Mat>,  // cell type topic x gene x pseudobulk sample
    y1_sum_di_vec: Vec<Mat>,  // cell type topic x gene x individual
    size_ip_vec: Vec<Mat>,    // cell type topic x individual x pseudobulk sample
    mixing: Vec<TopicMixing>, // per topic: how well pseudobulks pool individuals
    n_topics: usize,          // number of  cell types
    n_opt_iter: usize,        // iterative optimization
    a0: f32,                  // hyper parameters
    b0: f32,                  // hyper parameters
}

/// How well one topic's pseudobulks pool individuals. A pseudobulk that
/// holds too few individuals cannot separate its cell-state rate from their
/// multipliers, and is dropped from the topic.
#[derive(Clone, Default)]
pub struct TopicMixing {
    pub pseudobulks_kept: usize,
    pub pseudobulks_dropped: usize,
    /// topic weight (cells) kept and dropped
    pub cells_kept: f32,
    pub cells_dropped: f32,
    /// individuals in each kept pseudobulk
    pub individuals_per_pseudobulk: Vec<usize>,
}

pub struct CocoaStatArgs {
    pub n_genes: usize,
    pub n_topics: usize,
    pub n_indv: usize,
    pub n_samples: usize,
}

impl CocoaStat {
    pub fn new(
        numbers: CocoaStatArgs,
        n_opt_iter: Option<usize>,
        hyper_param: Option<(f32, f32)>,
    ) -> Self {
        let n_genes = numbers.n_genes;
        let n_topics = numbers.n_topics;
        let n_indv = numbers.n_indv;
        let n_samples = numbers.n_samples;

        let a0 = hyper_param.map(|x| x.0);
        let b0 = hyper_param.map(|x| x.1);

        Self {
            y1_sum_dp_vec: vec![Mat::zeros(n_genes, n_samples); n_topics],
            y1_sum_di_vec: vec![Mat::zeros(n_genes, n_indv); n_topics],
            size_ip_vec: vec![Mat::zeros(n_indv, n_samples); n_topics],
            mixing: vec![TopicMixing::default(); n_topics],
            n_topics,
            n_opt_iter: n_opt_iter.unwrap_or(100),
            a0: a0.unwrap_or(1.),
            b0: b0.unwrap_or(1.),
        }
    }

    pub fn y1_stat_mut(&mut self, k: usize) -> &mut Mat {
        &mut self.y1_sum_dp_vec[k]
    }

    pub fn indv_y1_stat_mut(&mut self, k: usize) -> &mut Mat {
        &mut self.y1_sum_di_vec[k]
    }

    pub fn indv_size_stat_mut(&mut self, k: usize) -> &mut Mat {
        &mut self.size_ip_vec[k]
    }

    pub fn y1_stat(&self, k: usize) -> &Mat {
        &self.y1_sum_dp_vec[k]
    }

    pub fn indv_y1_stat(&self, k: usize) -> &Mat {
        &self.y1_sum_di_vec[k]
    }

    pub fn mixing(&self, k: usize) -> &TopicMixing {
        &self.mixing[k]
    }

    pub fn mixing_mut(&mut self, k: usize) -> &mut TopicMixing {
        &mut self.mixing[k]
    }

    pub fn n_topics(&self) -> usize {
        self.n_topics
    }

    pub fn indv_size_stat(&self, k: usize) -> &Mat {
        &self.size_ip_vec[k]
    }
}

/// Compute two-sided p-value from z-score using normal CDF.
pub fn z_to_pvalue(z: f32) -> f32 {
    // p = erfc(|z| / sqrt(2))
    let p = (z.abs() as f64 / std::f64::consts::SQRT_2).compl_error();
    p as f32
}

////////////////////////////////////////////////////////
// Residual collider adjustment for topic proportions //
////////////////////////////////////////////////////////
//
// When cell type A is a collider (X -> A <- U), conditioning on A
// opens the spurious path X -> A <- U -> Y. We remove the exposure-
// driven component of topic logits before matching, breaking the
// X -> A edge.
//
// Method: adapted from residual collider stratification.
//   Hartwig et al. (2023) Eur J Epidemiol
//   "Avoiding collider bias in MR when performing stratified analyses"
//
// Background on collider bias with continuous conditioning:
//   Akimova et al. (2021) Sci Rep
//   "Gene-environment dependencies lead to collider bias in models
//    with polygenic scores"
//
// See also:
//   Cole et al. (2010) Int J Epidemiol
//   "Illustrating bias due to conditioning on a collider"
//
//   Davey Smith & Munafò (2019) Int J Epidemiol
//   "Contextualizing selection bias in Mendelian randomization"

/// Average each topic's log-proportion across cells belonging to the
/// same individual.
///
/// Returns (n_individuals x n_topics) matrix of per-individual means.
fn average_topic_log_proportions_per_individual(
    cell_topic_proportions: &Mat, // n_cells x n_topics (probability space)
    cell_to_individual: &[usize], // which individual each cell belongs to
    n_individuals: usize,
) -> Mat {
    let n_topics = cell_topic_proportions.ncols();
    let mut sum = Mat::zeros(n_individuals, n_topics);
    let mut count = vec![0usize; n_individuals];

    for (j, &indv) in cell_to_individual.iter().enumerate() {
        if indv >= n_individuals {
            continue; // skip unmatched cells
        }
        count[indv] += 1;
        for k in 0..n_topics {
            let val = cell_topic_proportions[(j, k)].max(1e-30).ln();
            sum[(indv, k)] += val;
        }
    }

    for i in 0..n_individuals {
        if count[i] > 0 {
            let n = count[i] as f32;
            for k in 0..n_topics {
                sum[(i, k)] /= n;
            }
        }
    }

    sum
}

/// For each topic, compute the mean log-proportion within each exposure
/// group and the grand mean across all individuals.
///
/// Returns:
///   - exposure_group_means: (n_exposure_groups x n_topics)
///   - grand_mean: (1 x n_topics)
fn average_topic_logits_per_exposure_group(
    individual_topic_logits: &Mat,       // n_individuals x n_topics
    individual_exposure_group: &[usize], // exposure group of each individual
) -> (Mat, Mat) {
    let n_individuals = individual_topic_logits.nrows();
    let n_topics = individual_topic_logits.ncols();
    let n_groups = individual_exposure_group.iter().max().map_or(0, |&m| m + 1);

    let mut group_sum = Mat::zeros(n_groups, n_topics);
    let mut group_count = vec![0usize; n_groups];
    let mut grand_sum = Mat::zeros(1, n_topics);

    for i in 0..n_individuals {
        let g = individual_exposure_group[i];
        group_count[g] += 1;
        for k in 0..n_topics {
            let val = individual_topic_logits[(i, k)];
            group_sum[(g, k)] += val;
            grand_sum[(0, k)] += val;
        }
    }

    for g in 0..n_groups {
        if group_count[g] > 0 {
            let n = group_count[g] as f32;
            for k in 0..n_topics {
                group_sum[(g, k)] /= n;
            }
        }
    }

    let n_total = n_individuals as f32;
    for k in 0..n_topics {
        grand_sum[(0, k)] /= n_total;
    }

    (group_sum, grand_sum)
}

/// Remove the exposure-driven shift from each cell's topic proportions.
///
/// For cell j in individual i with exposure group x:
///   log z'_jk = log z_jk - (group_mean_xk - grand_mean_k)
///
/// This breaks the X -> A (exposure -> cell type) edge in the collider
/// DAG while preserving within-individual cell-level variation.
///
/// The input `cell_topic_proportions` is in probability space (after
/// exp of logits). We take log, subtract the exposure-group shift,
/// and exp back. The downstream `sum_to_one_rows_inplace()` will
/// re-normalize to valid proportions.
///
/// Works for any number of exposure groups (binary or multi-category).
///
/// Returns per-topic max absolute shift across groups for logging.
pub fn remove_exposure_effect_from_topic_proportions(
    cell_topic_proportions: &mut Mat, // n_cells x n_topics, modified in place
    cell_to_individual: &[usize],     // which individual each cell belongs to
    individual_exposure_group: &[usize], // exposure group of each individual
) -> Vec<f32> {
    let n_topics = cell_topic_proportions.ncols();
    let n_cells = cell_topic_proportions.nrows();
    let n_individuals = individual_exposure_group.len();

    // Step 1: individual-level mean log-proportions
    let individual_topic_logits = average_topic_log_proportions_per_individual(
        cell_topic_proportions,
        cell_to_individual,
        n_individuals,
    );

    // Step 2: per-exposure-group means and grand mean
    let (group_means, grand_mean) = average_topic_logits_per_exposure_group(
        &individual_topic_logits,
        individual_exposure_group,
    );

    // Precompute multiplicative factors: exp(-(group_mean - grand_mean))
    // Since exp(log(z) - shift) = z * exp(-shift), we avoid per-cell log/exp
    let n_groups = group_means.nrows();
    let mut scale_per_group_topic = Mat::zeros(n_groups, n_topics);
    let mut max_shift_per_topic = vec![0f32; n_topics];
    for g in 0..n_groups {
        for k in 0..n_topics {
            let shift = group_means[(g, k)] - grand_mean[(0, k)];
            scale_per_group_topic[(g, k)] = (-shift).exp();
            let abs_shift = shift.abs();
            if abs_shift > max_shift_per_topic[k] {
                max_shift_per_topic[k] = abs_shift;
            }
        }
    }

    // Step 3: multiply each cell's proportions by the precomputed scale factor
    for j in 0..n_cells {
        let indv = cell_to_individual[j];
        if indv >= n_individuals {
            continue; // skip unmatched cells
        }
        let exp_group = individual_exposure_group[indv];
        for k in 0..n_topics {
            cell_topic_proportions[(j, k)] *= scale_per_group_topic[(exp_group, k)];
        }
    }

    max_shift_per_topic
}

/// True when every non-empty row is a hard one-hot assignment.
///
/// All-zero rows (e.g. NA cells) are skipped so they do not suppress the
/// hard-assignment warning; a matrix with no assigned row is not one-hot.
pub fn topics_look_one_hot(z: &Mat) -> bool {
    let mut n_pos = z
        .row_iter()
        .map(|r| r.iter().filter(|&&v| v > 1e-6).count())
        .filter(|&n| n > 0)
        .peekable();
    n_pos.peek().is_some() && n_pos.all(|n| n == 1)
}
