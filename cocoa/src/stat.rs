use crate::common::*;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::CalibrateTarget;
use matrix_param::traits::Inference;
use matrix_param::traits::*;
use rayon::prelude::*;
use special::Error;

pub struct CocoaStat {
    y1_sum_dp_vec: Vec<Mat>, // cell type topic x gene x pseudobulk sample
    y0_sum_dp_vec: Vec<Mat>, // cell type topic x gene x pseudobulk sample
    size_p_vec: Vec<DVec>,   // cell type topic x pseudobulk sample
    y1_sum_di_vec: Vec<Mat>, // cell type topic x gene x individual
    size_ip_vec: Vec<Mat>,   // cell type topic x individual x pseudobulk sample
    n_topics: usize,         // number of  cell types
    n_opt_iter: usize,       // iterative optimization
    a0: f32,                 // hyper parameters
    b0: f32,                 // hyper parameters
}

pub struct CocoaGammaOut {
    pub shared: GammaMatrix,
    pub residual: GammaMatrix,
    pub exposure: GammaMatrix,
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
            y0_sum_dp_vec: vec![Mat::zeros(n_genes, n_samples); n_topics],
            size_p_vec: vec![DVec::zeros(n_samples); n_topics],
            y1_sum_di_vec: vec![Mat::zeros(n_genes, n_indv); n_topics],
            size_ip_vec: vec![Mat::zeros(n_indv, n_samples); n_topics],
            n_topics,
            n_opt_iter: n_opt_iter.unwrap_or(100),
            a0: a0.unwrap_or(1.),
            b0: b0.unwrap_or(1.),
        }
    }

    pub fn y1_stat_mut(&mut self, k: usize) -> &mut Mat {
        &mut self.y1_sum_dp_vec[k]
    }

    pub fn y0_stat_mut(&mut self, k: usize) -> &mut Mat {
        &mut self.y0_sum_dp_vec[k]
    }

    pub fn size_stat_mut(&mut self, k: usize) -> &mut DVec {
        &mut self.size_p_vec[k]
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

    pub fn y0_stat(&self, k: usize) -> &Mat {
        &self.y0_sum_dp_vec[k]
    }

    pub fn size_stat(&self, k: usize) -> &DVec {
        &self.size_p_vec[k]
    }

    pub fn indv_y1_stat(&self, k: usize) -> &Mat {
        &self.y1_sum_di_vec[k]
    }

    pub fn indv_size_stat(&self, k: usize) -> &Mat {
        &self.size_ip_vec[k]
    }

    pub fn estimate_parameters(&self) -> anyhow::Result<Vec<CocoaGammaOut>> {
        let ret: Result<Vec<_>, _> = (0..self.n_topics)
            .into_par_iter()
            .map(|k| self.optimize_each_topic(k))
            .collect();
        let ret = ret?;
        info!("finished optimization for {} topics", self.n_topics);
        Ok(ret)
    }

    pub fn optimize_each_topic(&self, k: usize) -> anyhow::Result<CocoaGammaOut> {
        let y1_dp = self.y1_stat(k);
        let y0_dp = self.y0_stat(k);
        let y10_dp = y1_dp + y0_dp;
        let y1_di = self.indv_y1_stat(k);

        let size_p = self.size_stat(k);
        let size_ip = self.indv_size_stat(k);

        let n_genes = y1_dp.nrows();
        let n_pb = y1_dp.ncols();
        let n_indv = y1_di.ncols();

        let mut mu_param_dp = GammaMatrix::new((n_genes, n_pb), self.a0, self.b0);
        let mut gamma_param_dp = GammaMatrix::new((n_genes, n_pb), self.a0, self.b0);
        let mut tau_param_di = GammaMatrix::new((n_genes, n_indv), self.a0, self.b0);

        let mut denom_dp = Mat::zeros(n_genes, n_pb);
        let mut denom_di = Mat::zeros(n_genes, n_indv);

        for _opt_iter in 0..self.n_opt_iter {
            // shared component μ(d,p)
            //
            // y1(d,p) + y0(d,p)
            // -----------------------------------------
            // sum_i τ(d,i) * n(i,p) + γ(d,p) * n(s)

            let gamma_dp = gamma_param_dp.posterior_mean();
            let tau_di = tau_param_di.posterior_mean();

            // γ(d,p) * n(p): scale each column p by size_p[p]
            denom_dp.copy_from(gamma_dp);
            for p in 0..n_pb {
                denom_dp.column_mut(p).scale_mut(size_p[p]);
            }
            denom_dp += tau_di * size_ip;

            mu_param_dp.update_stat(&y10_dp, &denom_dp);
            mu_param_dp.calibrate_with(CalibrateTarget::MeanOnly);

            // matched component γ(d,p)
            //
            // y0(d,p)
            // -----------------------------------
            // μ(d,p) * n(s)

            let mu_dp = mu_param_dp.posterior_mean();

            // μ(d,p) * n(p): scale each column p by size_p[p]
            denom_dp.copy_from(mu_dp);
            for p in 0..n_pb {
                denom_dp.column_mut(p).scale_mut(size_p[p]);
            }

            gamma_param_dp.update_stat(y0_dp, &denom_dp);
            gamma_param_dp.calibrate_with(CalibrateTarget::MeanOnly);

            // individual-specific effect τ(d,i)
            //
            // y1(d,i)
            // ---------------------
            // sum_s μ(d,p) * n(i,p)

            denom_di.copy_from(&(mu_dp * size_ip.transpose()));
            tau_param_di.update_stat(y1_di, &denom_di);
            tau_param_di.calibrate_with(CalibrateTarget::MeanOnly);
        }

        // Final full calibration for downstream use (log_mean needed
        // for exposure contrast, all quantities needed for I/O export)
        mu_param_dp.calibrate();
        gamma_param_dp.calibrate();
        tau_param_di.calibrate();

        Ok(CocoaGammaOut {
            shared: mu_param_dp,
            residual: gamma_param_dp,
            exposure: tau_param_di,
        })
    }
}

/// Compute per-gene signed log exposure contrast:
///   mean(log τ_{exp1}) - mean(log τ_{exp0})
/// averaged across topics. Returns Vec<f32> of length n_genes.
pub fn compute_exposure_contrast(
    parameters: &[CocoaGammaOut],
    exposure_assignment: &[usize],
) -> Vec<f32> {
    let n_topics = parameters.len();
    let n_genes = parameters[0].exposure.posterior_log_mean().nrows();
    let n_indv = parameters[0].exposure.posterior_log_mean().ncols();

    let exp0_indvs: Vec<usize> = (0..n_indv)
        .filter(|&i| exposure_assignment[i] == 0)
        .collect();
    let exp1_indvs: Vec<usize> = (0..n_indv)
        .filter(|&i| exposure_assignment[i] == 1)
        .collect();

    let n0 = exp0_indvs.len() as f32;
    let n1 = exp1_indvs.len() as f32;

    let mut contrast = vec![0f32; n_genes];

    for param in parameters {
        let tau_log = param.exposure.posterior_log_mean();
        for g in 0..n_genes {
            let mean0: f32 = exp0_indvs.iter().map(|&i| tau_log[(g, i)]).sum::<f32>() / n0;
            let mean1: f32 = exp1_indvs.iter().map(|&i| tau_log[(g, i)]).sum::<f32>() / n1;
            contrast[g] += (mean1 - mean0) / n_topics as f32;
        }
    }

    contrast
}

/// Compute two-sided p-value from z-score using normal CDF.
pub fn z_to_pvalue(z: f32) -> f32 {
    // p = erfc(|z| / sqrt(2))
    let p = (z.abs() as f64 / std::f64::consts::SQRT_2).compl_error();
    p as f32
}

///////////////////////////////////////////////////////////////////////////
// Residual collider adjustment for topic proportions
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
///////////////////////////////////////////////////////////////////////////

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
