use crate::common::*;
use indicatif::ProgressIterator;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::Inference;
use matrix_param::traits::*;
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
        let mut ret = vec![];
        for k in 0..self.n_topics {
            ret.push(self.optimize_each_topic(k)?);
            info!("finished optimization {}/{}", k + 1, self.n_topics);
        }
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

        (0..self.n_opt_iter).progress().for_each(|_opt_iter| {
            // shared component μ(d,p)
            //
            // y1(d,p) + y0(d,p)
            // -----------------------------------------
            // sum_i τ(d,i) * n(i,p) + γ(d,p) * n(s)

            let gamma_dp = gamma_param_dp.posterior_mean();
            let tau_di = tau_param_di.posterior_mean();

            denom_dp.copy_from(gamma_dp);
            denom_dp.row_iter_mut().for_each(|mut row| {
                row.component_mul_assign(&size_p.transpose());
            });
            denom_dp += tau_di * size_ip;

            mu_param_dp.update_stat(&y10_dp, &denom_dp);
            mu_param_dp.calibrate();

            // matched component γ(d,p)
            //
            // y0(d,p)
            // -----------------------------------
            // μ(d,p) * n(s)

            let mu_dp = mu_param_dp.posterior_mean();

            denom_dp.copy_from(mu_dp);
            denom_dp.row_iter_mut().for_each(|mut row| {
                row.component_mul_assign(&size_p.transpose());
            });

            gamma_param_dp.update_stat(&y0_dp, &denom_dp);
            gamma_param_dp.calibrate();

            // individual-specific effect τ(d,i)
            //
            // y1(d,i)
            // ---------------------
            // sum_s μ(d,p) * n(i,p)

            denom_di.copy_from(&(mu_dp * size_ip.transpose()));
            tau_param_di.update_stat(&y1_di, &denom_di);
            tau_param_di.calibrate();
        });

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

    for k in 0..n_topics {
        let tau_log = parameters[k].exposure.posterior_log_mean();
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
