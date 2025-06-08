use crate::common::*;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::Inference;
use matrix_param::traits::*;
use matrix_util::traits::MatOps;

pub struct CocoaStat<'a> {
    y1_sum_ds_vec: Vec<Mat>,            // cell type topic x gene x sample
    y0_sum_ds_vec: Vec<Mat>,            // cell type topic x gene x sample
    size_s_vec: Vec<DVec>,              // cell type topic x sample
    sample_to_exposure: &'a Vec<usize>, // exposure assignment
    n_topics: usize,                    // number of  cell types
    n_exposures: usize,                 // exposure/treatment
    n_opt_iter: usize,                  // iterative optimization
    a0: f32,                            // hyper parameters
    b0: f32,                            // hyper parameters
}

fn num_categories(sample_to_exposure: &Vec<usize>) -> usize {
    match sample_to_exposure.iter().max() {
        Some(&sz) => sz + 1,
        _ => 1,
    }
}

pub struct CocoaGammaOut {
    pub shared: GammaMatrix,
    pub residual: GammaMatrix,
    pub exposure: GammaMatrix,
}

impl<'a> CocoaStat<'a> {
    pub fn new(
        n_genes: usize,
        n_topics: usize,
        sample_to_exposure: &'a Vec<usize>,
        n_opt_iter: Option<usize>,
        hyper_param: Option<(f32, f32)>,
    ) -> Self {
        let a0 = hyper_param.map(|x| x.0);
        let b0 = hyper_param.map(|x| x.1);
        let n_samples = sample_to_exposure.len();
        let n_exposures = num_categories(sample_to_exposure);

        Self {
            y1_sum_ds_vec: vec![Mat::zeros(n_genes, n_samples); n_topics],
            y0_sum_ds_vec: vec![Mat::zeros(n_genes, n_samples); n_topics],
            size_s_vec: vec![DVec::zeros(n_samples); n_topics],
            sample_to_exposure,
            n_topics,
            n_exposures,
            n_opt_iter: n_opt_iter.unwrap_or(100),
            a0: a0.unwrap_or(1.),
            b0: b0.unwrap_or(1.),
        }
    }

    pub fn y1_stat(&mut self, k: usize) -> &mut Mat {
        &mut self.y1_sum_ds_vec[k]
    }

    pub fn y0_stat(&mut self, k: usize) -> &mut Mat {
        &mut self.y0_sum_ds_vec[k]
    }

    pub fn size_stat(&mut self, k: usize) -> &mut DVec {
        &mut self.size_s_vec[k]
    }

    pub fn estimate_parameters(&self) -> anyhow::Result<Vec<CocoaGammaOut>> {
        let mut ret = vec![];
        for k in 0..self.n_topics {
            ret.push(self.optimize_each_topic(k)?);
        }
        Ok(ret)
    }

    pub fn optimize_each_topic(&self, k: usize) -> anyhow::Result<CocoaGammaOut> {
        let size_s = &self.size_s_vec[k];

        let n_genes = self.y1_sum_ds_vec[k].nrows();

        let y1_ds = self.y1_sum_ds_vec[k]
            .normalize_columns()
            .scale(n_genes as f32);
        let y0_ds = self.y0_sum_ds_vec[k]
            .normalize_columns()
            .scale(n_genes as f32);

        let y10_ds = &y1_ds + &y0_ds;

        debug_assert_eq!(size_s.nrows(), self.sample_to_exposure.len());
        debug_assert_eq!(size_s.nrows(), y1_ds.ncols());
        debug_assert_eq!(y1_ds.ncols(), y0_ds.ncols());
        debug_assert_eq!(y1_ds.nrows(), y0_ds.nrows());

        let n_exposures = self.n_exposures;

        let sample_to_exposure: Vec<_> = self
            .sample_to_exposure
            .iter()
            .enumerate()
            .filter_map(|(s, &x)| if x < n_exposures { Some((s, x)) } else { None })
            .collect();

        let mut y1_dx = Mat::zeros(y1_ds.nrows(), n_exposures);
        let mut y0_dx = Mat::zeros(y0_ds.nrows(), n_exposures);

        for &(s, x) in sample_to_exposure.iter() {
            if x < y1_dx.ncols() {
                let mut y = y1_dx.column_mut(x);
                y += &y1_ds.column(s);
                let mut y0 = y0_dx.column_mut(x);
                y0 += &y0_ds.column(s);
            }
        }

        // model 1: sum_s ( y[g,s] * log(μ[g,s] * τ[g,x(s)]) - n[s] * μ[g,s] * τ[g,x(s)] )
        // model 2: sum_s ( y0[g,s] * log(μ[g,s] * γ[g,x(s)]) - n[s] * μ[g,s] * γ[g,x(s)] )

        let n_genes = y1_ds.nrows();
        let n_samples = y1_ds.ncols();
        let n_exposures = y1_dx.ncols();

        let mut mu_param_ds = GammaMatrix::new((n_genes, n_samples), self.a0, self.b0);
        let mut tau_param_dx = GammaMatrix::new((n_genes, n_exposures), self.a0, self.b0);
        let mut gamma_param_dx = GammaMatrix::new((n_genes, n_exposures), self.a0, self.b0);

        // let mut gamma_param_ds = GammaMatrix::new((n_genes, n_samples), self.a0, self.b0);
        mu_param_ds.calibrate();
        tau_param_dx.calibrate();
        gamma_param_dx.calibrate();
        // gamma_param_ds.calibrate();

        let mut denom_ds = Mat::zeros(n_genes, n_samples);
        let mut denom_dx = Mat::zeros(n_genes, n_exposures);

        for _iter in 0..self.n_opt_iter {
            // 1. sample-specific component
            //           y1[g,s]             + y0[g,s]
            // μ[g,s] = ----------------------------------------
            //           size[s] * τ[g,x(s)] + size[s] * γ[g,s]
            let gamma_dx = gamma_param_dx.posterior_mean();
            let tau_dx = tau_param_dx.posterior_mean();
            // let gamma_ds = gamma_param_ds.posterior_mean();
            denom_ds.fill(0.);
            for &(s, x) in sample_to_exposure.iter() {
                let mut denom = denom_ds.column_mut(s);

                denom += &tau_dx.column(x).scale(size_s[s]);
                denom += &gamma_dx.column(x).scale(size_s[s]);
                // denom += &gamma_ds.column(s).scale(size_s[s]);
            }
            mu_param_ds.update_stat(&y10_ds, &denom_ds);
            mu_param_ds.calibrate();

            // 2. shared exposure component
            //           y1[g,x]
            // τ[g,x] = -------------------------------
            //           sum_s μ[g,s] I{x(s)=x} * n[s]
            let mu_ds = mu_param_ds.posterior_mean();

            denom_dx.fill(0.);
            for &(s, x) in sample_to_exposure.iter() {
                let mut denom = denom_dx.column_mut(x);
                denom += &mu_ds.column(s).scale(size_s[s]);
            }
            tau_param_dx.update_stat(&y1_dx, &denom_dx);
            tau_param_dx.calibrate();

            // 3. null exposure component
            //           y0[g,x]
            // γ[g,x] = -----------------------------
            //           sum_s μ[g,s] I{x(s)=x} n[s]
            gamma_param_dx.update_stat(&y0_dx, &denom_dx);
        }

        info!("finished optimization");

        Ok(CocoaGammaOut {
            shared: mu_param_ds,
            residual: gamma_param_dx,
            exposure: tau_param_dx,
        })
    }
}
