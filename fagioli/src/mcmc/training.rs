use anyhow::Result;
use log::info;
use nalgebra::DMatrix;

use crate::summary_stats::common::{BlockFitResult, RssBlockFitter, RssParams};
use crate::summary_stats::rss_svd::RssSvdNal;

use mcmc_util::engine::McmcConfig;
use mcmc_util::sparse_regression::{
    mcmc_sparse_regression, BernoulliNormalPrior, RegressionConfig, SoftmaxNormalPrior,
};

use super::config::{McmcFitConfig, McmcPriorType};

/// MCMC-based per-block fitter using mcmc-util's ESS sparse regression.
pub struct McmcSampler {
    pub config: McmcFitConfig,
}

impl McmcSampler {
    pub fn new(config: McmcFitConfig) -> Self {
        Self { config }
    }
}

impl RssBlockFitter for McmcSampler {
    fn fit_block(
        &self,
        x_block: &DMatrix<f32>,
        z_block: &DMatrix<f32>,
        rss_params: &RssParams,
        seed: u64,
    ) -> Result<BlockFitResult> {
        let t = z_block.ncols();
        let p = x_block.ncols();
        let cfg = &self.config;

        // RSS SVD projection into eigenspace
        let svd = RssSvdNal::from_genotypes(x_block, rss_params.max_rank, rss_params.lambda)?;
        let kk = svd.effective_rank();

        // LDSC intercept correction (in-place)
        let mut z_block = z_block.clone();
        if rss_params.ldsc_intercept && kk > 2 {
            let (intercepts, slopes) = svd.ldsc_correct_zscores_inplace(&mut z_block);
            for tt in 0..t {
                if intercepts[tt] > 1.01 || slopes[tt].abs() > 0.01 {
                    info!(
                        "    LDSC trait {}: intercept={:.3}, slope(h)={:.4}",
                        tt, intercepts[tt], slopes[tt],
                    );
                }
            }
        }

        // Project into eigenspace
        let y_tilde = svd.project_zscores(&z_block);
        let x_tilde = svd.x_design();

        let reg = RegressionConfig {
            estimate_residual_var: false,
            residual_var: 1.0, // fixed for RSS eigenspace
            estimate_effect_var: cfg.estimate_effect_var,
        };

        let mut pip_mat = DMatrix::<f32>::zeros(p, t);
        let mut eff_mean_mat = DMatrix::<f32>::zeros(p, t);
        let mut eff_std_mat = DMatrix::<f32>::zeros(p, t);

        // Fit each trait independently
        for tt in 0..t {
            let y_t = y_tilde.column(tt).into_owned();

            let mcmc_config = McmcConfig {
                n_samples: cfg.n_samples,
                warmup: cfg.warmup,
                thin: cfg.thin,
                seed: seed.wrapping_add(tt as u64),
            };

            let result = match cfg.prior_type {
                McmcPriorType::Susie => {
                    let prior = SoftmaxNormalPrior::new(cfg.logit_var, cfg.prior_var);
                    mcmc_sparse_regression(
                        x_tilde,
                        &y_t,
                        &prior,
                        &mcmc_config,
                        cfg.num_components,
                        &reg,
                    )
                }
                McmcPriorType::SpikeSlab => {
                    let prior = BernoulliNormalPrior::new(cfg.logit_var, cfg.prior_var);
                    mcmc_sparse_regression(
                        x_tilde,
                        &y_t,
                        &prior,
                        &mcmc_config,
                        cfg.num_components,
                        &reg,
                    )
                }
            };

            let std_beta =
                mcmc_util::sparse_regression::compute_posterior_std_beta(&result.samples, p);

            for j in 0..p {
                pip_mat[(j, tt)] = result.pip[j];
                eff_mean_mat[(j, tt)] = result.posterior_mean_beta[j];
                eff_std_mat[(j, tt)] = std_beta[j];
            }
        }

        Ok(BlockFitResult {
            pip: pip_mat,
            effect_mean: eff_mean_mat,
            effect_std: eff_std_mat,
            avg_elbo: f32::NAN,
        })
    }

    fn method_name(&self) -> &str {
        "MCMC"
    }
}
