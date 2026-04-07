use anyhow::Result;
use clap::{Args, ValueEnum};
use log::info;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;

use fagioli::sgvb::{fit_block, BlockFitResult, ComputeDevice, FitConfig, ModelType, PriorType};
use fagioli::summary_stats::common::{
    estimate_adaptive_prior_vars, parse_prior_vars, prepare_sumstat_input, report_top_hits,
    write_sumstat_output, CommonSumstatArgs,
};
use fagioli::summary_stats::polygenic_score::compute_all_polygenic_scores_ridge;
use fagioli::summary_stats::LdBlock;

/// SuSiE inference method for the fine-mapping step.
#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum SusieMethod {
    /// Classical coordinate ascent (IBSS) — fast, no GPU
    Cavi,
    /// Stochastic gradient variational Bayes — GPU-capable
    Sgvb,
}

#[derive(Args, Debug, Clone)]
pub struct PrsSusieArgs {
    #[command(flatten)]
    pub common: CommonSumstatArgs,

    // ── Ridge PRS ────────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "0.1",
        help = "Ridge regularization λ for PRS: inv(D + λI)"
    )]
    pub ridge_lambda: f32,

    // ── SuSiE method ─────────────────────────────────────────────────────
    #[arg(
        long,
        value_enum,
        default_value = "cavi",
        help = "SuSiE inference: 'cavi' (classical IBSS) or 'sgvb' (gradient-based)"
    )]
    pub method: SusieMethod,

    // ── CAVI-specific ────────────────────────────────────────────────────
    #[arg(long, default_value = "100", help = "Max CAVI/IBSS iterations")]
    pub max_cavi_iter: usize,

    #[arg(long, default_value = "1e-3", help = "CAVI ELBO convergence tolerance")]
    pub cavi_tol: f64,

    // ── SGVB-specific ────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "20",
        help = "Monte Carlo samples per SGVB gradient step"
    )]
    pub num_sgvb_samples: usize,

    #[arg(long, default_value = "0.01", help = "AdamW optimizer learning rate")]
    pub learning_rate: f64,

    #[arg(long, default_value = "1000", help = "Max gradient steps per block")]
    pub num_iterations: usize,

    #[arg(long, default_value = "1000", help = "Row minibatch size for SGVB")]
    pub batch_size: usize,

    #[arg(long, default_value = "50", help = "Trailing ELBO averaging window")]
    pub elbo_window: usize,

    #[arg(
        long,
        default_value = "1.0",
        help = "Dirichlet concentration for SuSiE selection prior"
    )]
    pub prior_alpha: f64,

    // ── Device (SGVB only) ───────────────────────────────────────────────
    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Hardware device: cpu, cuda, or metal (SGVB only)"
    )]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "GPU device index")]
    pub device_no: usize,
}

pub fn prs_susie(args: &PrsSusieArgs) -> Result<()> {
    info!("Starting prs-susie");

    // ── Step 1: Load genotypes, z-scores, LD blocks ─────────────────────
    let input = prepare_sumstat_input(&args.common)?;
    let t = input.zscores.ncols();

    // ── Step 2: Compute ridge PRS ─────────────────────────────────────────
    let yhat = compute_all_polygenic_scores_ridge(
        &input.geno.genotypes,
        &input.zscores,
        &input.blocks,
        args.ridge_lambda,
    )?;

    info!(
        "Ridge PRS: {} x {} (ridge_λ={})",
        yhat.nrows(),
        yhat.ncols(),
        args.ridge_lambda,
    );

    // ── Step 3: Prior variance ──────────────────────────────────────────
    let mut prior_vars = parse_prior_vars(&args.common.prior_var)?;
    if prior_vars.is_empty() {
        prior_vars =
            estimate_adaptive_prior_vars(&input, args.common.num_components, args.common.lambda);
    }

    info!(
        "Method: {:?}, L={}, prior_vars={:?}",
        args.method, args.common.num_components, &prior_vars,
    );

    // ── Step 4: Per-block SuSiE fine-mapping on yhat ~ X*beta ───────────
    let num_blocks = input.blocks.len();
    let num_jobs = if args.common.jobs == 0 {
        if args.method == SusieMethod::Sgvb && args.device != ComputeDevice::Cpu {
            1
        } else {
            rayon::current_num_threads()
        }
    } else {
        args.common.jobs
    };

    info!(
        "Fitting individual-level SuSiE for {} blocks ({} jobs)",
        num_blocks, num_jobs,
    );

    let device = args.device.to_device(args.device_no)?;

    let fit_block_fn = |(block_idx, block): (usize, &LdBlock)| -> (usize, BlockFitResult) {
        let block_m = block.num_snps();
        if block_m < 10 {
            return (
                block_idx,
                BlockFitResult {
                    pip: DMatrix::<f32>::zeros(block_m, t),
                    effect_mean: DMatrix::<f32>::zeros(block_m, t),
                    effect_std: DMatrix::<f32>::zeros(block_m, t),
                    avg_elbo: 0.0,
                },
            );
        }

        // Standardize block genotypes
        let mut x_block = input
            .geno
            .genotypes
            .columns(block.snp_start, block_m)
            .clone_owned();
        x_block.scale_columns_inplace();

        let seed = args.common.seed.wrapping_add(block_idx as u64);

        let result = match args.method {
            SusieMethod::Cavi => fit_block_cavi(&x_block, &yhat, args, &prior_vars),
            SusieMethod::Sgvb => {
                let config = build_sgvb_config(args, &prior_vars, seed);
                fit_block(&x_block, &yhat, None, &config, &device)
            }
        };

        let result = result.unwrap_or_else(|e| {
            log::warn!("Block {} failed: {}, using zeros", block_idx, e);
            BlockFitResult {
                pip: DMatrix::<f32>::zeros(block_m, t),
                effect_mean: DMatrix::<f32>::zeros(block_m, t),
                effect_std: DMatrix::<f32>::zeros(block_m, t),
                avg_elbo: f32::NEG_INFINITY,
            }
        });

        info!(
            "Block {}/{}: {} SNPs, avg_elbo={:.2}",
            block_idx + 1,
            num_blocks,
            block_m,
            result.avg_elbo,
        );

        (block_idx, result)
    };

    let mut block_results: Vec<(usize, BlockFitResult)> = if num_jobs <= 1 {
        input.blocks.iter().enumerate().map(fit_block_fn).collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_jobs)
            .build()?;
        pool.install(|| {
            input
                .blocks
                .par_iter()
                .enumerate()
                .map(fit_block_fn)
                .collect()
        })
    };

    block_results.sort_by_key(|(idx, _)| *idx);

    report_top_hits(
        &block_results,
        &input.blocks,
        &input.zscores,
        &input.geno,
        t,
    );

    // ── Step 5: Write output ──────────────────────────────────────────────
    let extra_params = serde_json::json!({
        "command": "prs-susie",
        "method": format!("{:?}", args.method),
        "ridge_lambda": args.ridge_lambda,
        "prior_vars": &prior_vars,
    });

    write_sumstat_output(&input, &block_results, &args.common, extra_params)?;

    info!("prs-susie completed successfully");
    Ok(())
}

/// Fit a single block using CAVI SuSiE on individual-level data.
///
/// Uses median of prior_vars grid (CAVI is deterministic, no grid search needed).
fn fit_block_cavi(
    x_block: &DMatrix<f32>,
    y_block: &DMatrix<f32>,
    args: &PrsSusieArgs,
    prior_vars: &[f32],
) -> Result<BlockFitResult> {
    use candle_util::candle_core::Device;
    use candle_util::sgvb::cavi_susie::{cavi_susie, CaviSusieParams};
    use matrix_util::traits::ConvertMatOps;

    let p = x_block.ncols();
    let t = y_block.ncols();
    let device = Device::Cpu;

    let x_tensor = x_block.to_tensor(&device)?;

    let prior_var = if prior_vars.is_empty() {
        0.2
    } else {
        let mut sorted = prior_vars.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2] as f64
    };

    let params = CaviSusieParams {
        num_components: args.common.num_components,
        max_iter: args.max_cavi_iter,
        tol: args.cavi_tol,
        prior_variance: prior_var,
        estimate_residual_variance: true,
        prior_weights: None,
    };

    let mut pip_mat = DMatrix::<f32>::zeros(p, t);
    let mut eff_mean_mat = DMatrix::<f32>::zeros(p, t);
    let mut eff_std_mat = DMatrix::<f32>::zeros(p, t);
    let mut total_elbo = 0.0f32;

    for tt in 0..t {
        let y_col = y_block.column(tt);
        let y_tensor = candle_util::candle_core::Tensor::from_slice(
            y_col.as_slice(),
            (y_col.len(),),
            &device,
        )?;

        let result = cavi_susie(&x_tensor, &y_tensor, &params)?;
        let beta = result.beta_mean();

        for j in 0..p {
            pip_mat[(j, tt)] = result.pip[j] as f32;
            eff_mean_mat[(j, tt)] = beta[j] as f32;
            let mut var_j = 0.0f64;
            for l in 0..result.alpha.len() {
                var_j += result.alpha[l][j] * (result.s2[l][j] + result.mu[l][j].powi(2));
            }
            var_j -= (beta[j]).powi(2);
            eff_std_mat[(j, tt)] = var_j.max(0.0).sqrt() as f32;
        }

        if let Some(&last_elbo) = result.elbo_trace.last() {
            total_elbo += last_elbo as f32;
        }
    }

    Ok(BlockFitResult {
        pip: pip_mat,
        effect_mean: eff_mean_mat,
        effect_std: eff_std_mat,
        avg_elbo: total_elbo / t as f32,
    })
}

/// Build SGVB FitConfig from CLI args.
fn build_sgvb_config(args: &PrsSusieArgs, prior_vars: &[f32], block_seed: u64) -> FitConfig {
    FitConfig {
        model_type: ModelType::Susie,
        prior_type: PriorType::Single,
        num_components: args.common.num_components,
        num_sgvb_samples: args.num_sgvb_samples,
        learning_rate: args.learning_rate,
        num_iterations: args.num_iterations,
        batch_size: args.batch_size,
        prior_vars: prior_vars.to_vec(),
        elbo_window: args.elbo_window,
        seed: block_seed,
        sigma2_inf: 0.0,
        prior_alpha: args.prior_alpha,
    }
}
