use anyhow::Result;
use clap::Args;
use log::info;
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;

use fagioli::sgvb::refinement::{refine_high_pip_variants, RefinementInput, RefinementParams};
use fagioli::sgvb::{
    fit_block_rss, BlockFitResult, BlockFitResultDetailed, ComputeDevice, FitConfig, ModelType,
    PriorType, RssParams,
};
use fagioli::summary_stats::common::{
    estimate_adaptive_prior_vars, parse_prior_vars, prepare_sumstat_input, write_sumstat_output,
    CommonSumstatArgs,
};
use fagioli::summary_stats::LdBlock;

#[derive(Args, Debug, Clone)]
pub struct FitSumstatSgvbArgs {
    #[command(flatten)]
    pub common: CommonSumstatArgs,

    // ── Model parameters (SGVB-specific) ─────────────────────────────────
    #[arg(
        long,
        default_value = "susie",
        help = "Fine-mapping model: 'susie', 'bisusie', or 'spike-slab'",
        long_help = "Fine-mapping model to use:\n\n\
            - susie: Sum of Single Effects with null absorber.\n\
            - bisusie: Bivariate SuSiE with separate predictor/outcome softmaxes.\n\
            - spike-slab: Independent per-SNP Bernoulli inclusion gates.\n\n\
            Default: susie."
    )]
    pub model: Box<str>,

    #[arg(
        long,
        default_value = "single",
        help = "Prior type: 'single' (grid search) or 'ash' (mixture-of-Gaussians)",
        long_help = "Prior type for effect sizes:\n\n\
            - single: Fixed single-Gaussian prior. Grid search over --prior-var.\n\
            - ash: Mixture-of-Gaussians (adaptive shrinkage) prior.\n\
              --prior-var grid becomes mixture components with learnable weights."
    )]
    pub prior_type: Box<str>,

    // ── SGVB training ────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "20",
        help = "Monte Carlo samples per SGVB gradient step"
    )]
    pub num_sgvb_samples: usize,

    #[arg(long, default_value = "0.01", help = "AdamW optimizer learning rate")]
    pub learning_rate: f64,

    #[arg(
        long,
        default_value = "1000",
        help = "Max gradient steps per LD block per prior variance"
    )]
    pub num_iterations: usize,

    #[arg(
        long,
        help = "Row minibatch size; omit to auto-scale by variant count \
                (full batch when N <= this value)"
    )]
    pub batch_size: Option<usize>,

    #[arg(
        long,
        default_value = "50",
        help = "Trailing window size for averaging ELBO"
    )]
    pub elbo_window: usize,

    #[arg(
        long,
        default_value = "0.0",
        help = "Infinitesimal polygenic prior variance (0 = disabled)"
    )]
    pub sigma2_inf: f32,

    #[arg(
        long,
        default_value = "1.0",
        help = "Dirichlet concentration for SuSiE selection prior (1.0 = uniform)"
    )]
    pub prior_alpha: f64,

    // ── Device ───────────────────────────────────────────────────────────
    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Hardware device: cpu, cuda, or metal"
    )]
    pub device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "GPU device index (for cuda or metal)"
    )]
    pub device_no: usize,

    // ── Refinement ────────────────────────────────────────────────────────
    #[arg(
        long,
        help = "Enable joint refinement of high-PIP variants across blocks"
    )]
    pub refine: bool,

    #[arg(
        long,
        default_value = "3000",
        help = "Max variants to include in joint refinement step"
    )]
    pub max_refine_variants: usize,
}

pub fn fit_sumstat_sgvb(args: &FitSumstatSgvbArgs) -> Result<()> {
    mkdir_parent(&args.common.output)?;
    info!("Starting fit-sumstat-sgvb");

    let device = args.device.to_device(args.device_no)?;
    let use_gpu = args.device != ComputeDevice::Cpu;

    // Override jobs: GPU → 1 job
    let num_jobs = if args.common.jobs == 0 {
        if use_gpu {
            1
        } else {
            rayon::current_num_threads()
        }
    } else {
        args.common.jobs
    };
    info!("Compute device: {:?}, jobs: {}", args.device, num_jobs);

    // ── Step 1-3: Shared pipeline ────────────────────────────────────────
    let input = prepare_sumstat_input(&args.common)?;
    let t = input.zscores.ncols();

    // ── Step 4: Build SGVB fit config ────────────────────────────────────
    let model_type: ModelType = args.model.parse()?;
    let prior_type: PriorType = args.prior_type.parse()?;
    let mut prior_vars = parse_prior_vars(&args.common.prior_var)?;

    if prior_vars.is_empty() {
        prior_vars =
            estimate_adaptive_prior_vars(&input, args.common.num_components, args.common.lambda);
    }

    let fit_config = FitConfig {
        model_type,
        prior_type,
        num_components: args.common.num_components,
        num_sgvb_samples: args.num_sgvb_samples,
        learning_rate: args.learning_rate,
        num_iterations: args.num_iterations,
        batch_size: args
            .batch_size
            .unwrap_or_else(|| matrix_util::utils::default_block_size(input.zscores.nrows())),
        prior_vars,
        elbo_window: args.elbo_window,
        seed: args.common.seed,
        sigma2_inf: args.sigma2_inf,
        prior_alpha: args.prior_alpha,
    };

    info!(
        "Model: {:?}, prior: {:?}, L={}, prior_vars={:?}, σ²_inf={:.2e}",
        model_type, prior_type, args.common.num_components, &fit_config.prior_vars, args.sigma2_inf,
    );

    // ── Step 5: Per-block RSS SGVB fine-mapping ──────────────────────────
    let num_blocks = input.blocks.len();
    info!(
        "Fitting RSS SGVB models for {} blocks ({} jobs)",
        num_blocks, num_jobs
    );

    let fit_block_fn = |(block_idx, block): (usize, &LdBlock)| -> (usize, BlockFitResultDetailed) {
        let block_m = block.num_snps();
        let num_priors = fit_config.num_prior_results();
        if block_m < 10 {
            let empty = BlockFitResultDetailed {
                per_prior_elbos: vec![0.0; num_priors],
                per_prior_pips: vec![DMatrix::<f32>::zeros(block_m, t); num_priors],
                per_prior_effects: vec![DMatrix::<f32>::zeros(block_m, t); num_priors],
                per_prior_stds: vec![DMatrix::<f32>::zeros(block_m, t); num_priors],
            };
            return (block_idx, empty);
        }

        let mut x_block = input
            .geno
            .genotypes
            .columns(block.snp_start, block_m)
            .clone_owned();
        x_block.scale_columns_inplace();

        let z_block = input.zscores.rows(block.snp_start, block_m).clone_owned();

        let mut block_config = fit_config.clone();
        block_config.seed = fit_config.seed.wrapping_add(block_idx as u64);

        let block_lambda = args.common.lambda.unwrap_or(0.1 / input.max_rank as f64);

        let rss_params = RssParams {
            max_rank: input.max_rank,
            lambda: block_lambda,
            ldsc_intercept: !args.common.no_ldsc_intercept,
        };

        let result = fit_block_rss(&x_block, &z_block, &block_config, &rss_params, &device)
            .unwrap_or_else(|e| {
                log::warn!("Block {} failed: {}, using zeros", block_idx, e);
                BlockFitResultDetailed {
                    per_prior_elbos: vec![f32::NEG_INFINITY; num_priors],
                    per_prior_pips: vec![DMatrix::<f32>::zeros(block_m, t); num_priors],
                    per_prior_effects: vec![DMatrix::<f32>::zeros(block_m, t); num_priors],
                    per_prior_stds: vec![DMatrix::<f32>::zeros(block_m, t); num_priors],
                }
            });

        info!(
            "Block {}/{}: {} SNPs, avg_elbo={:.2}",
            block_idx + 1,
            num_blocks,
            block_m,
            result.best_result().avg_elbo,
        );

        (block_idx, result)
    };

    let block_results: Vec<(usize, BlockFitResultDetailed)> = if num_jobs <= 1 {
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

    let mut sorted_results = block_results;
    sorted_results.sort_by_key(|(idx, _)| *idx);

    // ── Step 6: Prior aggregation ──────────────────────────────────────
    let globally_averaged: Vec<(usize, BlockFitResult)> = {
        let num_priors = fit_config.num_prior_results();
        let mut global_elbos = vec![0.0f32; num_priors];
        for (_idx, detailed) in &sorted_results {
            for (j, &e) in detailed.per_prior_elbos.iter().enumerate() {
                global_elbos[j] += e;
            }
        }
        if fit_config.prior_type == PriorType::Ash {
            info!("Ash prior ELBO (total): {:.1}", global_elbos[0]);
        } else {
            info!(
                "Prior grid ELBOs (total): {:?}",
                fit_config
                    .prior_vars
                    .iter()
                    .zip(global_elbos.iter())
                    .map(|(v, e)| format!("{:.3}:{:.1}", v, e))
                    .collect::<Vec<_>>()
            );
        }

        sorted_results
            .iter()
            .map(|(idx, d)| (*idx, d.best_result()))
            .collect()
    };

    fagioli::summary_stats::common::report_top_hits(
        &globally_averaged,
        &input.blocks,
        &input.zscores,
        &input.geno,
        t,
    );

    // ── Step 6b: Joint refinement of high-PIP variants ──────────────────
    let globally_averaged = if args.refine {
        refine_high_pip_variants(
            globally_averaged,
            &RefinementInput {
                blocks: &input.blocks,
                genotypes: &input.geno.genotypes,
                zscores: &input.zscores,
                snp_ids: &input.geno.snp_ids,
                num_traits: t,
            },
            &fit_config,
            &RefinementParams {
                max_variants: args.max_refine_variants,
                user_lambda: args.common.lambda,
                ldsc_intercept: !args.common.no_ldsc_intercept,
            },
            &device,
        )?
    } else {
        globally_averaged
    };

    // ── Step 7: Write output ──────────────────────────────────────────────
    let extra_params = serde_json::json!({
        "command": "fit-sumstat-sgvb",
        "method": "sgvb",
        "model": args.model,
        "prior_type": format!("{:?}", prior_type),
        "prior_vars": &fit_config.prior_vars,
        "num_sgvb_samples": args.num_sgvb_samples,
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_iterations,
        "batch_size": fit_config.batch_size,
        "elbo_window": args.elbo_window,
        "sigma2_inf": args.sigma2_inf,
        "refine": args.refine,
        "max_refine_variants": args.max_refine_variants,
    });

    write_sumstat_output(&input, &globally_averaged, &args.common, extra_params)?;

    info!("fit-sumstat-sgvb completed successfully");
    Ok(())
}
