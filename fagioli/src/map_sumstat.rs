use rustc_hash::FxHashMap;

use anyhow::Result;
use clap::Args;
use log::info;
use matrix_util::dmatrix_util::{subset_columns, subset_rows};
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;

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
pub struct MapSumstatArgs {
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
        default_value = "1000",
        help = "Row minibatch size (full batch when N <= this value)"
    )]
    pub batch_size: usize,

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

pub fn map_sumstat(args: &MapSumstatArgs) -> Result<()> {
    info!("Starting map-sumstat");

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
        batch_size: args.batch_size,
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
            &input.blocks,
            &input.geno.genotypes,
            &input.zscores,
            &input.geno.snp_ids,
            t,
            args.max_refine_variants,
            &fit_config,
            args.common.lambda,
            !args.common.no_ldsc_intercept,
            &device,
        )?
    } else {
        globally_averaged
    };

    // ── Step 7: Write output ──────────────────────────────────────────────
    let extra_params = serde_json::json!({
        "command": "map-sumstat",
        "method": "sgvb",
        "model": args.model,
        "prior_type": format!("{:?}", prior_type),
        "prior_vars": &fit_config.prior_vars,
        "num_sgvb_samples": args.num_sgvb_samples,
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "elbo_window": args.elbo_window,
        "sigma2_inf": args.sigma2_inf,
        "refine": args.refine,
        "max_refine_variants": args.max_refine_variants,
    });

    write_sumstat_output(&input, &globally_averaged, &args.common, extra_params)?;

    info!("map-sumstat completed successfully");
    Ok(())
}

// ── SGVB-specific helpers ───────────────────────────────────────────────────

/// Find the elbow/knee point in sorted (descending) candidates.
fn find_pip_elbow(candidates: &[(usize, f32)]) -> usize {
    let n = candidates.len();
    if n < 3 {
        return n;
    }
    debug_assert!(
        candidates.windows(2).all(|w| w[0].1 >= w[1].1),
        "find_pip_elbow requires descending-sorted input"
    );

    let mid_pip = candidates[n / 2].1;
    let null_thresh = (2.0 * mid_pip).max(0.01);
    let above_null = candidates.partition_point(|&(_, p)| p >= null_thresh);
    if above_null < 3 {
        return above_null;
    }

    let y1 = candidates[0].1 as f64;
    let x2 = (above_null - 1) as f64;
    let y2 = candidates[above_null - 1].1 as f64;

    let dy = y2 - y1;
    let line_len = (x2 * x2 + dy * dy).sqrt();
    if line_len < 1e-12 {
        return above_null;
    }

    let mut max_dist = 0.0f64;
    let mut elbow_idx = 0;
    for (i, &(_, pip)) in candidates.iter().enumerate().take(above_null - 1).skip(1) {
        let px = i as f64;
        let py = pip as f64 - y1;
        let dist = (px * dy - py * x2).abs() / line_len;
        if dist > max_dist {
            max_dist = dist;
            elbow_idx = i;
        }
    }

    elbow_idx + 1
}

/// Joint refinement: refit a single model on high-PIP variants selected by elbow.
#[allow(clippy::too_many_arguments)]
fn refine_high_pip_variants(
    mut globally_averaged: Vec<(usize, BlockFitResult)>,
    blocks: &[LdBlock],
    genotypes: &DMatrix<f32>,
    zscores: &DMatrix<f32>,
    snp_ids: &[Box<str>],
    t: usize,
    max_variants: usize,
    fit_config: &FitConfig,
    user_lambda: Option<f64>,
    ldsc_intercept: bool,
    device: &candle_util::candle_core::Device,
) -> Result<Vec<(usize, BlockFitResult)>> {
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for (block_idx, result) in &globally_averaged {
        let block = &blocks[*block_idx];
        for snp_j in 0..block.num_snps() {
            let max_pip = (0..t).map(|k| result.pip[(snp_j, k)]).fold(0f32, f32::max);
            candidates.push((block.snp_start + snp_j, max_pip));
        }
    }

    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

    let elbow_n = find_pip_elbow(&candidates);
    let elbow_pip = candidates
        .get(elbow_n.saturating_sub(1))
        .map_or(0.0, |c| c.1);
    candidates.truncate(elbow_n);

    if candidates.len() < 2 {
        info!(
            "Refinement: elbow at {} variants (PIP >= {:.4}), too few — skipping",
            elbow_n, elbow_pip,
        );
        return Ok(globally_averaged);
    }

    if candidates.len() > max_variants {
        log::warn!(
            "Refinement: elbow selected {} variants, capping at {}",
            candidates.len(),
            max_variants,
        );
        candidates.truncate(max_variants);
    }

    let p_sel = candidates.len();
    info!(
        "Refinement: elbow at {} variants (PIP >= {:.4}), fitting joint model",
        p_sel, elbow_pip,
    );

    let sel_indices = candidates.iter().map(|&(j, _)| j);
    let mut x_joint = subset_columns(genotypes, sel_indices)?;
    let sel_indices = candidates.iter().map(|&(j, _)| j);
    let z_joint = subset_rows(zscores, sel_indices)?;
    x_joint.scale_columns_inplace();

    let n = genotypes.nrows();
    let joint_max_rank = n.min(p_sel);
    let joint_lambda = user_lambda.unwrap_or(0.1 / joint_max_rank as f64);

    let mut joint_config = fit_config.clone();
    joint_config.num_components = joint_config.num_components.min(p_sel / 2).max(1);
    joint_config.seed = fit_config.seed.wrapping_add(999);

    let rss_params = RssParams {
        max_rank: joint_max_rank,
        lambda: joint_lambda,
        ldsc_intercept,
    };

    let joint_result = fit_block_rss(&x_joint, &z_joint, &joint_config, &rss_params, device)?;
    let refined = joint_result.best_result();

    info!(
        "Refinement: joint ELBO={:.2}, L={}",
        refined.avg_elbo, joint_config.num_components,
    );

    let mut hits: Vec<(f32, usize, usize)> = Vec::new();
    for (j_new, &(j_global, _)) in candidates.iter().enumerate() {
        for trait_k in 0..t {
            let pip = refined.pip[(j_new, trait_k)];
            if pip >= 0.5 {
                hits.push((pip, j_global, trait_k));
            }
        }
    }
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));
    for &(pip, global_snp, trait_k) in hits.iter().take(10) {
        let z = zscores[(global_snp, trait_k)];
        info!(
            "  ** refined {}: trait={}, pip={:.4}, z={:.2}",
            snp_ids[global_snp], trait_k, pip, z,
        );
    }

    let sel_lookup: FxHashMap<usize, usize> = candidates
        .iter()
        .enumerate()
        .map(|(j_new, &(j_global, _))| (j_global, j_new))
        .collect();

    for (block_idx, result) in &mut globally_averaged {
        let block = &blocks[*block_idx];
        for snp_j in 0..block.num_snps() {
            let global_snp = block.snp_start + snp_j;
            if let Some(&j_new) = sel_lookup.get(&global_snp) {
                for trait_k in 0..t {
                    result.pip[(snp_j, trait_k)] = refined.pip[(j_new, trait_k)];
                    result.effect_mean[(snp_j, trait_k)] = refined.effect_mean[(j_new, trait_k)];
                    result.effect_std[(snp_j, trait_k)] = refined.effect_std[(j_new, trait_k)];
                }
            }
        }
    }

    Ok(globally_averaged)
}
