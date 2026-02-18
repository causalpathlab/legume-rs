use anyhow::Result;
use clap::Args;
use log::info;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;
use rust_htslib::bgzf;
use rust_htslib::tpool::ThreadPool;
use std::io::Write;

use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};
use fagioli::sgvb::{fit_block_rss, BlockFitResult, FitConfig, ModelType};
use fagioli::summary_stats::{
    estimate_ld_blocks, load_ld_blocks_from_file, read_sumstat_zscores_with_n, LdBlock,
};

#[derive(Args, Debug, Clone)]
pub struct MapSumstatArgs {
    /// Summary statistics file (.sumstats.bed.gz)
    #[arg(long)]
    pub sumstat_file: String,

    /// PLINK BED file prefix (without .bed)
    #[arg(long)]
    pub bed_prefix: String,

    /// Chromosome
    #[arg(long)]
    pub chromosome: String,

    /// Left position bound (optional)
    #[arg(long)]
    pub left_bound: Option<u64>,

    /// Right position bound (optional)
    #[arg(long)]
    pub right_bound: Option<u64>,

    /// Max individuals to use from reference panel
    #[arg(long)]
    pub max_individuals: Option<usize>,

    // --- LD block parameters ---
    /// External LD block file (BED format: chr, start, end)
    #[arg(long)]
    pub ld_block_file: Option<String>,

    /// Number of landmark SNPs for Nystrom LD block estimation
    #[arg(long, default_value = "500")]
    pub num_landmarks: usize,

    /// Number of rSVD components for block estimation
    #[arg(long, default_value = "20")]
    pub num_ld_components: usize,

    /// Minimum block size in SNPs
    #[arg(long, default_value = "50")]
    pub min_block_snps: usize,

    /// Maximum block size in SNPs
    #[arg(long, default_value = "5000")]
    pub max_block_snps: usize,

    // --- RSS SVD parameters ---
    /// Maximum rank for rSVD of each LD block's genotype matrix
    #[arg(long, default_value = "50")]
    pub max_rank: usize,

    /// Regularization λ for D̃ = √(D² + λ); default 0.1/K
    #[arg(long)]
    pub lambda: Option<f64>,

    // --- Model parameters ---
    /// Fine-mapping model: susie, bisusie, multilevel-susie
    #[arg(long, default_value = "susie")]
    pub model: String,

    /// Number of SuSiE/BiSuSiE components (L)
    #[arg(long, default_value = "10")]
    pub num_components: usize,

    /// Comma-separated prior variances for coordinate search
    #[arg(long, default_value = "0.01,0.05,0.1,0.2,0.5,1.0")]
    pub prior_var: String,

    /// Number of SGVB Monte Carlo samples
    #[arg(long, default_value = "20")]
    pub num_sgvb_samples: usize,

    /// AdamW learning rate
    #[arg(long, default_value = "0.01")]
    pub learning_rate: f64,

    /// Number of training iterations per block
    #[arg(long, default_value = "500")]
    pub num_iterations: usize,

    /// Minibatch size (use minibatch when N > batch_size)
    #[arg(long, default_value = "1000")]
    pub batch_size: usize,

    /// Number of ELBO values to average for model selection
    #[arg(long, default_value = "50")]
    pub elbo_window: usize,

    /// Block size for MultiLevelSusieVar tree
    #[arg(long, default_value = "50")]
    pub ml_block_size: usize,

    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Verbose: report top significant SNPs per block (PIP >= 0.7)
    #[arg(long, short)]
    pub verbose: bool,

    /// Output prefix
    #[arg(short, long)]
    pub output: String,
}

pub fn map_sumstat(args: &MapSumstatArgs) -> Result<()> {
    info!("Starting map-sumstat");

    let num_threads = rayon::current_num_threads().max(1) as u32;
    let tpool = ThreadPool::new(num_threads)?;
    info!("Using {} threads", num_threads);

    // ── Step 1: Read reference panel genotypes ────────────────────────────
    let region = GenomicRegion::new(
        Some(args.chromosome.clone()),
        args.left_bound,
        args.right_bound,
    );

    let mut reader = BedReader::new(&args.bed_prefix)?;
    let geno = reader.read(args.max_individuals, Some(region))?;
    let n = geno.num_individuals();
    let m = geno.num_snps();

    info!("Loaded reference panel: {} individuals x {} SNPs", n, m);

    // ── Step 2: Read summary statistics ───────────────────────────────────
    let (zscores, median_n) = read_sumstat_zscores_with_n(&args.sumstat_file, &geno.snp_ids)?;
    let t = zscores.ncols();
    info!(
        "Z-scores: {} SNPs x {} traits, median N={}",
        zscores.nrows(),
        t,
        median_n,
    );

    // ── Step 3: Determine LD blocks ───────────────────────────────────────
    let blocks: Vec<LdBlock> = if let Some(ref block_file) = args.ld_block_file {
        info!("Loading LD blocks from {}", block_file);
        load_ld_blocks_from_file(block_file, &geno.positions, &geno.chromosomes)?
    } else if m > args.min_block_snps * 2 {
        info!("Estimating LD blocks via Nystrom + rSVD");
        estimate_ld_blocks(
            &geno.genotypes,
            &geno.positions,
            &geno.chromosomes,
            args.num_landmarks,
            args.num_ld_components,
            args.min_block_snps,
            args.max_block_snps,
            args.seed,
        )?
    } else {
        info!("Too few SNPs for block estimation, using single block");
        fagioli::summary_stats::create_uniform_blocks(m, m, &geno.positions, &geno.chromosomes)
    };

    let num_blocks = blocks.len();
    info!("Using {} LD blocks", num_blocks);

    // ── Step 4: Build fit config ────────────────────────────────────────
    let model_type: ModelType = args.model.parse()?;
    let prior_vars: Vec<f32> = args
        .prior_var
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let fit_config = FitConfig {
        model_type,
        num_components: args.num_components,
        num_sgvb_samples: args.num_sgvb_samples,
        learning_rate: args.learning_rate,
        num_iterations: args.num_iterations,
        batch_size: args.batch_size,
        prior_vars,
        elbo_window: args.elbo_window,
        seed: args.seed,
        ml_block_size: args.ml_block_size,
    };

    info!(
        "Model: {:?}, L={}, prior_vars={:?}",
        model_type, args.num_components, &fit_config.prior_vars
    );

    // ── Step 5: Per-block RSS fine-mapping (parallel) ────────────────────
    info!("Fitting RSS SGVB models for {} blocks", num_blocks);

    let block_results: Vec<(usize, BlockFitResult)> = blocks
        .par_iter()
        .enumerate()
        .map(|(block_idx, block)| {
            let block_m = block.num_snps();
            if block_m < 10 {
                let empty = BlockFitResult {
                    pip: DMatrix::<f32>::zeros(block_m, t),
                    effect_mean: DMatrix::<f32>::zeros(block_m, t),
                    effect_std: DMatrix::<f32>::zeros(block_m, t),
                    avg_elbo: 0.0,
                };
                return (block_idx, empty);
            }

            // Standardize block genotypes
            let mut x_block = geno
                .genotypes
                .columns(block.snp_start, block_m)
                .clone_owned();
            x_block.scale_columns_inplace();

            // Extract z-scores for this block
            let z_block = zscores.rows(block.snp_start, block_m).clone_owned();

            // Per-block seed for reproducibility
            let mut block_config = fit_config.clone();
            block_config.seed = fit_config.seed.wrapping_add(block_idx as u64);

            // λ: user-specified or default 0.1/K
            let block_lambda = args.lambda.unwrap_or(0.1 / args.max_rank as f64);

            let result =
                fit_block_rss(&x_block, &z_block, &block_config, args.max_rank, block_lambda)
                    .unwrap_or_else(|e| {
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

            // Report top significant SNPs (PIP >= 0.7) in verbose mode
            if args.verbose {
                let mut hits: Vec<(f32, usize, usize)> = Vec::new();
                for snp_j in 0..block_m {
                    for trait_k in 0..t {
                        let pip = result.pip[(snp_j, trait_k)];
                        if pip >= 0.7 {
                            hits.push((pip, snp_j, trait_k));
                        }
                    }
                }
                if !hits.is_empty() {
                    hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    let n_show = hits.len().min(5);
                    for &(pip, snp_j, trait_k) in &hits[..n_show] {
                        let global_snp = block.snp_start + snp_j;
                        let snp_id = &geno.snp_ids[global_snp];
                        let z = zscores[(global_snp, trait_k)];
                        let eff = result.effect_mean[(snp_j, trait_k)];
                        info!(
                            "  ** {}: trait={}, pip={:.4}, z={:.2}, effect={:.4}",
                            snp_id, trait_k, pip, z, eff,
                        );
                    }
                    if hits.len() > 5 {
                        info!("  ... and {} more with pip >= 0.7", hits.len() - 5);
                    }
                }
            }

            (block_idx, result)
        })
        .collect();

    // ── Step 6: Write results ─────────────────────────────────────────────
    let out_file = format!("{}.results.bed.gz", args.output);
    info!("Writing results to {}", out_file);

    let mut writer = bgzf::Writer::from_path(&out_file)?;
    writer.set_thread_pool(&tpool)?;

    // Header
    writeln!(
        writer,
        "#chr\tstart\tend\tsnp_id\ttrait_idx\tpip\teffect_mean\teffect_std\tz_marginal"
    )?;

    // Sort by block index for deterministic output
    let mut sorted_results = block_results;
    sorted_results.sort_by_key(|(idx, _)| *idx);

    for (block_idx, result) in &sorted_results {
        let block = &blocks[*block_idx];
        let block_m = block.num_snps();

        for snp_j in 0..block_m {
            let global_snp = block.snp_start + snp_j;
            let chr = &geno.chromosomes[global_snp];
            let pos = geno.positions[global_snp];
            let snp_id = &geno.snp_ids[global_snp];

            for trait_k in 0..t {
                let pip = result.pip[(snp_j, trait_k)];
                let eff_mean = result.effect_mean[(snp_j, trait_k)];
                let eff_std = result.effect_std[(snp_j, trait_k)];
                let z_marginal = zscores[(global_snp, trait_k)];

                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
                    chr,
                    pos,
                    pos + 1,
                    snp_id,
                    trait_k,
                    pip,
                    eff_mean,
                    eff_std,
                    z_marginal,
                )?;
            }
        }
    }

    writer.flush()?;
    info!("Results written: {}", out_file);

    // ── Step 7: Write parameters ──────────────────────────────────────────
    let param_file = format!("{}.parameters.json", args.output);
    let params = serde_json::json!({
        "command": "map-sumstat",
        "sumstat_file": args.sumstat_file,
        "bed_prefix": args.bed_prefix,
        "chromosome": args.chromosome,
        "num_individuals": n,
        "num_snps": m,
        "num_traits": t,
        "median_n": median_n,
        "num_blocks": num_blocks,
        "model": args.model,
        "num_components": args.num_components,
        "prior_vars": &fit_config.prior_vars,
        "num_sgvb_samples": args.num_sgvb_samples,
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "elbo_window": args.elbo_window,
        "max_rank": args.max_rank,
        "lambda": args.lambda.unwrap_or(0.1 / args.max_rank as f64),
        "seed": args.seed,
    });
    std::fs::write(&param_file, serde_json::to_string_pretty(&params)?)?;
    info!("Wrote parameters: {}", param_file);

    info!("map-sumstat completed successfully");
    Ok(())
}
