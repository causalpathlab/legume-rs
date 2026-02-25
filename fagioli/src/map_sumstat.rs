use std::collections::HashSet;

use anyhow::{ensure, Result};
use clap::Args;
use log::info;
use matrix_util::common_io::read_lines;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;
use rust_htslib::tpool::ThreadPool;

use fagioli::genotype::{BedReader, GenomicRegion, GenotypeMatrix, GenotypeReader};
use fagioli::io::results::{write_parameters, write_variant_results, VariantRow};
use fagioli::sgvb::{fit_block_rss, BlockFitResult, ComputeDevice, FitConfig, ModelType};
use fagioli::summary_stats::{
    estimate_ld_blocks, load_ld_blocks_from_file, read_sumstat_zscores_with_n, LdBlock,
    LdBlockParams,
};

#[derive(Args, Debug, Clone)]
pub struct MapSumstatArgs {
    // ── Input ────────────────────────────────────────────────────────────
    #[arg(long, help = "Summary statistics file (.sumstats.bed.gz)")]
    pub sumstat_file: String,

    #[arg(long, help = "PLINK BED file prefix (without .bed/.bim/.fam)")]
    pub bed_prefix: String,

    #[arg(long, help = "Chromosome to analyze")]
    pub chromosome: String,

    #[arg(long, help = "Left genomic position bound (bp)")]
    pub left_bound: Option<u64>,

    #[arg(long, help = "Right genomic position bound (bp)")]
    pub right_bound: Option<u64>,

    // ── Individual filtering ─────────────────────────────────────────────
    #[arg(long, help = "Max individuals to use from reference panel")]
    pub max_individuals: Option<usize>,

    #[arg(
        long,
        conflicts_with = "remove",
        help = "Keep only these individuals (like plink --keep)",
        long_help = "Keep only these individuals (like plink --keep).\n\n\
            Accepts a file path or a comma-separated list of IIDs.\n\
            File format: one individual per line, either \"FID IID\" (two columns)\n\
            or just \"IID\" (one column). Lines starting with # are skipped.\n\
            Gzipped files (.gz) are supported.\n\n\
            Examples:\n  \
              --keep samples.txt\n  \
              --keep ind1,ind2,ind3"
    )]
    pub keep: Option<String>,

    #[arg(
        long,
        conflicts_with = "keep",
        help = "Remove these individuals (like plink --remove)",
        long_help = "Remove these individuals (like plink --remove).\n\n\
            Accepts a file path or a comma-separated list of IIDs.\n\
            File format: one individual per line, either \"FID IID\" (two columns)\n\
            or just \"IID\" (one column). Lines starting with # are skipped.\n\
            Gzipped files (.gz) are supported.\n\n\
            Examples:\n  \
              --remove samples.txt\n  \
              --remove ind1,ind2,ind3"
    )]
    pub remove: Option<String>,

    // ── LD block parameters ──────────────────────────────────────────────
    #[arg(
        long,
        help = "External LD block file (BED: chr, start, end)",
        long_help = "External LD block file in BED format (chr, start, end).\n\
            If omitted, blocks are estimated from the genotype data via Nystrom + rSVD."
    )]
    pub ld_block_file: Option<String>,

    #[arg(
        long,
        default_value = "500",
        help = "Landmark SNPs for Nystrom LD block estimation"
    )]
    pub num_landmarks: usize,

    #[arg(
        long,
        default_value = "20",
        help = "rSVD components for LD block estimation"
    )]
    pub num_ld_components: usize,

    #[arg(long, default_value = "50", help = "Minimum LD block size in SNPs")]
    pub min_block_snps: usize,

    #[arg(long, default_value = "5000", help = "Maximum LD block size in SNPs")]
    pub max_block_snps: usize,

    // ── RSS SVD parameters ───────────────────────────────────────────────
    #[arg(long, default_value = "50", help = "Max rank for rSVD per LD block")]
    pub max_rank: usize,

    #[arg(
        long,
        help = "Regularization lambda; default: 0.1 / max_rank",
        long_help = "Regularization lambda for D_tilde = sqrt(D^2 + lambda).\n\
            Default: 0.1 / max_rank."
    )]
    pub lambda: Option<f64>,

    // ── Model parameters ─────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "susie",
        help = "Fine-mapping model: susie, bisusie, multilevel-susie"
    )]
    pub model: String,

    #[arg(
        long,
        default_value = "10",
        help = "Number of SuSiE/BiSuSiE components (L)"
    )]
    pub num_components: usize,

    #[arg(
        long,
        default_value = "0.01,0.05,0.1,0.2,0.5,1.0",
        help = "Comma-separated prior variances for coordinate search"
    )]
    pub prior_var: String,

    // ── SGVB training ────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "20",
        help = "SGVB Monte Carlo samples per iteration"
    )]
    pub num_sgvb_samples: usize,

    #[arg(long, default_value = "0.01", help = "AdamW learning rate")]
    pub learning_rate: f64,

    #[arg(long, default_value = "500", help = "Training iterations per block")]
    pub num_iterations: usize,

    #[arg(
        long,
        default_value = "1000",
        help = "Minibatch size (full batch if N <= batch_size)"
    )]
    pub batch_size: usize,

    #[arg(
        long,
        default_value = "50",
        help = "ELBO values to average for convergence"
    )]
    pub elbo_window: usize,

    #[arg(
        long,
        default_value = "50",
        help = "Block size for MultiLevelSusieVar tree"
    )]
    pub ml_block_size: usize,

    #[arg(
        long,
        default_value = "0.0",
        help = "Infinitesimal prior variance for polygenic background (0 = off, scaled by 1/p internally)"
    )]
    pub sigma2_inf: f32,

    // ── Device ───────────────────────────────────────────────────────────
    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Compute device: cpu, cuda, metal"
    )]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device number for cuda or metal")]
    pub device_no: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Parallel jobs (0 = auto: all CPUs for cpu, 1 for gpu)"
    )]
    pub jobs: usize,

    // ── Misc ─────────────────────────────────────────────────────────────
    #[arg(long, default_value = "42", help = "Random seed")]
    pub seed: u64,

    #[arg(short, long, help = "Output prefix for results and parameters")]
    pub output: String,
}

pub fn map_sumstat(args: &MapSumstatArgs) -> Result<()> {
    info!("Starting map-sumstat");

    let device = args.device.to_device(args.device_no)?;
    let use_gpu = args.device != ComputeDevice::Cpu;
    let num_jobs = if args.jobs == 0 {
        if use_gpu {
            1
        } else {
            rayon::current_num_threads()
        }
    } else {
        args.jobs
    };
    info!("Compute device: {:?}, jobs: {}", args.device, num_jobs);

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
    let mut geno = reader.read(args.max_individuals, Some(region))?;

    // Apply --keep / --remove individual filtering
    if let Some(ref keep_val) = args.keep {
        let keep_ids = parse_individual_ids(keep_val)?;
        let keep_indices: Vec<usize> = geno
            .individual_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| keep_ids.contains(id.as_ref()))
            .map(|(i, _)| i)
            .collect();
        let n_before = geno.num_individuals();
        filter_individuals(&mut geno, &keep_indices);
        info!(
            "Kept {}/{} individuals (--keep)",
            geno.num_individuals(),
            n_before
        );
    } else if let Some(ref remove_val) = args.remove {
        let remove_ids = parse_individual_ids(remove_val)?;
        let keep_indices: Vec<usize> = geno
            .individual_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| !remove_ids.contains(id.as_ref()))
            .map(|(i, _)| i)
            .collect();
        let n_before = geno.num_individuals();
        filter_individuals(&mut geno, &keep_indices);
        info!(
            "Removed {}/{} individuals (--remove)",
            n_before - geno.num_individuals(),
            n_before,
        );
    }

    let n = geno.num_individuals();
    let m = geno.num_snps();

    info!("Reference panel: {} individuals x {} SNPs", n, m);

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
            &LdBlockParams {
                num_landmarks: args.num_landmarks,
                num_components: args.num_ld_components,
                min_block_snps: args.min_block_snps,
                max_block_snps: args.max_block_snps,
                seed: args.seed,
            },
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
        sigma2_inf: args.sigma2_inf,
    };

    info!(
        "Model: {:?}, L={}, prior_vars={:?}, σ²_inf={:.2e}",
        model_type, args.num_components, &fit_config.prior_vars, args.sigma2_inf,
    );

    // ── Step 5: Per-block RSS fine-mapping ──────────────────────────────
    info!(
        "Fitting RSS SGVB models for {} blocks ({} jobs)",
        num_blocks, num_jobs
    );

    let fit_block_fn = |(block_idx, block): (usize, &LdBlock)| -> (usize, BlockFitResult) {
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

        let result = fit_block_rss(
            &x_block,
            &z_block,
            &block_config,
            args.max_rank,
            block_lambda,
            &device,
        )
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

        // Report top significant SNPs (PIP >= 0.7)
        {
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
    };

    let block_results: Vec<(usize, BlockFitResult)> = if num_jobs <= 1 {
        blocks.iter().enumerate().map(fit_block_fn).collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_jobs)
            .build()?;
        pool.install(|| blocks.par_iter().enumerate().map(fit_block_fn).collect())
    };

    // ── Step 6: Write results ─────────────────────────────────────────────
    // Sort by block index for deterministic output
    let mut sorted_results = block_results;
    sorted_results.sort_by_key(|(idx, _)| *idx);

    let variant_rows = build_sumstat_variant_rows(&sorted_results, &blocks, &zscores, t);
    write_variant_results(
        &format!("{}.results.bed.gz", args.output),
        &["trait_idx"],
        &variant_rows,
        &geno,
        &tpool,
    )?;

    // ── Step 7: Write parameters ──────────────────────────────────────────
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
        "sigma2_inf": args.sigma2_inf,
    });
    write_parameters(&format!("{}.parameters.json", args.output), &params)?;

    info!("map-sumstat completed successfully");
    Ok(())
}

/// Parse individual IDs from either a file path or a comma-separated list.
///
/// If the value contains a comma or doesn't point to an existing file,
/// it is treated as a comma-separated list of IIDs.
/// Otherwise, the file is read line-by-line (FID IID or just IID per line).
fn parse_individual_ids(value: &str) -> Result<HashSet<String>> {
    if value.contains(',') || !std::path::Path::new(value).is_file() {
        let ids: HashSet<String> = value
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        ensure!(!ids.is_empty(), "Empty individual ID list");
        info!("Parsed {} individual IDs from command line", ids.len());
        Ok(ids)
    } else {
        let lines = read_lines(value)?;
        let mut ids = HashSet::new();
        for line in &lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            // If two+ columns: FID IID → take IID (column 1)
            // If one column: take it as IID
            let iid = if fields.len() >= 2 {
                fields[1]
            } else {
                fields[0]
            };
            ids.insert(iid.to_string());
        }
        ensure!(!ids.is_empty(), "No individual IDs found in {}", value);
        info!("Read {} individual IDs from {}", ids.len(), value);
        Ok(ids)
    }
}

/// Subset a GenotypeMatrix to keep only the specified individual (row) indices.
fn filter_individuals(geno: &mut GenotypeMatrix, indices: &[usize]) {
    let m = geno.genotypes.ncols();
    geno.genotypes = DMatrix::from_fn(indices.len(), m, |i, j| geno.genotypes[(indices[i], j)]);
    geno.individual_ids = indices
        .iter()
        .map(|&i| geno.individual_ids[i].clone())
        .collect();
}

/// Build variant result rows from block-level RSS fine-mapping output.
fn build_sumstat_variant_rows(
    sorted_results: &[(usize, BlockFitResult)],
    blocks: &[LdBlock],
    zscores: &DMatrix<f32>,
    t: usize,
) -> Vec<VariantRow> {
    let mut rows = Vec::new();

    for (block_idx, result) in sorted_results {
        let block = &blocks[*block_idx];
        let block_m = block.num_snps();

        for snp_j in 0..block_m {
            let global_snp = block.snp_start + snp_j;

            for trait_k in 0..t {
                rows.push(VariantRow {
                    snp_idx: global_snp,
                    labels: vec![Box::from(trait_k.to_string())],
                    pip: result.pip[(snp_j, trait_k)],
                    effect_mean: result.effect_mean[(snp_j, trait_k)],
                    effect_std: result.effect_std[(snp_j, trait_k)],
                    z_marginal: zscores[(global_snp, trait_k)],
                });
            }
        }
    }

    rows
}
