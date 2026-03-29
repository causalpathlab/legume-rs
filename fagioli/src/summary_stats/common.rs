//! Shared pipeline infrastructure for summary-statistics fine-mapping.
//!
//! Both `map-sumstat` (SGVB) and `mcmc-sumstat` (MCMC) share the same input
//! parsing, LD block estimation, z-score adjustment, and output writing.
//! The per-block fitting strategy is abstracted via [`RssBlockFitter`].

use rustc_hash::FxHashSet as HashSet;

use anyhow::{ensure, Result};
use clap::Args;
use log::info;
use matrix_util::common_io::read_lines;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;
use rust_htslib::tpool::ThreadPool;

use crate::genotype::{BedReader, GenomicRegion, GenotypeMatrix, GenotypeReader};
use crate::io::results::{write_parameters, write_variant_results, VariantRow};
use crate::summary_stats::rss_svd::RssSvdNal;
use crate::summary_stats::{
    create_uniform_blocks, estimate_ld_blocks, load_ld_blocks_from_file,
    read_sumstat_zscores_with_n, LdBlock, LdBlockParams,
};

// ── Shared result types ─────────────────────────────────────────────────────

/// Result from fitting a single block (shared by SGVB and MCMC).
#[derive(Debug)]
pub struct BlockFitResult {
    /// Per-(SNP, trait) posterior inclusion probabilities, shape (p, k).
    pub pip: DMatrix<f32>,
    /// Posterior mean effect sizes, shape (p, k).
    pub effect_mean: DMatrix<f32>,
    /// Posterior std of effect sizes, shape (p, k).
    pub effect_std: DMatrix<f32>,
    /// Diagnostic score (ELBO for SGVB, NaN for MCMC).
    pub avg_elbo: f32,
}

/// RSS-specific parameters for per-block fitting.
pub struct RssParams {
    pub max_rank: usize,
    pub lambda: f64,
    pub ldsc_intercept: bool,
}

/// Build an adaptive prior_var grid centered on `h2 / num_components`.
///
/// When `n` is provided (RSS mode), the grid is scaled by `n` to convert
/// from per-SD variance to z-score-scale variance.
pub fn adaptive_prior_grid(h2_estimate: f32, num_components: usize, n: Option<u64>) -> Vec<f32> {
    let center = (h2_estimate / num_components as f32).max(0.01);
    let scale = n.unwrap_or(1) as f32;
    let multipliers = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0];
    let mut grid: Vec<f32> = multipliers.iter().map(|&m| center * scale * m).collect();
    grid.iter_mut().for_each(|v| *v = v.clamp(0.001, 1e6));
    grid
}

// ── Common CLI arguments ────────────────────────────────────────────────────

/// CLI arguments shared between SGVB and MCMC summary-statistics fine-mapping.
#[derive(Args, Debug, Clone)]
pub struct CommonSumstatArgs {
    // ── Input ────────────────────────────────────────────────────────────
    #[arg(
        long,
        help_heading = "Input",
        help = "GWAS summary statistics file (.sumstats.bed.gz)",
        long_help = "BGZF-compressed summary statistics file in BED-like format.\n\
            Required columns: chr, end (or pos), z (or zscore).\n\
            Optional columns: start, snp_id, a1/ea, a2/nea, trait_idx, n, beta, se, pvalue.\n\
            SNPs are matched to the reference panel by chr+position.\n\
            Strand-ambiguous alleles (A/T, C/G) are automatically dropped."
    )]
    pub sumstat_file: Box<str>,

    #[arg(
        long,
        help_heading = "Input",
        help = "PLINK BED file prefix for LD reference panel (without .bed/.bim/.fam)",
        long_help = "Path prefix for PLINK binary genotype files used as the LD reference panel.\n\
            Reads {prefix}.bed, {prefix}.bim, and {prefix}.fam.\n\
            The LD matrix R = X'X/n is computed from these genotypes.\n\
            SNP positions in .bim are used to match against summary statistics."
    )]
    pub bed_prefix: Box<str>,

    #[arg(
        long,
        help_heading = "Input",
        help = "Chromosome to analyze (must match chr column in .bim and sumstats)"
    )]
    pub chromosome: Box<str>,

    #[arg(
        long,
        help_heading = "Input",
        help = "Left genomic position bound in bp (inclusive, filters SNPs)"
    )]
    pub left_bound: Option<u64>,

    #[arg(
        long,
        help_heading = "Input",
        help = "Right genomic position bound in bp (inclusive, filters SNPs)"
    )]
    pub right_bound: Option<u64>,

    // ── Individual filtering ─────────────────────────────────────────────
    #[arg(
        long,
        help_heading = "Individual Filtering",
        help = "Subsample to at most N individuals from the reference panel"
    )]
    pub max_individuals: Option<usize>,

    #[arg(
        long,
        help_heading = "Individual Filtering",
        conflicts_with = "remove",
        help = "Keep only these individuals in the reference panel",
        long_help = "Keep only these individuals in the LD reference panel (like plink --keep).\n\n\
            Accepts a file path or a comma-separated list of IIDs.\n\
            File format: one individual per line, either \"FID IID\" (two columns)\n\
            or just \"IID\" (one column). Lines starting with # are skipped.\n\
            Gzipped files (.gz) are supported.\n\n\
            Examples:\n  \
              --keep samples.txt\n  \
              --keep ind1,ind2,ind3"
    )]
    pub keep: Option<Box<str>>,

    #[arg(
        long,
        help_heading = "Individual Filtering",
        conflicts_with = "keep",
        help = "Remove these individuals from the reference panel",
        long_help = "Remove these individuals from the LD reference panel (like plink --remove).\n\n\
            Accepts a file path or a comma-separated list of IIDs.\n\
            File format: one individual per line, either \"FID IID\" (two columns)\n\
            or just \"IID\" (one column). Lines starting with # are skipped.\n\
            Gzipped files (.gz) are supported.\n\n\
            Examples:\n  \
              --remove samples.txt\n  \
              --remove ind1,ind2,ind3"
    )]
    pub remove: Option<Box<str>>,

    // ── LD block parameters ──────────────────────────────────────────────
    #[arg(
        long,
        help_heading = "LD Blocks",
        help = "External LD block boundary file (BED: chr, start, end)",
        long_help = "External LD block file in BED format (chr, start, end).\n\
            Each block defines an independent fine-mapping region.\n\
            If omitted, LD blocks are automatically estimated from the\n\
            reference genotypes using Nystrom + rSVD embedding distances."
    )]
    pub ld_block_file: Option<Box<str>>,

    #[arg(
        long,
        help_heading = "LD Blocks",
        default_value = "500",
        help = "Number of landmark SNPs for Nystrom LD block estimation"
    )]
    pub num_landmarks: usize,

    #[arg(
        long,
        help_heading = "LD Blocks",
        default_value = "20",
        help = "Number of rSVD components for LD block estimation"
    )]
    pub num_ld_components: usize,

    #[arg(
        long,
        help_heading = "LD Blocks",
        default_value = "200",
        help = "Minimum LD block size in SNPs (smaller blocks are merged)"
    )]
    pub min_block_snps: usize,

    // ── RSS SVD parameters ───────────────────────────────────────────────
    #[arg(
        long,
        help_heading = "RSS Eigenspace",
        help = "Max rank for per-block rSVD (default: reference panel sample size n)"
    )]
    pub max_rank: Option<usize>,

    #[arg(
        long,
        help_heading = "RSS Eigenspace",
        help = "SVD regularization lambda (default: 0.1 / max_rank)"
    )]
    pub lambda: Option<f64>,

    // ── Model parameters ─────────────────────────────────────────────────
    #[arg(
        long,
        help_heading = "Model",
        default_value = "10",
        help = "Number of sparse components L (max causal SNPs per block)",
        long_help = "Number of sparse components (L) in the model.\n\
            Each component can select one causal SNP, so L is the maximum number\n\
            of causal variants the model can identify per LD block.\n\
            Used by both SGVB (SuSiE) and MCMC. Default: 10."
    )]
    pub num_components: usize,

    #[arg(
        long,
        help_heading = "Model",
        default_value = "",
        help = "Prior variance for effect sizes (comma-separated, empty = adaptive)",
        long_help = "Prior variance(s) for the effect size distribution.\n\n\
            If empty (default), an adaptive grid is built from LDSC h² estimation.\n\
            - SGVB: fits each grid value and selects the best by ELBO.\n\
            - MCMC: uses the median of the grid as a single prior variance\n\
              (or a fixed value if only one is given). Use --estimate-prior-var\n\
              to let the MCMC chain learn the prior variance from the data."
    )]
    pub prior_var: Box<str>,

    // ── Z-score adjustments ─────────────────────────────────────────────
    #[arg(
        long,
        help_heading = "Z-score Adjustments",
        default_value_t = false,
        help = "Disable PVE adjustment on z-scores"
    )]
    pub no_pve_adjust: bool,

    #[arg(
        long,
        help_heading = "Z-score Adjustments",
        default_value_t = false,
        help = "Disable per-block LDSC intercept correction on z-scores"
    )]
    pub no_ldsc_intercept: bool,

    // ── Misc ─────────────────────────────────────────────────────────────
    #[arg(long, default_value = "42", help = "Random seed for reproducibility")]
    pub seed: u64,

    #[arg(
        short,
        long,
        help = "Output file prefix (produces {prefix}.results.bed.gz and {prefix}.parameters.json)"
    )]
    pub output: Box<str>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of parallel block-fitting jobs (0 = auto)"
    )]
    pub jobs: usize,
}

// ── Block fitter trait ──────────────────────────────────────────────────────

/// Per-block fine-mapping fitter, generic over inference method.
pub trait RssBlockFitter: Sync {
    /// Fit a single LD block. Returns per-SNP PIPs and effect estimates.
    ///
    /// # Arguments
    /// * `x_block` - Standardized genotypes, shape (n, p_block).
    /// * `z_block` - Z-scores for this block, shape (p_block, T).
    /// * `rss_params` - RSS SVD parameters (max_rank, lambda, ldsc_intercept).
    /// * `seed` - Per-block RNG seed.
    fn fit_block(
        &self,
        x_block: &DMatrix<f32>,
        z_block: &DMatrix<f32>,
        rss_params: &RssParams,
        seed: u64,
    ) -> Result<BlockFitResult>;

    /// Method name for logging and output JSON.
    fn method_name(&self) -> &str;
}

// ── Preprocessed sumstat input ──────────────────────────────────────────────

/// Preprocessed summary statistics data, ready for per-block fitting.
pub struct SumstatInput {
    pub geno: GenotypeMatrix,
    pub zscores: DMatrix<f32>,
    pub blocks: Vec<LdBlock>,
    pub median_n: u64,
    pub max_rank: usize,
}

// ── Shared pipeline functions ───────────────────────────────────────────────

/// Read genotypes, summary statistics, apply PVE adjustment, estimate LD blocks.
pub fn prepare_sumstat_input(args: &CommonSumstatArgs) -> Result<SumstatInput> {
    // ── Read reference panel genotypes ────────────────────────────────────
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
    let max_rank = args.max_rank.unwrap_or(n);

    info!("Reference panel: {} individuals x {} SNPs", n, m);

    // ── Read summary statistics ──────────────────────────────────────────
    let (zscores, median_n) = read_sumstat_zscores_with_n(
        &args.sumstat_file,
        &geno.snp_ids,
        &geno.chromosomes,
        &geno.positions,
        &geno.allele1,
        &geno.allele2,
    )?;
    let t = zscores.ncols();
    info!(
        "Z-scores: {} SNPs x {} traits, median N={}",
        zscores.nrows(),
        t,
        median_n,
    );

    // ── PVE-adjusted z-scores ────────────────────────────────────────────
    let mut zscores = zscores;
    if !args.no_pve_adjust && median_n > 2 {
        let max_z_before = zscores.iter().map(|z| z.abs()).fold(0.0f32, f32::max);
        let nf = median_n as f32;
        zscores.iter_mut().for_each(|z| {
            let z2 = *z * *z;
            *z *= ((nf - 1.0) / (z2 + nf - 2.0)).sqrt();
        });
        let max_z_after = zscores.iter().map(|z| z.abs()).fold(0.0f32, f32::max);
        info!(
            "PVE adjustment: max|z| {:.2} -> {:.2} (median_n={})",
            max_z_before, max_z_after, median_n,
        );
    } else if args.no_pve_adjust {
        info!("PVE adjustment: disabled (--no-pve-adjust)");
    } else {
        info!("PVE adjustment: skipped (median_n={} <= 2)", median_n);
    }

    // ── Determine LD blocks ──────────────────────────────────────────────
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
                min_block_snps: Some(args.min_block_snps),
                max_block_snps: None,
                seed: args.seed,
            },
        )?
    } else {
        info!("Too few SNPs for block estimation, using single block");
        create_uniform_blocks(m, m, &geno.positions, &geno.chromosomes)
    };

    info!("Using {} LD blocks", blocks.len());

    Ok(SumstatInput {
        geno,
        zscores,
        blocks,
        median_n,
        max_rank,
    })
}

/// Parse comma-separated prior variance string, or return empty vec for adaptive.
pub fn parse_prior_vars(prior_var_str: &str) -> Result<Vec<f32>> {
    if prior_var_str.trim().is_empty() {
        Ok(Vec::new())
    } else {
        Ok(prior_var_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }
}

/// Estimate adaptive prior variance grid from LDSC h² across LD blocks.
///
/// Uses the nalgebra-based RSS SVD (`RssSvdNal`) for h² estimation,
/// so this works for both SGVB and MCMC paths without candle dependency.
pub fn estimate_adaptive_prior_vars(
    input: &SumstatInput,
    num_components: usize,
    lambda: Option<f64>,
) -> Vec<f32> {
    let t = input.zscores.ncols();
    let block_lambda = lambda.unwrap_or(0.1 / input.max_rank as f64);
    let mut h2_sum = vec![0.0f32; t];
    let mut n_blocks_used = 0usize;

    for block in &input.blocks {
        let block_m = block.num_snps();
        if block_m < 10 {
            continue;
        }
        let mut x_block = input
            .geno
            .genotypes
            .columns(block.snp_start, block_m)
            .clone_owned();
        x_block.scale_columns_inplace();
        let z_block = input.zscores.rows(block.snp_start, block_m).clone_owned();
        if let Ok(slopes) =
            RssSvdNal::estimate_block_h2(&x_block, &z_block, input.max_rank, block_lambda)
        {
            for (tt, &s) in slopes.iter().enumerate() {
                h2_sum[tt] += s;
            }
            n_blocks_used += 1;
        }
    }

    let mean_h2 = if n_blocks_used > 0 {
        h2_sum.iter().sum::<f32>() / t as f32
    } else {
        0.1
    };
    let h2_est = mean_h2.max(0.01);
    info!(
        "LDSC h² estimate: {:.4} (per-trait: {:?}, {} blocks)",
        h2_est,
        h2_sum
            .iter()
            .map(|h| format!("{:.4}", h))
            .collect::<Vec<_>>(),
        n_blocks_used,
    );

    adaptive_prior_grid(h2_est, num_components, Some(input.median_n))
}

/// Run per-block fitting in parallel and return sorted results.
pub fn run_blocks<F: RssBlockFitter>(
    input: &SumstatInput,
    fitter: &F,
    args: &CommonSumstatArgs,
) -> Result<Vec<(usize, BlockFitResult)>> {
    let num_blocks = input.blocks.len();
    let t = input.zscores.ncols();
    let num_jobs = if args.jobs == 0 {
        rayon::current_num_threads()
    } else {
        args.jobs
    };

    info!(
        "Fitting {} models for {} blocks ({} jobs)",
        fitter.method_name(),
        num_blocks,
        num_jobs,
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

        let mut x_block = input
            .geno
            .genotypes
            .columns(block.snp_start, block_m)
            .clone_owned();
        x_block.scale_columns_inplace();

        let z_block = input.zscores.rows(block.snp_start, block_m).clone_owned();

        let block_lambda = args.lambda.unwrap_or(0.1 / input.max_rank as f64);
        let rss_params = RssParams {
            max_rank: input.max_rank,
            lambda: block_lambda,
            ldsc_intercept: !args.no_ldsc_intercept,
        };

        let seed = args.seed.wrapping_add(block_idx as u64);

        let result = fitter
            .fit_block(&x_block, &z_block, &rss_params, seed)
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

    // Report top hits
    report_top_hits(
        &block_results,
        &input.blocks,
        &input.zscores,
        &input.geno,
        t,
    );

    Ok(block_results)
}

/// Write block-level results and parameters JSON.
pub fn write_sumstat_output(
    input: &SumstatInput,
    block_results: &[(usize, BlockFitResult)],
    args: &CommonSumstatArgs,
    extra_params: serde_json::Value,
) -> Result<()> {
    let t = input.zscores.ncols();

    let variant_rows = build_sumstat_variant_rows(block_results, &input.blocks, &input.zscores, t);

    let num_threads = rayon::current_num_threads().max(1) as u32;
    let tpool = ThreadPool::new(num_threads)?;

    write_variant_results(
        &format!("{}.results.bed.gz", args.output),
        &["trait_idx"],
        &variant_rows,
        &input.geno,
        &tpool,
    )?;

    let mut params = serde_json::json!({
        "sumstat_file": args.sumstat_file,
        "bed_prefix": args.bed_prefix,
        "chromosome": args.chromosome,
        "num_individuals": input.geno.num_individuals(),
        "num_snps": input.geno.num_snps(),
        "num_traits": t,
        "median_n": input.median_n,
        "num_blocks": input.blocks.len(),
        "num_components": args.num_components,
        "max_rank": input.max_rank,
        "lambda": args.lambda.unwrap_or(0.1 / input.max_rank as f64),
        "seed": args.seed,
        "pve_adjust": !args.no_pve_adjust,
        "ldsc_intercept": !args.no_ldsc_intercept,
    });

    if let (Some(base), Some(extra)) = (params.as_object_mut(), extra_params.as_object()) {
        for (k, v) in extra {
            base.insert(k.clone(), v.clone());
        }
    }

    write_parameters(&format!("{}.parameters.json", args.output), &params)?;

    Ok(())
}

/// Convenience: run blocks + write output in one call.
pub fn run_blocks_and_write<F: RssBlockFitter>(
    input: &SumstatInput,
    fitter: &F,
    args: &CommonSumstatArgs,
    extra_params: serde_json::Value,
) -> Result<()> {
    let block_results = run_blocks(input, fitter, args)?;
    write_sumstat_output(input, &block_results, args, extra_params)
}

// ── Helper functions ────────────────────────────────────────────────────────

pub fn report_top_hits(
    results: &[(usize, BlockFitResult)],
    blocks: &[LdBlock],
    zscores: &DMatrix<f32>,
    geno: &GenotypeMatrix,
    t: usize,
) {
    for (block_idx, result) in results {
        let block = &blocks[*block_idx];
        let block_m = block.num_snps();
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
}

/// Parse individual IDs from either a file path or a comma-separated list.
pub(crate) fn parse_individual_ids(value: &str) -> Result<HashSet<Box<str>>> {
    if value.contains(',') || !std::path::Path::new(value).is_file() {
        let ids: HashSet<Box<str>> = value
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(Box::from)
            .collect();
        ensure!(!ids.is_empty(), "Empty individual ID list");
        info!("Parsed {} individual IDs from command line", ids.len());
        Ok(ids)
    } else {
        let lines = read_lines(value)?;
        let mut ids: HashSet<Box<str>> = Default::default();
        for line in &lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            let iid = if fields.len() >= 2 {
                fields[1]
            } else {
                fields[0]
            };
            ids.insert(iid.into());
        }
        ensure!(!ids.is_empty(), "No individual IDs found in {}", value);
        info!("Read {} individual IDs from {}", ids.len(), value);
        Ok(ids)
    }
}

/// Subset a GenotypeMatrix to keep only the specified individual (row) indices.
pub(crate) fn filter_individuals(geno: &mut GenotypeMatrix, indices: &[usize]) {
    let m = geno.genotypes.ncols();
    geno.genotypes = DMatrix::from_fn(indices.len(), m, |i, j| geno.genotypes[(indices[i], j)]);
    geno.individual_ids = indices
        .iter()
        .map(|&i| geno.individual_ids[i].clone())
        .collect();
}

/// Build variant result rows from block-level fine-mapping output.
pub fn build_sumstat_variant_rows(
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
