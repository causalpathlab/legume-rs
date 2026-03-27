use rustc_hash::FxHashSet as HashSet;

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
use fagioli::sgvb::{
    adaptive_prior_grid, estimate_block_h2, fit_block_rss, BlockFitResult, BlockFitResultDetailed,
    ComputeDevice, FitConfig, ModelType, MultilevelConfig, RssParams, SnpCoordinates,
};
use fagioli::summary_stats::{
    estimate_ld_blocks, load_ld_blocks_from_file, read_sumstat_zscores_with_n, LdBlock,
    LdBlockParams,
};

#[derive(Args, Debug, Clone)]
pub struct MapSumstatArgs {
    // ── Input ────────────────────────────────────────────────────────────
    #[arg(
        long,
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
        help = "PLINK BED file prefix for LD reference panel (without .bed/.bim/.fam)",
        long_help = "Path prefix for PLINK binary genotype files used as the LD reference panel.\n\
            Reads {prefix}.bed, {prefix}.bim, and {prefix}.fam.\n\
            The LD matrix R = X'X/n is computed from these genotypes.\n\
            SNP positions in .bim are used to match against summary statistics."
    )]
    pub bed_prefix: Box<str>,

    #[arg(
        long,
        help = "Chromosome to analyze (must match chr column in .bim and sumstats)"
    )]
    pub chromosome: Box<str>,

    #[arg(
        long,
        help = "Left genomic position bound in bp (inclusive, filters SNPs)"
    )]
    pub left_bound: Option<u64>,

    #[arg(
        long,
        help = "Right genomic position bound in bp (inclusive, filters SNPs)"
    )]
    pub right_bound: Option<u64>,

    // ── Individual filtering ─────────────────────────────────────────────
    #[arg(
        long,
        help = "Subsample to at most N individuals from the reference panel"
    )]
    pub max_individuals: Option<usize>,

    #[arg(
        long,
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
        help = "External LD block boundary file (BED: chr, start, end)",
        long_help = "External LD block file in BED format (chr, start, end).\n\
            Each block defines an independent fine-mapping region.\n\
            If omitted, LD blocks are automatically estimated from the\n\
            reference genotypes using Nystrom + rSVD embedding distances."
    )]
    pub ld_block_file: Option<Box<str>>,

    #[arg(
        long,
        default_value = "500",
        help = "Number of landmark SNPs for Nystrom LD block estimation",
        long_help = "Number of randomly sampled landmark SNPs for the Nystrom\n\
            approximation used in LD block estimation. The landmarks form\n\
            the basis for projecting all SNPs into an embedding space where\n\
            LD breaks are detected as distance peaks. Default: 500."
    )]
    pub num_landmarks: usize,

    #[arg(
        long,
        default_value = "20",
        help = "Number of rSVD components for LD block estimation",
        long_help = "Rank of the randomized SVD computed on the landmark genotype\n\
            matrix. Defines the dimensionality of the SNP embedding space\n\
            for LD block boundary detection. Default: 20."
    )]
    pub num_ld_components: usize,

    #[arg(
        long,
        default_value = "50",
        help = "Minimum LD block size in SNPs (smaller blocks are merged)"
    )]
    pub min_block_snps: usize,

    #[arg(
        long,
        default_value = "5000",
        help = "Maximum LD block size in SNPs (larger blocks are split)"
    )]
    pub max_block_snps: usize,

    // ── RSS SVD parameters ───────────────────────────────────────────────
    #[arg(
        long,
        help = "Max rank for per-block rSVD (default: reference panel sample size n)",
        long_help = "Maximum rank for the randomized SVD of X/sqrt(n) within each LD block.\n\
            The RSS likelihood operates in this eigenspace (Zhu & Stephens 2017).\n\
            Default: sample size n (full rank). Lower values speed up fitting\n\
            for large reference panels at the cost of approximation accuracy."
    )]
    pub max_rank: Option<usize>,

    #[arg(
        long,
        help = "SVD regularization lambda (default: 0.1 / max_rank)",
        long_help = "Regularization parameter for the RSS eigenspace.\n\
            Applied as D_tilde = sqrt(D^2 + lambda) to the singular values,\n\
            preventing numerical instability from near-zero eigenvalues.\n\
            Default: 0.1 / max_rank."
    )]
    pub lambda: Option<f64>,

    // ── Model parameters ─────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "susie",
        help = "Fine-mapping model: 'susie' (univariate) or 'bisusie' (bivariate)",
        long_help = "Fine-mapping model to use:\n\
            - susie: Sum of Single Effects, each component selects one causal SNP\n\
            - bisusie: Bivariate SuSiE, each component has a pair of effects\n\
            Default: susie. Multilevel mode (--multilevel) only supports susie."
    )]
    pub model: Box<str>,

    #[arg(
        long,
        default_value = "10",
        help = "Number of SuSiE components L (max causal SNPs per block)",
        long_help = "Number of single-effect components (L) in the SuSiE model.\n\
            Each component can select one causal SNP, so L is the maximum number\n\
            of causal variants the model can identify per LD block. Default: 10."
    )]
    pub num_components: usize,

    #[arg(
        long,
        default_value = "",
        help = "Prior variance grid for effect sizes (comma-separated, empty = adaptive)",
        long_help = "Comma-separated prior variances for the effect size distribution.\n\
            The model is fit once for each value, and the best is selected by ELBO.\n\n\
            If empty (default), an adaptive grid is built automatically:\n\
            1. LDSC h² is estimated from z-scores and LD structure\n\
            2. A grid of 9 log-spaced points is placed around h²/L\n\
            3. Grid is scaled by n (sample size) for z-score-scale effects\n\n\
            Examples:\n  \
              --prior-var 0.05,0.1,0.2,0.3,0.5  (fixed grid)\n  \
              --prior-var ''  (adaptive from LDSC h², the default)"
    )]
    pub prior_var: Box<str>,

    // ── SGVB training ────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "20",
        help = "Monte Carlo samples per SGVB gradient step",
        long_help = "Number of Monte Carlo samples (S) drawn per gradient step in\n\
            Stochastic Gradient Variational Bayes. More samples reduce gradient\n\
            variance but increase per-step cost. Default: 20."
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
        help = "Row minibatch size (full batch when N <= this value)",
        long_help = "Number of individuals sampled per gradient step. When the total\n\
            number of individuals N exceeds this value, random minibatches are\n\
            drawn each iteration. The RSS eigenspace is already compact (K << N),\n\
            so minibatch is rarely needed here. Default: 1000."
    )]
    pub batch_size: usize,

    #[arg(
        long,
        default_value = "50",
        help = "Trailing window size for averaging ELBO (convergence diagnostic)",
        long_help = "Number of recent ELBO (evidence lower bound) values to average\n\
            for the reported convergence diagnostic. The average ELBO over the\n\
            last elbo_window iterations is used for prior variance selection\n\
            (best ELBO wins). Default: 50."
    )]
    pub elbo_window: usize,

    #[arg(
        long,
        help = "Use multilevel SuSiE with LD-aware hierarchical variable selection",
        long_help = "Enable multilevel SuSiE with LD-aware hierarchical softmax.\n\n\
            When enabled, LD sub-blocks are estimated within each outer LD block\n\
            using Nystrom + rSVD on the genotype matrix, and used as the level-0\n\
            partition for a recursive multilevel SuSiE model.\n\n\
            Benefits: improves optimization for large LD blocks (p > 200 SNPs)\n\
            by replacing a single flat softmax over all SNPs with a hierarchy of\n\
            smaller softmaxes that respect LD structure.\n\n\
            Constraints: only works with --model susie (not bisusie).\n\
            Automatically falls back to flat SuSiE for blocks with < 200 SNPs."
    )]
    pub multilevel: bool,

    #[arg(
        long,
        default_value = "20",
        help = "Min SNPs per LD sub-block in multilevel hierarchy",
        long_help = "Minimum number of SNPs per LD sub-block when estimating the\n\
            level-0 partition for multilevel SuSiE. Blocks smaller than this\n\
            are merged with neighbors. Only used when --multilevel is set.\n\
            Default: 20."
    )]
    pub multilevel_min_block: usize,

    #[arg(
        long,
        default_value = "500",
        help = "Max SNPs per LD sub-block in multilevel hierarchy",
        long_help = "Maximum number of SNPs per LD sub-block when estimating the\n\
            level-0 partition for multilevel SuSiE. Blocks larger than this\n\
            are split into uniform sub-blocks. Only used when --multilevel is set.\n\
            Default: 500."
    )]
    pub multilevel_max_block: usize,

    #[arg(
        long,
        default_value = "0.0",
        help = "Infinitesimal polygenic prior variance (0 = disabled)",
        long_help = "Prior variance for the infinitesimal polygenic background term.\n\
            When > 0, a dense Gaussian regression is fit alongside the sparse\n\
            SuSiE term to absorb polygenic signal. The per-SNP prior variance\n\
            is sigma2_inf / p (total variance spread across all SNPs).\n\
            Set to 0 to disable (default). Typical values: 0.001-0.01."
    )]
    pub sigma2_inf: f32,

    #[arg(
        long,
        default_value = "1.0",
        help = "Dirichlet concentration for SuSiE selection prior (1.0 = uniform)",
        long_help = "Concentration parameter for the per-SNP selection prior in SuSiE.\n\
            Each SNP gets prior inclusion probability prior_alpha / p.\n\
            - 1.0 (default): uniform prior, all SNPs equally likely a priori\n\
            - < 1.0: sparser prior, encourages fewer selected SNPs\n\
            - > 1.0: denser prior, more permissive selection"
    )]
    pub prior_alpha: f64,

    // ── Device ───────────────────────────────────────────────────────────
    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Hardware device for tensor computation: cpu, cuda, or metal"
    )]
    pub device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "GPU device index (for cuda or metal, 0-indexed)"
    )]
    pub device_no: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of parallel block-fitting jobs (0 = auto)",
        long_help = "Number of LD blocks to fit in parallel.\n\
            0 = automatic: uses all CPU cores for --device cpu, or 1 for GPU.\n\
            Set to 1 for sequential execution (useful for debugging)."
    )]
    pub jobs: usize,

    // ── Z-score adjustments ─────────────────────────────────────────────
    #[arg(
        long,
        default_value_t = false,
        help = "Disable PVE (proportion of variance explained) adjustment on z-scores",
        long_help = "Disable the PVE adjustment that shrinks z-scores toward zero:\n\
            z_adj = z * sqrt((n-1) / (z^2 + n - 2)).\n\
            This adjustment corrects for winner's curse by accounting for the\n\
            fact that large z-scores overestimate true effect sizes.\n\
            Enabled by default. Disable with this flag if z-scores are already adjusted."
    )]
    pub no_pve_adjust: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable per-block LDSC intercept correction on z-scores",
        long_help = "Disable local LDSC (LD Score regression) intercept correction.\n\
            When enabled (default), the LDSC intercept is estimated per LD block\n\
            and z-scores are rescaled by 1/sqrt(intercept) to correct for\n\
            confounding or population stratification. Disable if z-scores are\n\
            already corrected (e.g., from a BOLT-LMM or SAIGE GWAS)."
    )]
    pub no_ldsc_intercept: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Select best prior variance per block instead of globally",
        long_help = "Use per-block prior variance selection instead of global selection.\n\
            Default (false): the prior variance with the best total ELBO summed\n\
            across all blocks is used for all blocks.\n\
            When set: each block independently selects its best prior variance.\n\
            Per-block selection is more flexible but can overfit with few blocks."
    )]
    pub local_prior: bool,

    // ── Misc ─────────────────────────────────────────────────────────────
    #[arg(long, default_value = "42", help = "Random seed for reproducibility")]
    pub seed: u64,

    #[arg(
        short,
        long,
        help = "Output file prefix (produces {prefix}.results.bed.gz and {prefix}.parameters.json)"
    )]
    pub output: Box<str>,
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
    let max_rank = args.max_rank.unwrap_or(n);

    info!("Reference panel: {} individuals x {} SNPs", n, m);

    // ── Step 2: Read summary statistics ───────────────────────────────────
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

    // ── Step 2b: PVE-adjusted z-scores (Zhu & Stephens, Ann. Appl. Stat. 2017;
    //    Zou et al., PLoS Genet. 2022, Eq. 14-18).
    //    z_tilde_j = z_j * sqrt((n-1) / (z_j^2 + n - 2))
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
    let prior_vars: Vec<f32> = if args.prior_var.trim().is_empty() {
        Vec::new()
    } else {
        args.prior_var
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()?
    };

    // ── Step 4b: LDSC h² estimation → adaptive prior grid ──────────────
    let prior_vars: Vec<f32> = if prior_vars.is_empty() {
        // Estimate chromosome-wide h² via LDSC slopes across all blocks
        let block_lambda = args.lambda.unwrap_or(0.1 / max_rank as f64);
        let mut h2_sum = vec![0.0f32; t];
        let mut n_blocks_used = 0usize;
        for block in &blocks {
            let block_m = block.num_snps();
            if block_m < 10 {
                continue;
            }
            let mut x_block = geno
                .genotypes
                .columns(block.snp_start, block_m)
                .clone_owned();
            x_block.scale_columns_inplace();
            let z_block = zscores.rows(block.snp_start, block_m).clone_owned();
            if let Ok(slopes) =
                estimate_block_h2(&x_block, &z_block, max_rank, block_lambda, &device)
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
        adaptive_prior_grid(h2_est, args.num_components, Some(median_n))
    } else {
        prior_vars
    };

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
        sigma2_inf: args.sigma2_inf,
        prior_alpha: args.prior_alpha,
        multilevel: if args.multilevel {
            Some(MultilevelConfig {
                min_block_snps: args.multilevel_min_block,
                max_block_snps: args.multilevel_max_block,
                ..Default::default()
            })
        } else {
            None
        },
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

    let fit_block_fn = |(block_idx, block): (usize, &LdBlock)| -> (usize, BlockFitResultDetailed) {
        let block_m = block.num_snps();
        let num_priors = fit_config.prior_vars.len();
        if block_m < 10 {
            let empty = BlockFitResultDetailed {
                per_prior_elbos: vec![0.0; num_priors],
                per_prior_pips: vec![DMatrix::<f32>::zeros(block_m, t); num_priors],
                per_prior_effects: vec![DMatrix::<f32>::zeros(block_m, t); num_priors],
                per_prior_stds: vec![DMatrix::<f32>::zeros(block_m, t); num_priors],
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
        let block_lambda = args.lambda.unwrap_or(0.1 / max_rank as f64);

        let block_coords = SnpCoordinates {
            positions: &geno.positions[block.snp_start..block.snp_end],
            chromosomes: &geno.chromosomes[block.snp_start..block.snp_end],
        };

        let rss_params = RssParams {
            max_rank,
            lambda: block_lambda,
            ldsc_intercept: !args.no_ldsc_intercept,
            coords: Some(&block_coords),
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
        blocks.iter().enumerate().map(fit_block_fn).collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_jobs)
            .build()?;
        pool.install(|| blocks.par_iter().enumerate().map(fit_block_fn).collect())
    };

    // Sort by block index for deterministic output
    let mut sorted_results = block_results;
    sorted_results.sort_by_key(|(idx, _)| *idx);

    // ── Step 6: Prior selection via ELBO argmax ─────────────────────────
    let num_priors = fit_config.prior_vars.len();

    let globally_averaged: Vec<(usize, BlockFitResult)> = if !args.local_prior {
        // Global: sum ELBOs across all blocks, pick single best prior_var
        let mut global_elbos = vec![0.0f32; num_priors];
        for (_idx, detailed) in &sorted_results {
            for (j, &e) in detailed.per_prior_elbos.iter().enumerate() {
                global_elbos[j] += e;
            }
        }
        let best_idx = global_elbos
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        info!(
            "Global prior: prior_var={:.3} (total_elbo={:.1}), all: {:?}",
            fit_config.prior_vars[best_idx],
            global_elbos[best_idx],
            fit_config
                .prior_vars
                .iter()
                .zip(global_elbos.iter())
                .map(|(v, e)| format!("{:.3}:{:.1}", v, e))
                .collect::<Vec<_>>()
        );
        sorted_results
            .iter()
            .map(|(idx, d)| {
                (
                    *idx,
                    BlockFitResult {
                        pip: d.per_prior_pips[best_idx].clone(),
                        effect_mean: d.per_prior_effects[best_idx].clone(),
                        effect_std: d.per_prior_stds[best_idx].clone(),
                        avg_elbo: d.per_prior_elbos[best_idx],
                    },
                )
            })
            .collect()
    } else {
        // Local (default): each block picks its own best prior_var by ELBO argmax
        sorted_results
            .iter()
            .map(|(idx, d)| {
                let best_idx = d
                    .per_prior_elbos
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                (
                    *idx,
                    BlockFitResult {
                        pip: d.per_prior_pips[best_idx].clone(),
                        effect_mean: d.per_prior_effects[best_idx].clone(),
                        effect_std: d.per_prior_stds[best_idx].clone(),
                        avg_elbo: d.per_prior_elbos[best_idx],
                    },
                )
            })
            .collect()
    };

    // Report top hits from globally averaged results
    for (block_idx, result) in &globally_averaged {
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

    let variant_rows = build_sumstat_variant_rows(&globally_averaged, &blocks, &zscores, t);
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
        "max_rank": max_rank,
        "lambda": args.lambda.unwrap_or(0.1 / max_rank as f64),
        "seed": args.seed,
        "sigma2_inf": args.sigma2_inf,
        "pve_adjust": !args.no_pve_adjust,
        "ldsc_intercept": !args.no_ldsc_intercept,
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
fn parse_individual_ids(value: &str) -> Result<HashSet<Box<str>>> {
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
            // If two+ columns: FID IID → take IID (column 1)
            // If one column: take it as IID
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
