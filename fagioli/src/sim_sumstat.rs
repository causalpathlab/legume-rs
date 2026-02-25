use anyhow::Result;
use clap::Args;
use log::info;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rayon::prelude::*;
use rust_htslib::tpool::ThreadPool;

use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};
use fagioli::simulation::{
    compose_phenotype, generate_confounder_matrix, sample_cell_type_genetic_effects,
    CellTypeGeneticEffects, ConfounderParams,
};
use fagioli::summary_stats::{
    compute_block_ld_scores, compute_block_sumstats, compute_yty_diagonal, create_uniform_blocks,
    estimate_ld_blocks, load_ld_blocks_from_file, write_confounders, write_ground_truth,
    write_ld_blocks, LdBlock, LdBlockParams, LdScoreWriter, SumstatWriter,
};

#[derive(Args, Debug, Clone)]
pub struct SimSumstatArgs {
    // ── Input ────────────────────────────────────────────────────────────
    #[arg(long, help = "PLINK BED file prefix (without .bed/.bim/.fam)")]
    pub bed_prefix: String,

    #[arg(long, help = "Chromosome to simulate from")]
    pub chromosome: String,

    #[arg(long, help = "Left genomic position bound (bp)")]
    pub left_bound: Option<u64>,

    #[arg(long, help = "Right genomic position bound (bp)")]
    pub right_bound: Option<u64>,

    #[arg(long, help = "Max individuals to use from genotype file")]
    pub max_individuals: Option<usize>,

    #[arg(long, default_value = "42", help = "Random seed")]
    pub seed: u64,

    // ── Trait parameters ─────────────────────────────────────────────────
    #[arg(long, default_value = "10", help = "Number of traits to simulate")]
    pub num_traits: usize,

    #[arg(
        long,
        default_value = "5",
        help = "Shared causal SNPs per causal block"
    )]
    pub num_shared_causal: usize,

    #[arg(
        long,
        default_value = "3",
        help = "Per-trait independent causal SNPs per causal block"
    )]
    pub num_independent_causal: usize,

    #[arg(
        long,
        default_value = "0.4",
        help = "Sparse heritability (genetic variance from causal SNPs)"
    )]
    pub h2_sparse: f32,

    #[arg(
        long,
        default_value = "0.0",
        help = "Polygenic heritability (dense infinitesimal effects on all SNPs)"
    )]
    pub h2_polygenic: f32,

    #[arg(
        long,
        default_value = "1",
        help = "Number of LD blocks that contain causal SNPs"
    )]
    pub num_causal_blocks: usize,

    // ── Confounder parameters ────────────────────────────────────────────
    #[arg(
        long,
        default_value = "0",
        help = "Number of confounder columns (0 = none)"
    )]
    pub num_confounders: usize,

    #[arg(
        long,
        default_value = "5",
        help = "Hidden factors generating confounders"
    )]
    pub num_hidden_factors: usize,

    #[arg(
        long,
        default_value = "0.1",
        help = "Variance proportion explained by confounders"
    )]
    pub pve_confounders: f32,

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

    // ── Output ───────────────────────────────────────────────────────────
    #[arg(short, long, help = "Output prefix for summary stats and LD files")]
    pub output: String,
}

const SEED_CONFOUNDERS: u64 = 100;
const SEED_PHENOTYPE: u64 = 200;
const SEED_EFFECTS: u64 = 300;

pub fn sim_sumstat(args: &SimSumstatArgs) -> Result<()> {
    info!("Starting sim-sumstat");

    // Create htslib thread pool for parallel BGZF compression
    let num_threads = rayon::current_num_threads().max(1) as u32;
    let tpool = ThreadPool::new(num_threads)?;
    info!("Using {} threads", num_threads);

    // ── Step 1: Read genotypes ───────────────────────────────────────────
    let region = GenomicRegion::new(
        Some(args.chromosome.clone()),
        args.left_bound,
        args.right_bound,
    );

    let mut reader = BedReader::new(&args.bed_prefix)?;
    let geno = reader.read(args.max_individuals, Some(region))?;
    let n = geno.num_individuals();
    let m = geno.num_snps();
    let t = args.num_traits;

    info!("Loaded {} individuals x {} SNPs", n, m);

    // ── Step 2: Determine LD blocks ──────────────────────────────────────
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
        create_uniform_blocks(m, m, &geno.positions, &geno.chromosomes)
    };

    let num_blocks = blocks.len();
    info!("Using {} LD blocks", num_blocks);

    // Write LD blocks
    let blocks_file = format!("{}.ld_blocks.bed.gz", args.output);
    write_ld_blocks(&blocks_file, &blocks, Some(&tpool))?;

    // ── Step 3: Pass 1 — Build phenotypes ────────────────────────────────
    // Select causal blocks: pick num_causal_blocks blocks at random
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed + SEED_EFFECTS);
    let min_snps = args.num_shared_causal + args.num_independent_causal;
    let mut eligible: Vec<usize> = (0..num_blocks)
        .filter(|&i| blocks[i].num_snps() >= min_snps)
        .collect();

    use rand::seq::SliceRandom;
    eligible.shuffle(&mut rng);
    let num_causal_blocks = args.num_causal_blocks.min(eligible.len());
    let causal_block_indices: Vec<usize> = eligible.into_iter().take(num_causal_blocks).collect();

    info!(
        "{} of {} blocks contain causal SNPs",
        num_causal_blocks, num_blocks,
    );

    // Pass 1: Parallel computation of per-block genetic values
    let block_results: Vec<(usize, CellTypeGeneticEffects, DMatrix<f32>)> = causal_block_indices
        .par_iter()
        .map(|&block_idx| {
            let block = &blocks[block_idx];
            let block_m = block.num_snps();

            let block_seed = args.seed + SEED_EFFECTS + block_idx as u64 * 1000;
            let effects = sample_cell_type_genetic_effects(
                block_m,
                t,
                args.num_shared_causal,
                args.num_independent_causal,
                args.h2_sparse / num_causal_blocks.max(1) as f32,
                block_seed,
            )
            .expect("Failed to sample genetic effects");

            let x_block = geno
                .genotypes
                .columns(block.snp_start, block_m)
                .clone_owned();

            let g_block = compute_genetic_values(&x_block, &effects);
            (block_idx, effects, g_block)
        })
        .collect();

    // Accumulate genetic values (sequential reduction)
    let mut g_total = DMatrix::<f32>::zeros(n, t);
    let mut block_effects: Vec<(usize, CellTypeGeneticEffects)> = Vec::new();

    for (block_idx, effects, g_block) in block_results {
        g_total += g_block;
        block_effects.push((block_idx, effects));
    }

    // Sort block_effects by block index for deterministic output
    block_effects.sort_by_key(|(idx, _)| *idx);

    info!(
        "Accumulated genetic values from {} causal blocks",
        block_effects.len()
    );

    // ── Step 3b: Polygenic effects (dense infinitesimal) ─────────────────
    let g_poly = if args.h2_polygenic > 0.0 {
        info!(
            "Generating polygenic effects: h2_poly={:.3}, sigma2/p={:.2e}",
            args.h2_polygenic,
            1.0 / m as f64,
        );
        let mut rng_poly = rand::rngs::StdRng::seed_from_u64(args.seed + SEED_EFFECTS + 999999);
        let normal = rand_distr::Normal::new(0.0f32, (1.0 / m as f32).sqrt()).unwrap();

        // beta_poly: p x T, each element ~ N(0, 1/p)
        let beta_poly = DMatrix::from_fn(m, t, |_, _| {
            rand_distr::Distribution::sample(&normal, &mut rng_poly)
        });

        // G_poly = X * beta_poly (N x T)
        let g = &geno.genotypes * &beta_poly;
        info!("Polygenic genetic values: {} x {}", g.nrows(), g.ncols());
        Some(g)
    } else {
        None
    };

    // Generate confounders
    let conf_params = ConfounderParams {
        num_confounders: args.num_confounders,
        num_hidden_factors: args.num_hidden_factors,
        pve_confounders: args.pve_confounders,
    };
    let confounder_matrix =
        generate_confounder_matrix(n, &conf_params, args.seed + SEED_CONFOUNDERS)?;

    // Compose final phenotype
    let pve_conf = if args.num_confounders > 0 {
        args.pve_confounders
    } else {
        0.0
    };

    let phenotypes = if let Some(ref gp) = g_poly {
        // Manual composition: separate h2 for sparse and polygenic
        compose_phenotype_with_polygenic(
            &g_total,
            gp,
            &confounder_matrix,
            args.h2_sparse,
            args.h2_polygenic,
            pve_conf,
            t,
            args.seed + SEED_PHENOTYPE,
        )?
    } else {
        compose_phenotype(
            &g_total,
            &confounder_matrix,
            args.h2_sparse,
            pve_conf,
            t,
            args.seed + SEED_PHENOTYPE,
        )?
    };

    info!(
        "Phenotype matrix: {} x {}",
        phenotypes.nrows(),
        phenotypes.ncols()
    );

    // Write ground truth
    let gt_file = format!("{}.ground_truth.bed.gz", args.output);
    write_ground_truth(
        &gt_file,
        &block_effects,
        &blocks,
        &geno.snp_ids,
        &geno.chromosomes,
        &geno.positions,
        Some(&tpool),
    )?;

    // Write confounders (if any)
    if args.num_confounders > 0 {
        let conf_file = format!("{}.confounders.tsv.gz", args.output);
        write_confounders(&conf_file, &confounder_matrix)?;
    }

    // ── Step 4: Pass 2 — Summary statistics (parallel compute, sequential write)
    info!("Computing summary statistics block by block");

    let sumstat_file = format!("{}.sumstats.bed.gz", args.output);
    let mut sumstat_writer = SumstatWriter::new(
        &sumstat_file,
        &geno.snp_ids,
        &geno.chromosomes,
        &geno.positions,
        &geno.allele1,
        &geno.allele2,
        n,
        Some(&tpool),
    )?;

    let ld_score_file = format!("{}.ld_scores.bed.gz", args.output);
    let mut ld_writer = LdScoreWriter::new(
        &ld_score_file,
        &geno.snp_ids,
        &geno.chromosomes,
        &geno.positions,
        Some(&tpool),
    )?;

    let yty_diag = compute_yty_diagonal(&phenotypes);

    // Process blocks in batches: parallel compute, sequential write
    let batch_size = num_threads as usize * 2;
    for batch_start in (0..num_blocks).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_blocks);
        let batch_indices: Vec<usize> = (batch_start..batch_end).collect();

        // Parallel: compute sumstats + LD scores for batch of blocks
        let batch_results: Vec<_> = batch_indices
            .par_iter()
            .map(|&block_idx| {
                let block = &blocks[block_idx];
                let block_m = block.num_snps();
                let x_block = geno
                    .genotypes
                    .columns(block.snp_start, block_m)
                    .clone_owned();

                let sumstats =
                    compute_block_sumstats(&x_block, &phenotypes, &yty_diag, block.snp_start);
                let ld_scores = compute_block_ld_scores(&x_block, block.snp_start);

                (block_idx, sumstats, ld_scores)
            })
            .collect();

        // Sequential: write in block order (required for tabix sort order)
        for (block_idx, sumstats, ld_scores) in batch_results {
            sumstat_writer.write_block(&sumstats)?;
            ld_writer.write_block(&ld_scores)?;

            if (block_idx + 1) % 10 == 0 || block_idx + 1 == num_blocks {
                info!(
                    "Processed block {}/{} ({} SNPs)",
                    block_idx + 1,
                    num_blocks,
                    blocks[block_idx].num_snps()
                );
            }
        }
    }

    sumstat_writer.finish()?;
    ld_writer.finish()?;

    // ── Step 5: Write parameters ─────────────────────────────────────────
    let param_file = format!("{}.parameters.json", args.output);
    let params = serde_json::json!({
        "command": "sim-sumstat",
        "num_individuals": n,
        "num_snps": m,
        "num_traits": t,
        "num_blocks": num_blocks,
        "num_causal_blocks": num_causal_blocks,
        "num_shared_causal_per_block": args.num_shared_causal,
        "num_independent_causal_per_block": args.num_independent_causal,
        "h2_sparse": args.h2_sparse,
        "h2_polygenic": args.h2_polygenic,
        "num_confounders": args.num_confounders,
        "num_hidden_factors": args.num_hidden_factors,
        "pve_confounders": pve_conf,
        "ld_block_file": args.ld_block_file,
        "num_landmarks": args.num_landmarks,
        "num_ld_components": args.num_ld_components,
        "min_block_snps": args.min_block_snps,
        "max_block_snps": args.max_block_snps,
        "seed": args.seed,
        "bed_prefix": args.bed_prefix,
        "chromosome": args.chromosome,
    });
    std::fs::write(&param_file, serde_json::to_string_pretty(&params)?)?;
    info!("Wrote parameters: {}", param_file);

    info!("sim-sumstat completed successfully");
    Ok(())
}

/// Compute raw genetic values G (N x T) from block genotypes and effects.
///
/// For each trait t:
///   G_t = X_shared * beta_shared_t + X_indep_t * beta_indep_t
fn compute_genetic_values(
    x_block: &DMatrix<f32>,
    effects: &CellTypeGeneticEffects,
) -> DMatrix<f32> {
    let n = x_block.nrows();
    let t = effects.num_cell_types;
    let mut g = DMatrix::zeros(n, t);

    // Shared effects
    if !effects.shared_causal_indices.is_empty() {
        let s = effects.shared_causal_indices.len();
        let mut x_shared = DMatrix::zeros(n, s);
        for (j, &snp_idx) in effects.shared_causal_indices.iter().enumerate() {
            x_shared.set_column(j, &x_block.column(snp_idx));
        }

        // effects.shared_effect_sizes is T x S
        // We want G_shared = X_shared (N x S) * beta_shared^T (S x T)
        let beta_shared = effects.shared_effect_sizes.transpose(); // S x T
        g += x_shared * beta_shared;
    }

    // Independent effects
    for trait_idx in 0..t {
        if effects.independent_causal_indices[trait_idx].is_empty() {
            continue;
        }

        let indep_idx = &effects.independent_causal_indices[trait_idx];
        let num_indep = indep_idx.len();

        let mut x_indep = DMatrix::zeros(n, num_indep);
        for (j, &snp_idx) in indep_idx.iter().enumerate() {
            x_indep.set_column(j, &x_block.column(snp_idx));
        }

        // Extract effect sizes for this trait
        let beta: DMatrix<f32> = DMatrix::from_iterator(
            num_indep,
            1,
            effects
                .independent_effect_sizes
                .row(trait_idx)
                .iter()
                .take(num_indep)
                .copied(),
        );

        let g_indep = x_indep * beta;
        for i in 0..n {
            g[(i, trait_idx)] += g_indep[(i, 0)];
        }
    }

    g
}

/// Compose phenotype with separate sparse and polygenic genetic components.
///
/// Y_t = sqrt(h2_sparse) * std(G_sparse_t) + sqrt(h2_poly) * std(G_poly_t)
///     + sqrt(pve_conf) * std(C * gamma_t) + sqrt(pve_noise) * std(eps_t)
#[allow(clippy::too_many_arguments)]
fn compose_phenotype_with_polygenic(
    g_sparse: &DMatrix<f32>,
    g_poly: &DMatrix<f32>,
    confounder_matrix: &DMatrix<f32>,
    h2_sparse: f32,
    h2_poly: f32,
    pve_conf: f32,
    num_traits: usize,
    seed: u64,
) -> Result<DMatrix<f32>> {
    use matrix_util::traits::{MatOps, SampleOps};
    use rand_distr::{Distribution, Normal};

    let n = g_sparse.nrows();
    let t = num_traits;

    let h2_total = h2_sparse + h2_poly;
    if h2_total + pve_conf > 1.0 + 1e-6 {
        anyhow::bail!(
            "h2_sparse ({}) + h2_poly ({}) + pve_conf ({}) cannot exceed 1.0",
            h2_sparse,
            h2_poly,
            pve_conf,
        );
    }
    let pve_noise = (1.0 - h2_total - pve_conf).max(0.0);

    info!(
        "Composing phenotype: h2_sparse={:.3}, h2_poly={:.3}, pve_conf={:.3}, pve_noise={:.3}",
        h2_sparse, h2_poly, pve_conf, pve_noise
    );

    // Standardize and scale sparse genetic values
    let mut gs = g_sparse.clone();
    gs.scale_columns_inplace();
    gs *= h2_sparse.sqrt();

    // Standardize and scale polygenic genetic values
    let mut gp = g_poly.clone();
    gp.scale_columns_inplace();
    gp *= h2_poly.sqrt();

    // Confounder contribution
    let mut conf_component = DMatrix::zeros(n, t);
    if pve_conf > 0.0 && confounder_matrix.ncols() > 0 {
        let l = confounder_matrix.ncols();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, (1.0 / l as f64).sqrt()).unwrap();

        let gamma = DMatrix::from_fn(l, t, |_, _| normal.sample(&mut rng) as f32);
        conf_component = confounder_matrix * gamma;
        conf_component.scale_columns_inplace();
        conf_component *= pve_conf.sqrt();
    }

    // Noise
    let mut eps = DMatrix::<f32>::rnorm(n, t);
    eps.scale_columns_inplace();
    eps *= pve_noise.sqrt();

    // Combine: Y = G_sparse + G_poly + C + eps
    let mut y = gs;
    for j in 0..t {
        for i in 0..n {
            y[(i, j)] += gp[(i, j)] + conf_component[(i, j)] + eps[(i, j)];
        }
    }

    info!("Composed phenotype matrix: {} x {}", y.nrows(), y.ncols());
    Ok(y)
}
