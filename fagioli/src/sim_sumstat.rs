use anyhow::Result;
use clap::Args;
use log::info;
use nalgebra::DMatrix;
use rand::SeedableRng;

use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};
use fagioli::simulation::{
    compose_phenotype, generate_confounder_matrix, sample_cell_type_genetic_effects,
    ConfounderParams, CellTypeGeneticEffects,
};
use fagioli::summary_stats::{
    compute_block_ld_scores, compute_block_sumstats, compute_yty_diagonal, create_uniform_blocks,
    estimate_ld_blocks, load_ld_blocks_from_file, write_confounders, write_ground_truth,
    write_ld_blocks, LdBlock, LdScoreWriter, SumstatWriter,
};

#[derive(Args, Debug, Clone)]
pub struct SimSumstatArgs {
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

    /// Max individuals to use
    #[arg(long)]
    pub max_individuals: Option<usize>,

    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,

    // --- Trait parameters ---
    /// Number of traits
    #[arg(long, default_value = "10")]
    pub num_traits: usize,

    /// Number of shared causal SNPs per causal block
    #[arg(long, default_value = "5")]
    pub num_shared_causal: usize,

    /// Number of independent (per-trait) causal SNPs per causal block
    #[arg(long, default_value = "3")]
    pub num_independent_causal: usize,

    /// Heritability (genetic variance proportion)
    #[arg(long, default_value = "0.4")]
    pub genetic_variance: f32,

    /// Per-block probability of containing causal SNPs
    #[arg(long, default_value = "0.3")]
    pub causal_block_density: f32,

    // --- Confounder parameters ---
    /// Number of confounder columns (0 = no confounders)
    #[arg(long, default_value = "0")]
    pub num_confounders: usize,

    /// Number of hidden factors generating confounders
    #[arg(long, default_value = "5")]
    pub num_hidden_factors: usize,

    /// Proportion of variance explained by confounders
    #[arg(long, default_value = "0.1")]
    pub pve_confounders: f32,

    // --- LD block parameters ---
    /// External LD block file (BED format: chr, start, end). If omitted, estimate from data.
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

    // --- Output ---
    /// Output prefix
    #[arg(short, long)]
    pub output: String,
}

const SEED_CONFOUNDERS: u64 = 100;
const SEED_PHENOTYPE: u64 = 200;
const SEED_EFFECTS: u64 = 300;

pub fn sim_sumstat(args: &SimSumstatArgs) -> Result<()> {
    info!("Starting sim-sumstat");

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
            args.num_landmarks,
            args.num_ld_components,
            args.min_block_snps,
            args.max_block_snps,
            args.seed,
        )?
    } else {
        info!("Too few SNPs for block estimation, using single block");
        create_uniform_blocks(m, m, &geno.positions, &geno.chromosomes)
    };

    let num_blocks = blocks.len();
    info!("Using {} LD blocks", num_blocks);

    // Write LD blocks
    let blocks_file = format!("{}.ld_blocks.tsv.gz", args.output);
    write_ld_blocks(&blocks_file, &blocks)?;

    // ── Step 3: Pass 1 — Build phenotypes ────────────────────────────────
    // Determine which blocks are causal
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed + SEED_EFFECTS);
    let causal_block_mask: Vec<bool> = (0..num_blocks)
        .map(|_| rand::Rng::random_bool(&mut rng, args.causal_block_density as f64))
        .collect();

    let num_causal_blocks = causal_block_mask.iter().filter(|&&x| x).count();
    info!(
        "{} of {} blocks contain causal SNPs (density={:.2})",
        num_causal_blocks, num_blocks, args.causal_block_density
    );

    // Accumulate genetic values: G_total (N x T)
    let mut g_total = DMatrix::<f32>::zeros(n, t);
    let mut block_effects: Vec<(usize, CellTypeGeneticEffects)> = Vec::new();

    for (block_idx, block) in blocks.iter().enumerate() {
        if !causal_block_mask[block_idx] {
            continue;
        }

        let block_m = block.num_snps();
        if block_m < args.num_shared_causal + args.num_independent_causal {
            info!(
                "Block {} too small ({} SNPs) for causal effects, skipping",
                block_idx, block_m
            );
            continue;
        }

        // Sample causal effects for this block
        let block_seed = args.seed + SEED_EFFECTS + block_idx as u64 * 1000;
        let effects = sample_cell_type_genetic_effects(
            block_m,
            t,
            args.num_shared_causal,
            args.num_independent_causal,
            args.genetic_variance / num_causal_blocks.max(1) as f32,
            block_seed,
        )?;

        // Extract block genotypes
        let x_block = geno
            .genotypes
            .columns(block.snp_start, block_m)
            .clone_owned();

        // Compute genetic values: G_block = X_block * beta_block (for each trait)
        let g_block = compute_genetic_values(&x_block, &effects);
        g_total += g_block;

        block_effects.push((block_idx, effects));
    }

    info!(
        "Accumulated genetic values from {} causal blocks",
        block_effects.len()
    );

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
    let phenotypes = compose_phenotype(
        &g_total,
        &confounder_matrix,
        args.genetic_variance,
        pve_conf,
        t,
        args.seed + SEED_PHENOTYPE,
    )?;

    info!(
        "Phenotype matrix: {} x {}",
        phenotypes.nrows(),
        phenotypes.ncols()
    );

    // Write ground truth
    let gt_file = format!("{}.ground_truth.tsv.gz", args.output);
    write_ground_truth(
        &gt_file,
        &block_effects,
        &blocks,
        &geno.snp_ids,
        &geno.chromosomes,
        &geno.positions,
    )?;

    // Write confounders (if any)
    if args.num_confounders > 0 {
        let conf_file = format!("{}.confounders.tsv.gz", args.output);
        write_confounders(&conf_file, &confounder_matrix)?;
    }

    // ── Step 4: Pass 2 — Summary statistics ──────────────────────────────
    info!("Computing summary statistics block by block");

    let sumstat_file = format!("{}.sumstats.tsv.gz", args.output);
    let mut sumstat_writer = SumstatWriter::new(
        &sumstat_file,
        &geno.snp_ids,
        &geno.chromosomes,
        &geno.positions,
        n,
    )?;

    let ld_score_file = format!("{}.ld_scores.tsv.gz", args.output);
    let mut ld_writer = LdScoreWriter::new(
        &ld_score_file,
        &geno.snp_ids,
        &geno.chromosomes,
        &geno.positions,
    )?;

    let yty_diag = compute_yty_diagonal(&phenotypes);

    for (block_idx, block) in blocks.iter().enumerate() {
        let block_m = block.num_snps();
        let x_block = geno
            .genotypes
            .columns(block.snp_start, block_m)
            .clone_owned();

        // Summary statistics
        let sumstats = compute_block_sumstats(&x_block, &phenotypes, &yty_diag, block.snp_start);
        sumstat_writer.write_block(&sumstats)?;

        // LD scores
        let ld_scores = compute_block_ld_scores(&x_block, block.snp_start);
        ld_writer.write_block(&ld_scores)?;

        if (block_idx + 1) % 10 == 0 || block_idx + 1 == num_blocks {
            info!(
                "Processed block {}/{} ({} SNPs)",
                block_idx + 1,
                num_blocks,
                block_m
            );
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
        "genetic_variance": args.genetic_variance,
        "causal_block_density": args.causal_block_density,
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
