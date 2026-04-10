use anyhow::Result;
use clap::Args;
use log::info;
use nalgebra::DMatrix;
use rust_htslib::tpool::ThreadPool;

use rayon::prelude::*;

use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};
use fagioli::io::gene_annotations::{load_bed_annotations, load_gtf};
use fagioli::simulation::{
    generate_confounder_matrix, generate_gene_expressions, generate_mediated_phenotype,
    sample_confounder_mixing_y, sample_mediation_effects, simulate_gene_annotations,
    split_discovery_replication, subset_dvector, subset_rows, ConfounderParams, ExpressionParams,
    MediationEffectParams, PhenotypeParams,
};
use fagioli::summary_stats::{
    compute_block_ld_scores, compute_block_sumstats, compute_yty_diagonal, create_uniform_blocks,
    estimate_ld_blocks, load_ld_blocks_from_file, write_confounders, write_gene_table,
    write_ld_blocks, write_mediation_ground_truth, EqtlSumstatWriter, GeneTableInput, LdBlock,
    LdBlockParams, LdScoreWriter, SnpWriterParams, SumstatRecord, SumstatWriter,
};
use matrix_util::dmatrix_util::subset_columns;

#[derive(Args, Debug, Clone)]
pub struct SimMediationArgs {
    // ── Input ────────────────────────────────────────────────────────────
    #[arg(long, help = "PLINK BED file prefix (without .bed/.bim/.fam)")]
    pub bed_prefix: Box<str>,

    #[arg(long, help = "Chromosome to simulate from")]
    pub chromosome: Box<str>,

    #[arg(long, help = "Left genomic position bound (bp)")]
    pub left_bound: Option<u64>,

    #[arg(long, help = "Right genomic position bound (bp)")]
    pub right_bound: Option<u64>,

    #[arg(long, help = "Max individuals to use from genotype file")]
    pub max_individuals: Option<usize>,

    #[arg(long, default_value = "42", help = "Random seed")]
    pub seed: u64,

    // ── Gene model ───────────────────────────────────────────────────────
    #[arg(
        long,
        help = "GFF/GTF or BED file for gene annotations (overrides --num-genes)"
    )]
    pub gff_file: Option<Box<str>>,

    #[arg(
        long,
        default_value = "100",
        help = "Number of genes to simulate (ignored if --gff-file)"
    )]
    pub num_genes: usize,

    #[arg(
        long,
        default_value = "1000000",
        help = "Cis window in bp around each gene TSS"
    )]
    pub cis_window: u64,

    #[arg(long, default_value = "3", help = "Number of cis-eQTL SNPs per gene")]
    pub n_eqtl_per_gene: usize,

    // ── Mediation architecture ───────────────────────────────────────────
    #[arg(
        long,
        default_value = "10",
        help = "Number of mediator genes (SNP → M → Y)"
    )]
    pub num_mediator_genes: usize,

    #[arg(long, help = "Of mediator genes, how many are observed (default: all)")]
    pub num_observed_mediators: Option<usize>,

    #[arg(
        long,
        default_value = "0.3",
        help = "Expression heritability from cis-eQTLs"
    )]
    pub h2_eqtl: f32,

    #[arg(
        long,
        default_value = "0.2",
        help = "PVE of Y from mediated path (through gene expression)"
    )]
    pub h2_mediated: f32,

    #[arg(
        long,
        default_value = "0.0",
        help = "PVE of Y from direct SNP effects (horizontal pleiotropy)"
    )]
    pub h2_direct: f32,

    // ── Confounder parameters ────────────────────────────────────────────
    #[arg(
        long,
        default_value = "0.1",
        help = "PVE of gene expression from confounders"
    )]
    pub h2_conf_m: f32,

    #[arg(long, default_value = "0.1", help = "PVE of Y from confounders")]
    pub h2_conf_y: f32,

    #[arg(
        long,
        default_value = "5",
        help = "Number of confounder columns (0 = none)"
    )]
    pub num_confounders: usize,

    #[arg(
        long,
        default_value = "3",
        help = "Hidden factors generating confounders"
    )]
    pub num_hidden_factors: usize,

    // ── Collider bias ────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "0",
        help = "Non-mediator genes with confounder-correlated expression (colliders)"
    )]
    pub num_collider_genes: usize,

    #[arg(
        long,
        default_value = "0.8",
        help = "Correlation between collider gene confounder effect and Y confounder effect"
    )]
    pub collider_confounder_correlation: f32,

    // ── Winner's curse ───────────────────────────────────────────────────
    #[arg(
        long,
        help = "Discovery cohort size for eQTL (enables discovery/replication split)"
    )]
    pub n_eqtl_discovery: Option<usize>,

    #[arg(
        long,
        default_value = "5e-8",
        help = "P-value threshold for eQTL instrument selection in discovery"
    )]
    pub eqtl_pvalue_threshold: f64,

    // ── LD block parameters ──────────────────────────────────────────────
    #[arg(
        long,
        help = "External LD block file (BED: chr, start, end)",
        long_help = "External LD block file in BED format (chr, start, end).\n\
            If omitted, blocks are estimated from the genotype data via Nystrom + rSVD."
    )]
    pub ld_block_file: Option<Box<str>>,

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
    #[arg(short, long, help = "Output prefix for all generated files")]
    pub output: Box<str>,
}

const SEED_CONFOUNDERS: u64 = 100;
const SEED_PHENOTYPE: u64 = 200;
const SEED_EFFECTS: u64 = 300;
const SEED_EXPRESSION: u64 = 400;
const SEED_GAMMA_Y: u64 = 500;
const SEED_SPLIT: u64 = 600;

/// Remap local cis-SNP indices back to global SNP indices.
fn remap_sumstat_indices(records: Vec<SumstatRecord>, cis_snps: &[usize]) -> Vec<SumstatRecord> {
    records
        .into_iter()
        .map(|r| SumstatRecord {
            snp_idx: cis_snps[r.snp_idx],
            ..r
        })
        .collect()
}

pub fn sim_mediation(args: &SimMediationArgs) -> Result<()> {
    info!("Starting sim-mediation");

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
                min_block_snps: Some(args.min_block_snps),
                max_block_snps: Some(args.max_block_snps),
                seed: args.seed,
            },
        )?
    } else {
        info!("Too few SNPs for block estimation, using single block");
        create_uniform_blocks(m, m, &geno.positions, &geno.chromosomes)
    };

    let num_blocks = blocks.len();
    info!("Using {} LD blocks", num_blocks);

    let blocks_file = format!("{}.ld_blocks.bed.gz", args.output);
    write_ld_blocks(&blocks_file, &blocks, Some(&tpool))?;

    // ── Step 3: Gene annotations ─────────────────────────────────────────
    let genes = if let Some(ref gff_file) = args.gff_file {
        info!("Reading gene annotations from {}", gff_file);
        if gff_file.ends_with(".bed") || gff_file.ends_with(".bed.gz") {
            load_bed_annotations(gff_file, Some(&args.chromosome), args.cis_window)?
        } else {
            load_gtf(
                gff_file,
                Some(&args.chromosome),
                args.left_bound,
                args.right_bound,
                args.cis_window,
            )?
        }
    } else {
        info!("Simulating {} gene annotations", args.num_genes);
        let region_start = args.left_bound.unwrap_or(0);
        let region_end = args.right_bound.unwrap_or(region_start + 10_000_000);
        simulate_gene_annotations(
            args.num_genes,
            &args.chromosome,
            region_start,
            region_end,
            args.cis_window,
            args.seed,
        )
    };
    let num_genes = genes.genes.len();
    info!("Using {} genes", num_genes);

    // ── Step 4: Sample mediation effects ─────────────────────────────────
    let num_observed_mediators = args
        .num_observed_mediators
        .unwrap_or(args.num_mediator_genes);

    let effects = sample_mediation_effects(&MediationEffectParams {
        genes: &genes,
        snp_positions: &geno.positions,
        snp_chromosomes: &geno.chromosomes,
        n_eqtl_per_gene: args.n_eqtl_per_gene,
        num_causal: args.num_mediator_genes,
        num_observed_causal: num_observed_mediators,
        num_collider: args.num_collider_genes,
        seed: args.seed + SEED_EFFECTS,
    })?;

    // ── Step 5: Generate shared confounder matrix ────────────────────────
    // pve_confounders controls the signal strength of the raw confounder matrix C;
    // the per-component h² (h2_conf_m, h2_conf_y) are applied after standardization,
    // so we need C strong enough for whichever layer uses more confounder variance.
    let conf_params = ConfounderParams {
        num_confounders: args.num_confounders,
        num_hidden_factors: args.num_hidden_factors,
        pve_confounders: args.h2_conf_m.max(args.h2_conf_y),
    };
    let confounder_matrix =
        generate_confounder_matrix(n, &conf_params, args.seed + SEED_CONFOUNDERS)?;

    // ── Step 6: Generate γ_y (before expressions, for collider correlation)
    let gamma_y = sample_confounder_mixing_y(args.num_confounders, args.seed + SEED_GAMMA_Y);

    // ── Step 7: Generate gene expressions ────────────────────────────────
    let expressions = generate_gene_expressions(&ExpressionParams {
        genotypes: &geno.genotypes,
        effects: &effects,
        confounders: &confounder_matrix,
        gamma_y: &gamma_y,
        h2_eqtl: args.h2_eqtl,
        h2_conf_m: args.h2_conf_m,
        collider_correlation: args.collider_confounder_correlation,
        seed: args.seed + SEED_EXPRESSION,
    })?;

    // ── Step 8: Generate complex trait Y ─────────────────────────────────
    let phenotype = generate_mediated_phenotype(&PhenotypeParams {
        expressions: &expressions,
        effects: &effects,
        genotypes: &geno.genotypes,
        confounders: &confounder_matrix,
        gamma_y: &gamma_y,
        h2_mediated: args.h2_mediated,
        h2_direct: args.h2_direct,
        h2_conf_y: args.h2_conf_y,
        seed: args.seed + SEED_PHENOTYPE,
    })?;

    let y_matrix = DMatrix::from_column_slice(n, 1, phenotype.as_slice());

    info!(
        "Phenotype Y: {} individuals, variance={:.3}",
        n,
        phenotype.dot(&phenotype) / n as f32,
    );

    // ── Step 9: Write ground truth and gene table ────────────────────────
    let gene_ids: Vec<Box<str>> = genes
        .genes
        .iter()
        .map(|g| Box::from(g.gene_id.to_string()))
        .collect();
    let gene_names: Vec<Option<Box<str>>> =
        genes.genes.iter().map(|g| g.gene_name.clone()).collect();
    let gene_chromosomes: Vec<Box<str>> =
        genes.genes.iter().map(|g| g.chromosome.clone()).collect();
    let gene_tss: Vec<u64> = genes.genes.iter().map(|g| g.tss).collect();

    let gt_file = format!("{}.ground_truth.bed.gz", args.output);
    write_mediation_ground_truth(
        &gt_file,
        &effects,
        &gene_ids,
        &geno.snp_ids,
        &geno.chromosomes,
        &geno.positions,
        Some(&tpool),
    )?;

    let num_cis_snps: Vec<usize> = effects.iter().map(|eff| eff.num_cis_snps).collect();

    let gene_file = format!("{}.genes.bed.gz", args.output);
    write_gene_table(
        &gene_file,
        &GeneTableInput {
            effects: &effects,
            gene_ids: &gene_ids,
            gene_names: &gene_names,
            gene_chromosomes: &gene_chromosomes,
            gene_tss: &gene_tss,
            num_cis_snps: &num_cis_snps,
        },
        Some(&tpool),
    )?;

    if args.num_confounders > 0 {
        let conf_file = format!("{}.confounders.tsv.gz", args.output);
        write_confounders(&conf_file, &confounder_matrix)?;
    }

    // ── Step 10: GWAS summary statistics (SNP → Y) ──────────────────────
    info!("Computing GWAS summary statistics");

    let gwas_file = format!("{}.gwas.sumstats.bed.gz", args.output);
    let snp_params = SnpWriterParams {
        path: &gwas_file,
        snp_ids: &geno.snp_ids,
        chromosomes: &geno.chromosomes,
        positions: &geno.positions,
        allele1: &geno.allele1,
        allele2: &geno.allele2,
        num_individuals: n,
        tpool: Some(&tpool),
    };
    let mut gwas_writer = SumstatWriter::new(&snp_params)?;

    let ld_score_file = format!("{}.ld_scores.bed.gz", args.output);
    let mut ld_writer = LdScoreWriter::new(
        &ld_score_file,
        &geno.snp_ids,
        &geno.chromosomes,
        &geno.positions,
        Some(&tpool),
    )?;

    let yty_diag = compute_yty_diagonal(&y_matrix);

    let batch_size = num_threads as usize * 2;
    for batch_start in (0..num_blocks).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_blocks);

        let batch_results: Vec<_> = (batch_start..batch_end)
            .into_par_iter()
            .map(|block_idx| {
                let block = &blocks[block_idx];
                let block_m = block.num_snps();
                let x_block = geno
                    .genotypes
                    .columns(block.snp_start, block_m)
                    .clone_owned();

                let sumstats =
                    compute_block_sumstats(&x_block, &y_matrix, &yty_diag, block.snp_start);
                let ld_scores = compute_block_ld_scores(&x_block, block.snp_start);

                (block_idx, sumstats, ld_scores)
            })
            .collect();

        for (block_idx, sumstats, ld_scores) in batch_results {
            gwas_writer.write_block(&sumstats)?;
            ld_writer.write_block(&ld_scores)?;

            if (block_idx + 1) % 10 == 0 || block_idx + 1 == num_blocks {
                info!(
                    "GWAS: processed block {}/{} ({} SNPs)",
                    block_idx + 1,
                    num_blocks,
                    blocks[block_idx].num_snps(),
                );
            }
        }
    }

    gwas_writer.finish()?;
    ld_writer.finish()?;

    // ── Step 11: eQTL summary statistics ─────────────────────────────────
    let observed_genes: Vec<usize> = effects
        .iter()
        .enumerate()
        .filter(|(_, eff)| eff.is_observed && !eff.eqtl_snp_indices.is_empty())
        .map(|(i, _)| i)
        .collect();

    if let Some(n_disc) = args.n_eqtl_discovery {
        // ── Winner's curse mode: discovery / replication split ────────
        info!(
            "Winner's curse mode: splitting {} individuals into discovery ({}) / replication ({})",
            n,
            n_disc,
            n.saturating_sub(n_disc),
        );

        let (disc_idx, rep_idx) = split_discovery_replication(n, n_disc, args.seed + SEED_SPLIT);

        let disc_file = format!("{}.eqtl.discovery.sumstats.bed.gz", args.output);
        let rep_file = format!("{}.eqtl.replication.sumstats.bed.gz", args.output);

        let mut disc_writer = EqtlSumstatWriter::new(&SnpWriterParams {
            path: &disc_file,
            snp_ids: &geno.snp_ids,
            chromosomes: &geno.chromosomes,
            positions: &geno.positions,
            allele1: &geno.allele1,
            allele2: &geno.allele2,
            num_individuals: disc_idx.len(),
            tpool: Some(&tpool),
        })?;
        let mut rep_writer = EqtlSumstatWriter::new(&SnpWriterParams {
            path: &rep_file,
            snp_ids: &geno.snp_ids,
            chromosomes: &geno.chromosomes,
            positions: &geno.positions,
            allele1: &geno.allele1,
            allele2: &geno.allele2,
            num_individuals: rep_idx.len(),
            tpool: Some(&tpool),
        })?;

        let mut n_selected = 0usize;

        for &g in &observed_genes {
            let cis_snps = genes.cis_snp_indices(g, &geno.positions, &geno.chromosomes);
            if cis_snps.is_empty() {
                continue;
            }

            let x_cis_full = subset_columns(&geno.genotypes, cis_snps.iter().copied())?;

            let x_disc = subset_rows(&x_cis_full, &disc_idx);
            let x_rep = subset_rows(&x_cis_full, &rep_idx);
            let m_g_disc = subset_dvector(&expressions[g], &disc_idx);
            let m_g_rep = subset_dvector(&expressions[g], &rep_idx);

            let m_disc_mat = DMatrix::from_column_slice(disc_idx.len(), 1, m_g_disc.as_slice());
            let m_rep_mat = DMatrix::from_column_slice(rep_idx.len(), 1, m_g_rep.as_slice());

            let mty_disc = compute_yty_diagonal(&m_disc_mat);
            let mty_rep = compute_yty_diagonal(&m_rep_mat);

            let disc_remapped = remap_sumstat_indices(
                compute_block_sumstats(&x_disc, &m_disc_mat, &mty_disc, 0),
                &cis_snps,
            );
            let rep_remapped = remap_sumstat_indices(
                compute_block_sumstats(&x_rep, &m_rep_mat, &mty_rep, 0),
                &cis_snps,
            );

            n_selected += disc_remapped
                .iter()
                .filter(|r| r.pvalue < args.eqtl_pvalue_threshold)
                .count();

            disc_writer.write_gene_block(&disc_remapped, g, &gene_ids[g])?;
            rep_writer.write_gene_block(&rep_remapped, g, &gene_ids[g])?;
        }

        disc_writer.finish()?;
        rep_writer.finish()?;

        info!(
            "eQTL: {} instruments selected at p < {} in discovery",
            n_selected, args.eqtl_pvalue_threshold,
        );
    } else {
        // ── Standard mode: single eQTL sumstats file ─────────────────
        info!(
            "Computing eQTL sumstats for {} observed genes",
            observed_genes.len(),
        );

        let eqtl_file = format!("{}.eqtl.sumstats.bed.gz", args.output);
        let mut eqtl_writer = EqtlSumstatWriter::new(&SnpWriterParams {
            path: &eqtl_file,
            snp_ids: &geno.snp_ids,
            chromosomes: &geno.chromosomes,
            positions: &geno.positions,
            allele1: &geno.allele1,
            allele2: &geno.allele2,
            num_individuals: n,
            tpool: Some(&tpool),
        })?;

        for &g in &observed_genes {
            let cis_snps = genes.cis_snp_indices(g, &geno.positions, &geno.chromosomes);
            if cis_snps.is_empty() {
                continue;
            }

            let x_cis = subset_columns(&geno.genotypes, cis_snps.iter().copied())?;
            let m_g = DMatrix::from_column_slice(n, 1, expressions[g].as_slice());
            let mty_diag = compute_yty_diagonal(&m_g);
            let remapped = remap_sumstat_indices(
                compute_block_sumstats(&x_cis, &m_g, &mty_diag, 0),
                &cis_snps,
            );

            eqtl_writer.write_gene_block(&remapped, g, &gene_ids[g])?;
        }

        eqtl_writer.finish()?;
    }

    // ── Step 12: Write parameters ────────────────────────────────────────
    let param_file = format!("{}.parameters.json", args.output);
    let params = serde_json::json!({
        "command": "sim-mediation",
        "num_individuals": n,
        "num_snps": m,
        "num_genes": num_genes,
        "num_blocks": num_blocks,
        "num_mediator_genes": args.num_mediator_genes,
        "num_observed_mediators": num_observed_mediators,
        "num_collider_genes": args.num_collider_genes,
        "collider_confounder_correlation": args.collider_confounder_correlation,
        "n_eqtl_per_gene": args.n_eqtl_per_gene,
        "cis_window": args.cis_window,
        "h2_eqtl": args.h2_eqtl,
        "h2_mediated": args.h2_mediated,
        "h2_direct": args.h2_direct,
        "h2_conf_m": args.h2_conf_m,
        "h2_conf_y": args.h2_conf_y,
        "num_confounders": args.num_confounders,
        "num_hidden_factors": args.num_hidden_factors,
        "n_eqtl_discovery": args.n_eqtl_discovery,
        "eqtl_pvalue_threshold": args.eqtl_pvalue_threshold,
        "ld_block_file": args.ld_block_file,
        "seed": args.seed,
        "bed_prefix": args.bed_prefix,
        "chromosome": args.chromosome,
    });
    std::fs::write(&param_file, serde_json::to_string_pretty(&params)?)?;
    info!("Wrote parameters: {}", param_file);

    info!("sim-mediation completed successfully");
    Ok(())
}
