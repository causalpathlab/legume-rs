use anyhow::Result;
use clap::Args;
use log::info;
use matrix_util::traits::SampleOps;
use nalgebra::DMatrix;

use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};
use fagioli::simulation::{
    sample_cell_type_fractions, sample_cell_type_genetic_effects, simulate_phenotypes,
    PhenotypeSimulationParams,
};

#[derive(Args, Debug, Clone)]
pub struct SimulationArgs {
    /// PLINK BED file prefix (without .bed)
    #[arg(long)]
    pub bed_prefix: String,

    /// Output prefix
    #[arg(short, long)]
    pub output: String,

    /// Output format: parquet or tsv
    #[arg(long, default_value = "parquet")]
    pub format: String,

    /// Chromosome (required to prevent memory overflow)
    #[arg(long)]
    pub chromosome: String,

    /// Left position bound (optional)
    #[arg(long)]
    pub left_bound: Option<u64>,

    /// Right position bound (optional)
    #[arg(long)]
    pub right_bound: Option<u64>,

    /// Max individuals
    #[arg(long)]
    pub max_individuals: Option<usize>,

    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,

    // === Phase 2: Gene-level eQTL and single-cell simulation ===
    /// Simulation mode: 'aggregated' (Phase 1) or 'single-cell' (Phase 2)
    #[arg(long, default_value = "single-cell")]
    pub mode: String,

    /// GFF/GTF file for gene annotations (required for single-cell mode)
    #[arg(long)]
    pub gff_file: Option<String>,

    /// Number of genes to simulate (if not using GFF)
    #[arg(long, default_value = "1000")]
    pub num_genes: usize,

    /// Number of cell types
    #[arg(long, default_value = "5")]
    pub num_cell_types: usize,

    /// Number of latent factors for gene correlations
    #[arg(long, default_value = "10")]
    pub num_factors: usize,

    /// Gene loading standard deviation (factor model)
    #[arg(long, default_value = "1.0")]
    pub gene_loading_std: f32,

    /// Factor score standard deviation (factor model)
    #[arg(long, default_value = "1.0")]
    pub factor_score_std: f32,

    /// Proportion of genes with eQTL
    #[arg(long, default_value = "0.4")]
    pub eqtl_gene_proportion: f32,

    /// Of eGenes, proportion with shared causal variants
    #[arg(long, default_value = "0.6")]
    pub shared_eqtl_proportion: f32,

    /// Of eGenes, proportion with independent causal variants
    #[arg(long, default_value = "0.4")]
    pub independent_eqtl_proportion: f32,

    /// Number of shared causal SNPs per gene
    #[arg(long, default_value = "1")]
    pub num_shared_causal_per_gene: usize,

    /// Number of independent causal SNPs per gene (per cell type)
    #[arg(long, default_value = "1")]
    pub num_independent_causal_per_gene: usize,

    /// Genetic variance (h²_genetic)
    #[arg(long, default_value = "0.4")]
    pub genetic_variance: f32,

    /// Cis window size (bp)
    #[arg(long, default_value = "1000000")]
    pub cis_window: u64,

    /// Mean cells per individual (Poisson)
    #[arg(long, default_value = "1000")]
    pub mean_cells_per_individual: f64,

    /// Sequencing depth per cell (total UMI)
    #[arg(long, default_value = "5000")]
    pub depth_per_cell: f64,

    /// Dirichlet alpha for cell type fractions (comma-separated or single value)
    #[arg(long, value_delimiter = ',', default_value = "1.0")]
    pub dirichlet_alpha: Vec<f32>,

    /// Sparse matrix backend: zarr or hdf5
    #[arg(long, default_value = "zarr")]
    pub backend: String,

    // === Phase 1 backward compatibility ===
    /// [Phase 1 only] Number of shared causal variants
    #[arg(long, default_value = "5")]
    pub num_shared_causal: usize,

    /// [Phase 1 only] Number of independent causal variants per cell type
    #[arg(long, default_value = "5")]
    pub num_independent_causal: usize,

    /// [Phase 1 only] SNP heritability
    #[arg(long, default_value = "0.5")]
    pub h2_snp: f32,

    /// [Phase 1 only] Covariate heritability
    #[arg(long, default_value = "0.2")]
    pub h2_covariate: f32,

    /// [Phase 1 only] Number of covariates
    #[arg(long, default_value = "2")]
    pub num_covariates: usize,
}

// Seed offsets for reproducible sub-simulations within Phase 2
const SEED_OFFSET_FACTOR_MODEL: u64 = 100;
const SEED_OFFSET_EQTL: u64 = 200;
const SEED_OFFSET_SC_PHENOTYPES: u64 = 300;

/// Read genotypes from PLINK BED file with region filtering
fn read_genotypes(args: &SimulationArgs) -> Result<fagioli::genotype::GenotypeMatrix> {
    let region_str = match (args.left_bound, args.right_bound) {
        (Some(left), Some(right)) => format!("chr={}, {}..{}", args.chromosome, left, right),
        (Some(left), None) => format!("chr={}, {}+", args.chromosome, left),
        (None, Some(right)) => format!("chr={}, ..{}", args.chromosome, right),
        (None, None) => format!("chr={}", args.chromosome),
    };
    info!(
        "Reading genotypes from {} ({})",
        args.bed_prefix, region_str
    );

    let mut reader = BedReader::new(&args.bed_prefix)?;
    let region = GenomicRegion::new(
        Some(args.chromosome.clone()),
        args.left_bound,
        args.right_bound,
    );

    let geno = reader.read(args.max_individuals, Some(region))?;
    info!(
        "Loaded {} individuals × {} SNPs",
        geno.num_individuals(),
        geno.num_snps()
    );
    Ok(geno)
}

pub fn sim_eqtl(args: &SimulationArgs) -> Result<()> {
    info!("Starting simulation (mode: {})", args.mode);

    // Validate format
    let use_parquet = match args.format.to_lowercase().as_str() {
        "parquet" => true,
        "tsv" => false,
        _ => anyhow::bail!("format must be 'parquet' or 'tsv'"),
    };

    // Branch based on mode
    match args.mode.to_lowercase().as_str() {
        "single-cell" | "sc" => sim_eqtl_phase2(args, use_parquet),
        "aggregated" | "phase1" => sim_eqtl_phase1(args, use_parquet),
        _ => anyhow::bail!("mode must be 'single-cell' or 'aggregated'"),
    }
}

/// Phase 1: Individual-level aggregated phenotypes
fn sim_eqtl_phase1(args: &SimulationArgs, use_parquet: bool) -> Result<()> {
    info!("Running Phase 1: Aggregated phenotype simulation");

    let geno = read_genotypes(args)?;
    let n = geno.num_individuals();
    let m = geno.num_snps();

    if args.num_shared_causal + args.num_independent_causal > m {
        anyhow::bail!(
            "num_shared_causal ({}) + num_independent_causal ({}) > num_snps ({})",
            args.num_shared_causal,
            args.num_independent_causal,
            m
        );
    }

    // Sample cell fractions
    info!("Sampling cell type fractions");
    let cell_frac =
        sample_cell_type_fractions(n, args.num_cell_types, &args.dirichlet_alpha, args.seed)?;

    // Sample genetic effects
    info!(
        "Sampling genetic effects ({} shared, {} independent)",
        args.num_shared_causal, args.num_independent_causal
    );
    let gen_eff = sample_cell_type_genetic_effects(
        m,
        args.num_cell_types,
        args.num_shared_causal,
        args.num_independent_causal,
        args.h2_snp,
        args.seed + 1,
    )?;

    // Generate covariates
    let cov = if args.num_covariates > 0 {
        info!("Generating {} covariates", args.num_covariates);
        let mut c = DMatrix::zeros(n, args.num_covariates);
        c.set_column(0, &DMatrix::from_element(n, 1, 1.0).column(0)); // intercept
        if args.num_covariates > 1 {
            let rand_cov = DMatrix::rnorm(n, args.num_covariates - 1);
            for j in 0..(args.num_covariates - 1) {
                c.set_column(j + 1, &rand_cov.column(j));
            }
        }
        Some(c)
    } else {
        None
    };

    // Simulate phenotypes
    info!("Simulating phenotypes");
    let pheno = simulate_phenotypes(PhenotypeSimulationParams {
        genotypes: &geno.genotypes,
        cell_fractions: &cell_frac,
        genetic_effects: &gen_eff,
        covariates: cov.as_ref(),
        h2_genetic: args.h2_snp,
        h2_covariate: args.h2_covariate,
        seed: args.seed + 2,
    })?;

    // Write outputs
    info!(
        "Writing outputs to {} (format: {})",
        args.output, args.format
    );
    write_outputs(args, &geno, &cell_frac, &gen_eff, &pheno, use_parquet)?;

    info!("Done!");
    Ok(())
}

/// Phase 2: Gene-level eQTL with single-cell counts
fn sim_eqtl_phase2(args: &SimulationArgs, use_parquet: bool) -> Result<()> {
    use fagioli::simulation::{
        generate_sc_phenotypes, load_gtf, sample_sc_eqtl_effects, simulate_factor_model,
        simulate_gene_annotations, GeneticArchitectureParams, ScPhenotypeParams,
    };

    info!("Running Phase 2: Gene-level eQTL with single-cell simulation");

    let geno = read_genotypes(args)?;
    let n = geno.num_individuals();

    // Read or simulate gene annotations
    let genes = if let Some(ref gff_file) = args.gff_file {
        info!("Reading gene annotations from {}", gff_file);
        load_gtf(
            gff_file,
            Some(&args.chromosome),
            args.left_bound,
            args.right_bound,
            args.cis_window,
        )?
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

    info!("Using {} genes", genes.genes.len());

    // Sample cell type fractions
    info!("Sampling cell type fractions");
    let cell_frac =
        sample_cell_type_fractions(n, args.num_cell_types, &args.dirichlet_alpha, args.seed)?;

    // Step 1: Simulate factor model
    info!("Simulating factor model");
    let factor_model = simulate_factor_model(
        genes.genes.len(),
        args.num_factors,
        args.num_cell_types,
        args.gene_loading_std,
        args.factor_score_std,
        args.seed + SEED_OFFSET_FACTOR_MODEL,
    )?;

    // Step 2: Sample gene-level eQTL effects
    info!("Sampling gene-level eQTL effects");
    let eqtl_params = GeneticArchitectureParams {
        eqtl_gene_proportion: args.eqtl_gene_proportion,
        shared_eqtl_proportion: args.shared_eqtl_proportion,
        independent_eqtl_proportion: args.independent_eqtl_proportion,
        num_shared_causal_per_gene: args.num_shared_causal_per_gene,
        num_independent_causal_per_gene: args.num_independent_causal_per_gene,
        genetic_variance: args.genetic_variance,
        cis_window: args.cis_window,
    };

    let eqtl_effects = sample_sc_eqtl_effects(
        &genes,
        &geno.positions,
        &geno.chromosomes,
        args.num_cell_types,
        &eqtl_params,
        args.seed + SEED_OFFSET_EQTL,
    )?;

    // Step 3: Generate single-cell phenotypes
    info!("Generating single-cell phenotypes");
    let sc_params = ScPhenotypeParams {
        mean_cells_per_individual: args.mean_cells_per_individual,
        depth_per_cell: args.depth_per_cell,
        overdispersion: 0.1,
    };

    let sc_data = generate_sc_phenotypes(
        &factor_model,
        &eqtl_effects,
        &geno.genotypes,
        &cell_frac,
        &sc_params,
        args.seed + SEED_OFFSET_SC_PHENOTYPES,
    )?;

    // Write outputs
    info!("Writing Phase 2 outputs to {}", args.output);
    write_phase2_outputs(
        args,
        &geno,
        &genes,
        &cell_frac,
        &factor_model,
        &eqtl_effects,
        &sc_data,
        use_parquet,
    )?;

    info!("Done!");
    Ok(())
}

fn write_outputs(
    args: &SimulationArgs,
    geno: &fagioli::genotype::GenotypeMatrix,
    cell_frac: &DMatrix<f32>,
    gen_eff: &fagioli::simulation::CellTypeGeneticEffects,
    pheno: &fagioli::simulation::SimulatedPhenotypes,
    use_parquet: bool,
) -> Result<()> {
    use matrix_util::common_io::open_buf_writer;
    use matrix_util::traits::IoOps;
    use std::io::Write;

    let ext = if use_parquet { "parquet" } else { "tsv" };

    // Generate cell type names
    let cell_type_names: Vec<Box<str>> = (0..args.num_cell_types)
        .map(|i| Box::from(format!("cell_type_{}", i)))
        .collect();

    // Phenotypes (N × K)
    let pheno_file = format!("{}.phenotypes.{}", args.output, ext);
    if use_parquet {
        pheno.phenotypes.to_parquet_with_names(
            &pheno_file,
            (Some(&geno.individual_ids), Some("individual")),
            Some(&cell_type_names),
        )?;
    } else {
        pheno.phenotypes.to_tsv(&pheno_file)?;
    }
    info!("Wrote phenotypes: {}", pheno_file);

    // Cell fractions (N × K)
    let frac_file = format!("{}.cell_fractions.{}", args.output, ext);
    if use_parquet {
        cell_frac.to_parquet_with_names(
            &frac_file,
            (Some(&geno.individual_ids), Some("individual")),
            Some(&cell_type_names),
        )?;
    } else {
        cell_frac.to_tsv(&frac_file)?;
    }
    info!("Wrote cell fractions: {}", frac_file);

    // Causal SNPs (TSV.GZ)
    let causal_file = format!("{}.causal_snps.tsv.gz", args.output);
    let mut writer = open_buf_writer(&causal_file)?;
    writeln!(
        writer,
        "type\tcell_type\tsnp_id\tchromosome\tposition\teffect_size"
    )?;

    // Write shared causal SNPs
    for ct in 0..gen_eff.num_cell_types {
        for (j, &snp_idx) in gen_eff.shared_causal_indices.iter().enumerate() {
            writeln!(
                writer,
                "shared\t{}\t{}\t{}\t{}\t{}",
                ct,
                geno.snp_ids[snp_idx],
                geno.chromosomes[snp_idx],
                geno.positions[snp_idx],
                gen_eff.shared_effect_sizes[(ct, j)]
            )?;
        }
    }

    // Write independent causal SNPs
    for ct in 0..gen_eff.num_cell_types {
        for (j, &snp_idx) in gen_eff.independent_causal_indices[ct].iter().enumerate() {
            writeln!(
                writer,
                "independent\t{}\t{}\t{}\t{}\t{}",
                ct,
                geno.snp_ids[snp_idx],
                geno.chromosomes[snp_idx],
                geno.positions[snp_idx],
                gen_eff.independent_effect_sizes[(ct, j)]
            )?;
        }
    }
    writer.flush()?;
    info!("Wrote causal SNPs: {}", causal_file);

    // Parameters (JSON)
    let param_file = format!("{}.parameters.json", args.output);
    let params = serde_json::json!({
        "num_individuals": geno.num_individuals(),
        "num_snps": geno.num_snps(),
        "num_cell_types": args.num_cell_types,
        "num_shared_causal": args.num_shared_causal,
        "num_independent_causal": args.num_independent_causal,
        "h2_snp": args.h2_snp,
        "h2_covariate": args.h2_covariate,
        "seed": args.seed,
        "format": args.format,
    });
    std::fs::write(&param_file, serde_json::to_string_pretty(&params)?)?;
    info!("Wrote parameters: {}", param_file);

    Ok(())
}

fn write_phase2_outputs(
    args: &SimulationArgs,
    geno: &fagioli::genotype::GenotypeMatrix,
    genes: &fagioli::simulation::GeneAnnotations,
    cell_frac: &DMatrix<f32>,
    factor_model: &fagioli::simulation::FactorModel,
    eqtl_effects: &fagioli::simulation::ScEqtlEffects,
    sc_data: &fagioli::simulation::ScCountData,
    use_parquet: bool,
) -> Result<()> {
    use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
    use matrix_util::common_io::open_buf_writer;
    use matrix_util::traits::IoOps;
    use std::io::Write;

    let ext = if use_parquet { "parquet" } else { "tsv" };

    // 1. Single-cell count matrix (backend)
    let backend = match args.backend.to_lowercase().as_str() {
        "zarr" => SparseIoBackend::Zarr,
        "hdf5" => SparseIoBackend::HDF5,
        _ => anyhow::bail!("backend must be 'zarr' or 'hdf5'"),
    };

    let backend_ext = match backend {
        SparseIoBackend::Zarr => "zarr",
        SparseIoBackend::HDF5 => "h5",
    };

    let backend_file = format!("{}.counts.{}", args.output, backend_ext);
    info!("Creating sparse count matrix backend: {}", backend_file);

    let mtx_shape = (sc_data.num_genes, sc_data.num_cells, sc_data.triplets.len());
    let mut sparse_backend = create_sparse_from_triplets(
        &sc_data.triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&backend),
    )?;

    // Add gene names (rows) with format: {ensembl}_{name} if gene_name available
    let gene_names: Vec<Box<str>> = genes
        .genes
        .iter()
        .map(|g| {
            if let Some(ref name) = g.gene_name {
                Box::from(format!("{}_{}", g.gene_id, name))
            } else {
                Box::from(g.gene_id.to_string())
            }
        })
        .collect();
    sparse_backend.register_row_names_vec(&gene_names);

    // Add cell IDs (columns) with format: {barcode}@{individual}
    let cell_names: Vec<Box<str>> = (0..sc_data.num_cells)
        .map(|i| {
            let ind_id = &geno.individual_ids[sc_data.cell_individuals[i]];
            Box::from(format!("cell_{}@{}", i, ind_id))
        })
        .collect();
    sparse_backend.register_column_names_vec(&cell_names);

    info!(
        "Created backend with {} genes × {} cells: {}",
        sc_data.num_genes, sc_data.num_cells, backend_file
    );

    // 2. Cell annotations (TSV.GZ)
    let cell_anno_file = format!("{}.cells.tsv.gz", args.output);
    let mut writer = open_buf_writer(&cell_anno_file)?;
    writeln!(writer, "cell_id\tindividual_id\tcell_type")?;
    for (cell_idx, (&ind_id, &ct)) in sc_data
        .cell_individuals
        .iter()
        .zip(&sc_data.cell_types)
        .enumerate()
    {
        let cell_id = format!("cell_{}@{}", cell_idx, geno.individual_ids[ind_id]);
        writeln!(
            writer,
            "{}\t{}\tcell_type_{}",
            cell_id, geno.individual_ids[ind_id], ct
        )?;
    }
    writer.flush()?;
    info!("Wrote cell annotations: {}", cell_anno_file);

    // 2b. Cell-to-individual mapping (TSV.GZ)
    let mapping_file = format!("{}.cell_to_individual.tsv.gz", args.output);
    let mut writer = open_buf_writer(&mapping_file)?;
    writeln!(writer, "cell_id\tindividual_id\tindividual_index")?;
    for (cell_idx, &ind_idx) in sc_data.cell_individuals.iter().enumerate() {
        let cell_id = format!("cell_{}@{}", cell_idx, geno.individual_ids[ind_idx]);
        writeln!(
            writer,
            "{}\t{}\t{}",
            cell_id, geno.individual_ids[ind_idx], ind_idx
        )?;
    }
    writer.flush()?;
    info!("Wrote cell-to-individual mapping: {}", mapping_file);

    // 3. Gene annotations (TSV.GZ)
    let gene_anno_file = format!("{}.genes.tsv.gz", args.output);
    let mut writer = open_buf_writer(&gene_anno_file)?;
    writeln!(
        writer,
        "gene_idx\tgene_id\tgene_name\tchromosome\ttss\tstrand"
    )?;
    for (idx, gene) in genes.genes.iter().enumerate() {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}",
            idx,
            gene.gene_id,
            gene.gene_name.as_ref().map(|s| s.as_ref()).unwrap_or("NA"),
            gene.chromosome,
            gene.tss,
            gene.strand
        )?;
    }
    writer.flush()?;
    info!("Wrote gene annotations: {}", gene_anno_file);

    // 4. Cell type fractions (parquet/tsv)
    let cell_type_names: Vec<Box<str>> = (0..args.num_cell_types)
        .map(|i| Box::from(format!("cell_type_{}", i)))
        .collect();

    let frac_file = format!("{}.cell_fractions.{}", args.output, ext);
    if use_parquet {
        cell_frac.to_parquet_with_names(
            &frac_file,
            (Some(&geno.individual_ids), Some("individual")),
            Some(&cell_type_names),
        )?;
    } else {
        cell_frac.to_tsv(&frac_file)?;
    }
    info!("Wrote cell fractions: {}", frac_file);

    // 5. Factor model (parquet/tsv)
    let gene_ids: Vec<Box<str>> = genes
        .genes
        .iter()
        .map(|g| Box::from(g.gene_id.to_string()))
        .collect();
    let factor_names: Vec<Box<str>> = (0..args.num_factors)
        .map(|i| Box::from(format!("factor_{}", i)))
        .collect();

    let loadings_file = format!("{}.gene_loadings.{}", args.output, ext);
    if use_parquet {
        factor_model.gene_loadings.to_parquet_with_names(
            &loadings_file,
            (Some(&gene_ids), Some("gene")),
            Some(&factor_names),
        )?;
    } else {
        factor_model.gene_loadings.to_tsv(&loadings_file)?;
    }
    info!("Wrote gene loadings: {}", loadings_file);

    let scores_file = format!("{}.factor_celltype.{}", args.output, ext);
    if use_parquet {
        factor_model.factor_celltype.to_parquet_with_names(
            &scores_file,
            (Some(&factor_names), Some("factor")),
            Some(&cell_type_names),
        )?;
    } else {
        factor_model.factor_celltype.to_tsv(&scores_file)?;
    }
    info!("Wrote factor-celltype scores: {}", scores_file);

    // 6. eQTL effects (TSV.GZ)
    let eqtl_file = format!("{}.eqtl_effects.tsv.gz", args.output);
    let mut writer = open_buf_writer(&eqtl_file)?;
    writeln!(writer, "gene_idx\tgene_id\teqtl_type\tcell_type\tsnp_idx\tsnp_id\tchromosome\tposition\teffect_size")?;

    for gene_effects in &eqtl_effects.genes {
        let gene_idx = gene_effects.gene_idx;
        let gene_id = &gene_effects.gene_id;

        // Shared causal SNPs
        for causal_snp in &gene_effects.shared_causal_snps {
            for (ct, &effect) in causal_snp.effect_sizes.iter().enumerate() {
                if effect.abs() > 1e-10 {
                    writeln!(
                        writer,
                        "{}\t{}\tshared\t{}\t{}\t{}\t{}\t{}\t{}",
                        gene_idx,
                        gene_id,
                        ct,
                        causal_snp.snp_idx,
                        geno.snp_ids[causal_snp.snp_idx],
                        geno.chromosomes[causal_snp.snp_idx],
                        causal_snp.position,
                        effect
                    )?;
                }
            }
        }

        // Independent causal SNPs
        for (ct, snps) in gene_effects.independent_causal_snps.iter().enumerate() {
            for causal_snp in snps {
                let effect = causal_snp.effect_sizes[ct];
                if effect.abs() > 1e-10 {
                    writeln!(
                        writer,
                        "{}\t{}\tindependent\t{}\t{}\t{}\t{}\t{}\t{}",
                        gene_idx,
                        gene_id,
                        ct,
                        causal_snp.snp_idx,
                        geno.snp_ids[causal_snp.snp_idx],
                        geno.chromosomes[causal_snp.snp_idx],
                        causal_snp.position,
                        effect
                    )?;
                }
            }
        }
    }
    writer.flush()?;
    info!("Wrote eQTL effects: {}", eqtl_file);

    // 7. Parameters (JSON)
    let param_file = format!("{}.parameters.json", args.output);
    let params = serde_json::json!({
        "mode": "single-cell",
        "num_individuals": geno.num_individuals(),
        "num_snps": geno.num_snps(),
        "num_genes": genes.genes.len(),
        "num_cell_types": args.num_cell_types,
        "num_factors": args.num_factors,
        "num_cells": sc_data.num_cells,
        "num_nonzero_counts": sc_data.triplets.len(),
        "num_egenes": eqtl_effects.num_egenes(),
        "num_shared_egenes": eqtl_effects.num_shared_egenes(),
        "num_independent_egenes": eqtl_effects.num_independent_egenes(),
        "eqtl_gene_proportion": args.eqtl_gene_proportion,
        "shared_eqtl_proportion": args.shared_eqtl_proportion,
        "independent_eqtl_proportion": args.independent_eqtl_proportion,
        "num_shared_causal_per_gene": args.num_shared_causal_per_gene,
        "num_independent_causal_per_gene": args.num_independent_causal_per_gene,
        "genetic_variance": args.genetic_variance,
        "cis_window": args.cis_window,
        "mean_cells_per_individual": args.mean_cells_per_individual,
        "depth_per_cell": args.depth_per_cell,
        "gene_loading_std": args.gene_loading_std,
        "factor_score_std": args.factor_score_std,
        "backend": args.backend,
        "seed": args.seed,
        "format": args.format,
    });
    std::fs::write(&param_file, serde_json::to_string_pretty(&params)?)?;
    info!("Wrote parameters: {}", param_file);

    Ok(())
}
