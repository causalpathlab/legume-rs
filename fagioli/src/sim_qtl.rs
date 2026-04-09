use anyhow::Result;
use clap::Args;
use log::info;

use data_beans::sparse_io::SparseIoBackend;
use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};
use fagioli::io::sim_output::{write_sim_qtl_outputs, SimQtlOutputParams};
use fagioli::simulation::{
    sample_cell_type_fractions, simulate_factor_model, GeneticArchitectureParams, ScPhenotypeParams,
};

#[derive(Args, Debug, Clone)]
pub struct SimQtlArgs {
    // ── Input / Output ───────────────────────────────────────────────────
    #[arg(long, help = "PLINK BED file prefix (without .bed/.bim/.fam)")]
    pub bed_prefix: Box<str>,

    #[arg(short, long, help = "Output prefix for all generated files")]
    pub output: Box<str>,

    #[arg(
        long,
        default_value = "parquet",
        help = "Output format: parquet or tsv"
    )]
    pub format: Box<str>,

    // ── Genomic region ───────────────────────────────────────────────────
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
        help = "GFF/GTF file for gene annotations (overrides --num-genes)"
    )]
    pub gff_file: Option<Box<str>>,

    #[arg(
        long,
        default_value = "1000",
        help = "Number of genes to simulate (ignored if --gff-file)"
    )]
    pub num_genes: usize,

    #[arg(long, default_value = "5", help = "Number of cell types")]
    pub num_cell_types: usize,

    #[arg(
        long,
        default_value = "10",
        help = "Latent factors for gene-gene correlations"
    )]
    pub num_factors: usize,

    #[arg(
        long,
        default_value = "1.0",
        help = "Gene loading std dev (factor model)"
    )]
    pub gene_loading_std: f32,

    #[arg(
        long,
        default_value = "1.0",
        help = "Factor score std dev (factor model)"
    )]
    pub factor_score_std: f32,

    // ── Genetic architecture ─────────────────────────────────────────────
    #[arg(
        long,
        default_value = "0.4",
        help = "Proportion of genes with eQTL effects"
    )]
    pub eqtl_gene_proportion: f32,

    #[arg(
        long,
        default_value = "0.6",
        help = "Of eQTL genes, proportion with shared causal variants"
    )]
    pub shared_eqtl_proportion: f32,

    #[arg(
        long,
        default_value = "0.4",
        help = "Of eQTL genes, proportion with cell-type-specific variants"
    )]
    pub independent_eqtl_proportion: f32,

    #[arg(long, default_value = "1", help = "Shared causal SNPs per gene")]
    pub num_shared_causal_per_gene: usize,

    #[arg(
        long,
        default_value = "1",
        help = "Cell-type-specific causal SNPs per gene"
    )]
    pub num_independent_causal_per_gene: usize,

    #[arg(
        long,
        default_value = "0.4",
        help = "Heritability (genetic variance proportion)"
    )]
    pub genetic_variance: f32,

    #[arg(long, default_value = "1000000", help = "Cis window size in bp")]
    pub cis_window: u64,

    // ── Single-cell parameters ───────────────────────────────────────────
    #[arg(
        long,
        default_value = "0.5",
        help = "Variance proportion from cell type vs individual"
    )]
    pub pve_cell_type: f32,

    #[arg(
        long,
        default_value = "1000",
        help = "Mean cells per individual (Poisson-sampled)"
    )]
    pub mean_cells_per_individual: f64,

    #[arg(
        long,
        default_value = "5000",
        help = "Sequencing depth per cell (total UMI)"
    )]
    pub depth_per_cell: f64,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1.0",
        help = "Dirichlet alpha for cell type fractions (comma-separated)"
    )]
    pub dirichlet_alpha: Vec<f32>,

    #[arg(
        long,
        default_value = "zarr",
        help = "Sparse matrix backend: zarr or hdf5"
    )]
    pub backend: Box<str>,
}

const SEED_FACTOR_MODEL: u64 = 100;
const SEED_SC: u64 = 300;

fn read_genotypes(args: &SimQtlArgs) -> Result<fagioli::genotype::GenotypeMatrix> {
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

pub fn sim_qtl(args: &SimQtlArgs) -> Result<()> {
    use fagioli::io::gene_annotations::load_gtf;
    use fagioli::simulation::simulate_gene_annotations;

    let use_parquet = match args.format.to_lowercase().as_str() {
        "parquet" => true,
        "tsv" => false,
        _ => anyhow::bail!("format must be 'parquet' or 'tsv'"),
    };

    info!("Starting simulation");
    let geno = read_genotypes(args)?;
    let n = geno.num_individuals();

    // Gene annotations (from GFF or simulated)
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

    // Cell type fractions
    let cell_frac =
        sample_cell_type_fractions(n, args.num_cell_types, &args.dirichlet_alpha, args.seed)?;

    // Factor model
    let factor_model = simulate_factor_model(
        genes.genes.len(),
        args.num_factors,
        args.num_cell_types,
        args.gene_loading_std,
        args.factor_score_std,
        args.seed + SEED_FACTOR_MODEL,
    )?;

    // Per-gene eQTL + individual linear model → single-cell counts
    let arch_params = GeneticArchitectureParams {
        eqtl_gene_proportion: args.eqtl_gene_proportion,
        shared_eqtl_proportion: args.shared_eqtl_proportion,
        independent_eqtl_proportion: args.independent_eqtl_proportion,
        num_shared_causal_per_gene: args.num_shared_causal_per_gene,
        num_independent_causal_per_gene: args.num_independent_causal_per_gene,
        cis_window: args.cis_window,
    };

    let sc_params = ScPhenotypeParams {
        mean_cells_per_individual: args.mean_cells_per_individual,
        depth_per_cell: args.depth_per_cell,
        h2_genetic: args.genetic_variance,
        pve_cell_type: args.pve_cell_type,
    };

    let (sc_data, gene_effects) = fagioli::simulation::sample_sc_counts(
        &genes,
        &geno,
        &factor_model,
        &cell_frac,
        &arch_params,
        &sc_params,
        args.seed + SEED_SC,
    )?;

    // Write outputs
    info!("Writing outputs to {}", args.output);
    let backend = match args.backend.to_lowercase().as_str() {
        "zarr" => SparseIoBackend::Zarr,
        "hdf5" => SparseIoBackend::HDF5,
        _ => anyhow::bail!("backend must be 'zarr' or 'hdf5'"),
    };
    let output_params = SimQtlOutputParams {
        output_prefix: &args.output,
        backend,
        num_cell_types: args.num_cell_types,
        num_factors: args.num_factors,
    };
    write_sim_qtl_outputs(
        &output_params,
        &geno,
        &genes,
        &cell_frac,
        &factor_model,
        &gene_effects,
        &sc_data,
        use_parquet,
    )?;

    // Write simulation-specific parameters
    let param_file = format!("{}.parameters.json", args.output);
    let params = serde_json::json!({
        "num_individuals": geno.num_individuals(),
        "num_snps": geno.num_snps(),
        "num_genes": genes.genes.len(),
        "num_cell_types": args.num_cell_types,
        "num_factors": args.num_factors,
        "num_cells": sc_data.num_cells,
        "num_nonzero_counts": sc_data.triplets.len(),
        "eqtl_gene_proportion": args.eqtl_gene_proportion,
        "shared_eqtl_proportion": args.shared_eqtl_proportion,
        "independent_eqtl_proportion": args.independent_eqtl_proportion,
        "num_shared_causal_per_gene": args.num_shared_causal_per_gene,
        "num_independent_causal_per_gene": args.num_independent_causal_per_gene,
        "genetic_variance": args.genetic_variance,
        "pve_cell_type": args.pve_cell_type,
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

    info!("Done!");
    Ok(())
}
