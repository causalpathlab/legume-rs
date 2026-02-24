use anyhow::Result;
use clap::Args;
use log::info;

use fagioli::genotype::bed_writer::write_plink;
use fagioli::simulation::genotype_sim::{simulate_wright_fisher, WrightFisherParams};

#[derive(Args, Debug, Clone)]
pub struct SimGenoArgs {
    #[arg(short, long, help = "Output prefix for PLINK files (.bed/.bim/.fam)")]
    pub output: String,

    #[arg(long, default_value = "200", help = "Number of diploid individuals")]
    pub num_individuals: usize,

    #[arg(
        long,
        default_value = "2000",
        help = "Number of initial segregating sites"
    )]
    pub num_snps: usize,

    #[arg(long, default_value = "1", help = "Chromosome label")]
    pub chromosome: String,

    #[arg(long, default_value = "1000", help = "Effective population size")]
    pub ne: usize,

    #[arg(long, default_value = "100", help = "Number of generations")]
    pub num_generations: usize,

    #[arg(
        long,
        default_value = "1e-8",
        help = "Recombination rate per bp per generation"
    )]
    pub recombination_rate: f64,

    #[arg(long, default_value = "1000", help = "Base-pair spacing between SNPs")]
    pub snp_spacing: u64,

    #[arg(
        long,
        default_value = "0.05",
        help = "Minimum initial minor allele frequency"
    )]
    pub maf_min: f32,

    #[arg(
        long,
        default_value = "0.5",
        help = "Maximum initial minor allele frequency"
    )]
    pub maf_max: f32,

    #[arg(long, default_value = "42", help = "Random seed")]
    pub seed: u64,
}

pub fn sim_geno(args: &SimGenoArgs) -> Result<()> {
    info!("Starting sim-geno");

    let params = WrightFisherParams {
        num_individuals: args.num_individuals,
        num_snps: args.num_snps,
        ne: args.ne,
        num_generations: args.num_generations,
        recombination_rate: args.recombination_rate,
        snp_spacing: args.snp_spacing,
        maf_min: args.maf_min,
        maf_max: args.maf_max,
        chromosome: args.chromosome.clone(),
        seed: args.seed,
    };

    info!(
        "Wright-Fisher: N_e={}, T={}, n={}, M={}, r={:.1e}, spacing={}bp",
        params.ne,
        params.num_generations,
        params.num_individuals,
        params.num_snps,
        params.recombination_rate,
        params.snp_spacing,
    );

    let geno = simulate_wright_fisher(&params)?;
    info!(
        "Simulated {} individuals x {} segregating SNPs",
        geno.num_individuals(),
        geno.num_snps()
    );

    write_plink(&args.output, &geno)?;
    info!(
        "Wrote PLINK files: {}.bed, {}.bim, {}.fam",
        args.output, args.output, args.output
    );

    info!("sim-geno completed successfully");
    Ok(())
}
