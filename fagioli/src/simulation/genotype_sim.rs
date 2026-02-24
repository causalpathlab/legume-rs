use anyhow::{bail, Result};
use nalgebra::DMatrix;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Poisson, Uniform};

use crate::genotype::GenotypeMatrix;

/// Parameters for Wright-Fisher forward simulation.
pub struct WrightFisherParams {
    /// Number of diploid individuals to sample from the final generation
    pub num_individuals: usize,
    /// Number of initial segregating sites
    pub num_snps: usize,
    /// Effective population size (diploid)
    pub ne: usize,
    /// Number of discrete generations to simulate
    pub num_generations: usize,
    /// Per-bp per-generation recombination rate
    pub recombination_rate: f64,
    /// Base-pair spacing between adjacent SNPs
    pub snp_spacing: u64,
    /// Minimum initial minor allele frequency
    pub maf_min: f32,
    /// Maximum initial minor allele frequency
    pub maf_max: f32,
    /// Chromosome label
    pub chromosome: String,
    /// Random seed
    pub seed: u64,
}

/// Run a Wright-Fisher forward simulation and return a `GenotypeMatrix`.
///
/// Algorithm:
/// 1. Initialize `2*ne` haplotypes with per-site allele freq ~ Uniform(maf_min, maf_max).
/// 2. For each generation: sample parents, recombine, replace population.
/// 3. Sample `num_individuals` diploids, sum haplotype pairs → dosage 0/1/2.
/// 4. Filter fixed (monomorphic) sites.
pub fn simulate_wright_fisher(params: &WrightFisherParams) -> Result<GenotypeMatrix> {
    if params.num_individuals > params.ne {
        bail!(
            "num_individuals ({}) cannot exceed ne ({})",
            params.num_individuals,
            params.ne
        );
    }
    if params.maf_min <= 0.0 || params.maf_max > 0.5 || params.maf_min > params.maf_max {
        bail!(
            "Invalid MAF range: [{}, {}]",
            params.maf_min,
            params.maf_max
        );
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);
    let m = params.num_snps;
    let n_hap = 2 * params.ne; // number of haplotypes

    // ── Step 1: Initialize haplotypes ────────────────────────────────────
    let maf_dist = Uniform::new(params.maf_min as f64, params.maf_max as f64)?;
    let mut haplotypes: Vec<Vec<u8>> = vec![vec![0u8; m]; n_hap];

    for site in 0..m {
        let freq = maf_dist.sample(&mut rng);
        for hap in haplotypes.iter_mut() {
            hap[site] = if rng.random_bool(freq) { 1 } else { 0 };
        }
    }

    // ── Step 2: Evolve for T generations ─────────────────────────────────
    let genome_length_bp = if m > 1 {
        (m as f64 - 1.0) * params.snp_spacing as f64
    } else {
        0.0
    };
    let expected_crossovers = params.recombination_rate * genome_length_bp;
    let parent_dist = Uniform::new(0usize, params.ne)?;

    for _gen in 0..params.num_generations {
        let mut new_haplotypes: Vec<Vec<u8>> = Vec::with_capacity(n_hap);

        for _ in 0..params.ne {
            // Two parents, each contributes one recombined gamete
            for _ in 0..2 {
                let parent = parent_dist.sample(&mut rng);
                let hap_a = &haplotypes[2 * parent];
                let hap_b = &haplotypes[2 * parent + 1];
                let gamete = recombine(
                    hap_a,
                    hap_b,
                    m,
                    expected_crossovers,
                    params.snp_spacing,
                    &mut rng,
                );
                new_haplotypes.push(gamete);
            }
        }

        haplotypes = new_haplotypes;
    }

    // ── Step 3: Sample individuals and compute dosages ───────────────────
    let n = params.num_individuals;
    // Use the first n diploids (they're already random from WF sampling)
    let mut dosages = DMatrix::<f32>::zeros(n, m);

    for i in 0..n {
        let h1 = &haplotypes[2 * i];
        let h2 = &haplotypes[2 * i + 1];
        for j in 0..m {
            dosages[(i, j)] = (h1[j] + h2[j]) as f32;
        }
    }

    // ── Step 4: Filter fixed sites ───────────────────────────────────────
    let mut keep: Vec<usize> = Vec::new();
    for j in 0..m {
        let col = dosages.column(j);
        let sum: f32 = col.iter().sum();
        let freq = sum / (2.0 * n as f32);
        if freq > 0.0 && freq < 1.0 {
            keep.push(j);
        }
    }

    let m_kept = keep.len();
    let mut filtered = DMatrix::<f32>::zeros(n, m_kept);
    let mut positions = Vec::with_capacity(m_kept);
    let mut snp_ids = Vec::with_capacity(m_kept);

    for (new_j, &old_j) in keep.iter().enumerate() {
        filtered.set_column(new_j, &dosages.column(old_j));
        let pos = 1 + old_j as u64 * params.snp_spacing;
        positions.push(pos);
        snp_ids.push(format!("{}:{}", params.chromosome, pos).into());
    }

    let individual_ids: Vec<Box<str>> = (0..n).map(|i| format!("ind{}", i).into()).collect();

    Ok(GenotypeMatrix {
        genotypes: filtered,
        individual_ids,
        snp_ids,
        chromosomes: vec![params.chromosome.clone().into(); m_kept],
        positions,
        allele1: vec!["A".into(); m_kept],
        allele2: vec!["T".into(); m_kept],
    })
}

/// Recombine two parental haplotypes into a single gamete.
fn recombine(
    hap_a: &[u8],
    hap_b: &[u8],
    num_sites: usize,
    expected_crossovers: f64,
    snp_spacing: u64,
    rng: &mut rand::rngs::StdRng,
) -> Vec<u8> {
    if num_sites <= 1 || expected_crossovers < 1e-12 {
        // No recombination possible
        return if rng.random_bool(0.5) {
            hap_a.to_vec()
        } else {
            hap_b.to_vec()
        };
    }

    // Number of crossovers ~ Poisson(expected_crossovers)
    let n_crossovers = if expected_crossovers > 0.0 {
        let pois = Poisson::new(expected_crossovers).unwrap();
        pois.sample(rng) as usize
    } else {
        0
    };

    if n_crossovers == 0 {
        return if rng.random_bool(0.5) {
            hap_a.to_vec()
        } else {
            hap_b.to_vec()
        };
    }

    // Sample crossover positions in bp, then convert to site indices
    let genome_len = (num_sites as u64 - 1) * snp_spacing;
    let pos_dist = Uniform::new(1u64, genome_len + 1).unwrap();
    let mut xover_sites: Vec<usize> = (0..n_crossovers)
        .map(|_| {
            let bp = pos_dist.sample(rng);
            // Convert bp position to the site index after which the crossover occurs
            (bp / snp_spacing) as usize
        })
        .collect();
    xover_sites.sort_unstable();
    xover_sites.dedup();

    // Walk along sites, switching source at each crossover
    let mut gamete = vec![0u8; num_sites];
    let mut use_a = rng.random_bool(0.5);
    let mut xover_idx = 0;

    for site in 0..num_sites {
        while xover_idx < xover_sites.len() && xover_sites[xover_idx] < site {
            use_a = !use_a;
            xover_idx += 1;
        }
        gamete[site] = if use_a { hap_a[site] } else { hap_b[site] };
    }

    gamete
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_simulation() {
        let params = WrightFisherParams {
            num_individuals: 50,
            num_snps: 100,
            ne: 100,
            num_generations: 10,
            recombination_rate: 1e-8,
            snp_spacing: 1000,
            maf_min: 0.05,
            maf_max: 0.5,
            chromosome: "1".to_string(),
            seed: 42,
        };

        let geno = simulate_wright_fisher(&params).unwrap();
        assert_eq!(geno.num_individuals(), 50);
        assert!(geno.num_snps() > 0);
        assert!(geno.num_snps() <= 100);

        // All values should be 0, 1, or 2
        for i in 0..geno.num_individuals() {
            for j in 0..geno.num_snps() {
                let v = geno.genotypes[(i, j)];
                assert!(v == 0.0 || v == 1.0 || v == 2.0, "unexpected value {}", v);
            }
        }

        // No monomorphic sites
        for j in 0..geno.num_snps() {
            let col = geno.genotypes.column(j);
            let sum: f32 = col.iter().sum();
            let freq = sum / (2.0 * geno.num_individuals() as f32);
            assert!(freq > 0.0 && freq < 1.0, "fixed site at col {}", j);
        }
    }

    #[test]
    fn test_deterministic_seed() {
        let params = WrightFisherParams {
            num_individuals: 20,
            num_snps: 50,
            ne: 50,
            num_generations: 5,
            recombination_rate: 1e-8,
            snp_spacing: 1000,
            maf_min: 0.1,
            maf_max: 0.5,
            chromosome: "1".to_string(),
            seed: 123,
        };

        let g1 = simulate_wright_fisher(&params).unwrap();
        let g2 = simulate_wright_fisher(&params).unwrap();

        assert_eq!(g1.genotypes, g2.genotypes);
    }
}
