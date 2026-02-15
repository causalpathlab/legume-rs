use anyhow::Result;
use log::info;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use super::gene_annotations::GeneAnnotations;
use genomic_data::gff::GeneId;

/// Genetic architecture parameters for eQTL simulation
#[derive(Debug, Clone)]
pub struct GeneticArchitectureParams {
    /// Proportion of genes with detectable eQTL (0.0 to 1.0)
    pub eqtl_gene_proportion: f32,

    /// Of genes WITH eQTL, proportion with shared causal variants (0.0 to 1.0)
    pub shared_eqtl_proportion: f32,

    /// Of genes WITH eQTL, proportion with cell-type-specific causal variants (0.0 to 1.0)
    pub independent_eqtl_proportion: f32,

    /// Number of shared causal SNPs (for genes with shared eQTL)
    pub num_shared_causal_per_gene: usize,

    /// Number of independent causal SNPs per cell type (for genes with independent eQTL)
    pub num_independent_causal_per_gene: usize,

    /// Total genetic variance (h²_genetic)
    pub genetic_variance: f32,

    /// Cis window size (bp)
    pub cis_window: u64,
}

impl Default for GeneticArchitectureParams {
    fn default() -> Self {
        Self {
            eqtl_gene_proportion: 0.4,
            shared_eqtl_proportion: 0.6,
            independent_eqtl_proportion: 0.4,
            num_shared_causal_per_gene: 1,
            num_independent_causal_per_gene: 1,
            genetic_variance: 0.4,
            cis_window: 1_000_000, // 1Mb
        }
    }
}

/// Causal SNP with effect sizes per cell type
#[derive(Debug, Clone)]
pub struct CausalSnp {
    pub snp_idx: usize, // Global SNP index
    pub position: u64,
    pub effect_sizes: Vec<f32>, // Per cell type (K values)
}

/// Gene-level eQTL effects (both shared and independent)
#[derive(Debug, Clone)]
pub struct GeneEqtlEffects {
    pub gene_id: GeneId,
    pub gene_idx: usize,

    /// Shared causal variants (across all cell types)
    pub shared_causal_snps: Vec<CausalSnp>,

    /// Independent causal variants (per cell type)
    pub independent_causal_snps: Vec<Vec<CausalSnp>>, // K vectors

    pub num_cell_types: usize,
}

impl GeneEqtlEffects {
    /// Create an empty GeneEqtlEffects (no eQTL for this gene)
    fn empty(gene_id: GeneId, gene_idx: usize, num_cell_types: usize) -> Self {
        Self {
            gene_id,
            gene_idx,
            shared_causal_snps: vec![],
            independent_causal_snps: vec![vec![]; num_cell_types],
            num_cell_types,
        }
    }
}

/// All genes' eQTL effects
#[derive(Debug, Clone)]
pub struct ScEqtlEffects {
    pub genes: Vec<GeneEqtlEffects>,
    pub num_genes: usize,
    pub num_cell_types: usize,
}

impl ScEqtlEffects {
    /// Get eQTL effects for a specific gene
    pub fn get_gene(&self, gene_idx: usize) -> Option<&GeneEqtlEffects> {
        self.genes.get(gene_idx)
    }

    /// Count genes with any eQTL
    pub fn num_egenes(&self) -> usize {
        self.genes
            .iter()
            .filter(|g| {
                !g.shared_causal_snps.is_empty()
                    || g.independent_causal_snps
                        .iter()
                        .any(|snps| !snps.is_empty())
            })
            .count()
    }

    /// Count genes with shared eQTL
    pub fn num_shared_egenes(&self) -> usize {
        self.genes
            .iter()
            .filter(|g| !g.shared_causal_snps.is_empty())
            .count()
    }

    /// Count genes with independent eQTL
    pub fn num_independent_egenes(&self) -> usize {
        self.genes
            .iter()
            .filter(|g| {
                g.independent_causal_snps
                    .iter()
                    .any(|snps| !snps.is_empty())
            })
            .count()
    }
}

/// Sample eQTL effects for all genes in parallel
pub fn sample_sc_eqtl_effects(
    genes: &GeneAnnotations,
    snp_positions: &[u64],
    snp_chromosomes: &[Box<str>],
    num_cell_types: usize,
    params: &GeneticArchitectureParams,
    seed: u64,
) -> Result<ScEqtlEffects> {
    info!("Sampling eQTL effects for {} genes", genes.genes.len());
    info!("  eQTL gene proportion: {:.2}", params.eqtl_gene_proportion);
    info!(
        "  Shared eQTL proportion: {:.2}",
        params.shared_eqtl_proportion
    );
    info!(
        "  Independent eQTL proportion: {:.2}",
        params.independent_eqtl_proportion
    );

    // Parallel across genes!
    let gene_effects: Vec<GeneEqtlEffects> = genes
        .genes
        .par_iter()
        .enumerate()
        .map(|(gene_idx, gene)| {
            let gene_seed = seed.wrapping_add(gene_idx as u64);
            sample_gene_eqtl_effects(
                gene_idx,
                &gene.gene_id,
                genes,
                snp_positions,
                snp_chromosomes,
                num_cell_types,
                params,
                gene_seed,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let effects = ScEqtlEffects {
        genes: gene_effects,
        num_genes: genes.genes.len(),
        num_cell_types,
    };

    info!("  Total eGenes: {}", effects.num_egenes());
    info!("  Shared eGenes: {}", effects.num_shared_egenes());
    info!("  Independent eGenes: {}", effects.num_independent_egenes());

    Ok(effects)
}

/// Sample eQTL effects for a single gene
#[allow(clippy::too_many_arguments)]
fn sample_gene_eqtl_effects(
    gene_idx: usize,
    gene_id: &GeneId,
    genes: &GeneAnnotations,
    snp_positions: &[u64],
    snp_chromosomes: &[Box<str>],
    num_cell_types: usize,
    params: &GeneticArchitectureParams,
    seed: u64,
) -> Result<GeneEqtlEffects> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Step 1: Does this gene have any eQTL?
    let has_eqtl = rng.random_bool(params.eqtl_gene_proportion as f64);
    if !has_eqtl {
        return Ok(GeneEqtlEffects::empty(
            gene_id.clone(),
            gene_idx,
            num_cell_types,
        ));
    }

    // Step 2: Get SNPs in cis window
    let cis_snps = genes.cis_snp_indices(gene_idx, snp_positions, snp_chromosomes);

    if cis_snps.is_empty() {
        return Ok(GeneEqtlEffects::empty(
            gene_id.clone(),
            gene_idx,
            num_cell_types,
        ));
    }

    // Step 3: Sample architecture type
    let has_shared = rng.random_bool(params.shared_eqtl_proportion as f64);
    let has_independent = rng.random_bool(params.independent_eqtl_proportion as f64);

    let num_shared = if has_shared {
        params.num_shared_causal_per_gene.min(cis_snps.len())
    } else {
        0
    };

    let num_independent = if has_independent {
        params.num_independent_causal_per_gene.min(cis_snps.len())
    } else {
        0
    };

    // Step 4: Variance allocation
    let total_causal = num_shared + num_independent;
    if total_causal == 0 {
        return Ok(GeneEqtlEffects::empty(
            gene_id.clone(),
            gene_idx,
            num_cell_types,
        ));
    }

    let shared_var = params.genetic_variance * (num_shared as f32 / total_causal as f32);
    let indep_var = params.genetic_variance * (num_independent as f32 / total_causal as f32);

    // Step 5: Sample shared causal SNPs using index sampling (avoids full shuffle)
    let shared_sampled = rand::seq::index::sample(&mut rng, cis_snps.len(), num_shared);
    let shared_set: std::collections::HashSet<usize> = shared_sampled.iter().collect();

    let shared_causal_snps = if num_shared > 0 {
        // Variance per shared effect: σ²_shared / (K × S)
        let effect_std = (shared_var / (num_cell_types as f32 * num_shared as f32)).sqrt();
        let normal = Normal::new(0.0, effect_std as f64)?;

        shared_sampled
            .iter()
            .map(|idx| {
                let snp_idx = cis_snps[idx];
                let effect_sizes: Vec<f32> = (0..num_cell_types)
                    .map(|_| normal.sample(&mut rng) as f32)
                    .collect();
                CausalSnp {
                    snp_idx,
                    position: snp_positions[snp_idx],
                    effect_sizes,
                }
            })
            .collect()
    } else {
        vec![]
    };

    // Step 6: Sample independent causal SNPs (from non-shared cis SNPs)
    let available_for_indep: Vec<usize> = (0..cis_snps.len())
        .filter(|i| !shared_set.contains(i))
        .map(|i| cis_snps[i])
        .collect();

    let independent_causal_snps = if num_independent > 0 && !available_for_indep.is_empty() {
        // Variance per independent effect: σ²_independent / I
        let effect_std = (indep_var / num_independent as f32).sqrt();
        let normal = Normal::new(0.0, effect_std as f64)?;
        let n_pick = num_independent.min(available_for_indep.len());

        (0..num_cell_types)
            .map(|k| {
                // Use index::sample to avoid cloning the full vec per cell type
                let sampled = rand::seq::index::sample(&mut rng, available_for_indep.len(), n_pick);

                sampled
                    .iter()
                    .map(|idx| {
                        let snp_idx = available_for_indep[idx];
                        let mut effect_sizes = vec![0.0; num_cell_types];
                        effect_sizes[k] = normal.sample(&mut rng) as f32;
                        CausalSnp {
                            snp_idx,
                            position: snp_positions[snp_idx],
                            effect_sizes,
                        }
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![vec![]; num_cell_types]
    };

    Ok(GeneEqtlEffects {
        gene_id: gene_id.clone(),
        gene_idx,
        shared_causal_snps,
        independent_causal_snps,
        num_cell_types,
    })
}

#[cfg(test)]
mod tests {
    use super::super::gene_annotations::simulate_gene_annotations;
    use super::*;

    #[test]
    fn test_sample_eqtl_effects() {
        let genes = simulate_gene_annotations(100, "22", 20_000_000, 30_000_000, 1_000_000, 42);

        // Simulate SNP positions
        let snp_positions: Vec<u64> = (0..1000).map(|i| 20_000_000 + i * 10_000).collect();
        let snp_chromosomes: Vec<Box<str>> = vec![Box::from("22"); 1000];

        let params = GeneticArchitectureParams {
            eqtl_gene_proportion: 0.5,
            shared_eqtl_proportion: 0.6,
            independent_eqtl_proportion: 0.4,
            num_shared_causal_per_gene: 1,
            num_independent_causal_per_gene: 1,
            genetic_variance: 0.4,
            cis_window: 1_000_000,
        };

        let effects =
            sample_sc_eqtl_effects(&genes, &snp_positions, &snp_chromosomes, 3, &params, 42)
                .unwrap();

        assert_eq!(effects.num_genes, 100);
        assert_eq!(effects.num_cell_types, 3);

        // Should have some eGenes
        let num_egenes = effects.num_egenes();
        assert!(num_egenes > 0 && num_egenes < 100);

        println!("eGenes: {}/{}", num_egenes, effects.num_genes);
        println!("Shared: {}", effects.num_shared_egenes());
        println!("Independent: {}", effects.num_independent_egenes());
    }

    #[test]
    fn test_no_eqtl_genes() {
        let genes = simulate_gene_annotations(10, "22", 20_000_000, 30_000_000, 1_000_000, 42);
        let snp_positions: Vec<u64> = vec![25_000_000];
        let snp_chromosomes: Vec<Box<str>> = vec![Box::from("22")];

        let params = GeneticArchitectureParams {
            eqtl_gene_proportion: 0.0, // No eQTL
            ..Default::default()
        };

        let effects =
            sample_sc_eqtl_effects(&genes, &snp_positions, &snp_chromosomes, 2, &params, 42)
                .unwrap();

        assert_eq!(effects.num_egenes(), 0);
    }

    #[test]
    fn test_single_cell_type() {
        let genes = simulate_gene_annotations(20, "22", 20_000_000, 30_000_000, 1_000_000, 42);
        let snp_positions: Vec<u64> = (0..200).map(|i| 20_000_000 + i * 50_000).collect();
        let snp_chromosomes: Vec<Box<str>> = vec![Box::from("22"); 200];

        let params = GeneticArchitectureParams {
            eqtl_gene_proportion: 1.0,
            shared_eqtl_proportion: 0.5,
            independent_eqtl_proportion: 0.5,
            num_shared_causal_per_gene: 1,
            num_independent_causal_per_gene: 1,
            genetic_variance: 0.3,
            cis_window: 1_000_000,
        };

        let effects = sample_sc_eqtl_effects(
            &genes,
            &snp_positions,
            &snp_chromosomes,
            1, // single cell type
            &params,
            42,
        )
        .unwrap();

        assert_eq!(effects.num_cell_types, 1);
        // With 1 cell type, shared and independent are effectively equivalent
        assert!(effects.num_egenes() > 0);
    }

    #[test]
    fn test_parallel_determinism() {
        let genes = simulate_gene_annotations(50, "22", 20_000_000, 30_000_000, 1_000_000, 42);
        let snp_positions: Vec<u64> = (0..500).map(|i| 20_000_000 + i * 20_000).collect();
        let snp_chromosomes: Vec<Box<str>> = vec![Box::from("22"); 500];
        let params = GeneticArchitectureParams::default();

        let e1 = sample_sc_eqtl_effects(&genes, &snp_positions, &snp_chromosomes, 3, &params, 42)
            .unwrap();
        let e2 = sample_sc_eqtl_effects(&genes, &snp_positions, &snp_chromosomes, 3, &params, 42)
            .unwrap();

        assert_eq!(e1.num_egenes(), e2.num_egenes());
        assert_eq!(e1.num_shared_egenes(), e2.num_shared_egenes());
        assert_eq!(e1.num_independent_egenes(), e2.num_independent_egenes());

        // Check individual gene effects match
        for (g1, g2) in e1.genes.iter().zip(e2.genes.iter()) {
            assert_eq!(g1.shared_causal_snps.len(), g2.shared_causal_snps.len());
            for (s1, s2) in g1
                .shared_causal_snps
                .iter()
                .zip(g2.shared_causal_snps.iter())
            {
                assert_eq!(s1.snp_idx, s2.snp_idx);
                assert_eq!(s1.effect_sizes, s2.effect_sizes);
            }
        }
    }
}
