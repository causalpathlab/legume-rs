use anyhow::Result;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use log::info;
use nalgebra::DMatrix;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{weighted::WeightedIndex, Distribution, Normal, Poisson};
use rayon::prelude::*;

use matrix_util::traits::MatOps;

use super::cell_type_effects::CellTypeGeneticEffects;
use super::factor_model::FactorModel;
use super::gene_annotations::GeneAnnotations;
use super::individual_linear_model::{simulate_individual_linear_model, PhenotypeSimulationParams};
use crate::genotype::GenotypeMatrix;

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Genetic architecture parameters for per-gene eQTL simulation
#[derive(Debug, Clone)]
pub struct GeneticArchitectureParams {
    /// Proportion of genes with detectable eQTL (0.0 to 1.0)
    pub eqtl_gene_proportion: f32,
    /// Of eQTL genes, proportion with shared causal variants (0.0 to 1.0)
    pub shared_eqtl_proportion: f32,
    /// Of eQTL genes, proportion with cell-type-specific causal variants (0.0 to 1.0)
    pub independent_eqtl_proportion: f32,
    /// Number of shared causal SNPs per gene
    pub num_shared_causal_per_gene: usize,
    /// Number of independent causal SNPs per gene per cell type
    pub num_independent_causal_per_gene: usize,
    /// Cis window size (bp) around TSS/TES
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
            cis_window: 1_000_000,
        }
    }
}

/// Parameters for single-cell count generation
#[derive(Debug, Clone)]
pub struct ScPhenotypeParams {
    pub mean_cells_per_individual: f64,
    pub depth_per_cell: f64,
    /// Heritability for per-gene individual linear model
    pub h2_genetic: f32,
    /// Proportion of log-rate variance from cell type identity (factor model base means)
    /// vs individual-level phenotypes. 0.0 = all individual, 1.0 = all cell type.
    pub pve_cell_type: f32,
}

impl Default for ScPhenotypeParams {
    fn default() -> Self {
        Self {
            mean_cells_per_individual: 1000.0,
            depth_per_cell: 5000.0,
            h2_genetic: 0.4,
            pve_cell_type: 0.5,
        }
    }
}

/// Single-cell count data with cell annotations
#[derive(Debug, Clone)]
pub struct ScCountData {
    pub triplets: Vec<(u64, u64, f32)>,
    pub cell_individuals: Vec<usize>,
    pub cell_types: Vec<usize>,
    pub num_genes: usize,
    pub num_cells: usize,
    pub num_cell_types: usize,
    pub num_individuals: usize,
    /// Individual-level log-rates: one N×G matrix per cell type
    pub individual_log_rates: Vec<DMatrix<f32>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-gene eQTL sampling (cis SNPs → CellTypeGeneticEffects)
// ─────────────────────────────────────────────────────────────────────────────

/// For one gene: find cis SNPs, sample causal variants, return effects.
/// Returns `None` for non-eQTL genes.
#[allow(clippy::too_many_arguments)]
fn sample_gene_effects(
    gene_idx: usize,
    genes: &GeneAnnotations,
    snp_positions: &[u64],
    snp_chromosomes: &[Box<str>],
    num_cell_types: usize,
    params: &GeneticArchitectureParams,
    h2_genetic: f32,
    seed: u64,
) -> Result<Option<CellTypeGeneticEffects>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    if !rng.random_bool(params.eqtl_gene_proportion as f64) {
        return Ok(None);
    }

    let cis_snps = genes.cis_snp_indices(gene_idx, snp_positions, snp_chromosomes);
    if cis_snps.is_empty() {
        return Ok(None);
    }

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

    let total_causal = num_shared + num_independent;
    if total_causal == 0 {
        return Ok(None);
    }

    let shared_var = h2_genetic * (num_shared as f32 / total_causal as f32);
    let indep_var = h2_genetic * (num_independent as f32 / total_causal as f32);

    // Shared causal SNPs
    let shared_sampled = rand::seq::index::sample(&mut rng, cis_snps.len(), num_shared);
    let shared_set: std::collections::HashSet<usize> = shared_sampled.iter().collect();

    let (shared_causal_indices, shared_effect_sizes) = if num_shared > 0 {
        let effect_std = (shared_var / (num_cell_types as f32 * num_shared as f32)).sqrt();
        let normal = Normal::new(0.0, effect_std as f64)?;

        let indices: Vec<usize> = shared_sampled.iter().map(|idx| cis_snps[idx]).collect();
        let effects = DMatrix::from_fn(num_cell_types, num_shared, |_ct, _j| {
            normal.sample(&mut rng) as f32
        });
        (indices, effects)
    } else {
        (Vec::new(), DMatrix::zeros(num_cell_types, 0))
    };

    // Independent causal SNPs (from remaining cis SNPs)
    let available: Vec<usize> = (0..cis_snps.len())
        .filter(|i| !shared_set.contains(i))
        .map(|i| cis_snps[i])
        .collect();

    let (independent_causal_indices, independent_effect_sizes) =
        if num_independent > 0 && !available.is_empty() {
            let effect_std = (indep_var / num_independent as f32).sqrt();
            let normal = Normal::new(0.0, effect_std as f64)?;
            let n_pick = num_independent.min(available.len());

            let mut indices_per_ct = Vec::with_capacity(num_cell_types);
            let mut effects = DMatrix::zeros(num_cell_types, n_pick);

            for ct in 0..num_cell_types {
                let sampled = rand::seq::index::sample(&mut rng, available.len(), n_pick);
                let ct_indices: Vec<usize> = sampled.iter().map(|idx| available[idx]).collect();
                for (j, _) in sampled.iter().enumerate() {
                    effects[(ct, j)] = normal.sample(&mut rng) as f32;
                }
                indices_per_ct.push(ct_indices);
            }
            (indices_per_ct, effects)
        } else {
            (
                vec![Vec::new(); num_cell_types],
                DMatrix::zeros(num_cell_types, 0),
            )
        };

    Ok(Some(CellTypeGeneticEffects {
        shared_causal_indices,
        shared_effect_sizes,
        independent_causal_indices,
        independent_effect_sizes,
        num_cell_types,
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Sample single-cell counts with per-gene eQTL model.
///
/// For each gene g:
/// 1. Find cis SNPs (TSS/TES ± cis_window), sample causal variants
/// 2. Call `simulate_individual_linear_model` → N×K phenotypes
///    - eQTL genes: genetic signal + noise
///    - Non-eQTL genes: noise only
///
/// Then per individual i, cell type k:
///   log_rate_gk = sqrt(pve_ct) * standardize(M_gk) + sqrt(1-pve_ct) * Y_g(i,k)
///   rates_gk = exp(log_rate_gk)
///
/// Returns (ScCountData, per-gene effects for ground truth output).
pub fn sample_sc_counts(
    genes: &GeneAnnotations,
    geno: &GenotypeMatrix,
    factor_model: &FactorModel,
    cell_fractions: &DMatrix<f32>,
    arch_params: &GeneticArchitectureParams,
    sc_params: &ScPhenotypeParams,
    seed: u64,
) -> Result<(ScCountData, Vec<Option<CellTypeGeneticEffects>>)> {
    let genotypes = &geno.genotypes;
    let snp_positions = geno.positions.as_slice();
    let snp_chromosomes = geno.chromosomes.as_slice();
    let num_individuals = genotypes.nrows();
    let num_cell_types = cell_fractions.ncols();
    let num_genes = factor_model.cell_type_means().nrows();

    info!(
        "Sampling single-cell counts: {} individuals × {} genes × {} cell types",
        num_individuals, num_genes, num_cell_types
    );
    info!(
        "  h2={:.3}, pve_cell_type={:.3}, cis_window={}",
        sc_params.h2_genetic, sc_params.pve_cell_type, arch_params.cis_window
    );

    // Standardize base means (factor model) so variance is controlled
    let mut base_means = factor_model.cell_type_means(); // G × K
    base_means.scale_columns_inplace();
    let pve_ct = sc_params.pve_cell_type;
    let pve_indv = 1.0 - pve_ct;
    base_means *= pve_ct.sqrt();

    // Empty effects for non-eQTL genes
    let empty_effects = CellTypeGeneticEffects {
        shared_causal_indices: Vec::new(),
        shared_effect_sizes: DMatrix::zeros(num_cell_types, 0),
        independent_causal_indices: vec![Vec::new(); num_cell_types],
        independent_effect_sizes: DMatrix::zeros(num_cell_types, 0),
        num_cell_types,
    };

    // Step 1: Per-gene — sample effects + compute phenotypes (parallel over genes)
    let per_gene: Vec<(DMatrix<f32>, Option<CellTypeGeneticEffects>)> = (0..num_genes)
        .into_par_iter()
        .map(|gene_idx| {
            let gene_seed = seed.wrapping_add(gene_idx as u64);

            let gene_effects = sample_gene_effects(
                gene_idx,
                genes,
                snp_positions,
                snp_chromosomes,
                num_cell_types,
                arch_params,
                sc_params.h2_genetic,
                gene_seed,
            )
            .expect("Failed to sample gene effects");

            let effects_ref = gene_effects.as_ref().unwrap_or(&empty_effects);
            let pheno = simulate_individual_linear_model(PhenotypeSimulationParams {
                genotypes,
                genetic_effects: effects_ref,
                h2_genetic: sc_params.h2_genetic,
                seed: gene_seed,
            })
            .expect("Failed to simulate phenotypes for gene");

            // Scale individual phenotypes by sqrt(1 - pve_cell_type)
            let mut y = pheno.phenotypes;
            y *= pve_indv.sqrt();

            (y, gene_effects)
        })
        .collect();

    let gene_phenotypes: Vec<DMatrix<f32>> = per_gene.iter().map(|(p, _)| p.clone()).collect();
    let gene_effects: Vec<Option<CellTypeGeneticEffects>> =
        per_gene.into_iter().map(|(_, e)| e).collect();

    let num_egenes = gene_effects.iter().filter(|e| e.is_some()).count();
    info!("  eGenes: {}/{}", num_egenes, num_genes);

    // Step 2: Sample cells per individual
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(num_genes as u64));
    let n_cells_dist = Poisson::new(sc_params.mean_cells_per_individual)?;
    let cells_per_ind: Vec<usize> = (0..num_individuals)
        .map(|_| n_cells_dist.sample(&mut rng) as usize)
        .collect();

    let total_cells: usize = cells_per_ind.iter().sum();
    info!("Total cells: {}", total_cells);

    // Step 3: Per-individual SC sampling (parallel)
    let pb_style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} individuals")
        .expect("Invalid progress bar template");

    let results: Vec<_> = (0..num_individuals)
        .into_par_iter()
        .progress_with_style(pb_style)
        .map(|i| {
            // log_rate_gk = sqrt(pve_ct)*standardize(M_gk) + sqrt(1-pve_ct)*Y_g(i,k)
            // Both components are already scaled above.
            let mut log_rates = DMatrix::zeros(num_genes, num_cell_types);
            for g in 0..num_genes {
                for k in 0..num_cell_types {
                    log_rates[(g, k)] = base_means[(g, k)] + gene_phenotypes[g][(i, k)];
                }
            }

            // Exponentiate
            let rates: Vec<Vec<f32>> = (0..num_cell_types)
                .map(|k| (0..num_genes).map(|g| log_rates[(g, k)].exp()).collect())
                .collect();

            let sc = sample_one_individual(
                &rates,
                &cell_fractions.row(i).iter().copied().collect::<Vec<_>>(),
                cells_per_ind[i],
                sc_params.depth_per_cell,
                seed.wrapping_add(num_genes as u64 + 1 + i as u64),
            );

            (sc, log_rates)
        })
        .collect();

    // Flatten into triplets + assemble individual-level log-rates (N×G per cell type)
    let mut global_cell = 0_u64;
    let mut triplets = Vec::new();
    let mut cell_individuals = Vec::new();
    let mut cell_types_vec = Vec::new();
    let mut individual_log_rates: Vec<DMatrix<f32>> = (0..num_cell_types)
        .map(|_| DMatrix::zeros(num_individuals, num_genes))
        .collect();

    for (ind, ((cell_counts, ct_assignments), log_rates)) in results.into_iter().enumerate() {
        // Store log-rates for this individual (log_rates is G×K)
        for k in 0..num_cell_types {
            for g in 0..num_genes {
                individual_log_rates[k][(ind, g)] = log_rates[(g, k)];
            }
        }

        for (counts, &ct) in cell_counts.iter().zip(&ct_assignments) {
            for &(gene, count) in counts {
                triplets.push((gene as u64, global_cell, count));
            }
            cell_individuals.push(ind);
            cell_types_vec.push(ct);
            global_cell += 1;
        }
    }

    info!("Generated {} non-zero counts", triplets.len());

    let sc_data = ScCountData {
        triplets,
        cell_individuals,
        cell_types: cell_types_vec,
        num_genes,
        num_cells: global_cell as usize,
        num_cell_types,
        num_individuals,
        individual_log_rates,
    };

    Ok((sc_data, gene_effects))
}

/// Sample cells for one individual. Returns (counts_per_cell, cell_type_assignments).
fn sample_one_individual(
    rates: &[Vec<f32>],
    fractions: &[f32],
    num_cells: usize,
    depth: f64,
    seed: u64,
) -> (Vec<Vec<(usize, f32)>>, Vec<usize>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let num_ct = rates.len();

    let fracs: Vec<f32> = fractions
        .iter()
        .map(|&v| if v.is_finite() && v > 0.0 { v } else { 0.0 })
        .collect();
    let sum: f32 = fracs.iter().sum();
    let fracs = if sum <= 0.0 {
        log::warn!("Invalid cell fractions, using uniform");
        vec![1.0 / num_ct as f32; num_ct]
    } else {
        fracs
    };

    let ct_dist = WeightedIndex::new(&fracs).unwrap();
    let ct_assignments: Vec<usize> = (0..num_cells).map(|_| ct_dist.sample(&mut rng)).collect();

    let counts: Vec<Vec<(usize, f32)>> = ct_assignments
        .iter()
        .map(|&ct| {
            let lambda = &rates[ct];
            let total: f32 = lambda.iter().sum();
            let scale = (depth as f32) / total.max(1e-8);

            lambda
                .iter()
                .enumerate()
                .filter_map(|(g, &lam)| {
                    let scaled = (lam * scale).max(1e-8);
                    Poisson::new(scaled as f64).ok().and_then(|d| {
                        let c = d.sample(&mut rng);
                        if c > 0.5 {
                            Some((g, c as f32))
                        } else {
                            None
                        }
                    })
                })
                .collect()
        })
        .collect();

    (counts, ct_assignments)
}

#[cfg(test)]
mod tests {
    use super::super::factor_model::simulate_factor_model;
    use super::super::gene_annotations::simulate_gene_annotations;
    use super::*;
    use rand::Rng;

    #[test]
    fn test_sample_sc_counts() {
        let num_genes = 50;
        let num_individuals = 10;
        let num_cell_types = 3;
        let num_snps = 200;

        let genes =
            simulate_gene_annotations(num_genes, "22", 20_000_000, 30_000_000, 1_000_000, 42);
        let positions: Vec<u64> = (0..num_snps)
            .map(|i| 20_000_000 + i as u64 * 50_000)
            .collect();
        let chromosomes: Vec<Box<str>> = vec![Box::from("22"); num_snps];

        let mut genotypes = DMatrix::from_element(num_individuals, num_snps, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for i in 0..num_individuals {
            for j in 0..num_snps {
                genotypes[(i, j)] = rng.random_range(0..=2) as f32;
            }
        }

        let geno = crate::genotype::GenotypeMatrix {
            individual_ids: (0..num_individuals)
                .map(|i| format!("ind_{i}").into_boxed_str())
                .collect(),
            snp_ids: (0..num_snps)
                .map(|i| format!("snp_{i}").into_boxed_str())
                .collect(),
            allele1: vec![Box::from("A"); num_snps],
            allele2: vec![Box::from("T"); num_snps],
            genotypes,
            positions,
            chromosomes,
        };

        let factor_model =
            simulate_factor_model(num_genes, 5, num_cell_types, 1.0, 1.0, 42).unwrap();
        let cell_fractions =
            DMatrix::from_element(num_individuals, num_cell_types, 1.0 / num_cell_types as f32);

        let arch_params = GeneticArchitectureParams {
            eqtl_gene_proportion: 0.5,
            ..Default::default()
        };

        let (sc, gene_effects) = sample_sc_counts(
            &genes,
            &geno,
            &factor_model,
            &cell_fractions,
            &arch_params,
            &ScPhenotypeParams {
                mean_cells_per_individual: 100.0,
                depth_per_cell: 1000.0,
                h2_genetic: 0.4,
                pve_cell_type: 0.5,
            },
            42,
        )
        .unwrap();

        assert!(sc.num_cells > 0);
        assert_eq!(sc.num_genes, num_genes);
        assert_eq!(sc.num_cell_types, num_cell_types);
        assert_eq!(sc.num_individuals, num_individuals);
        assert!(!sc.triplets.is_empty());

        // Should have some eQTL and some non-eQTL genes
        let num_egenes = gene_effects.iter().filter(|e| e.is_some()).count();
        assert!(num_egenes > 0 && num_egenes < num_genes);
    }
}
