use anyhow::Result;
use log::{info, warn};
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, weighted::WeightedIndex};
use rayon::prelude::*;
use indicatif::{ParallelProgressIterator, ProgressStyle};

use super::factor_model::FactorModel;
use super::gene_eqtl::ScEqtlEffects;

/// Single-cell count data with cell annotations
#[derive(Debug, Clone)]
pub struct ScCountData {
    /// Sparse triplets: (gene_idx, cell_idx, count)
    pub triplets: Vec<(u64, u64, f32)>,

    /// Cell annotations: individual_id for each cell
    pub cell_individuals: Vec<usize>,

    /// Cell annotations: cell type for each cell
    pub cell_types: Vec<usize>,

    /// Total number of genes
    pub num_genes: usize,

    /// Total number of cells
    pub num_cells: usize,

    /// Number of cell types
    pub num_cell_types: usize,

    /// Number of individuals
    pub num_individuals: usize,
}

/// Cells generated for a single individual
struct IndividualCells {
    /// Sparse counts per cell: Vec of (gene_idx, count) per cell
    triplets: Vec<Vec<(usize, f32)>>,
    /// Cell type assignment for each cell
    cell_types: Vec<usize>,
}

/// Parameters for single-cell phenotype generation
#[derive(Debug, Clone)]
pub struct ScPhenotypeParams {
    /// Number of cells per individual (sampled from Poisson)
    pub mean_cells_per_individual: f64,

    /// Sequencing depth per cell (total UMI counts)
    pub depth_per_cell: f64,

    /// Overdispersion parameter for Poisson sampling (not used yet, for future NB)
    pub overdispersion: f32,
}

impl Default for ScPhenotypeParams {
    fn default() -> Self {
        Self {
            mean_cells_per_individual: 1000.0,
            depth_per_cell: 5000.0,
            overdispersion: 0.1,
        }
    }
}

/// Generate single-cell count data with genetic effects
///
/// # Model
/// For each individual i:
/// 1. Sample number of cells: n_i ~ Poisson(mean_cells_per_individual)
/// 2. Sample cell type proportions for each cell: c ~ Multinomial(π_i)
/// 3. Compute cell-type-specific expression:
///    - Base: M_gk = W × Z (from factor model)
///    - Genetic: Add eQTL effects from genotypes
/// 4. For each cell c of individual i with cell type k:
///    - λ_gc = exp(M_gk + Σ_j X_ij * β_gjk) where β_gjk is effect of SNP j on gene g in cell type k
///    - Scale λ to sum to depth_per_cell
///    - Sample counts: Y_gc ~ Poisson(λ_gc)
///
/// # Arguments
/// * `factor_model` - Base expression model (W × Z)
/// * `eqtl_effects` - Gene-level eQTL effects (shared + independent)
/// * `genotypes` - Genotype matrix (N × M), values in {0, 1, 2}
/// * `cell_fractions` - Cell type fractions per individual (N × K)
/// * `params` - Simulation parameters
/// * `seed` - Random seed
///
pub fn generate_sc_phenotypes(
    factor_model: &FactorModel,
    eqtl_effects: &ScEqtlEffects,
    genotypes: &DMatrix<f32>,
    cell_fractions: &DMatrix<f32>,
    params: &ScPhenotypeParams,
    seed: u64,
) -> Result<ScCountData> {
    let num_individuals = genotypes.nrows();
    let num_genes = eqtl_effects.num_genes;
    let num_cell_types = eqtl_effects.num_cell_types;

    info!("Generating single-cell phenotypes:");
    info!("  {} individuals", num_individuals);
    info!("  {} genes", num_genes);
    info!("  {} cell types", num_cell_types);
    info!("  Mean cells/individual: {}", params.mean_cells_per_individual);
    info!("  Depth/cell: {}", params.depth_per_cell);

    // Step 1: Compute base cell type means from factor model (G × K)
    let base_means = factor_model.cell_type_means();
    info!("Computed base expression from factor model");

    // Step 2: Sample cells per individual and generate counts in parallel
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let cells_per_individual_dist = Poisson::new(params.mean_cells_per_individual)?;

    let cells_per_individual: Vec<usize> = (0..num_individuals)
        .map(|_| cells_per_individual_dist.sample(&mut rng) as usize)
        .collect();

    let total_cells: usize = cells_per_individual.iter().sum();
    info!("Sampling {} total cells across {} individuals", total_cells, num_individuals);

    // Progress bar setup
    let pb_style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} individuals")
        .expect("Invalid progress bar template");

    // Step 3: Generate counts for each individual's cells in parallel
    let results: Vec<_> = (0..num_individuals)
        .into_par_iter()
        .progress_with_style(pb_style)
        .map(|ind_idx| {
            let ind_seed = seed.wrapping_add(ind_idx as u64);
            generate_individual_cells(
                ind_idx,
                cells_per_individual[ind_idx],
                &base_means,
                eqtl_effects,
                genotypes.row(ind_idx).transpose(),
                cell_fractions.row(ind_idx).transpose(),
                params,
                ind_seed,
            )
        })
        .collect();

    // Step 4: Combine results from all individuals
    let mut global_cell_idx = 0_u64;
    let mut all_triplets = Vec::new();
    let mut cell_individuals = Vec::new();
    let mut cell_types = Vec::new();

    for (ind_idx, ind_cells) in results.into_iter().enumerate() {
        for (local_cell_idx, triplet_vec) in ind_cells.triplets.into_iter().enumerate() {
            for (gene_idx, count) in triplet_vec {
                all_triplets.push((gene_idx as u64, global_cell_idx, count));
            }
            cell_individuals.push(ind_idx);
            cell_types.push(ind_cells.cell_types[local_cell_idx]);
            global_cell_idx += 1;
        }
    }

    info!("Generated {} non-zero counts", all_triplets.len());

    Ok(ScCountData {
        triplets: all_triplets,
        cell_individuals,
        cell_types,
        num_genes,
        num_cells: global_cell_idx as usize,
        num_cell_types,
        num_individuals,
    })
}

/// Generate cells for a single individual
///
fn generate_individual_cells(
    _ind_idx: usize,
    num_cells: usize,
    base_means: &DMatrix<f32>,
    eqtl_effects: &ScEqtlEffects,
    genotype_row: DVector<f32>,
    cell_fraction_row: DVector<f32>,
    params: &ScPhenotypeParams,
    seed: u64,
) -> IndividualCells {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let num_genes = eqtl_effects.num_genes;
    let num_cell_types = eqtl_effects.num_cell_types;

    // Validate cell fractions: must be finite and non-negative with at least one positive
    let fractions: Vec<f32> = cell_fraction_row
        .iter()
        .map(|&v| if v.is_finite() && v >= 0.0 { v } else { 0.0 })
        .collect();
    let frac_sum: f32 = fractions.iter().sum();
    let fractions = if frac_sum <= 0.0 {
        // Fallback to uniform if all fractions are zero/invalid
        warn!("Cell fractions are all zero/invalid for individual, using uniform");
        vec![1.0 / num_cell_types as f32; num_cell_types]
    } else {
        fractions
    };

    // Sample cell type assignments using cell fractions as probabilities
    let cell_type_dist = WeightedIndex::new(fractions.iter().copied())
        .expect("Failed to create WeightedIndex after validation");

    // Sample each cell's type independently from categorical distribution
    let cell_type_assignments: Vec<usize> = (0..num_cells)
        .map(|_| cell_type_dist.sample(&mut rng))
        .collect();

    // Compute cell-type-specific expression with genetic effects
    let mut cell_type_lambdas = vec![Vec::new(); num_cell_types];

    for cell_type_idx in 0..num_cell_types {
        let mut lambda_g = Vec::with_capacity(num_genes);

        for gene_idx in 0..num_genes {
            // Start with base expression
            let mut log_expr = base_means[(gene_idx, cell_type_idx)];

            // Add genetic effects from this individual's genotypes
            let gene_effects = &eqtl_effects.genes[gene_idx];

            // Shared causal SNPs (same SNPs, potentially different effects per cell type)
            for causal_snp in &gene_effects.shared_causal_snps {
                debug_assert!(causal_snp.snp_idx < genotype_row.len(),
                    "snp_idx {} out of bounds (genotype len {})", causal_snp.snp_idx, genotype_row.len());
                let genotype = genotype_row[causal_snp.snp_idx];
                let effect = causal_snp.effect_sizes[cell_type_idx];
                log_expr += genotype * effect;
            }

            // Independent causal SNPs (cell-type-specific SNPs)
            for causal_snp in &gene_effects.independent_causal_snps[cell_type_idx] {
                debug_assert!(causal_snp.snp_idx < genotype_row.len(),
                    "snp_idx {} out of bounds (genotype len {})", causal_snp.snp_idx, genotype_row.len());
                let genotype = genotype_row[causal_snp.snp_idx];
                let effect = causal_snp.effect_sizes[cell_type_idx];
                log_expr += genotype * effect;
            }

            // Convert to expression level (exponentiate if needed, or keep linear)
            // For now, keep linear and ensure positive
            let expr = log_expr.max(0.01);
            lambda_g.push(expr);
        }

        cell_type_lambdas[cell_type_idx] = lambda_g;
    }

    // Generate counts for each cell
    let mut cell_triplets = Vec::new();

    for &cell_type_idx in &cell_type_assignments {
        let lambda_g: &Vec<f32> = &cell_type_lambdas[cell_type_idx];

        // Scale to target depth
        let total_lambda: f32 = lambda_g.iter().sum();
        let scale = (params.depth_per_cell as f32) / total_lambda.max(1e-8);

        let mut cell_counts = Vec::new();

        for (gene_idx, &lambda_val) in lambda_g.iter().enumerate() {
            let scaled_lambda = (lambda_val * scale).max(1e-8);

            match Poisson::new(scaled_lambda as f64) {
                Ok(poisson_dist) => {
                    let count = poisson_dist.sample(&mut rng);
                    if count > 0.5 {
                        cell_counts.push((gene_idx, count as f32));
                    }
                }
                Err(_) => {
                    warn!("Poisson sampling failed for gene {} (lambda={}), skipping", gene_idx, scaled_lambda);
                }
            }
        }

        cell_triplets.push(cell_counts);
    }

    IndividualCells {
        triplets: cell_triplets,
        cell_types: cell_type_assignments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::factor_model::simulate_factor_model;
    use super::super::gene_eqtl::{sample_sc_eqtl_effects, GeneticArchitectureParams};
    use super::super::gene_annotations::simulate_gene_annotations;
    use rand::Rng;

    #[test]
    fn test_generate_sc_phenotypes() {
        // Setup
        let num_genes = 50;
        let num_individuals = 10;
        let num_cell_types = 3;
        let num_factors = 5;

        // Simulate factor model
        let factor_model = simulate_factor_model(
            num_genes,
            num_factors,
            num_cell_types,
            1.0,
            1.0,
            42,
        ).unwrap();

        // Simulate gene annotations
        let genes = simulate_gene_annotations(
            num_genes,
            "22",
            20_000_000,
            30_000_000,
            1_000_000,
            42,
        );

        // Simulate SNPs
        let snp_positions: Vec<u64> = (0..200).map(|i| 20_000_000 + i * 50_000).collect();
        let snp_chromosomes: Vec<Box<str>> = vec![Box::from("22"); 200];

        // Simulate eQTL effects
        let eqtl_params = GeneticArchitectureParams {
            eqtl_gene_proportion: 0.5,
            shared_eqtl_proportion: 0.6,
            independent_eqtl_proportion: 0.4,
            num_shared_causal_per_gene: 1,
            num_independent_causal_per_gene: 1,
            genetic_variance: 0.4,
            cis_window: 1_000_000,
        };

        let eqtl_effects = sample_sc_eqtl_effects(
            &genes,
            &snp_positions,
            &snp_chromosomes,
            num_cell_types,
            &eqtl_params,
            42,
        ).unwrap();

        // Simulate genotypes
        let mut genotypes = DMatrix::from_element(num_individuals, 200, 0.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for i in 0..num_individuals {
            for j in 0..200 {
                genotypes[(i, j)] = rng.random_range(0..=2) as f32;
            }
        }

        // Simulate cell fractions
        let cell_fractions = DMatrix::from_element(num_individuals, num_cell_types, 1.0 / num_cell_types as f32);

        // Generate sc phenotypes
        let params = ScPhenotypeParams {
            mean_cells_per_individual: 100.0,
            depth_per_cell: 1000.0,
            overdispersion: 0.1,
        };

        let sc_data = generate_sc_phenotypes(
            &factor_model,
            &eqtl_effects,
            &genotypes,
            &cell_fractions,
            &params,
            42,
        ).unwrap();

        // Assertions
        assert!(sc_data.num_cells > 0);
        assert_eq!(sc_data.num_genes, num_genes);
        assert_eq!(sc_data.num_cell_types, num_cell_types);
        assert_eq!(sc_data.num_individuals, num_individuals);
        assert_eq!(sc_data.cell_individuals.len(), sc_data.num_cells);
        assert_eq!(sc_data.cell_types.len(), sc_data.num_cells);
        assert!(sc_data.triplets.len() > 0);

        println!("Generated {} cells with {} non-zero counts",
                 sc_data.num_cells, sc_data.triplets.len());
        println!("Average counts per cell: {:.1}",
                 sc_data.triplets.len() as f32 / sc_data.num_cells as f32);
    }
}
