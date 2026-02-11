use anyhow::Result;
use log::info;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma, Normal};

/// Cell type-specific genetic effects with hybrid shared/independent model
#[derive(Debug, Clone)]
pub struct CellTypeGeneticEffects {
    /// Shared causal variant indices (same across all cell types)
    pub shared_causal_indices: Vec<usize>,

    /// Shared effect sizes: K × S matrix (S = number of shared causal variants)
    /// Each row k contains effect sizes for cell type k
    pub shared_effect_sizes: DMatrix<f32>,

    /// Independent causal variant indices for each cell type (K vectors)
    pub independent_causal_indices: Vec<Vec<usize>>,

    /// Independent effect sizes: K × I matrix (I = number of independent causal variants per cell type)
    pub independent_effect_sizes: DMatrix<f32>,

    /// Number of cell types
    pub num_cell_types: usize,
}

/// Sample cell type fractions for each individual using Dirichlet distribution
///
/// # Arguments
/// * `num_individuals` - Number of individuals
/// * `num_cell_types` - Number of cell types (K)
/// * `alpha` - Dirichlet concentration parameters (length K or 1 for symmetric)
/// * `seed` - Random seed
///
/// # Returns
/// N × K matrix where each row sums to 1 (cell type fractions per individual)
///
/// # Model
/// π_i ~ Dirichlet(α) for i = 1,...,N
pub fn sample_cell_type_fractions(
    num_individuals: usize,
    num_cell_types: usize,
    alpha: &[f32],
    seed: u64,
) -> Result<DMatrix<f32>> {
    info!(
        "Sampling cell type fractions: {} individuals, {} cell types",
        num_individuals, num_cell_types
    );

    // Validate alpha
    if alpha.is_empty() || (alpha.len() != 1 && alpha.len() != num_cell_types) {
        anyhow::bail!(
            "Alpha must have length 1 (symmetric) or {} (per cell type), got {}",
            num_cell_types,
            alpha.len()
        );
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Sample from Dirichlet by sampling Gamma and normalizing
    // If X_i ~ Gamma(α_i, 1), then X_i / Σ_j X_j ~ Dirichlet(α)
    let alphas: Vec<f32> = if alpha.len() == 1 {
        vec![alpha[0]; num_cell_types]
    } else {
        alpha.to_vec()
    };

    let gamma_dists: Vec<Gamma<f64>> = alphas
        .iter()
        .map(|&a| Gamma::new(a as f64, 1.0).unwrap())
        .collect();

    let gamma_samples = DMatrix::from_fn(num_individuals, num_cell_types, |_, j| {
        gamma_dists[j].sample(&mut rng) as f32
    });

    // Normalize rows to sum to 1 using matrix-util trait
    let fractions = gamma_samples.sum_to_one_rows();

    info!("Successfully sampled cell type fractions");
    Ok(fractions)
}

/// Select causal variants and sample cell type-specific genetic effects
///
/// # Arguments
/// * `num_snps` - Total number of SNPs
/// * `num_cell_types` - Number of cell types (K)
/// * `num_shared_causal` - Number of shared causal variants (across all cell types)
/// * `num_independent_causal` - Number of independent causal variants (per cell type)
/// * `genetic_variance` - Total genetic variance (σ²_g)
/// * `seed` - Random seed
///
/// # Returns
/// CellTypeGeneticEffects with shared and independent causal variants and effect sizes
///
/// # Model
/// Hybrid model with both shared and independent effects:
/// - Shared variants: S variants shared across all K cell types
///   - β_jk ~ N(0, σ²_shared / (K × S)) for j ∈ C_shared, k = 1,...,K
/// - Independent variants: I variants per cell type (different per cell type)
///   - β_jk ~ N(0, σ²_independent / I) for j ∈ C_k, k = 1,...,K
/// - Variance allocation: σ²_shared + σ²_independent = σ²_g
///   Split proportionally based on number of causal variants
pub fn sample_cell_type_genetic_effects(
    num_snps: usize,
    num_cell_types: usize,
    num_shared_causal: usize,
    num_independent_causal: usize,
    genetic_variance: f32,
    seed: u64,
) -> Result<CellTypeGeneticEffects> {
    info!(
        "Sampling cell type genetic effects: {} SNPs, {} cell types, {} shared causal, {} independent causal",
        num_snps, num_cell_types, num_shared_causal, num_independent_causal
    );

    if num_shared_causal + num_independent_causal > num_snps {
        anyhow::bail!(
            "num_shared_causal ({}) + num_independent_causal ({}) cannot exceed num_snps ({})",
            num_shared_causal,
            num_independent_causal,
            num_snps
        );
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Allocate variance proportionally based on number of causal variants
    let total_causal = num_shared_causal + num_independent_causal;
    let shared_variance = if total_causal > 0 {
        genetic_variance * (num_shared_causal as f32 / total_causal as f32)
    } else {
        0.0
    };
    let independent_variance = if total_causal > 0 {
        genetic_variance * (num_independent_causal as f32 / total_causal as f32)
    } else {
        0.0
    };

    // Sample shared causal variants
    let (shared_causal_indices, shared_effect_sizes) = if num_shared_causal > 0 {
        let mut all_indices: Vec<usize> = (0..num_snps).collect();
        all_indices.shuffle(&mut rng);
        let shared_indices = all_indices[..num_shared_causal].to_vec();

        info!("Selected {} shared causal variants", shared_indices.len());

        // Variance per shared effect: σ²_shared / (K × S)
        let effect_variance = shared_variance / (num_cell_types as f32 * num_shared_causal as f32);
        let effect_std = effect_variance.sqrt();

        let normal = Normal::new(0.0, effect_std as f64)
            .map_err(|e| anyhow::anyhow!("Failed to create Normal distribution: {}", e))?;

        let mut effects = DMatrix::zeros(num_cell_types, num_shared_causal);
        for k in 0..num_cell_types {
            for j in 0..num_shared_causal {
                effects[(k, j)] = normal.sample(&mut rng) as f32;
            }
        }

        (shared_indices, effects)
    } else {
        (Vec::new(), DMatrix::zeros(num_cell_types, 0))
    };

    // Sample independent causal variants
    let (independent_causal_indices, independent_effect_sizes) = if num_independent_causal > 0 {
        let mut causal_per_type = Vec::new();
        let mut all_effects = Vec::new();

        // Variance per independent effect: σ²_independent / I
        let effect_variance = independent_variance / num_independent_causal as f32;
        let effect_std = effect_variance.sqrt();

        let normal = Normal::new(0.0, effect_std as f64)
            .map_err(|e| anyhow::anyhow!("Failed to create Normal distribution: {}", e))?;

        // Get available indices (exclude shared causal variants to avoid overlap)
        let available_indices: Vec<usize> = (0..num_snps)
            .filter(|idx| !shared_causal_indices.contains(idx))
            .collect();

        if available_indices.len() < num_independent_causal * num_cell_types {
            info!(
                "Warning: Only {} SNPs available for independent sampling (need {})",
                available_indices.len(),
                num_independent_causal * num_cell_types
            );
        }

        for _k in 0..num_cell_types {
            // Select causal variants for this cell type
            let mut indices = available_indices.clone();
            indices.shuffle(&mut rng);
            let causal_k: Vec<usize> =
                indices[..num_independent_causal.min(indices.len())].to_vec();

            // Sample effect sizes for this cell type
            let effects_k: Vec<f32> = (0..causal_k.len())
                .map(|_| normal.sample(&mut rng) as f32)
                .collect();

            causal_per_type.push(causal_k);
            all_effects.push(effects_k);
        }

        info!(
            "Selected independent causal variants for {} cell types",
            num_cell_types
        );

        // Convert to DMatrix
        let mut effects = DMatrix::zeros(num_cell_types, num_independent_causal);
        for k in 0..num_cell_types {
            for j in 0..num_independent_causal.min(all_effects[k].len()) {
                effects[(k, j)] = all_effects[k][j];
            }
        }

        (causal_per_type, effects)
    } else {
        (
            vec![Vec::new(); num_cell_types],
            DMatrix::zeros(num_cell_types, 0),
        )
    };

    Ok(CellTypeGeneticEffects {
        shared_causal_indices,
        shared_effect_sizes,
        independent_causal_indices,
        independent_effect_sizes,
        num_cell_types,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_cell_type_fractions() {
        let fractions = sample_cell_type_fractions(100, 5, &[1.0], 42).unwrap();

        assert_eq!(fractions.nrows(), 100);
        assert_eq!(fractions.ncols(), 5);

        // Check rows sum to 1
        for i in 0..100 {
            let sum: f32 = fractions.row(i).iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fractions_deterministic_with_seed() {
        let f1 = sample_cell_type_fractions(50, 3, &[1.0], 42).unwrap();
        let f2 = sample_cell_type_fractions(50, 3, &[1.0], 42).unwrap();
        assert_eq!(f1, f2);

        let f3 = sample_cell_type_fractions(50, 3, &[1.0], 999).unwrap();
        assert_ne!(f1, f3);
    }

    #[test]
    fn test_shared_causal_only() {
        let effects = sample_cell_type_genetic_effects(1000, 3, 10, 0, 0.5, 42).unwrap();

        assert_eq!(effects.num_cell_types, 3);
        assert_eq!(effects.shared_causal_indices.len(), 10);
        assert_eq!(effects.shared_effect_sizes.nrows(), 3);
        assert_eq!(effects.shared_effect_sizes.ncols(), 10);

        // No independent effects
        assert_eq!(effects.independent_causal_indices.len(), 3);
        assert!(effects.independent_causal_indices[0].is_empty());
    }

    #[test]
    fn test_independent_causal_only() {
        let effects = sample_cell_type_genetic_effects(1000, 3, 0, 10, 0.5, 42).unwrap();

        assert_eq!(effects.num_cell_types, 3);
        assert_eq!(effects.independent_causal_indices.len(), 3);

        // Each cell type should have different causal indices
        assert_ne!(
            effects.independent_causal_indices[0],
            effects.independent_causal_indices[1]
        );
        assert_ne!(
            effects.independent_causal_indices[1],
            effects.independent_causal_indices[2]
        );

        assert_eq!(effects.independent_effect_sizes.nrows(), 3);
        assert_eq!(effects.independent_effect_sizes.ncols(), 10);

        // No shared effects
        assert!(effects.shared_causal_indices.is_empty());
    }

    #[test]
    fn test_hybrid_model() {
        let effects = sample_cell_type_genetic_effects(1000, 3, 5, 5, 0.5, 42).unwrap();

        assert_eq!(effects.num_cell_types, 3);

        // Should have both shared and independent effects
        assert_eq!(effects.shared_causal_indices.len(), 5);
        assert_eq!(effects.shared_effect_sizes.nrows(), 3);
        assert_eq!(effects.shared_effect_sizes.ncols(), 5);

        assert_eq!(effects.independent_causal_indices.len(), 3);
        assert_eq!(effects.independent_effect_sizes.nrows(), 3);
        assert_eq!(effects.independent_effect_sizes.ncols(), 5);

        // Shared and independent indices should not overlap
        for idx in &effects.shared_causal_indices {
            for k in 0..3 {
                assert!(!effects.independent_causal_indices[k].contains(idx));
            }
        }
    }
}
