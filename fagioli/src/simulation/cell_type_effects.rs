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

/// Compute raw genetic values G (N x T) from block genotypes and effects.
///
/// For each trait t:
///   G_t = X_shared * beta_shared_t + X_indep_t * beta_indep_t
pub fn compute_genetic_values(
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

////////////////////////////////////////////////////////////////////////////////
// Correlated (factor-structured) genetic architecture
////////////////////////////////////////////////////////////////////////////////

/// Offset giving the independent-effects arm its own RNG stream, so that the
/// two samplers agree on it at a common seed.
const SEED_INDEPENDENT_ARM: u64 = 0x9E37_79B9_7F4A_7C15;

/// Genome-wide trait loadings for a factor genetic architecture.
///
/// [`sample_cell_type_genetic_effects`] gives shared causal *loci* whose effect
/// sizes are drawn independently per trait, so `E[Cov_g(t,t')] = 0` -- shared
/// loci, not shared genetics, and no genetic correlation to recover. Real
/// pleiotropy is correlated, and multi-trait methods (cross-trait LDSC, MTAG,
/// GenomicSEM) cannot be validated against a simulator that always returns
/// `r_g = 0`.
///
/// This structure supplies the missing piece: effects at shared causal variants
/// are `β_j = Λ f_j`, giving
///
/// ```text
/// Cov_g = Λ (Σ_j f_j f_j') Λ'
/// ```
///
/// which is genuinely low-rank across traits. `Λ` is drawn **once** and reused
/// for every block, because a genetic factor is a property of the traits, not
/// of a locus -- per-block loadings would give each block its own geometry and
/// largely cancel when summed genome-wide.
#[derive(Debug, Clone)]
pub struct TraitFactorLoadings {
    /// `Λ`, shape (T, H_g). Rows are traits, columns genetic factors.
    pub lambda: DMatrix<f32>,
}

impl TraitFactorLoadings {
    /// Draw `Λ` with i.i.d. standard normal entries, scaled so each trait's
    /// loading vector has unit norm in expectation.
    pub fn sample(num_traits: usize, num_factors: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).expect("standard normal");
        let scale = 1.0 / (num_factors.max(1) as f32).sqrt();
        let lambda = DMatrix::from_fn(num_traits, num_factors, |_, _| {
            normal.sample(&mut rng) as f32 * scale
        });
        Self { lambda }
    }

    pub fn num_traits(&self) -> usize {
        self.lambda.nrows()
    }

    pub fn num_factors(&self) -> usize {
        self.lambda.ncols()
    }
}

/// Sample genetic effects whose shared component is correlated across traits.
///
/// Identical to [`sample_cell_type_genetic_effects`] except that the shared
/// effect sizes come from the factor model `β_j = Λ f_j` rather than from
/// independent per-trait draws. Independent effects are unchanged: they sit at
/// different variants per trait and so contribute only to the diagonal of
/// `Cov_g`.
pub fn sample_factor_genetic_effects(
    num_snps: usize,
    num_shared_causal: usize,
    num_independent_causal: usize,
    loadings: &TraitFactorLoadings,
    genetic_variance: f32,
    seed: u64,
) -> Result<CellTypeGeneticEffects> {
    let num_traits = loadings.num_traits();
    let num_factors = loadings.num_factors();

    if num_shared_causal + num_independent_causal > num_snps {
        anyhow::bail!(
            "num_shared_causal ({}) + num_independent_causal ({}) cannot exceed num_snps ({})",
            num_shared_causal,
            num_independent_causal,
            num_snps
        );
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let total_causal = num_shared_causal + num_independent_causal;
    let (shared_variance, independent_variance) = if total_causal > 0 {
        (
            genetic_variance * (num_shared_causal as f32 / total_causal as f32),
            genetic_variance * (num_independent_causal as f32 / total_causal as f32),
        )
    } else {
        (0.0, 0.0)
    };

    // ── Shared: correlated through Λ ──────────────────────────────────────
    let (shared_causal_indices, shared_effect_sizes) = if num_shared_causal > 0 {
        let mut all: Vec<usize> = (0..num_snps).collect();
        all.shuffle(&mut rng);
        let indices = all[..num_shared_causal].to_vec();

        // Per-variant factor scores f_j, then β_j = Λ f_j.
        let effect_variance =
            shared_variance / (num_traits as f32 * num_shared_causal as f32);
        let f_std = (effect_variance / num_factors.max(1) as f32).sqrt() * num_factors as f32;
        let normal = Normal::new(0.0, f_std as f64)
            .map_err(|e| anyhow::anyhow!("Failed to create Normal distribution: {}", e))?;

        let f_scores = DMatrix::from_fn(num_factors, num_shared_causal, |_, _| {
            normal.sample(&mut rng) as f32
        });
        // (T, H) x (H, S) -> (T, S)
        let effects = &loadings.lambda * f_scores;

        info!(
            "Sampled {} shared causal variants through {} genetic factor(s)",
            indices.len(),
            num_factors,
        );
        (indices, effects)
    } else {
        (Vec::new(), DMatrix::zeros(num_traits, 0))
    };

    // ── Independent: unchanged, distinct variants per trait ───────────────
    // Drawn from its own stream. The shared arm above consumes H*S normals
    // where `sample_cell_type_genetic_effects` consumes K*S, so sharing one
    // stream would leave the two samplers with different trait-specific
    // effects at the same seed -- and the contrast between them is exactly
    // what the tests measure.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ SEED_INDEPENDENT_ARM);
    let (independent_causal_indices, independent_effect_sizes) = if num_independent_causal > 0 {
        let effect_variance = independent_variance / num_independent_causal as f32;
        let normal = Normal::new(0.0, (effect_variance.sqrt()) as f64)
            .map_err(|e| anyhow::anyhow!("Failed to create Normal distribution: {}", e))?;

        let available: Vec<usize> = (0..num_snps)
            .filter(|j| !shared_causal_indices.contains(j))
            .collect();

        let mut indices = Vec::with_capacity(num_traits);
        let mut effects = DMatrix::zeros(num_traits, num_independent_causal);
        for t in 0..num_traits {
            let mut pool = available.clone();
            pool.shuffle(&mut rng);
            let take = num_independent_causal.min(pool.len());
            indices.push(pool[..take].to_vec());
            for j in 0..take {
                effects[(t, j)] = normal.sample(&mut rng) as f32;
            }
        }
        (indices, effects)
    } else {
        (vec![Vec::new(); num_traits], DMatrix::zeros(num_traits, 0))
    };

    Ok(CellTypeGeneticEffects {
        shared_causal_indices,
        shared_effect_sizes,
        independent_causal_indices,
        independent_effect_sizes,
        num_cell_types: num_traits,
    })
}

/// True genetic covariance `Cov_g = Σ_j β_j β_j'` implied by a block's effects.
///
/// Shared variants contribute `B_sh B_sh'` in full; independent variants sit at
/// distinct SNPs per trait and so contribute only to the diagonal. Summing this
/// over causal blocks gives the genome-wide `Cov_g` that cross-trait LDSC and
/// any trait-geometry estimate should recover.
pub fn genetic_covariance(effects: &CellTypeGeneticEffects) -> DMatrix<f32> {
    let t = effects.num_cell_types;
    let mut cov = &effects.shared_effect_sizes * effects.shared_effect_sizes.transpose();
    for tt in 0..t {
        if tt < effects.independent_effect_sizes.nrows() {
            let own: f32 = effects
                .independent_effect_sizes
                .row(tt)
                .iter()
                .map(|v| v * v)
                .sum();
            cov[(tt, tt)] += own;
        }
    }
    cov
}

#[cfg(test)]
mod factor_tests {
    use super::*;

    fn to_corr(m: &DMatrix<f32>) -> DMatrix<f32> {
        let t = m.nrows();
        DMatrix::from_fn(t, t, |i, j| {
            let d = (m[(i, i)] * m[(j, j)]).sqrt();
            if d > 0.0 {
                (m[(i, j)] / d).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
    }

    fn mean_abs_offdiag(m: &DMatrix<f32>) -> f32 {
        let t = m.nrows();
        let mut s = 0.0;
        let mut n = 0;
        for i in 0..t {
            for j in (i + 1)..t {
                s += m[(i, j)].abs();
                n += 1;
            }
        }
        s / n.max(1) as f32
    }

    /// The defect this architecture exists to fix: shared *loci* with
    /// independently drawn per-trait effects give no genetic correlation.
    #[test]
    fn test_independent_architecture_has_no_genetic_correlation() {
        let t = 12;
        // Average over blocks so this is about the expectation, not one draw.
        let mut cov = DMatrix::<f32>::zeros(t, t);
        for b in 0..40 {
            let e = sample_cell_type_genetic_effects(500, t, 20, 5, 0.4, 1000 + b).unwrap();
            cov += genetic_covariance(&e);
        }
        let rg = to_corr(&cov);
        let m = mean_abs_offdiag(&rg);
        println!("independent architecture: mean |r_g| = {m:.4}");
        assert!(
            m < 0.15,
            "shared loci with independent effects should give r_g ~ 0, got {m}"
        );
    }

    /// The same configuration, with and without a genome-wide Λ. The contrast is
    /// the point; the absolute level is not, because trait-specific
    /// (independent) effects legitimately dilute `r_g` by adding variance to the
    /// diagonal only.
    #[test]
    fn test_factor_architecture_produces_genetic_correlation() {
        let t = 12;
        let loadings = TraitFactorLoadings::sample(t, 3, 7);

        let mut cov_factor = DMatrix::<f32>::zeros(t, t);
        let mut cov_indep = DMatrix::<f32>::zeros(t, t);
        for b in 0..40 {
            cov_factor += genetic_covariance(
                &sample_factor_genetic_effects(500, 20, 5, &loadings, 0.4, 1000 + b).unwrap(),
            );
            cov_indep += genetic_covariance(
                &sample_cell_type_genetic_effects(500, t, 20, 5, 0.4, 1000 + b).unwrap(),
            );
        }

        let m_factor = mean_abs_offdiag(&to_corr(&cov_factor));
        let m_indep = mean_abs_offdiag(&to_corr(&cov_indep));
        println!(
            "mean |r_g| with 3 factors {m_factor:.4} vs independent {m_indep:.4} \
             (ratio {:.1}x)",
            m_factor / m_indep.max(1e-6)
        );

        assert!(
            m_factor > 10.0 * m_indep,
            "the factor architecture must dominate the independent one: \
             {m_factor} vs {m_indep}"
        );
        assert!(
            m_factor > 0.1,
            "factor architecture should give clearly nonzero r_g, got {m_factor}"
        );
    }

    /// Without trait-specific effects there is nothing to dilute, so the same
    /// loadings give a much stronger correlation — which is what confirms the
    /// dilution above is the independent effects and not a defect in Λ.
    #[test]
    fn test_independent_effects_dilute_but_do_not_destroy_rg() {
        let t = 12;
        let loadings = TraitFactorLoadings::sample(t, 3, 7);

        let level = |n_indep: usize| {
            let mut cov = DMatrix::<f32>::zeros(t, t);
            for b in 0..40 {
                cov += genetic_covariance(
                    &sample_factor_genetic_effects(500, 20, n_indep, &loadings, 0.4, 1000 + b)
                        .unwrap(),
                );
            }
            mean_abs_offdiag(&to_corr(&cov))
        };

        let none = level(0);
        let some = level(5);
        let lots = level(40);
        println!("mean |r_g| by independent-causal count: 0 -> {none:.4}, 5 -> {some:.4}, 40 -> {lots:.4}");

        assert!(none > some, "trait-specific effects should dilute r_g");
        assert!(some > lots, "more trait-specific effects should dilute it further");
        assert!(none > 0.3, "undiluted r_g should be substantial, got {none}");
    }

    /// The implied `r_g` must track the loadings that generated it, not merely
    /// be nonzero — otherwise the simulator states a geometry it does not honour.
    #[test]
    fn test_implied_rg_tracks_the_loadings() {
        let t = 12;
        let loadings = TraitFactorLoadings::sample(t, 2, 11);
        let mut cov = DMatrix::<f32>::zeros(t, t);
        for b in 0..60 {
            let e =
                sample_factor_genetic_effects(500, 30, 0, &loadings, 0.4, 5000 + b).unwrap();
            cov += genetic_covariance(&e);
        }
        let rg = to_corr(&cov);
        // With no independent effects, Cov_g ∝ Λ F F' Λ' and F F' → cI, so the
        // implied correlation should approach corr(ΛΛ').
        let target = to_corr(&(&loadings.lambda * loadings.lambda.transpose()));

        let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..t {
            for j in (i + 1)..t {
                let (a, b) = (rg[(i, j)] as f64, target[(i, j)] as f64);
                sxy += a * b;
                sxx += a * a;
                syy += b * b;
            }
        }
        let corr = sxy / (sxx * syy).sqrt();
        println!("implied r_g vs corr(ΛΛ'): {corr:.4}");
        assert!(
            corr > 0.9,
            "implied r_g should follow the generating loadings, got {corr}"
        );
    }

    /// A single factor makes every trait pair correlated through one axis.
    #[test]
    fn test_one_factor_gives_a_single_shared_axis() {
        let t = 8;
        let loadings = TraitFactorLoadings::sample(t, 1, 13);
        let mut cov = DMatrix::<f32>::zeros(t, t);
        for b in 0..40 {
            let e = sample_factor_genetic_effects(400, 25, 0, &loadings, 0.4, 9000 + b).unwrap();
            cov += genetic_covariance(&e);
        }
        let rg = to_corr(&cov);
        // Rank-1: every |r_g| should be near 1.
        let m = mean_abs_offdiag(&rg);
        println!("one factor: mean |r_g| = {m:.4}");
        assert!(m > 0.85, "a single factor should give near-perfect |r_g|, got {m}");
    }
}
