use anyhow::Result;
use log::info;
use nalgebra::DMatrix;
use matrix_util::traits::{MatOps, SampleOps};

use super::cell_type_effects::CellTypeGeneticEffects;

/// Simulated phenotype data
#[derive(Debug, Clone)]
pub struct SimulatedPhenotypes {
    /// Phenotypes (N × K)
    pub phenotypes: DMatrix<f32>,
    /// Genetic values per cell type (N × K)
    pub genetic_values: DMatrix<f32>,
    /// Covariate values (N × 1), shared across cell types
    pub covariate_values: DMatrix<f32>,
    /// Noise per cell type (N × K)
    pub environmental_noise: DMatrix<f32>,
    /// Covariate effects (P × 1)
    pub covariate_effects: DMatrix<f32>,
}

/// Parameters for phenotype generation
pub struct PhenotypeSimulationParams<'a> {
    pub genotypes: &'a DMatrix<f32>,
    pub cell_fractions: &'a DMatrix<f32>,
    pub genetic_effects: &'a CellTypeGeneticEffects,
    pub covariates: Option<&'a DMatrix<f32>>,
    pub h2_genetic: f32,
    pub h2_covariate: f32,
    pub seed: u64,
}

/// Generate cell type-specific phenotypes
///
/// # Model
/// Y_ik = G_ik + C_i + ε_ik, for i=1..N, k=1..K
/// where:
/// - G_ik = X_i^shared β_k^shared + X_i^independent β_k^independent (genetic value)
/// - C_i = Z_i γ (covariate effect, shared across cell types)
/// - ε_ik ~ N(0,1) (noise per cell type)
pub fn simulate_phenotypes(params: PhenotypeSimulationParams) -> Result<SimulatedPhenotypes> {
    let genotypes = params.genotypes;
    let cell_fractions = params.cell_fractions;
    let genetic_effects = params.genetic_effects;
    let covariates = params.covariates;
    let h2_genetic = params.h2_genetic;
    let h2_covariate = params.h2_covariate;
    let n = genotypes.nrows();
    let k = genetic_effects.num_cell_types;

    // Validate
    if cell_fractions.nrows() != n {
        anyhow::bail!("cell_fractions rows ({}) != genotypes rows ({})",
                      cell_fractions.nrows(), n);
    }
    if cell_fractions.ncols() != k {
        anyhow::bail!("cell_fractions cols ({}) != num_cell_types ({})",
                      cell_fractions.ncols(), k);
    }
    if let Some(z) = covariates {
        if z.nrows() != n {
            anyhow::bail!("covariates rows ({}) != genotypes rows ({})", z.nrows(), n);
        }
    }
    if h2_genetic + h2_covariate > 1.0 {
        anyhow::bail!("h2_genetic + h2_covariate must be <= 1");
    }

    info!("Generating cell type-specific phenotypes: {} individuals × {} cell types", n, k);
    info!("h²_g={:.3}, h²_c={:.3}, h²_ε={:.3}",
          h2_genetic, h2_covariate, 1.0 - h2_genetic - h2_covariate);

    // G: N × K, G_ik = X_i^shared β_k^shared + X_i^independent β_k^independent
    let mut g_raw = DMatrix::zeros(n, k);

    for ct in 0..k {
        // Shared effects
        if !genetic_effects.shared_causal_indices.is_empty() {
            let shared_idx = &genetic_effects.shared_causal_indices;

            // X_shared: N × S
            let mut x_shared = DMatrix::zeros(n, shared_idx.len());
            for (j, &snp_idx) in shared_idx.iter().enumerate() {
                x_shared.set_column(j, &genotypes.column(snp_idx));
            }

            // β_shared_k: S × 1
            let beta_shared_k = DMatrix::from_iterator(
                shared_idx.len(), 1,
                genetic_effects.shared_effect_sizes.row(ct).iter().copied()
            );

            // G_shared_k = X_shared × β_shared_k (N × 1)
            let g_shared_k = x_shared * beta_shared_k;

            for i in 0..n {
                g_raw[(i, ct)] += g_shared_k[(i, 0)];
            }
        }

        // Independent effects
        if !genetic_effects.independent_causal_indices[ct].is_empty() {
            let indep_idx = &genetic_effects.independent_causal_indices[ct];

            // X_independent: N × I
            let mut x_indep = DMatrix::zeros(n, indep_idx.len());
            for (j, &snp_idx) in indep_idx.iter().enumerate() {
                x_indep.set_column(j, &genotypes.column(snp_idx));
            }

            // β_independent_k: I × 1
            let beta_indep_k = DMatrix::from_iterator(
                indep_idx.len(), 1,
                genetic_effects.independent_effect_sizes.row(ct).iter().take(indep_idx.len()).copied()
            );

            // G_independent_k = X_independent × β_independent_k (N × 1)
            let g_indep_k = x_indep * beta_indep_k;

            for i in 0..n {
                g_raw[(i, ct)] += g_indep_k[(i, 0)];
            }
        }
    }

    // C: N × 1, shared across cell types
    let (c_raw, cov_effects) = if let Some(z) = covariates {
        let gamma = DMatrix::rnorm(z.ncols(), 1);
        (z * &gamma, gamma)
    } else {
        (DMatrix::zeros(n, 1), DMatrix::zeros(0, 1))
    };

    // ε: N × K
    let eps_raw = DMatrix::rnorm(n, k);

    // Standardize
    let h2_noise = 1.0 - h2_genetic - h2_covariate;

    let mut g_std = g_raw;
    g_std.scale_columns_inplace();
    g_std *= h2_genetic.sqrt();

    let c_std = if covariates.is_some() {
        let mut c = c_raw;
        c.scale_columns_inplace();
        c *= h2_covariate.sqrt();
        c
    } else {
        DMatrix::zeros(n, 1)
    };

    let mut eps_std = eps_raw;
    eps_std.scale_columns_inplace();
    eps_std *= h2_noise.sqrt();

    // Y = G + C + ε (broadcast C across cell types)
    let mut y = g_std.clone();
    for ct in 0..k {
        for i in 0..n {
            y[(i, ct)] += c_std[(i, 0)] + eps_std[(i, ct)];
        }
    }

    info!("Phenotypes generated successfully");

    Ok(SimulatedPhenotypes {
        phenotypes: y,
        genetic_values: g_std,
        covariate_values: c_std,
        environmental_noise: eps_std,
        covariate_effects: cov_effects,
    })
}
