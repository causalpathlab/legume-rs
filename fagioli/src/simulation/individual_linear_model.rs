use anyhow::Result;
use log::info;
use matrix_util::traits::{MatOps, SampleOps};
use nalgebra::DMatrix;

use super::cell_type_effects::CellTypeGeneticEffects;

// ─────────────────────────────────────────────────────────────────────────────
// Individual-level linear model (per cell type)
// ─────────────────────────────────────────────────────────────────────────────

/// Simulated phenotype data
#[derive(Debug, Clone)]
pub struct IndvPhenotypes {
    /// Phenotypes (N × K)
    pub phenotypes: DMatrix<f32>,
    /// Genetic values per cell type (N × K)
    pub genetic_values: DMatrix<f32>,
    /// Noise per cell type (N × K)
    pub environmental_noise: DMatrix<f32>,
}

/// Parameters for phenotype generation
pub struct PhenotypeSimulationParams<'a> {
    pub genotypes: &'a DMatrix<f32>,
    pub genetic_effects: &'a CellTypeGeneticEffects,
    pub h2_genetic: f32,
    pub seed: u64,
}

/// Generate cell type-specific phenotypes
///
/// # Model
/// Y_ik = sqrt(h2) * standardize(G_ik) + sqrt(1-h2) * standardize(ε_ik)
/// where:
/// - G_ik = X_i^shared β_k^shared + X_i^independent β_k^independent (genetic value)
/// - ε_ik ~ N(0,1) (noise per cell type)
pub fn simulate_individual_linear_model(
    params: PhenotypeSimulationParams,
) -> Result<IndvPhenotypes> {
    let genotypes = params.genotypes;
    let genetic_effects = params.genetic_effects;
    let h2_genetic = params.h2_genetic;
    let n = genotypes.nrows();
    let k = genetic_effects.num_cell_types;

    if h2_genetic > 1.0 || h2_genetic < 0.0 {
        anyhow::bail!("h2_genetic must be in [0, 1], got {}", h2_genetic);
    }

    info!(
        "Generating cell type-specific phenotypes: {} individuals × {} cell types",
        n, k
    );
    info!(
        "h²_g={:.3}, h²_ε={:.3}",
        h2_genetic,
        1.0 - h2_genetic
    );

    // G: N × K, G_ik = X_i^shared β_k^shared + X_i^independent β_k^independent
    let mut g_raw = DMatrix::zeros(n, k);

    for ct in 0..k {
        // Shared effects
        if !genetic_effects.shared_causal_indices.is_empty() {
            let shared_idx = &genetic_effects.shared_causal_indices;

            let mut x_shared = DMatrix::zeros(n, shared_idx.len());
            for (j, &snp_idx) in shared_idx.iter().enumerate() {
                x_shared.set_column(j, &genotypes.column(snp_idx));
            }

            let beta_shared_k = DMatrix::from_iterator(
                shared_idx.len(),
                1,
                genetic_effects.shared_effect_sizes.row(ct).iter().copied(),
            );

            let g_shared_k = x_shared * beta_shared_k;

            for i in 0..n {
                g_raw[(i, ct)] += g_shared_k[(i, 0)];
            }
        }

        // Independent effects
        if !genetic_effects.independent_causal_indices[ct].is_empty() {
            let indep_idx = &genetic_effects.independent_causal_indices[ct];

            let mut x_indep = DMatrix::zeros(n, indep_idx.len());
            for (j, &snp_idx) in indep_idx.iter().enumerate() {
                x_indep.set_column(j, &genotypes.column(snp_idx));
            }

            let beta_indep_k = DMatrix::from_iterator(
                indep_idx.len(),
                1,
                genetic_effects
                    .independent_effect_sizes
                    .row(ct)
                    .iter()
                    .take(indep_idx.len())
                    .copied(),
            );

            let g_indep_k = x_indep * beta_indep_k;

            for i in 0..n {
                g_raw[(i, ct)] += g_indep_k[(i, 0)];
            }
        }
    }

    // ε: N × K
    let eps_raw = DMatrix::rnorm(n, k);

    // Standardize and scale by variance components
    let h2_noise = 1.0 - h2_genetic;

    let mut g_std = g_raw;
    g_std.scale_columns_inplace();
    g_std *= h2_genetic.sqrt();

    let mut eps_std = eps_raw;
    eps_std.scale_columns_inplace();
    eps_std *= h2_noise.sqrt();

    // Y = G + ε
    let mut y = g_std.clone();
    for ct in 0..k {
        for i in 0..n {
            y[(i, ct)] += eps_std[(i, ct)];
        }
    }

    info!("Phenotypes generated successfully");

    Ok(IndvPhenotypes {
        phenotypes: y,
        genetic_values: g_std,
        environmental_noise: eps_std,
    })
}
