use anyhow::Result;
use log::info;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Factor model for gene-gene correlations via W × Z
///
/// Model: Cell type mean expression = W × Z
/// - W: G × F (gene loadings)
/// - Z: F × K (factor scores per cell type)
/// - M = W × Z: G × K (cell type mean expression)
#[derive(Debug, Clone)]
pub struct FactorModel {
    /// Gene loadings: G × F
    pub gene_loadings: DMatrix<f32>,

    /// Factor-celltype scores: F × K
    pub factor_celltype: DMatrix<f32>,

    /// Number of factors
    pub num_factors: usize,
}

impl FactorModel {
    /// Compute cell type mean expression: M = W × Z
    pub fn cell_type_means(&self) -> DMatrix<f32> {
        &self.gene_loadings * &self.factor_celltype
    }

    /// Compute individual aggregated expression: Y = M × Π^T
    /// where Π is N × K (individuals × cell type fractions)
    pub fn individual_expression(&self, cell_fractions: &DMatrix<f32>) -> DMatrix<f32> {
        self.cell_type_means() * cell_fractions.transpose()
    }
}

/// Simulate a factor model with random gene loadings and factor scores
///
/// # Arguments
/// * `num_genes` - Number of genes (G)
/// * `num_factors` - Number of latent factors (F)
/// * `num_cell_types` - Number of cell types (K)
/// * `gene_loading_std` - Standard deviation for gene loadings
/// * `factor_score_std` - Standard deviation for factor scores
/// * `seed` - Random seed
///
/// # Returns
/// FactorModel with W ~ N(0, gene_loading_std²) and Z ~ N(0, factor_score_std²)
pub fn simulate_factor_model(
    num_genes: usize,
    num_factors: usize,
    num_cell_types: usize,
    gene_loading_std: f32,
    factor_score_std: f32,
    seed: u64,
) -> Result<FactorModel> {
    info!(
        "Simulating factor model: {} genes, {} factors, {} cell types",
        num_genes, num_factors, num_cell_types
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f64, 1.0).unwrap();

    // Sample gene loadings W: G × F from N(0, σ_w²)
    let gene_loadings = DMatrix::from_fn(num_genes, num_factors, |_, _| {
        normal.sample(&mut rng) as f32 * gene_loading_std
    });

    // Sample factor-celltype scores Z: F × K from N(0, σ_z²)
    let mut factor_celltype = DMatrix::from_fn(num_factors, num_cell_types, |_, _| {
        normal.sample(&mut rng) as f32 * factor_score_std
    });

    // Orthogonalize factors (optional, makes factors independent)
    // Using Gram-Schmidt on factor-celltype matrix
    factor_celltype = orthogonalize_columns(&factor_celltype);

    info!("Factor model simulated successfully");

    Ok(FactorModel {
        gene_loadings,
        factor_celltype,
        num_factors,
    })
}

/// Orthogonalize columns using Gram-Schmidt process
fn orthogonalize_columns(mat: &DMatrix<f32>) -> DMatrix<f32> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    let mut result = DMatrix::zeros(nrows, ncols);

    for j in 0..ncols {
        let mut v = mat.column(j).clone_owned();

        // Subtract projections onto previous columns
        for i in 0..j {
            let u = result.column(i);
            let proj = u.dot(&v) / u.dot(&u);
            v -= u * proj;
        }

        // Normalize
        let norm = v.norm();
        if norm > 1e-10 {
            v /= norm;
        }

        result.set_column(j, &v);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_factor_model() {
        let model = simulate_factor_model(100, 5, 3, 1.0, 1.0, 42).unwrap();

        assert_eq!(model.gene_loadings.nrows(), 100);
        assert_eq!(model.gene_loadings.ncols(), 5);
        assert_eq!(model.factor_celltype.nrows(), 5);
        assert_eq!(model.factor_celltype.ncols(), 3);
        assert_eq!(model.num_factors, 5);
    }

    #[test]
    fn test_cell_type_means() {
        let model = simulate_factor_model(100, 5, 3, 1.0, 1.0, 42).unwrap();
        let means = model.cell_type_means();

        // Should be G × K
        assert_eq!(means.nrows(), 100);
        assert_eq!(means.ncols(), 3);
    }

    #[test]
    fn test_individual_expression() {
        let model = simulate_factor_model(100, 5, 3, 1.0, 1.0, 42).unwrap();

        // Simulate cell fractions: 50 individuals × 3 cell types
        let cell_fractions = DMatrix::from_element(50, 3, 1.0 / 3.0);

        let expr = model.individual_expression(&cell_fractions);

        // Should be G × N
        assert_eq!(expr.nrows(), 100);
        assert_eq!(expr.ncols(), 50);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let m1 = simulate_factor_model(50, 3, 2, 1.0, 1.0, 123).unwrap();
        let m2 = simulate_factor_model(50, 3, 2, 1.0, 1.0, 123).unwrap();
        assert_eq!(m1.gene_loadings, m2.gene_loadings);
        assert_eq!(m1.factor_celltype, m2.factor_celltype);

        // Different seed should give different results
        let m3 = simulate_factor_model(50, 3, 2, 1.0, 1.0, 999).unwrap();
        assert_ne!(m1.gene_loadings, m3.gene_loadings);
    }

    #[test]
    fn test_orthogonalization() {
        let mat = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 0.0, 0.0, 1.0]);

        let orth = orthogonalize_columns(&mat);

        // Check columns are orthogonal
        let col0 = orth.column(0);
        let col1 = orth.column(1);
        let dot_product = col0.dot(&col1);

        assert!(dot_product.abs() < 1e-6);
    }
}
