use anyhow::Result;
use log::info;
use matrix_util::traits::{MatOps, SampleOps};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Parameters for low-rank confounder generation
#[derive(Debug, Clone)]
pub struct ConfounderParams {
    /// Number of observed confounder columns (L)
    pub num_confounders: usize,
    /// Low-rank dimension for generating confounders (R)
    pub num_hidden_factors: usize,
    /// Proportion of phenotype variance explained by confounders
    pub pve_confounders: f32,
}

/// Generate low-rank confounder matrix C (N x L).
///
/// C = orthogonalize(R_NxR) * Lambda_RxL, then column-standardize.
/// R is a random matrix, orthogonalized via QR decomposition.
pub fn generate_confounder_matrix(
    n: usize,
    params: &ConfounderParams,
    seed: u64,
) -> Result<DMatrix<f32>> {
    let l = params.num_confounders;
    let r = params.num_hidden_factors.min(n).min(l);

    if l == 0 {
        return Ok(DMatrix::zeros(n, 0));
    }

    info!(
        "Generating confounder matrix: {} individuals, {} confounders, {} hidden factors",
        n, l, r
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate random N x R matrix and orthogonalize via QR
    let random_mat = DMatrix::from_fn(n, r, |_, _| normal.sample(&mut rng) as f32);

    let qr = random_mat.qr();
    let q = qr.q(); // N x R orthogonal columns

    // Generate R x L mixing matrix
    let lambda = DMatrix::from_fn(r, l, |_, _| normal.sample(&mut rng) as f32);

    // C = Q * Lambda -> N x L
    let mut c = q * lambda;

    // Column-standardize
    c.scale_columns_inplace();

    info!("Generated confounder matrix: {} x {}", c.nrows(), c.ncols());
    Ok(c)
}

/// Compose final phenotype from genetic values, confounders, and noise.
///
/// Y_t = sqrt(h2) * standardize(G_t) + sqrt(pve_conf) * standardize(C * gamma_t)
///     + sqrt(1 - h2 - pve_conf) * standardize(eps_t)
pub fn compose_phenotype(
    genetic_values: &DMatrix<f32>,
    confounder_matrix: &DMatrix<f32>,
    h2: f32,
    pve_conf: f32,
    num_traits: usize,
    seed: u64,
) -> Result<DMatrix<f32>> {
    let n = genetic_values.nrows();
    let t = num_traits;

    if h2 + pve_conf > 1.0 + 1e-6 {
        anyhow::bail!(
            "h2 ({}) + pve_conf ({}) cannot exceed 1.0",
            h2,
            pve_conf
        );
    }
    let pve_noise = (1.0 - h2 - pve_conf).max(0.0);

    info!(
        "Composing phenotype: h2={:.3}, pve_conf={:.3}, pve_noise={:.3}",
        h2, pve_conf, pve_noise
    );

    // Standardize genetic values
    let mut g_std = genetic_values.clone();
    g_std.scale_columns_inplace();
    g_std *= h2.sqrt();

    // Confounder contribution
    let mut conf_component = DMatrix::zeros(n, t);
    if pve_conf > 0.0 && confounder_matrix.ncols() > 0 {
        let l = confounder_matrix.ncols();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, (1.0 / l as f64).sqrt()).unwrap();

        // gamma: L x T mixing weights
        let gamma = DMatrix::from_fn(l, t, |_, _| normal.sample(&mut rng) as f32);
        conf_component = confounder_matrix * gamma;
        conf_component.scale_columns_inplace();
        conf_component *= pve_conf.sqrt();
    }

    // Noise
    let mut eps = DMatrix::<f32>::rnorm(n, t);
    eps.scale_columns_inplace();
    eps *= pve_noise.sqrt();

    // Combine: Y = G + C + eps
    let mut y = g_std;
    for j in 0..t {
        for i in 0..n {
            y[(i, j)] += conf_component[(i, j)] + eps[(i, j)];
        }
    }

    info!("Composed phenotype matrix: {} x {}", y.nrows(), y.ncols());
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_confounder_matrix() {
        let params = ConfounderParams {
            num_confounders: 10,
            num_hidden_factors: 5,
            pve_confounders: 0.1,
        };
        let c = generate_confounder_matrix(100, &params, 42).unwrap();
        assert_eq!(c.nrows(), 100);
        assert_eq!(c.ncols(), 10);

        // Columns should be approximately standardized (mean ~0, std ~1)
        for j in 0..10 {
            let col = c.column(j);
            let mean: f32 = col.iter().sum::<f32>() / col.len() as f32;
            assert!(mean.abs() < 0.2, "Column {} mean too large: {}", j, mean);
        }
    }

    #[test]
    fn test_generate_empty_confounders() {
        let params = ConfounderParams {
            num_confounders: 0,
            num_hidden_factors: 0,
            pve_confounders: 0.0,
        };
        let c = generate_confounder_matrix(100, &params, 42).unwrap();
        assert_eq!(c.nrows(), 100);
        assert_eq!(c.ncols(), 0);
    }

    #[test]
    fn test_compose_phenotype_no_confounders() {
        let n = 200;
        let t = 5;
        let g = DMatrix::<f32>::rnorm(n, t);
        let c = DMatrix::zeros(n, 0);

        let y = compose_phenotype(&g, &c, 0.4, 0.0, t, 42).unwrap();
        assert_eq!(y.nrows(), n);
        assert_eq!(y.ncols(), t);
    }

    #[test]
    fn test_compose_phenotype_with_confounders() {
        let n = 200;
        let t = 5;
        let g = DMatrix::<f32>::rnorm(n, t);
        let params = ConfounderParams {
            num_confounders: 10,
            num_hidden_factors: 5,
            pve_confounders: 0.1,
        };
        let c = generate_confounder_matrix(n, &params, 42).unwrap();

        let y = compose_phenotype(&g, &c, 0.4, 0.1, t, 43).unwrap();
        assert_eq!(y.nrows(), n);
        assert_eq!(y.ncols(), t);
    }

    #[test]
    fn test_compose_phenotype_h2_plus_pve_too_large() {
        let n = 100;
        let t = 3;
        let g = DMatrix::<f32>::rnorm(n, t);
        let c = DMatrix::zeros(n, 0);

        let result = compose_phenotype(&g, &c, 0.8, 0.5, t, 42);
        assert!(result.is_err());
    }
}
