use nalgebra::DMatrix;
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Summary statistics for one SNP-trait pair
#[derive(Debug, Clone)]
pub struct SumstatRecord {
    pub trait_idx: usize,
    pub snp_idx: usize,
    pub beta: f32,
    pub se: f32,
    pub z: f32,
    pub pvalue: f64,
}

/// Compute marginal OLS for one block of SNPs against all T traits.
///
/// For each SNP j and trait t:
///   beta_hat = (X_j' Y_t) / (X_j' X_j)
///   RSS = Y_t'Y_t - beta_hat * X_j'Y_t
///   SE = sqrt(RSS / (n-2)) / sqrt(X_j'X_j)
///   z = beta_hat / SE
///   p from Student's t(n-2)
pub fn compute_block_sumstats(
    x_block: &DMatrix<f32>,
    phenotypes: &DMatrix<f32>,
    yty_diag: &[f32],
    global_snp_offset: usize,
) -> Vec<SumstatRecord> {
    let n = x_block.nrows() as f32;
    let m_b = x_block.ncols();
    let t = phenotypes.ncols();
    let df = (x_block.nrows() as f64) - 2.0;

    // X'X diagonal: M_b vector
    let xtx_diag: Vec<f32> = (0..m_b)
        .map(|j| {
            let col = x_block.column(j);
            col.dot(&col)
        })
        .collect();

    // X'Y: M_b x T (BLAS dgemm)
    let xty = x_block.transpose() * phenotypes;

    // Student's t distribution for p-values
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();

    let mut records = Vec::with_capacity(m_b * t);

    for j in 0..m_b {
        let xtx_j = xtx_diag[j];
        if xtx_j < 1e-10 {
            // Monomorphic SNP or near-zero variance
            for trait_idx in 0..t {
                records.push(SumstatRecord {
                    trait_idx,
                    snp_idx: global_snp_offset + j,
                    beta: 0.0,
                    se: f32::NAN,
                    z: 0.0,
                    pvalue: 1.0,
                });
            }
            continue;
        }

        for trait_idx in 0..t {
            let xty_jt = xty[(j, trait_idx)];
            let beta = xty_jt / xtx_j;

            // RSS = yTy - beta * xTy
            let rss = (yty_diag[trait_idx] - beta * xty_jt).max(0.0);
            let se = (rss / (n - 2.0)).sqrt() / xtx_j.sqrt();

            let z = if se > 1e-10 { beta / se } else { 0.0 };

            // Two-sided p-value from Student's t
            let pvalue = if z.is_finite() {
                2.0 * (1.0 - t_dist.cdf(z.abs() as f64))
            } else {
                1.0
            };

            records.push(SumstatRecord {
                trait_idx,
                snp_idx: global_snp_offset + j,
                beta,
                se,
                z,
                pvalue,
            });
        }
    }

    records
}

/// Precompute Y'Y diagonal (one value per trait).
pub fn compute_yty_diagonal(phenotypes: &DMatrix<f32>) -> Vec<f32> {
    let t = phenotypes.ncols();
    (0..t)
        .map(|j| {
            let col = phenotypes.column(j);
            col.dot(&col)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_util::traits::SampleOps;

    #[test]
    fn test_compute_block_sumstats_null() {
        // Under the null (no true signal), z-scores should be ~N(0,1)
        let n = 1000;
        let m = 50;
        let t = 3;

        let x = DMatrix::<f32>::rnorm(n, m);
        let y = DMatrix::<f32>::rnorm(n, t);
        let yty = compute_yty_diagonal(&y);

        let records = compute_block_sumstats(&x, &y, &yty, 0);
        assert_eq!(records.len(), m * t);

        // Check z-scores are roughly N(0,1) under null
        let z_vals: Vec<f32> = records.iter().map(|r| r.z).collect();
        let mean_z: f32 = z_vals.iter().sum::<f32>() / z_vals.len() as f32;
        let var_z: f32 =
            z_vals.iter().map(|z| (z - mean_z).powi(2)).sum::<f32>() / z_vals.len() as f32;

        assert!(
            mean_z.abs() < 0.3,
            "Mean z-score under null too large: {}",
            mean_z
        );
        assert!(
            (var_z - 1.0).abs() < 0.5,
            "Variance of z-scores under null too far from 1: {}",
            var_z
        );
    }

    #[test]
    fn test_compute_block_sumstats_signal() {
        let n = 500;
        let m = 10;

        let x = DMatrix::<f32>::rnorm(n, m);
        // Y = X[:,0] * 2.0 + noise (strong signal on first SNP)
        let mut y = DMatrix::<f32>::rnorm(n, 1);
        for i in 0..n {
            y[(i, 0)] += 2.0 * x[(i, 0)];
        }
        let yty = compute_yty_diagonal(&y);

        let records = compute_block_sumstats(&x, &y, &yty, 100);
        assert_eq!(records.len(), m);

        // First SNP should have large |z|
        let z_first = records[0].z.abs();
        assert!(z_first > 5.0, "Causal SNP z-score too small: {}", z_first);

        // SNP index should have correct offset
        assert_eq!(records[0].snp_idx, 100);
        assert_eq!(records[5].snp_idx, 105);
    }
}
