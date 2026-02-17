use anyhow::Result;
use log::info;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;

use super::ld_block::LdBlock;

/// Minimum singular value ratio (relative to max) to keep in SVD truncation.
const SVD_TRUNCATION_RATIO: f32 = 1e-4;

/// Compute polygenic scores for a single LD block.
///
/// Given block genotypes X_b (N × M_b) and z-scores z_b (M_b × T):
/// 1. Standardize X_b columns, scale by 1/√n
/// 2. Thin SVD: X_scaled = U D V'
/// 3. Ŷ_b = U D⁻¹ V' z_b (N × T)
///
/// Small singular values are truncated for numerical stability.
pub fn compute_block_polygenic_scores(
    x_block: &DMatrix<f32>,
    z_block: &DMatrix<f32>,
) -> Result<DMatrix<f32>> {
    let n = x_block.nrows();
    let m = x_block.ncols();
    let t = z_block.ncols();

    if m == 0 || n == 0 {
        return Ok(DMatrix::<f32>::zeros(n, t));
    }

    // Standardize columns and scale by 1/sqrt(n)
    let mut x_scaled = x_block.clone();
    x_scaled.scale_columns_inplace();
    let scale = 1.0 / (n as f32).sqrt();
    x_scaled *= scale;

    // Thin SVD
    let svd = x_scaled.svd(true, true);
    let u = svd
        .u
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("SVD failed to compute U"))?;
    let vt = svd
        .v_t
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("SVD failed to compute V'"))?;
    let singular_values = &svd.singular_values;

    // Determine truncation threshold
    let max_sv = singular_values.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_sv * SVD_TRUNCATION_RATIO;

    // Compute U * D^{-1} * V' * z_block
    // First: V' * z_block -> (k × M_b) * (M_b × T) -> k × T
    let vt_z = vt * z_block; // k × T

    // Apply D^{-1}: element-wise divide each row of vt_z by singular value
    let k = singular_values.len();
    let mut dinv_vt_z = DMatrix::<f32>::zeros(k, t);
    for i in 0..k {
        let sv = singular_values[i];
        if sv > threshold {
            let inv_sv = 1.0 / sv;
            for j in 0..t {
                dinv_vt_z[(i, j)] = vt_z[(i, j)] * inv_sv;
            }
        }
        // else: zero row (truncated)
    }

    // U * (D^{-1} V' z) -> (N × k) * (k × T) -> N × T
    let yhat = u * dinv_vt_z;
    Ok(yhat)
}

/// Compute polygenic scores across all LD blocks (parallel).
///
/// Returns Ŷ (N × T) = Σ_b Ŷ_b, the sum of per-block polygenic scores.
pub fn compute_all_polygenic_scores(
    genotypes: &DMatrix<f32>,
    zscores: &DMatrix<f32>,
    blocks: &[LdBlock],
) -> Result<DMatrix<f32>> {
    let n = genotypes.nrows();
    let t = zscores.ncols();

    info!(
        "Computing polygenic scores: {} individuals, {} traits, {} blocks",
        n,
        t,
        blocks.len()
    );

    // Parallel computation per block
    let block_results: Vec<DMatrix<f32>> = blocks
        .par_iter()
        .map(|block| {
            let block_m = block.num_snps();

            // Extract block genotypes
            let x_block = genotypes.columns(block.snp_start, block_m).clone_owned();

            // Extract block z-scores
            let z_block = zscores.rows(block.snp_start, block_m).clone_owned();

            compute_block_polygenic_scores(&x_block, &z_block)
                .expect("Failed to compute block polygenic scores")
        })
        .collect();

    // Sum across blocks
    let mut yhat = DMatrix::<f32>::zeros(n, t);
    for block_yhat in block_results {
        yhat += block_yhat;
    }

    info!(
        "Polygenic scores computed: {} x {}",
        yhat.nrows(),
        yhat.ncols()
    );

    Ok(yhat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_util::traits::SampleOps;

    #[test]
    fn test_block_polygenic_scores_shape() {
        let n = 50;
        let m = 20;
        let t = 5;

        let x = DMatrix::<f32>::rnorm(n, m);
        let z = DMatrix::<f32>::rnorm(m, t);

        let yhat = compute_block_polygenic_scores(&x, &z).unwrap();
        assert_eq!(yhat.nrows(), n);
        assert_eq!(yhat.ncols(), t);
    }

    #[test]
    fn test_all_polygenic_scores() {
        let n = 50;
        let m = 60;
        let t = 3;

        let genotypes = DMatrix::<f32>::rnorm(n, m);
        let zscores = DMatrix::<f32>::rnorm(m, t);

        let blocks = vec![
            LdBlock {
                block_idx: 0,
                snp_start: 0,
                snp_end: 30,
                chr: Box::from("chr1"),
                bp_start: 0,
                bp_end: 29000,
            },
            LdBlock {
                block_idx: 1,
                snp_start: 30,
                snp_end: 60,
                chr: Box::from("chr1"),
                bp_start: 30000,
                bp_end: 59000,
            },
        ];

        let yhat = compute_all_polygenic_scores(&genotypes, &zscores, &blocks).unwrap();
        assert_eq!(yhat.nrows(), n);
        assert_eq!(yhat.ncols(), t);

        // Should be non-zero
        assert!(yhat.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_empty_block() {
        let n = 10;
        let t = 3;
        let x = DMatrix::<f32>::zeros(n, 0);
        let z = DMatrix::<f32>::zeros(0, t);
        let yhat = compute_block_polygenic_scores(&x, &z).unwrap();
        assert_eq!(yhat.nrows(), n);
        assert_eq!(yhat.ncols(), t);
    }
}
