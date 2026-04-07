use anyhow::Result;
use log::info;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;

use super::ld_block::LdBlock;

/// Minimum singular value ratio (relative to max) to keep in SVD truncation.
const SVD_TRUNCATION_RATIO: f32 = 1e-4;

/// Core SVD-based PRS: Ŷ = U * diag(w(d)) * V' * z
///
/// `sv_weight` maps (singular_value, max_singular_value) to a weight.
fn compute_block_prs_svd(
    x_block: &DMatrix<f32>,
    z_block: &DMatrix<f32>,
    sv_weight: impl Fn(f32, f32) -> f32,
) -> Result<DMatrix<f32>> {
    let n = x_block.nrows();
    let m = x_block.ncols();
    let t = z_block.ncols();

    if m == 0 || n == 0 {
        return Ok(DMatrix::<f32>::zeros(n, t));
    }

    let mut x_scaled = x_block.clone();
    x_scaled.scale_columns_inplace();
    x_scaled *= 1.0 / (n as f32).sqrt();

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

    let max_sv = singular_values.iter().cloned().fold(0.0f32, f32::max);
    let vt_z = vt * z_block;
    let k = singular_values.len();
    let mut weighted_vt_z = vt_z;
    for i in 0..k {
        weighted_vt_z
            .row_mut(i)
            .scale_mut(sv_weight(singular_values[i], max_sv));
    }

    Ok(u * weighted_vt_z)
}

/// Compute polygenic scores for a single LD block using truncated SVD inverse.
pub fn compute_block_polygenic_scores(
    x_block: &DMatrix<f32>,
    z_block: &DMatrix<f32>,
) -> Result<DMatrix<f32>> {
    compute_block_prs_svd(x_block, z_block, |d, max_sv| {
        let threshold = max_sv * SVD_TRUNCATION_RATIO;
        if d > threshold {
            1.0 / d
        } else {
            0.0
        }
    })
}

/// Compute ridge-regularized polygenic scores for a single LD block.
///
/// Uses `1/(d + λ)` instead of `1/d` with truncation.
pub fn compute_block_polygenic_scores_ridge(
    x_block: &DMatrix<f32>,
    z_block: &DMatrix<f32>,
    lambda: f32,
) -> Result<DMatrix<f32>> {
    compute_block_prs_svd(x_block, z_block, |d, _| 1.0 / (d + lambda))
}

/// Parallel block-level PRS computation.
fn compute_all_prs(
    genotypes: &DMatrix<f32>,
    zscores: &DMatrix<f32>,
    blocks: &[LdBlock],
    block_fn: impl Fn(&DMatrix<f32>, &DMatrix<f32>) -> Result<DMatrix<f32>> + Sync,
) -> Result<DMatrix<f32>> {
    let n = genotypes.nrows();
    let t = zscores.ncols();

    let block_results: Result<Vec<DMatrix<f32>>> = blocks
        .par_iter()
        .map(|block| {
            let block_m = block.num_snps();
            let x_block = genotypes.columns(block.snp_start, block_m).clone_owned();
            let z_block = zscores.rows(block.snp_start, block_m).clone_owned();
            block_fn(&x_block, &z_block)
        })
        .collect();

    let mut yhat = DMatrix::<f32>::zeros(n, t);
    for block_yhat in block_results? {
        yhat += block_yhat;
    }
    Ok(yhat)
}

/// Compute ridge-regularized polygenic scores across all LD blocks (parallel).
pub fn compute_all_polygenic_scores_ridge(
    genotypes: &DMatrix<f32>,
    zscores: &DMatrix<f32>,
    blocks: &[LdBlock],
    lambda: f32,
) -> Result<DMatrix<f32>> {
    info!(
        "Computing ridge PRS: {} individuals, {} traits, {} blocks, λ={:.4}",
        genotypes.nrows(),
        zscores.ncols(),
        blocks.len(),
        lambda,
    );
    let yhat = compute_all_prs(genotypes, zscores, blocks, |x, z| {
        compute_block_polygenic_scores_ridge(x, z, lambda)
    })?;
    info!("Ridge PRS computed: {} x {}", yhat.nrows(), yhat.ncols());
    Ok(yhat)
}

/// Compute polygenic scores across all LD blocks (parallel).
pub fn compute_all_polygenic_scores(
    genotypes: &DMatrix<f32>,
    zscores: &DMatrix<f32>,
    blocks: &[LdBlock],
) -> Result<DMatrix<f32>> {
    info!(
        "Computing polygenic scores: {} individuals, {} traits, {} blocks",
        genotypes.nrows(),
        zscores.ncols(),
        blocks.len(),
    );
    let yhat = compute_all_prs(genotypes, zscores, blocks, compute_block_polygenic_scores)?;
    info!(
        "Polygenic scores computed: {} x {}",
        yhat.nrows(),
        yhat.ncols(),
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
