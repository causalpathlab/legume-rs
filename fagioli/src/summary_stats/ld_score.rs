use nalgebra::DMatrix;

/// LD score for a single SNP
#[derive(Debug, Clone)]
pub struct LdScoreRecord {
    pub snp_idx: usize,
    pub l2: f32,
    pub num_snps_in_block: usize,
}

/// Compute within-block LD scores.
///
/// For each SNP j in the block:
///   l_j = sum_k r^2_{jk}
/// where r_{jk} = (X_j' X_k) / (n * sigma_j * sigma_k)
///
/// Implementation: standardize X_block columns, compute R = X'X / n,
/// then l_j = sum_k R[j,k]^2
pub fn compute_block_ld_scores(
    x_block: &DMatrix<f32>,
    global_snp_offset: usize,
) -> Vec<LdScoreRecord> {
    let n = x_block.nrows() as f32;
    let m_b = x_block.ncols();

    if m_b == 0 {
        return Vec::new();
    }

    // Standardize columns (mean 0, variance 1)
    let mut x_std = x_block.clone();
    let nrows = x_std.nrows();
    for j in 0..m_b {
        let mean = x_std.column(j).iter().sum::<f32>() / n;
        for i in 0..nrows {
            x_std[(i, j)] -= mean;
        }
        let var: f32 = x_std.column(j).iter().map(|v| v * v).sum::<f32>() / n;
        let std = var.sqrt();
        if std > 1e-10 {
            for i in 0..nrows {
                x_std[(i, j)] /= std;
            }
        }
    }

    // Correlation matrix: R = X_std' X_std / n (M_b x M_b)
    let r_mat = x_std.transpose() * &x_std;
    // r_mat is scaled by n, so each element is r_jk * n
    // We need r_jk = r_mat[j,k] / n

    // LD scores: l_j = sum_k (r_mat[j,k] / n)^2
    let mut records = Vec::with_capacity(m_b);
    for j in 0..m_b {
        let mut l2 = 0.0f32;
        for k in 0..m_b {
            let r_jk = r_mat[(j, k)] / n;
            l2 += r_jk * r_jk;
        }

        records.push(LdScoreRecord {
            snp_idx: global_snp_offset + j,
            l2,
            num_snps_in_block: m_b,
        });
    }

    records
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_util::traits::SampleOps;

    #[test]
    fn test_ld_scores_identity() {
        // Independent SNPs: LD scores should be approximately 1.0
        // (only r^2_{jj} = 1 contributes)
        let n = 5000;
        let m = 20;
        let x = DMatrix::<f32>::rnorm(n, m);

        let scores = compute_block_ld_scores(&x, 0);
        assert_eq!(scores.len(), m);

        for record in &scores {
            // Each score should be close to 1.0 (self-LD)
            // plus small noise from finite sample correlations
            assert!(
                record.l2 > 0.5 && record.l2 < 3.0,
                "LD score out of range: {} for SNP {}",
                record.l2,
                record.snp_idx
            );
        }
    }

    #[test]
    fn test_ld_scores_correlated() {
        // Create correlated SNPs (should have higher LD scores)
        let n = 1000;
        let m = 10;
        let base = DMatrix::<f32>::rnorm(n, 1);
        let noise = DMatrix::<f32>::rnorm(n, m);

        // All SNPs are correlated with the base
        let mut x = noise * 0.3;
        for j in 0..m {
            for i in 0..n {
                x[(i, j)] += 0.7 * base[(i, 0)];
            }
        }

        let scores = compute_block_ld_scores(&x, 0);

        // LD scores should be significantly above 1.0
        let mean_l2: f32 = scores.iter().map(|r| r.l2).sum::<f32>() / m as f32;
        assert!(
            mean_l2 > 2.0,
            "Mean LD score for correlated SNPs too low: {}",
            mean_l2
        );
    }

    #[test]
    fn test_ld_scores_offset() {
        let n = 100;
        let m = 5;
        let x = DMatrix::<f32>::rnorm(n, m);

        let scores = compute_block_ld_scores(&x, 500);

        assert_eq!(scores[0].snp_idx, 500);
        assert_eq!(scores[4].snp_idx, 504);
        assert_eq!(scores[0].num_snps_in_block, 5);
    }
}
