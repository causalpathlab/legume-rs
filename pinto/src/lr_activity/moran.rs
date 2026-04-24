//! Global Moran's I per gene over a fixed edge list.

use crate::util::common::*;

/// Compute global Moran's I for every row of `x_gn` (genes × cells) over the
/// given edge list. Each edge `(i, j)` is treated as symmetric; the W matrix is
/// unweighted 0/1 so `W_sum = 2 · n_edges`.
///
///   I_g = (N / W_sum) · Σ_{(i,j)} w_ij (x_ig − x̄_g)(x_jg − x̄_g)
///                                     / Σ_i (x_ig − x̄_g)²
///
/// With symmetric W and `W_sum = 2·|E|`, the edge sum (counted once per
/// unordered pair) contributes 2× the directed sum, so I reduces to
///
///   I_g = (N / |E|) · Σ_e (x_ie − x̄_g)(x_je − x̄_g) / Σ_i (x_ig − x̄_g)²
///
/// Genes with zero (row) variance are returned as `NaN`.
pub fn global_moran_per_gene(x_gn: &Mat, edges: &[(usize, usize)]) -> DVec {
    let g = x_gn.nrows();
    let n = x_gn.ncols() as f32;
    let n_edges = edges.len() as f32;

    let mut means = DVec::zeros(g);
    for j in 0..x_gn.ncols() {
        means += x_gn.column(j);
    }
    means /= n;

    let mut den = DVec::zeros(g);
    for j in 0..x_gn.ncols() {
        let col = x_gn.column(j) - &means;
        den += col.map(|v| v * v);
    }

    // Center on the fly per edge — avoids allocating a second G×N matrix.
    let mut num = DVec::zeros(g);
    for &(i, j) in edges {
        let ci = x_gn.column(i) - &means;
        let cj = x_gn.column(j) - &means;
        num += ci.component_mul(&cj);
    }

    if n_edges == 0.0 {
        return DVec::from_element(g, f32::NAN);
    }

    let scale = n / n_edges;
    DVec::from_iterator(
        g,
        (0..g).map(|gi| {
            let d = den[gi];
            if d > 0.0 {
                scale * num[gi] / d
            } else {
                f32::NAN
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moran_perfect_positive() {
        // Linear chain 0-1-2-...-9 with x = 0,1,2,...,9
        // Neighboring values highly correlated -> I ≈ 1
        let n = 10;
        let mut x = Mat::zeros(1, n);
        for j in 0..n {
            x[(0, j)] = j as f32;
        }
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let i_g = global_moran_per_gene(&x, &edges);
        assert!(
            i_g[0] > 0.7,
            "expected strong positive autocorrelation, got {}",
            i_g[0]
        );
    }

    #[test]
    fn moran_perfect_negative() {
        // Alternating pattern: neighbors anti-correlated -> I negative
        let n = 10;
        let mut x = Mat::zeros(1, n);
        for j in 0..n {
            x[(0, j)] = if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let i_g = global_moran_per_gene(&x, &edges);
        assert!(
            i_g[0] < -0.7,
            "expected strong negative autocorrelation, got {}",
            i_g[0]
        );
    }

    #[test]
    fn moran_zero_variance_is_nan() {
        let n = 5;
        let x = Mat::from_element(1, n, 3.0);
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let i_g = global_moran_per_gene(&x, &edges);
        assert!(i_g[0].is_nan());
    }

    #[test]
    fn moran_multiple_genes() {
        let n = 6;
        let mut x = Mat::zeros(2, n);
        // gene 0: positive autocorr
        for j in 0..n {
            x[(0, j)] = j as f32;
        }
        // gene 1: alternating
        for j in 0..n {
            x[(1, j)] = if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let i_g = global_moran_per_gene(&x, &edges);
        assert!(i_g[0] > 0.5);
        assert!(i_g[1] < -0.5);
    }
}
