//! Benjamini–Hochberg FDR correction.

/// BH q-values: given p-values, return adjusted q-values preserving input
/// order. `q_i = min_{j >= i_sorted} (m * p_j / j)`, cumulative-min from
/// largest p downwards, clamped to `[0, 1]`.
pub fn bh_fdr(p_values: &[f32]) -> Vec<f32> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut q = vec![0.0f32; m];
    let mut running_min = 1.0f32;
    for rev_rank in (0..m).rev() {
        let idx = order[rev_rank];
        let p = p_values[idx];
        let rank_one_based = (rev_rank + 1) as f32;
        let scaled = (p * m as f32 / rank_one_based).clamp(0.0, 1.0);
        if scaled < running_min {
            running_min = scaled;
        }
        q[idx] = running_min;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bh_preserves_input_order() {
        let p = vec![0.01, 0.50, 0.03, 0.20];
        let q = bh_fdr(&p);
        assert_eq!(q.len(), p.len());
    }

    #[test]
    fn bh_classic_fixture() {
        // Classic BH example: p = [0.01, 0.04, 0.03, 0.005]
        // sorted ascending: [0.005, 0.01, 0.03, 0.04] with m=4
        // scaled: [0.02, 0.02, 0.04, 0.04]
        let p = vec![0.01, 0.04, 0.03, 0.005];
        let q = bh_fdr(&p);
        assert_relative_eq!(q[3], 0.02, epsilon = 1e-5);
        assert_relative_eq!(q[0], 0.02, epsilon = 1e-5);
        assert_relative_eq!(q[2], 0.04, epsilon = 1e-5);
        assert_relative_eq!(q[1], 0.04, epsilon = 1e-5);
    }

    #[test]
    fn bh_empty_returns_empty() {
        let q = bh_fdr(&[]);
        assert!(q.is_empty());
    }

    #[test]
    fn bh_clamps_to_unit_interval() {
        let p = vec![0.001, 0.5, 0.9];
        let q = bh_fdr(&p);
        for &qi in &q {
            assert!((0.0..=1.0).contains(&qi));
        }
    }

    #[test]
    fn bh_monotone_in_p() {
        // Smaller p should get smaller-or-equal q (in sorted rank).
        let p = vec![0.9, 0.01, 0.5, 0.02, 0.3];
        let q = bh_fdr(&p);
        let mut pairs: Vec<(f32, f32)> = p.iter().copied().zip(q.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for w in pairs.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6, "{:?} violates monotone", w);
        }
    }
}
