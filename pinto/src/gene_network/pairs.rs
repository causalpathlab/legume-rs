use crate::link_community::profiles::compute_gene_totals;
use crate::util::common::*;

/// Version of visit_gene_pair_deltas that accepts a pre-allocated buffer.
/// Reuse this buffer across multiple cells to avoid allocation overhead.
#[inline]
pub fn visit_gene_pair_deltas_with_buffer(
    rows: &[usize],
    vals: &[f32],
    gene_adj: &[Vec<(usize, usize)>],
    gene_means: &DVec,
    use_log1p: bool,
    gene_vals: &mut [f32],
    mut on_delta: impl FnMut(usize, f32),
) {
    // Populate dense array with transformed values
    for (&g, &v) in rows.iter().zip(vals.iter()) {
        gene_vals[g] = if use_log1p { v.ln_1p() } else { v };
    }

    // Visit gene pairs
    for &g1 in rows.iter() {
        let t_g1 = gene_vals[g1];
        for &(g2, edge_idx) in gene_adj[g1].iter() {
            let t_g2 = gene_vals[g2];
            if t_g2 >= 0.0 {
                // g2 is expressed in this cell
                let delta = t_g1 * t_g2 - gene_means[g1] * gene_means[g2];
                on_delta(edge_idx, delta);
            }
        }
    }

    // Clear for next use
    for &g in rows.iter() {
        gene_vals[g] = -1.0;
    }
}

/// Compute gene-level raw means: μ_g = E[x_g] across all cells.
///
/// Delegates to `compute_gene_totals` and divides by n_cells.
pub fn compute_gene_raw_means(data_vec: &SparseIoVec, block_size: usize) -> anyhow::Result<DVec> {
    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    info!(
        "Computing gene raw means across {} cells, {} genes",
        n_cells, n_genes
    );

    let totals = compute_gene_totals(data_vec, block_size)?;
    let means = DVec::from_iterator(
        n_genes,
        totals.into_iter().map(|t| t as f32 / n_cells as f32),
    );
    Ok(means)
}

/// Find elbow threshold from a set of values using the kneedle method.
///
/// Sorts positive values descending, normalizes rank and value to [0, 1],
/// then finds the point with maximum perpendicular distance from the line
/// connecting the first and last points.
///
/// Returns (threshold, elbow_rank). Threshold is 0.0 if no clear elbow
/// exists (all values similar or too few points).
pub fn elbow_threshold(values: &[f32]) -> (f32, usize) {
    // Only consider positive values for elbow detection
    let mut positive: Vec<f32> = values.iter().copied().filter(|&v| v > 0.0).collect();
    if positive.len() < 3 {
        return (0.0, 0);
    }

    positive.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let n = positive.len();

    // Normalize: x = rank/(n-1) in [0,1], y = value/max in [0,1]
    let y_max = positive[0] as f64;
    if y_max < 1e-12 {
        return (0.0, 0);
    }

    let x1 = 1.0_f64;
    let y1 = positive[n - 1] as f64 / y_max;

    // Line from (0, 1) to (1, y1)
    let dx = x1;
    let dy = y1 - 1.0;
    let line_len = (dx * dx + dy * dy).sqrt();

    let mut max_dist = 0.0_f64;
    let mut elbow_idx = 0;

    for (i, &val) in positive.iter().enumerate().take(n - 1).skip(1) {
        let px = i as f64 / (n - 1) as f64;
        let py = val as f64 / y_max - 1.0;
        let dist = (px * dy - py * dx).abs() / line_len;
        if dist > max_dist {
            max_dist = dist;
            elbow_idx = i;
        }
    }

    (positive[elbow_idx], elbow_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience wrapper for tests: allocates a buffer internally.
    fn visit_gene_pair_deltas(
        rows: &[usize],
        vals: &[f32],
        gene_adj: &[Vec<(usize, usize)>],
        gene_means: &DVec,
        use_log1p: bool,
        on_delta: impl FnMut(usize, f32),
    ) {
        let n_genes = gene_means.len();
        let mut gene_vals = vec![-1.0f32; n_genes];
        visit_gene_pair_deltas_with_buffer(
            rows,
            vals,
            gene_adj,
            gene_means,
            use_log1p,
            &mut gene_vals,
            on_delta,
        )
    }

    #[test]
    fn test_elbow_threshold_clear_elbow() {
        // High values followed by a sharp drop → elbow should detect the transition
        let values = vec![100.0, 90.0, 80.0, 70.0, 10.0, 5.0, 2.0, 1.0];
        let (threshold, rank) = elbow_threshold(&values);
        assert!(threshold > 0.0, "threshold should be positive");
        assert!(rank > 0 && rank < values.len(), "rank={}", rank);
    }

    #[test]
    fn test_elbow_threshold_all_zeros() {
        let values = vec![0.0, 0.0, 0.0];
        let (threshold, rank) = elbow_threshold(&values);
        assert_eq!(threshold, 0.0);
        assert_eq!(rank, 0);
    }

    #[test]
    fn test_elbow_threshold_too_few() {
        // Fewer than 3 positive values → no elbow
        let values = vec![1.0, 2.0];
        let (threshold, rank) = elbow_threshold(&values);
        assert_eq!(threshold, 0.0);
        assert_eq!(rank, 0);
    }

    #[test]
    fn test_elbow_threshold_with_negatives() {
        // Negative values are ignored
        let values = vec![100.0, 50.0, 10.0, -5.0, -10.0];
        let (threshold, _rank) = elbow_threshold(&values);
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_elbow_threshold_uniform() {
        // All equal → elbow at some interior point
        let values = vec![5.0; 10];
        let (threshold, _rank) = elbow_threshold(&values);
        // With uniform values the curve is flat, threshold is one of the values
        assert_eq!(threshold, 5.0);
    }

    #[test]
    fn test_visit_gene_pair_deltas() {
        // 3 genes: 0, 1, 2. Edges: (0,1) idx=0, (0,2) idx=1
        let gene_adj: Vec<Vec<(usize, usize)>> = vec![
            vec![(1, 0), (2, 1)], // gene 0 → gene 1 (edge 0), gene 2 (edge 1)
            vec![],               // gene 1
            vec![],               // gene 2
        ];
        let gene_means = DVec::from_vec(vec![1.0, 2.0, 3.0]);

        // Cell has genes 0 and 1 present (raw counts)
        let rows = vec![0usize, 1];
        let vals = vec![4.0f32, 6.0];

        let mut deltas = vec![];
        visit_gene_pair_deltas(
            &rows,
            &vals,
            &gene_adj,
            &gene_means,
            false,
            |edge_idx, delta| {
                deltas.push((edge_idx, delta));
            },
        );

        // Edge (0,1): 4.0 * 6.0 - 1.0 * 2.0 = 22.0
        assert_eq!(deltas.len(), 1); // gene 2 not present → edge 1 skipped
        assert_eq!(deltas[0].0, 0);
        assert!((deltas[0].1 - 22.0).abs() < 1e-6);
    }

    #[test]
    fn test_visit_gene_pair_deltas_log1p() {
        let gene_adj: Vec<Vec<(usize, usize)>> = vec![vec![(1, 0)], vec![]];
        // When use_log1p=true, gene_means should be log1p-scale means
        let gene_means = DVec::from_vec(vec![1.0f32.ln_1p(), 2.0f32.ln_1p()]);

        let rows = vec![0usize, 1];
        let vals = vec![1.0f32, 2.0];

        let mut deltas = vec![];
        visit_gene_pair_deltas(
            &rows,
            &vals,
            &gene_adj,
            &gene_means,
            true,
            |edge_idx, delta| {
                deltas.push((edge_idx, delta));
            },
        );

        let expected = 1.0f32.ln_1p() * 2.0f32.ln_1p() - gene_means[0] * gene_means[1];
        assert_eq!(deltas.len(), 1);
        assert!((deltas[0].1 - expected).abs() < 1e-6);
    }
}
