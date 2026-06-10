use super::*;
use approx::assert_relative_eq;

#[test]
fn test_aggregate_rows_sums_match() {
    // 6 features, 3 samples
    let data = DMatrix::from_row_slice(
        6,
        3,
        &[
            1.0, 2.0, 3.0, // feature 0
            4.0, 5.0, 6.0, // feature 1
            7.0, 8.0, 9.0, // feature 2
            10.0, 11.0, 12.0, // feature 3
            13.0, 14.0, 15.0, // feature 4
            16.0, 17.0, 18.0, // feature 5
        ],
    );

    let fc = FeatureCoarsening {
        fine_to_coarse: vec![0, 0, 1, 1, 2, 2],
        coarse_to_fine: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        num_coarse: 3,
    };

    let agg = fc.aggregate_rows_ds(&data);
    assert_eq!(agg.nrows(), 3);
    assert_eq!(agg.ncols(), 3);

    // Group 0: features 0+1
    assert_relative_eq!(agg[(0, 0)], 5.0);
    assert_relative_eq!(agg[(0, 1)], 7.0);
    assert_relative_eq!(agg[(0, 2)], 9.0);

    // Group 1: features 2+3
    assert_relative_eq!(agg[(1, 0)], 17.0);

    // Group 2: features 4+5
    assert_relative_eq!(agg[(2, 0)], 29.0);

    // Column sums should be preserved
    let orig_col_sum: f32 = data.column(0).iter().sum();
    let agg_col_sum: f32 = agg.column(0).iter().sum();
    assert_relative_eq!(orig_col_sum, agg_col_sum);
}

#[test]
fn test_aggregate_columns_nd() {
    // 2 samples, 4 features → 2 groups
    let data = DMatrix::from_row_slice(
        2,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, // sample 0
            5.0, 6.0, 7.0, 8.0, // sample 1
        ],
    );

    let fc = FeatureCoarsening {
        fine_to_coarse: vec![0, 0, 1, 1],
        coarse_to_fine: vec![vec![0, 1], vec![2, 3]],
        num_coarse: 2,
    };

    let agg = fc.aggregate_columns_nd(&data);
    assert_eq!(agg.nrows(), 2);
    assert_eq!(agg.ncols(), 2);
    assert_relative_eq!(agg[(0, 0)], 3.0); // 1+2
    assert_relative_eq!(agg[(0, 1)], 7.0); // 3+4
    assert_relative_eq!(agg[(1, 0)], 11.0); // 5+6
    assert_relative_eq!(agg[(1, 1)], 15.0); // 7+8
}

#[test]
fn test_expand_logits_preserves_probabilities() {
    // 2 topics, 3 coarse features → expand to 6 fine features
    // Groups: {0,1}, {2,3}, {4,5}
    let logits = DMatrix::from_row_slice(
        3,
        2,
        &[
            -1.2, -0.8, // coarse 0
            -0.5, -1.5, // coarse 1
            -1.0, -1.0, // coarse 2
        ],
    );

    let fc = FeatureCoarsening {
        fine_to_coarse: vec![0, 0, 1, 1, 2, 2],
        coarse_to_fine: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        num_coarse: 3,
    };

    let expanded = fc.expand_log_dict_dk(&logits, 6);
    assert_eq!(expanded.nrows(), 6);
    assert_eq!(expanded.ncols(), 2);

    let ln2 = 2.0f32.ln();

    // For each topic, sum of exp(expanded) within each group
    // should equal exp(coarse logit)
    for k in 0..2 {
        for (c, group) in fc.coarse_to_fine.iter().enumerate() {
            let coarse_prob: f32 = logits[(c, k)].exp();
            let fine_sum: f32 = group.iter().map(|&f| expanded[(f, k)].exp()).sum();
            assert_relative_eq!(fine_sum, coarse_prob, epsilon = 1e-6);
        }
    }

    // Each fine feature in a group of size 2 gets logit - ln(2)
    assert_relative_eq!(expanded[(0, 0)], -1.2 - ln2, epsilon = 1e-6);
    assert_relative_eq!(expanded[(1, 0)], -1.2 - ln2, epsilon = 1e-6);
}

#[test]
fn test_compute_feature_coarsening() {
    use matrix_util::traits::SampleOps;

    // Create a D×S matrix with D=500 features, S=50 samples
    let d = 500;
    let s = 50;
    let data = DMatrix::<f32>::rnorm(d, s);

    let fc = compute_feature_coarsening(&data, 50).unwrap();

    // All features should be assigned
    assert_eq!(fc.fine_to_coarse.len(), d);

    // Coarse features should be reasonable
    assert!(fc.num_coarse > 0);
    assert!(fc.num_coarse <= 64); // 2^6 = 64 max for sort_dim=6

    // Every fine feature should appear in exactly one coarse group
    let mut counts = vec![0usize; d];
    for group in &fc.coarse_to_fine {
        for &f in group {
            counts[f] += 1;
        }
    }
    assert!(counts.iter().all(|&c| c == 1));

    // fine_to_coarse should be consistent with coarse_to_fine
    for (c, group) in fc.coarse_to_fine.iter().enumerate() {
        for &f in group {
            assert_eq!(fc.fine_to_coarse[f], c);
        }
    }
}

#[test]
fn test_integration_coarsen_expand_roundtrip() {
    // Simulated data: D=500, N=300, K=5, max_features=50
    use matrix_util::traits::SampleOps;

    let d = 500;
    let s = 100;
    let k = 5;

    // Create pseudobulk sketch
    let sketch = DMatrix::<f32>::rnorm(d, s);
    let fc = compute_feature_coarsening(&sketch, 50).unwrap();

    // Create a fake log-dictionary at coarse resolution
    let coarse_logits = DMatrix::<f32>::rnorm(fc.num_coarse, k);

    // Expand to fine resolution
    let expanded = fc.expand_log_dict_dk(&coarse_logits, d);
    assert_eq!(expanded.nrows(), d);
    assert_eq!(expanded.ncols(), k);

    // For each topic, sum of exp(expanded) within each group
    // should equal exp(coarse logit)
    for kk in 0..k {
        for (c, group) in fc.coarse_to_fine.iter().enumerate() {
            let coarse_val = coarse_logits[(c, kk)].exp();
            let fine_sum: f32 = group.iter().map(|&f| expanded[(f, kk)].exp()).sum();
            assert_relative_eq!(fine_sum, coarse_val, epsilon = 1e-4);
        }
    }
}

#[test]
fn test_skip_when_d_small() {
    // If D <= max_features, coarsening should still work but produce ~D groups
    use matrix_util::traits::SampleOps;
    let d = 30;
    let s = 20;
    let data = DMatrix::<f32>::rnorm(d, s);
    let fc = compute_feature_coarsening(&data, 50).unwrap();
    // Should produce fewer groups than D (binary hashing)
    assert!(fc.num_coarse <= d);
    assert!(fc.num_coarse > 0);
}
