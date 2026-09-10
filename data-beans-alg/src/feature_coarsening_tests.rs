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

/// Source axis [g0 g1 g2 g3], groups {g0,g1} and {g2,g3}. The new axis is
/// [g1 gX g3 gY g0]: two source genes reordered, one dropped, two new.
fn grown_fixture() -> (FeatureCoarsening, Vec<Option<usize>>, Vec<Vec<f32>>) {
    let source = FeatureCoarsening::from_fine_to_coarse(vec![0, 0, 1, 1], 2).unwrap();
    let remap = vec![Some(1), None, Some(3), None, Some(0)];
    // Two pseudobulks: group 0's genes lean to the first, group 1's to the
    // second; gX leans to the first, gY to the second.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let unit = vec![
        vec![s, -s],  // g1
        vec![s, -s],  // gX
        vec![-s, s],  // g3
        vec![-s, s],  // gY
        vec![s, -s],  // g0
    ];
    (source, remap, unit)
}

#[test]
fn grown_known_features_keep_their_group_and_new_ones_join_the_nearest() {
    let (source, remap, unit) = grown_fixture();
    let grown = source.grow_by_profile(&remap, &unit).unwrap();
    assert_eq!(grown.fine_to_coarse, vec![0, 0, 1, 1, 0]);
    assert_eq!(grown.num_coarse, 2, "the group count is what consumers are keyed to");
    assert_eq!(grown.coarse_to_fine[0], vec![0, 1, 4]);
    assert_eq!(grown.coarse_to_fine[1], vec![2, 3]);
}

#[test]
fn grown_group_with_no_surviving_member_attracts_nothing_but_keeps_its_index() {
    // Three source groups; group 2's only gene is absent from the new axis.
    let source = FeatureCoarsening::from_fine_to_coarse(vec![0, 1, 2], 3).unwrap();
    let remap = vec![Some(0), Some(1), None];
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let unit = vec![vec![s, -s], vec![-s, s], vec![-s, s]];
    let grown = source.grow_by_profile(&remap, &unit).unwrap();
    assert_eq!(grown.num_coarse, 3);
    assert_eq!(grown.fine_to_coarse, vec![0, 1, 1]);
    assert!(grown.coarse_to_fine[2].is_empty());
}

#[test]
fn grown_feature_without_a_profile_goes_to_the_largest_group() {
    let source = FeatureCoarsening::from_fine_to_coarse(vec![0, 0, 1], 2).unwrap();
    let remap = vec![Some(0), Some(1), Some(2), None];
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let unit = vec![vec![s, -s], vec![s, -s], vec![-s, s], vec![0.0, 0.0]];
    let grown = source.grow_by_profile(&remap, &unit).unwrap();
    assert_eq!(grown.fine_to_coarse[3], 0);
}

#[test]
fn grown_axis_with_nothing_in_common_is_refused() {
    let source = FeatureCoarsening::from_fine_to_coarse(vec![0, 1], 2).unwrap();
    let err = match source.grow_by_profile(&[None, None], &[vec![1.0], vec![1.0]]) {
        Ok(_) => panic!("an axis with nothing in common must be refused"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("no feature"), "{err}");
}

#[test]
fn from_fine_to_coarse_rejects_a_group_index_out_of_range() {
    assert!(FeatureCoarsening::from_fine_to_coarse(vec![0, 2], 2).is_err());
}

/// Turning the grouping off should be sayable, not encoded as a number whose
/// literal reading ("at most zero features") is the opposite of its meaning.
/// The numeric spelling stays, because recorded runs and existing scripts use
/// it, so the two have to agree.
mod switching_it_off {
    use super::FeatureCoarseningArgs;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: FeatureCoarseningArgs,
    }

    fn parse(extra: &[&str]) -> Result<FeatureCoarseningArgs, clap::Error> {
        Cli::try_parse_from(["x"].iter().copied().chain(extra.iter().copied())).map(|c| c.args)
    }

    #[test]
    fn the_named_switch_and_the_zero_agree() {
        assert_eq!(parse(&[]).unwrap().cap().map(std::num::NonZeroUsize::get), Some(1000));
        assert!(parse(&["--no-feature-coarsening"]).unwrap().cap().is_none());
        assert!(parse(&["--max-coarse-features", "0"]).unwrap().cap().is_none());
        assert_eq!(
            parse(&["--max-coarse-features", "250"]).unwrap().cap().map(std::num::NonZeroUsize::get),
            Some(250)
        );
    }

    /// One says group at most N, the other says do not group. Asking for both
    /// is a contradiction rather than a precedence puzzle.
    #[test]
    fn asking_for_both_is_refused() {
        assert!(parse(&["--no-feature-coarsening", "--max-coarse-features", "250"]).is_err());
    }

}
