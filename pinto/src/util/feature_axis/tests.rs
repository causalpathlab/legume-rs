//! What a row means, and the fold each answer implies.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::<str>::from(*s)).collect()
}

/// Two channel rows per feature, deliberately NOT adjacent and not in feature order,
/// because nothing guarantees a producer emits them that way.
fn channelized() -> Vec<Box<str>> {
    names(&[
        "FEATURE1/count/spliced",
        "FEATURE2/count/unspliced",
        "FEATURE1/count/unspliced",
        "FEATURE2/count/spliced",
    ])
}

#[test]
fn a_matrix_without_channels_resolves_to_the_identity_axis() {
    let rows = names(&["FEATURE1", "FEATURE2", "FEATURE3"]);
    let axis = FeatureAxis::resolve(&rows).unwrap();

    assert!(!axis.is_channelized());
    assert_eq!(axis.n_features(), 3);
    assert_eq!(axis.feature_names(), rows.as_slice());
    for r in 0..3 {
        assert_eq!(axis.feature_of_row(r), r);
        assert!(!axis.row_is_nascent(r));
    }
}

#[test]
fn a_fully_channelized_matrix_pairs_the_two_tracks_of_each_feature() {
    let axis = FeatureAxis::resolve(&channelized()).unwrap();

    assert!(axis.is_channelized());
    assert_eq!(axis.n_features(), 2);
    assert_eq!(
        axis.feature_names(),
        names(&["FEATURE1", "FEATURE2"]).as_slice()
    );
    assert_eq!(axis.feature_of_row(0), axis.feature_of_row(2));
    assert_eq!(axis.feature_of_row(1), axis.feature_of_row(3));
    assert!(axis.row_is_nascent(1) && axis.row_is_nascent(2));
    assert!(!axis.row_is_nascent(0) && !axis.row_is_nascent(3));
}

/// The failure `cage` cannot absorb. Break it by falling back to the identity
/// axis on a mixed matrix and the `total` row becomes a third feature whose counts
/// are already inside the other two.
#[test]
fn a_mixed_matrix_is_a_hard_error_naming_the_offenders() {
    let rows = names(&[
        "FEATURE1/count/spliced",
        "FEATURE1/count/unspliced",
        "FEATURE1/count/total",
    ]);
    let err = FeatureAxis::resolve(&rows).unwrap_err().to_string();
    assert!(err.contains("FEATURE1/count/total"), "{err}");
    assert!(err.contains("total"), "{err}");
}

/// The regression Stage 0 exists to prevent, stated as an equality: a
/// two-channel matrix must fold to exactly the single-channel matrix whose
/// counts are its per-feature sums.
#[test]
fn pooling_two_channels_equals_the_single_channel_matrix() {
    let axis = FeatureAxis::resolve(&channelized()).unwrap();

    // rows: G1/s, G2/u, G1/u, G2/s   over three columns
    let two = Mat::from_row_slice(
        4,
        3,
        &[
            1.0, 2.0, 3.0, // FEATURE1 spliced
            10.0, 20.0, 30.0, // FEATURE2 unspliced
            4.0, 5.0, 6.0, // FEATURE1 unspliced
            40.0, 50.0, 60.0, // FEATURE2 spliced
        ],
    );
    let pooled = axis.pool_rows_opt(&two).expect("channelized axis folds");
    let expect = Mat::from_row_slice(2, 3, &[5.0, 7.0, 9.0, 50.0, 70.0, 90.0]);
    assert_eq!(pooled, expect);

    assert_eq!(axis.pool_totals(&[1.0, 10.0, 4.0, 40.0]), vec![5.0, 50.0]);

    // Same property on a sparse profile: the two rows of FEATURE1 merge into one
    // entry, and the result is ascending by feature id.
    let obs = axis.pool_profile(vec![(3, 40.0), (0, 1.0), (2, 4.0)]);
    assert_eq!(obs, vec![(0, 5.0), (1, 40.0)]);
}

#[test]
fn the_identity_axis_folds_are_pass_throughs() {
    let rows = names(&["FEATURE1", "FEATURE2"]);
    let axis = FeatureAxis::resolve(&rows).unwrap();
    let m = Mat::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

    // `None`, not a copy: the identity axis must not allocate a fold at all.
    assert!(axis.pool_rows_opt(&m).is_none());
    assert_eq!(axis.pool_totals(&[1.0, 2.0]), vec![1.0, 2.0]);
    // Unsorted input stays untouched — an identity fold must not even reorder.
    assert_eq!(
        axis.pool_profile(vec![(1, 2.0), (0, 1.0)]),
        vec![(1, 2.0), (0, 1.0)]
    );
}

/// HVG ranks ROWS, so it can pick one track of a feature and drop the other. Break
/// the promotion and the projection sees half a feature.
#[test]
fn hvg_weights_are_promoted_to_whole_features() {
    let axis = FeatureAxis::resolve(&channelized()).unwrap();
    // only FEATURE1's spliced row was selected
    let mut w = vec![1.0, 0.0, 0.0, 0.0];
    let n_features = axis.promote_row_weights(&mut w);

    assert_eq!(n_features, 1, "one feature carries weight, not one row");
    assert_eq!(
        w,
        vec![1.0, 0.0, 1.0, 0.0],
        "FEATURE1's nascent row joins it"
    );
}

/// `δ` is identified only by the contrast, so a feature with counts on one track
/// is not identified — and neither is any feature at all when there are no
/// channels to contrast.
#[test]
fn delta_is_identified_only_where_both_tracks_carry_counts() {
    let axis = FeatureAxis::resolve(&channelized()).unwrap();
    //          G1/s   G2/u  G1/u  G2/s
    let totals = [7.0, 0.0, 3.0, 9.0];
    assert_eq!(axis.delta_identified(&totals), vec![true, false]);

    let flat = FeatureAxis::resolve(&names(&["FEATURE1", "FEATURE2"])).unwrap();
    assert_eq!(flat.delta_identified(&[1.0, 1.0]), vec![false, false]);
}

/// Three features, one of them single-track, and the tracks interleaved.
///
/// The asymmetry is deliberate. With two features perfectly alternating,
/// `row_to_feature` is `[0,1,0,1]`, which any positional guess such as
/// `row % n_features` reproduces by accident, so a broken broadcast still passes.
/// Here `row_to_feature` is `[0,1,0,2,1]` and `row % 3` is `[0,1,2,0,1]`, so the
/// coincidence is gone.
fn lopsided() -> Vec<Box<str>> {
    names(&[
        "FEATURE1/count/spliced",
        "FEATURE2/count/unspliced",
        "FEATURE1/count/unspliced",
        "FEATURE3/count/spliced",
        "FEATURE2/count/spliced",
    ])
}

#[test]
fn a_per_feature_vector_spreads_back_over_that_feature_s_rows() {
    let axis = FeatureAxis::resolve(&lopsided()).unwrap();
    assert_eq!(axis.n_features(), 3);
    assert_eq!(axis.n_rows(), 5);

    let by_feature: Vec<f32> = vec![0.25, 0.5, 0.75];
    let by_row = axis.broadcast_to_rows(&by_feature);
    assert_eq!(by_row.len(), axis.n_rows());
    for r in 0..axis.n_rows() {
        assert_eq!(by_row[r], by_feature[axis.feature_of_row(r)], "row {r}");
    }
    // Explicit, so the test states the mapping rather than restating the code.
    assert_eq!(by_row, vec![0.25, 0.5, 0.25, 0.75, 0.5]);

    // Both tracks of a feature must agree. A projection that weights one track
    // differently from the other has split the feature.
    let g1: Vec<f32> = (0..axis.n_rows())
        .filter(|&r| axis.feature_of_row(r) == 0)
        .map(|r| by_row[r])
        .collect();
    assert_eq!(g1, vec![0.25, 0.25]);
}

#[test]
fn broadcasting_on_the_identity_axis_is_a_pass_through() {
    let axis = FeatureAxis::resolve(&names(&["A", "B", "C"])).unwrap();
    // Both element types, since the broadcast is generic and the count filter
    // uses the f64 form while the weights use the f32 one.
    assert_eq!(
        axis.broadcast_to_rows(&[1.0f32, 2.0, 3.0]),
        vec![1.0f32, 2.0, 3.0]
    );
    assert_eq!(
        axis.broadcast_to_rows(&[1.0f64, 2.0, 3.0]),
        vec![1.0f64, 2.0, 3.0]
    );
}

#[test]
fn a_count_threshold_keeps_or_drops_a_feature_as_one_unit() {
    let axis = FeatureAxis::resolve(&lopsided()).unwrap();
    // Per row: FEATURE1 splits 3 / 1, FEATURE2 splits 1 / 1, FEATURE3 has 2 on its only
    // track. At a threshold of 2 the row-wise answer keeps FEATURE1's mature row
    // and FEATURE3, so it splits one feature down the middle and drops FEATURE2 even
    // though its total clears the bar.
    let row_totals: Vec<f64> = vec![3.0, 1.0, 1.0, 2.0, 1.0];
    let per_feature = axis.pool_totals(&row_totals);
    assert_eq!(per_feature, vec![4.0, 2.0, 2.0]);
    let spread = axis.broadcast_to_rows(&per_feature);

    let kept_rowwise: Vec<usize> = (0..axis.n_rows())
        .filter(|&r| row_totals[r] >= 2.0)
        .collect();
    let kept_pooled: Vec<usize> = (0..axis.n_rows()).filter(|&r| spread[r] >= 2.0).collect();
    assert_eq!(kept_rowwise, vec![0, 3], "row-wise keeps one lone track");
    assert_eq!(
        kept_pooled,
        vec![0, 1, 2, 3, 4],
        "pooled keeps every feature whole"
    );

    // And every row of a feature agrees, so no feature is half-filtered.
    for r in 0..axis.n_rows() {
        assert_eq!(spread[r], per_feature[axis.feature_of_row(r)]);
    }
}
