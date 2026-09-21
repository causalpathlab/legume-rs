//! Tests for the topic-residualization helpers and the p-value transform.
//! Recovery tests for the group model live in `group_tests.rs`.

use super::*;
use legume_numeric::matrix::traits::MatOps;

#[test]
fn z_to_pvalue_matches_the_two_sided_normal_tail() {
    assert!((z_to_pvalue(0.0) - 1.0).abs() < 1e-6, "z=0 must give p=1");
    assert!(
        (z_to_pvalue(1.959_964) - 0.05).abs() < 1e-3,
        "z=1.96 must give p~0.05"
    );
    assert!(
        (z_to_pvalue(2.575_829) - 0.01).abs() < 1e-3,
        "z=2.58 must give p~0.01"
    );
    // Two-sided, so the sign of z cannot matter.
    assert!((z_to_pvalue(1.5) - z_to_pvalue(-1.5)).abs() < 1e-7);
    // Monotone decreasing in |z|.
    assert!(z_to_pvalue(3.0) < z_to_pvalue(2.0));
    assert!(z_to_pvalue(2.0) < z_to_pvalue(1.0));
}

//////////////////////////////////////
// Residual collider stratification //
//////////////////////////////////////

#[test]
fn removing_the_exposure_effect_equalizes_group_means() {
    // Two individuals per group, three cells each. Group 1 has topic 0
    // inflated by exp(0.8); the adjustment must remove exactly that.
    let n_topics = 2;
    let cell_to_individual = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let individual_exposure_group = vec![0usize, 0, 1, 1];
    let lift = 0.8f32;

    let n_cells = cell_to_individual.len();
    let mut props = Mat::zeros(n_cells, n_topics);
    for (j, &i) in cell_to_individual.iter().enumerate() {
        // Within-individual variation that the adjustment must preserve.
        let jitter = 1.0 + 0.05 * ((j % 3) as f32);
        let group_lift = if individual_exposure_group[i] == 1 {
            lift.exp()
        } else {
            1.0
        };
        props[(j, 0)] = 0.4 * jitter * group_lift;
        props[(j, 1)] = 0.6 * jitter;
    }

    let before = super::average_topic_log_proportions_per_individual(
        &props,
        &cell_to_individual,
        individual_exposure_group.len(),
    );
    let (before_groups, _) =
        super::average_topic_logits_per_exposure_group(&before, &individual_exposure_group);
    let gap_before = before_groups[(1, 0)] - before_groups[(0, 0)];
    assert!(
        (gap_before - lift).abs() < 1e-4,
        "fixture did not plant the intended gap: {gap_before}"
    );

    let max_shift = remove_exposure_effect_from_topic_proportions(
        &mut props,
        &cell_to_individual,
        &individual_exposure_group,
    );

    // Balanced groups, so each sits half the gap from the grand mean.
    assert!(
        (max_shift[0] - lift / 2.0).abs() < 1e-4,
        "reported shift {} != half the planted gap",
        max_shift[0]
    );
    assert!(max_shift[1] < 1e-4, "untouched topic reported a shift");

    let after = super::average_topic_log_proportions_per_individual(
        &props,
        &cell_to_individual,
        individual_exposure_group.len(),
    );
    let (after_groups, _) =
        super::average_topic_logits_per_exposure_group(&after, &individual_exposure_group);
    let gap_after = after_groups[(1, 0)] - after_groups[(0, 0)];
    assert!(
        gap_after.abs() < 1e-4,
        "exposure gap survived the adjustment: {gap_after}"
    );
}

#[test]
fn removing_the_exposure_effect_preserves_within_individual_variation() {
    // The adjustment scales every cell in a group by one constant per topic,
    // so ratios between cells of the same individual must be untouched.
    let cell_to_individual = vec![0, 0, 1, 1];
    let individual_exposure_group = vec![0usize, 1];
    let mut props = Mat::from_row_slice(4, 2, &[0.2, 0.8, 0.5, 0.5, 0.3, 0.7, 0.6, 0.4]);
    let ratio_before = props[(0, 0)] / props[(1, 0)];

    remove_exposure_effect_from_topic_proportions(
        &mut props,
        &cell_to_individual,
        &individual_exposure_group,
    );

    let ratio_after = props[(0, 0)] / props[(1, 0)];
    assert!(
        (ratio_before - ratio_after).abs() < 1e-5,
        "within-individual ratio changed: {ratio_before} -> {ratio_after}"
    );
}

#[test]
fn cells_with_no_matching_individual_are_left_alone() {
    // `cell_to_individual` may point past the end for unmatched cells; those
    // must be skipped rather than panicking or being silently rescaled.
    let cell_to_individual = vec![0, 1, 99];
    let individual_exposure_group = vec![0usize, 1];
    let mut props = Mat::from_row_slice(3, 1, &[0.3, 0.7, 0.5]);

    remove_exposure_effect_from_topic_proportions(
        &mut props,
        &cell_to_individual,
        &individual_exposure_group,
    );

    assert!(
        (props[(2, 0)] - 0.5).abs() < 1e-7,
        "unmatched cell was modified: {}",
        props[(2, 0)]
    );
}

// Break these would catch: treating an empty row as "not one-hot" (suppresses
// the hard-assignment warning), or classifying soft rows as one-hot.
#[test]
fn topics_look_one_hot_accepts_hard_rows_and_skips_empty() {
    let z = Mat::from_row_slice(
        3,
        2,
        &[
            1.0, 0.0, // hard
            0.0, 0.0, // empty / NA — must not suppress
            0.0, 1.0, // hard
        ],
    );
    assert!(topics_look_one_hot(&z));
}

#[test]
fn topics_look_one_hot_rejects_soft_mixture() {
    let z = Mat::from_row_slice(1, 2, &[0.6, 0.4]);
    assert!(!topics_look_one_hot(&z));
}

#[test]
fn topics_look_one_hot_rejects_all_empty_rows() {
    let z = Mat::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
    assert!(!topics_look_one_hot(&z));
}

#[test]
fn one_hot_topics_are_unchanged_after_residualize_and_renorm() {
    // Hard assignments: residualize then sum_to_one must leave one-hot rows intact.
    let cell_to_individual = vec![0, 0, 1, 1];
    let individual_exposure_group = vec![0usize, 1];
    let mut props = Mat::from_row_slice(
        4,
        2,
        &[
            1.0, 0.0, // indv 0, cell 0
            0.0, 1.0, // indv 0, cell 1
            1.0, 0.0, // indv 1, cell 0
            0.0, 1.0, // indv 1, cell 1
        ],
    );
    let before = props.clone();

    remove_exposure_effect_from_topic_proportions(
        &mut props,
        &cell_to_individual,
        &individual_exposure_group,
    );
    props.sum_to_one_rows_inplace();

    for i in 0..4 {
        for k in 0..2 {
            assert!(
                (props[(i, k)] - before[(i, k)]).abs() < 1e-5,
                "one-hot row {i} changed at topic {k}: {} -> {}",
                before[(i, k)],
                props[(i, k)]
            );
        }
    }
}
