//! Genes whose exposure effect cannot be estimated are flagged and left out,
//! so the p-values of the rest keep their null distribution.

use super::*;

/// 6 individuals, levels 0,0,0,1,1,1; all have cells.
fn design() -> (Vec<usize>, Vec<bool>) {
    (vec![0, 0, 0, 1, 1, 1], vec![true; 6])
}

#[test]
fn genes_with_no_counts_or_an_all_zero_level_are_flagged() {
    let (x, used) = design();
    // gene 0 fine, gene 1 all zero, gene 2 zero at level 1
    let y = Mat::from_row_slice(
        3,
        6,
        &[
            1.0, 2.0, 0.0, 3.0, 1.0, 2.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            4.0, 2.0, 1.0, 0.0, 0.0, 0.0,
        ],
    );
    let flags = topic_flags(&y, &used, &x, 2, 3);
    assert_eq!(
        flags,
        vec![None, Some(Flag::NoCounts), Some(Flag::LevelZero)]
    );
}

#[test]
fn a_level_with_too_few_individuals_flags_the_whole_topic() {
    let (x, mut used) = design();
    used[4] = false;
    used[5] = false;
    let y = Mat::from_element(2, 6, 1.0);
    let flags = topic_flags(&y, &used, &x, 2, 3);
    assert_eq!(flags, vec![Some(Flag::LevelSparse); 2]);
}

#[test]
fn the_contrast_averages_estimable_topics_and_drops_genes_with_none() {
    // two topics, two genes, one non-reference level
    let psi_a = Mat::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 5.0]);
    let psi_b = Mat::from_row_slice(2, 2, &[0.0, 3.0, 0.0, 7.0]);
    let flags = vec![
        vec![None, Some(Flag::LevelZero)],
        vec![None, Some(Flag::NoCounts)],
    ];
    let mask = EstimableMask::from_topic_flags(&flags);
    let c = mask.mean_log_effect([&psi_a, &psi_b].into_iter());
    assert!((c[(0, 0)] - 2.0).abs() < 1e-6);
    assert!(c[(1, 0)].is_nan());
    assert_eq!(mask.gene_flag(1), Some(Flag::LevelZero));
    assert_eq!(mask.gene_flag(0), None);
}

#[test]
fn a_topic_flagged_for_one_gene_is_left_out_of_that_gene_only() {
    let psi_a = Mat::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 5.0]);
    let psi_b = Mat::from_row_slice(2, 2, &[0.0, 3.0, 0.0, 7.0]);
    let flags = vec![vec![None, None], vec![None, Some(Flag::LevelZero)]];
    let mask = EstimableMask::from_topic_flags(&flags);
    let c = mask.mean_log_effect([&psi_a, &psi_b].into_iter());
    assert!((c[(0, 0)] - 2.0).abs() < 1e-6);
    assert!((c[(1, 0)] - 5.0).abs() < 1e-6);
}

#[test]
fn run_level_flags_cover_every_gene() {
    let flags = vec![vec![None, None]];
    let mut mask = EstimableMask::from_topic_flags(&flags);
    mask.flag_all(Flag::WeakOverlap);
    assert_eq!(mask.gene_flag(0), Some(Flag::WeakOverlap));
    let psi = Mat::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 2.0]);
    assert!(mask.mean_log_effect([&psi].into_iter())[(0, 0)].is_nan());
}

#[test]
fn a_permutation_null_that_barely_moves_is_degenerate() {
    let x = vec![0, 0, 0, 1, 1, 1];
    let stuck: Vec<Vec<usize>> = vec![x.clone(); 50];
    assert!(permutations_degenerate(&stuck, 20));
    // all 20 balanced labelings of 6 individuals
    let all: Vec<Vec<usize>> = (0u32..64)
        .filter(|m| m.count_ones() == 3)
        .map(|m| (0..6).map(|i| ((m >> i) & 1) as usize).collect())
        .collect();
    assert_eq!(all.len(), 20);
    assert!(!permutations_degenerate(&all, 20));
    assert!(permutations_degenerate(&all[..19], 20));
}
