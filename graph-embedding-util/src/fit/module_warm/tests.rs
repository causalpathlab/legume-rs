use super::*;
use crate::fit::config::ParentModulesOwned;

/// Three planted co-expression blocks over 12 pseudobulks are recovered as three
/// modules, up to label permutation.
fn planted() -> DMatrix<f32> {
    let (d, s) = (30usize, 12usize);
    let mut m = DMatrix::<f32>::zeros(d, s);
    for i in 0..d {
        let block = i / 10;
        for j in 0..s {
            let on = j / 4 == block;
            m[(i, j)] = if on { 50.0 } else { 2.0 } + ((i * 7 + j * 3) % 5) as f32;
        }
    }
    m
}

#[test]
fn warm_start_recovers_planted_blocks() {
    let labels = warm_start_module_labels(&planted(), 3, 11);
    assert_eq!(labels.len(), 30);
    for block in 0..3 {
        let first = labels[block * 10];
        assert!(
            labels[block * 10..(block + 1) * 10]
                .iter()
                .all(|&l| l == first),
            "block {block} split: {labels:?}"
        );
    }
    let mut distinct = labels.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 3);
}

#[test]
fn warm_start_is_seed_reproducible() {
    let a = warm_start_module_labels(&planted(), 3, 5);
    let b = warm_start_module_labels(&planted(), 3, 5);
    assert_eq!(a, b);
}

#[test]
fn wide_profiles_are_sketched() {
    // More pseudobulks than the direct limit → the Gaussian sketch path.
    let (d, s) = (12usize, WARM_PROFILE_MAX_DIM + 8);
    let mut m = DMatrix::<f32>::zeros(d, s);
    for i in 0..d {
        for j in 0..s {
            let on = (j % 2 == 0) == (i < 6);
            m[(i, j)] = if on { 40.0 } else { 1.0 };
        }
    }
    let labels = warm_start_module_labels(&m, 2, 3);
    assert!(labels[..6].iter().all(|&l| l == labels[0]));
    assert!(labels[6..].iter().all(|&l| l == labels[6]));
    assert_ne!(labels[0], labels[6]);
}

/// Warm start from a PARENT: matched features take the parent's membership rows
/// verbatim; unmatched ones are initialized through the parent's modules from
/// their profile neighbours (identical profile ⇒ identical row).
#[test]
fn parent_warm_start_carries_matched_rows_and_initializes_the_rest() {
    let parent_pi = DMatrix::<f32>::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let parent_mu = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let parent_rho = &parent_pi * &parent_mu;
    // New axis: gene 0 = parent 2, gene 1 = parent 0, gene 2 = unseen with gene 1's
    // profile, gene 3 = unseen with parent-1-like profile (parent 1 is MISSING here,
    // so it is initialized from the matched neighbours only).
    let row_to_parent = vec![Some(2), Some(0), None, None];
    let profiles = DMatrix::<f32>::from_row_slice(
        4,
        4,
        &[
            5.0, 5.0, 5.0, 5.0, //
            9.0, 1.0, 9.0, 1.0, //
            9.0, 1.0, 9.0, 1.0, //
            1.0, 9.0, 1.0, 9.0,
        ],
    );
    let logits = parent_module_logits(
        &ParentModulesOwned {
            rho: parent_rho,
            pi: parent_pi.clone(),
            mu: parent_mu,
            row_to_parent,
            knobs: crate::transfer::AlignKnobs {
                k: 2,
                similarity_floor: 0.5,
            },
        },
        &profiles,
    );
    assert_eq!(logits.nrows(), 4);
    assert_eq!(logits.ncols(), 2);
    assert_eq!(logits.row(0), parent_pi.row(2));
    assert_eq!(logits.row(1), parent_pi.row(0));
    assert_eq!(
        logits.row(2),
        parent_pi.row(0),
        "same profile as gene 1 → parent 0's row"
    );
    // Anti-correlated with every matched gene → diffuse module average of the parent.
    let avg: Vec<f32> = (0..2)
        .map(|m| parent_pi.column(m).iter().sum::<f32>() / 3.0)
        .collect();
    for m in 0..2 {
        assert!((logits[(3, m)] - avg[m]).abs() < 1e-6);
    }
}
