use super::*;

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
