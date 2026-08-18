use crate::link_community::dict_merge::*;
use crate::util::common::*;

/// Build a dictionary with a clean block structure plus a pile of loud,
/// uninformative rows.
///
/// Signal rows: two genuinely different community groups (columns 0,1 vs 2,3),
/// contrast +-1, so the within-group cosine is +1 and the between-group cosine
/// is -1 once each row is centred.
///
/// Noise rows: no group structure at all, but a LARGE per-column swing shared by
/// every noise row. This is what an undetected gene looks like after the
/// Poisson-Gamma fit -- its log-rate is set by each community's exposure rather
/// than by data, so it is both uninformative AND high-variance. Because cosine is
/// dominated by the largest-magnitude rows, these rows decide the merge unless
/// they are filtered out.
fn dictionary_with_loud_noise(
    n_signal: usize,
    n_noise: usize,
    noise_scale: f32,
) -> (Mat, Vec<bool>) {
    let n = n_signal + n_noise;
    let mut m = Mat::zeros(n, 4);
    for g in 0..n_signal {
        // group A high, group B low (or the reverse) -- real contrast, unit scale
        let flip = if g % 2 == 0 { 1.0 } else { -1.0 };
        m[(g, 0)] = flip;
        m[(g, 1)] = flip;
        m[(g, 2)] = -flip;
        m[(g, 3)] = -flip;
    }
    for g in n_signal..n {
        // identical across noise rows, so it is a single shared direction, and
        // it cuts ACROSS the true grouping (cols 0,2 vs 1,3)
        m[(g, 0)] = noise_scale;
        m[(g, 1)] = -noise_scale;
        m[(g, 2)] = noise_scale;
        m[(g, 3)] = -noise_scale;
    }
    let keep: Vec<bool> = (0..n).map(|g| g < n_signal).collect();
    (m, keep)
}

/// Cosine at which columns 0 and 1 (same true group) were joined.
fn merge_height_for_true_pair(merges: &[BhcMerge]) -> f64 {
    merges
        .iter()
        // `cosine_merge` always emits `left < right`.
        .find(|m| m.left == 0 && m.right == 1)
        .map(|m| m.log_bf)
        .expect("columns 0 and 1 should merge with each other")
}

#[test]
fn loud_undetected_genes_hijack_the_merge_when_not_filtered() {
    // 20 real genes against 200 loud noise genes at 5x amplitude.
    let (m, keep) = dictionary_with_loud_noise(20, 200, 5.0);

    // Unfiltered: the noise direction dominates, so the true pair (0,1) is NOT
    // the first thing to merge -- 0 pairs with 2 instead, which is the noise
    // grouping, not the signal grouping.
    let unfiltered = cosine_merge(&m, None);
    let first = &unfiltered[0];
    let first_pair = (first.left.min(first.right), first.left.max(first.right));
    assert_ne!(
        first_pair,
        (0, 1),
        "unfiltered merge should be hijacked by the loud noise rows"
    );

    // Filtered to the detected genes: the true pair merges first, at cosine 1.
    let filtered = cosine_merge(&m, Some(&keep));
    let first = &filtered[0];
    let first_pair = (first.left.min(first.right), first.left.max(first.right));
    assert_eq!(
        first_pair,
        (0, 1),
        "filtered merge should recover the true grouping first"
    );
    assert!(
        merge_height_for_true_pair(&filtered) > 0.99,
        "same-group columns should merge at cosine ~1 once noise is filtered"
    );
}

#[test]
fn filtering_puts_the_right_communities_together_at_a_default_cut() {
    let (m, keep) = dictionary_with_loud_noise(20, 200, 5.0);
    // True grouping is {0,1} and {2,3}. Both arms may well produce TWO groups at
    // a 0.9 cut, so counting them proves nothing -- what differs is WHICH columns
    // land together.
    let grouped_together =
        |lab: &[i32], a: usize, b: usize| -> bool { lab[a] >= 0 && lab[a] == lab[b] };

    let filtered = cosine_cut(&cosine_merge(&m, Some(&keep)), 4, 0.9);
    assert!(
        grouped_together(&filtered, 0, 1) && grouped_together(&filtered, 2, 3),
        "filtered cut should recover the true grouping {{0,1}},{{2,3}}"
    );
    assert!(
        !grouped_together(&filtered, 0, 2),
        "filtered cut must not merge across the true groups"
    );

    let unfiltered = cosine_cut(&cosine_merge(&m, None), 4, 0.9);
    assert!(
        grouped_together(&unfiltered, 0, 2),
        "unfiltered cut should follow the loud noise rows and merge 0 with 2"
    );
}

#[test]
fn an_all_false_mask_yields_no_merges() {
    let (m, _) = dictionary_with_loud_noise(4, 0, 1.0);
    let none_kept = vec![false; m.nrows()];
    assert!(cosine_merge(&m, Some(&none_kept)).is_empty());
}

#[test]
fn a_full_mask_matches_passing_none() {
    let (m, _) = dictionary_with_loud_noise(8, 8, 2.0);
    let all_kept = vec![true; m.nrows()];
    let a = cosine_merge(&m, None);
    let b = cosine_merge(&m, Some(&all_kept));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!((x.left, x.right), (y.left, y.right));
        assert!((x.log_bf - y.log_bf).abs() < 1e-6);
    }
}

///////////////////////////////////////////
// moved from dict_merge.rs's inline mod //
///////////////////////////////////////////

/// Build an `(n_genes × k)` matrix from K column vectors.
fn mat_from_columns(cols: &[Vec<f32>]) -> Mat {
    let n_genes = cols[0].len();
    let k = cols.len();
    Mat::from_fn(n_genes, k, |g, j| cols[j][g])
}

#[test]
fn merges_two_identical_columns_first() {
    // Three columns: 0 and 1 identical, 2 anti-correlated.
    let cols = vec![
        vec![1.0, 0.0, -1.0, 0.5],
        vec![1.0, 0.0, -1.0, 0.5],
        vec![-1.0, 0.0, 1.0, -0.5],
    ];
    let m = mat_from_columns(&cols);
    let merges = cosine_merge(&m, None);
    assert_eq!(merges.len(), 2);
    // First merge should be (0, 1) with cosine very close to 1.
    let first = &merges[0];
    assert!(
        (first.left == 0 && first.right == 1) || (first.left == 1 && first.right == 0),
        "first merge should join cols 0 and 1, got ({}, {})",
        first.left,
        first.right
    );
    assert!(
        (first.log_bf - 1.0).abs() < 1e-6,
        "identical-column merge should have cosine ≈ 1, got {}",
        first.log_bf
    );
    assert_eq!(first.n_samples, 2);
    assert_eq!(first.id, 3);
}

#[test]
fn tree_shape_invariants() {
    // Random-ish 6 columns over 8 genes. Just check structural invariants.
    let cols: Vec<Vec<f32>> = (0..6)
        .map(|j| {
            (0..8)
                .map(|g| ((g + j) as f32 * 0.7).sin())
                .collect::<Vec<_>>()
        })
        .collect();
    let m = mat_from_columns(&cols);
    let merges = cosine_merge(&m, None);
    let k = 6usize;
    assert_eq!(merges.len(), k - 1);
    // Ids are exactly k, k+1, ..., 2k-2 in order.
    for (step, m) in merges.iter().enumerate() {
        assert_eq!(m.id as usize, k + step);
        assert!(m.left < m.id);
        assert!(m.right < m.id);
        assert!(m.left != m.right);
    }
    // Root node n_samples == K.
    assert_eq!(merges.last().unwrap().n_samples as usize, k);
    // Cosine similarity is monotonically non-increasing as we go up
    // an average-linkage UPGMA tree (the so-called Lance-Williams
    // monotonicity for UPGMA on a similarity).
    for w in merges.windows(2) {
        assert!(
            w[0].log_bf + 1e-9 >= w[1].log_bf,
            "UPGMA similarity should be non-increasing: {} -> {}",
            w[0].log_bf,
            w[1].log_bf
        );
    }
}

#[test]
fn cut_at_threshold_recovers_groups() {
    // Two groups of similar columns: {0, 1, 2} and {3, 4}; group means
    // anti-correlated. Each within-group column has small jitter so
    // cosine within a group is close to 1; across groups close to -1.
    let g0a = vec![1.0, 0.0, -1.0, 0.5, 0.2];
    let g0b = vec![1.01, 0.02, -0.99, 0.51, 0.18];
    let g0c = vec![0.99, -0.01, -1.02, 0.49, 0.22];
    let g1a = vec![-1.0, 0.0, 1.0, -0.5, -0.2];
    let g1b = vec![-1.02, 0.01, 0.98, -0.49, -0.22];
    let m = mat_from_columns(&[g0a, g0b, g0c, g1a, g1b]);
    let merges = cosine_merge(&m, None);
    let k = 5usize;
    // Cut at cosine ≥ 0.5: only within-group merges happen.
    let labels = cosine_cut(&merges, k, 0.5);
    assert_eq!(labels.len(), k);
    let n_super: usize = labels.iter().copied().filter(|&v| v >= 0).count();
    let max_label = labels.iter().copied().max().unwrap();
    assert_eq!(n_super, k);
    assert_eq!(max_label, 1, "expected exactly 2 super-clusters");
    // Cols 0, 1, 2 share a label; cols 3, 4 share a label; the two are different.
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn empty_for_k_lt_two() {
    let m = Mat::from_element(4, 1, 0.5);
    assert!(cosine_merge(&m, None).is_empty());
}
