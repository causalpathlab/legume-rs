use super::*;

#[test]
fn unseen_rows_exclude_matched_and_hidden() {
    // new rows: 0 matched, 1 unseen, 2 hidden (matched before the hide), 3 unseen, 4 matched
    let before_hide = vec![Some(3), None, Some(7), None, Some(1)];
    let hidden: HashSet<usize> = [2usize].into_iter().collect();
    assert_eq!(unseen_rows(&before_hide, &hidden), vec![1, 3]);
    // A hidden row that was ALSO unmatched is still not "unseen": it was withheld.
    let hidden2: HashSet<usize> = [1usize, 2].into_iter().collect();
    assert_eq!(unseen_rows(&before_hide, &hidden2), vec![3]);
}

#[test]
fn profiles_sum_counts_per_cluster() {
    let labels = vec![0usize, 1, 0];
    let f0: Vec<u32> = vec![0, 2];
    let c0: Vec<f32> = vec![1.0, 2.0];
    let f1: Vec<u32> = vec![1];
    let c1: Vec<f32> = vec![5.0];
    let f2: Vec<u32> = vec![0, 1];
    let c2: Vec<f32> = vec![3.0, 4.0];
    let cells = vec![(0usize, &f0[..], &c0[..]), (1, &f1[..], &c1[..]), (2, &f2[..], &c2[..])];
    let p = profiles_by_cluster(3, 2, &labels, cells.into_iter());
    assert_eq!(p.nrows(), 3);
    assert_eq!(p.ncols(), 2);
    assert_eq!(p[(0, 0)], 4.0); // gene 0: cells 0 and 2 → 1 + 3
    assert_eq!(p[(1, 0)], 4.0); // gene 1: cell 2
    assert_eq!(p[(1, 1)], 5.0); // gene 1: cell 1
    assert_eq!(p[(2, 0)], 2.0);
    assert_eq!(p[(2, 1)], 0.0);
}

#[test]
fn union_remap_places_unseen_after_training_genes_and_drops_the_rest() {
    let new_to_train = vec![Some(3), None, None, None, Some(1)];
    let unseen = vec![1usize, 3]; // row 2 is hidden/dropped
    let m = union_remap(&new_to_train, &unseen, 10);
    assert_eq!(m, vec![Some(3), Some(10), None, Some(11), Some(1)]);
}

/// Rates equal to the observed composition score the composition's own entropy,
/// and equal the null when the null is that same composition.
#[test]
fn initialized_score_is_a_multinomial_per_count() {
    let h = 2;
    // Two initialized genes with rows that make the rates proportional to (1, 3)
    // for a cell with θ = (0, 0): exp(bias) = (1, 3).
    let rows = DMatrix::<f32>::zeros(2, h);
    let bias = [0f32, 3f32.ln()];
    let theta = Mat::zeros(2, h);
    let b_cell = [0.7f32, -0.2];
    // cell 0 observes (1, 3) → composition (0.25, 0.75); cell 1 observes nothing.
    let obs = vec![vec![(0u32, 1.0f32), (1, 3.0)], vec![]];
    let null = [0.25f32, 0.75];
    let s = score_initialized(&rows, &bias, &theta, &b_cell, &obs, &null);
    let p = [0.25f32, 0.75];
    let want = (1.0 * p[0].ln() + 3.0 * p[1].ln()) / 4.0;
    assert_eq!(s[0].count, 4.0);
    assert!((s[0].llik_per_count - want).abs() < 1e-6, "{} vs {want}", s[0].llik_per_count);
    assert!((s[0].null_llik_per_count - want).abs() < 1e-6);
    assert_eq!(s[1].count, 0.0);
    assert!(s[1].llik_per_count.is_nan());
    assert!(s[1].null_llik_per_count.is_nan());
    // The cell bias cancels in the composition: changing it leaves the score.
    let s2 = score_initialized(&rows, &bias, &theta, &[5.0, 5.0], &obs, &null);
    assert!((s2[0].llik_per_count - s[0].llik_per_count).abs() < 1e-6);
    // A latent that favours gene 1 raises its share and lowers the score for
    // a cell whose counts favour gene 0.
    let rows2 = DMatrix::<f32>::from_row_slice(2, h, &[0.0, 0.0, 2.0, 0.0]);
    let theta2 = Mat::from_row_slice(2, h, &[1.0, 0.0, 1.0, 0.0]);
    let obs2 = vec![vec![(0u32, 3.0f32), (1, 1.0)], vec![]];
    let s3 = score_initialized(&rows2, &bias, &theta2, &b_cell, &obs2, &null);
    let s4 = score_initialized(&rows, &bias, &theta, &b_cell, &obs2, &null);
    assert!(s3[0].llik_per_count < s4[0].llik_per_count);
}
