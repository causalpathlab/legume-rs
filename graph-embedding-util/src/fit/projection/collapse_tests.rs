//! The phase-2 row collapse: module-only rows fold into one row per module,
//! and the per-cell objective changes only by a constant in `θ`.

use super::*;

/// Six features, H = 2: rows 0, 3 residual; rows 1, 2, 5 module-only in
/// module 7; row 4 module-only in module 9.
fn fixture() -> (RowCollapse, Vec<f32>, Vec<f32>) {
    let module_only = [false, true, true, false, true, true];
    let labels = [0, 7, 7, 1, 9, 7];
    let c = RowCollapse::from_modules(&module_only, &labels).expect("some module-only rows");
    // Module-only rows carry their module's row; biases are their own shares.
    let feat = vec![
        0.3, -0.1, // 0
        0.5, 0.2, // 1  module 7
        0.5, 0.2, // 2  module 7
        -0.4, 0.6, // 3
        0.1, -0.7, // 4  module 9
        0.5, 0.2, // 5  module 7
    ];
    let b = vec![0.1, -1.0, -0.5, 0.2, 0.0, -2.0];
    (c, feat, b)
}

#[test]
fn residual_rows_stay_and_a_module_becomes_one_row() {
    let (c, _, _) = fixture();
    assert_eq!(c.n_rows, 4, "two residual rows + two modules");
    let r = &c.row_of;
    assert!(r[1] == r[2] && r[2] == r[5], "module 7 is one row: {r:?}");
    assert!(r[0] != r[3] && r[0] != r[1] && r[3] != r[1] && r[4] != r[1]);
    assert!(r.iter().all(|&x| (x as usize) < c.n_rows));
}

#[test]
fn no_module_only_rows_means_no_collapse() {
    assert!(RowCollapse::from_modules(&[false, false], &[0, 1]).is_none());
}

#[test]
fn the_reduced_dictionary_is_the_module_row_and_the_members_log_sum_exp() {
    let (c, feat, b) = fixture();
    let (rf, rb) = c.reduce_dictionary(&feat, &b, 2);
    let m7 = c.row_of[1] as usize;
    assert_eq!(&rf[m7 * 2..m7 * 2 + 2], &[0.5, 0.2]);
    let lse = ((-1.0f32).exp() + (-0.5f32).exp() + (-2.0f32).exp()).ln();
    assert!((rb[m7] - lse).abs() < 1e-6);
    let r3 = c.row_of[3] as usize;
    assert_eq!(&rf[r3 * 2..r3 * 2 + 2], &[-0.4, 0.6]);
    assert_eq!(rb[r3], 0.2);
}

#[test]
fn a_cells_counts_are_summed_per_reduced_row() {
    let (c, _, _) = fixture();
    let (f, x) = c.reduce_edges(&[1, 2, 3, 5], &[1.0, 2.0, 4.0, 0.5]);
    let m7 = c.row_of[1];
    let r3 = c.row_of[3];
    let got: std::collections::HashMap<u32, f32> = f.iter().copied().zip(x).collect();
    assert_eq!(got.len(), 2);
    assert_eq!(got[&m7], 3.5);
    assert_eq!(got[&r3], 4.0);
    assert!(f.windows(2).all(|w| w[0] < w[1]), "ascending: {f:?}");
}

/// Per-cell profiled multinomial NLL `N · lse(s) − Σ n·s` and the exact
/// intercept `ln N − lse(s)`, with `s_f = ⟨θ, e_f⟩ + b_f`.
fn nll_and_intercept(theta: &[f32], feat: &[f32], b: &[f32], f: &[u32], x: &[f32]) -> (f64, f64) {
    let h = theta.len();
    let s: Vec<f64> = (0..b.len())
        .map(|g| {
            let e = &feat[g * h..(g + 1) * h];
            f64::from(e.iter().zip(theta).map(|(a, t)| a * t).sum::<f32>() + b[g])
        })
        .collect();
    let mx = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = mx + s.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
    let n: f64 = x.iter().map(|&v| f64::from(v)).sum();
    let data: f64 = f
        .iter()
        .zip(x)
        .map(|(&g, &v)| f64::from(v) * s[g as usize])
        .sum();
    (n * lse - data, n.ln() - lse)
}

#[test]
fn the_collapse_changes_the_objective_by_a_constant_in_theta() {
    let (c, feat, b) = fixture();
    let (rf, rb) = c.reduce_dictionary(&feat, &b, 2);
    let (f, x) = (vec![0u32, 1, 2, 4, 5], vec![3.0f32, 1.0, 2.0, 5.0, 4.0]);
    let (rfi, rx) = c.reduce_edges(&f, &x);
    let mut gaps = Vec::new();
    for theta in [[0.0f32, 0.0], [1.3, -0.7], [-2.0, 0.4]] {
        let (full, c_full) = nll_and_intercept(&theta, &feat, &b, &f, &x);
        let (red, c_red) = nll_and_intercept(&theta, &rf, &rb, &rfi, &rx);
        assert!(
            (c_full - c_red).abs() < 1e-5,
            "intercept {c_full} vs {c_red}"
        );
        gaps.push(full - red);
    }
    assert!(
        gaps.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-5),
        "NLL gap depends on θ: {gaps:?}"
    );
}
