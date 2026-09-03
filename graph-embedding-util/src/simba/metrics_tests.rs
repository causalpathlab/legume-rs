use super::*;
use candle_util::candle_core::{Device, Tensor};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// An independent f64 transcription of `compare_entities` for one gene's
/// score column `x` over the cells.
fn reference(x: &[f64], n_top: usize, t: f64) -> (f64, f64, f64, f64) {
    let n = x.len() as f64;
    let mean_exp = x.iter().map(|v| v.exp()).sum::<f64>() / n;
    let norm: Vec<f64> = x.iter().map(|v| v - mean_exp.ln()).collect();
    let z: f64 = x.iter().map(|v| (v / t).exp()).sum();
    let p: Vec<f64> = x.iter().map(|v| (v / t).exp() / z).collect();
    let mut sorted = norm.clone();
    sorted.sort_by(f64::total_cmp);
    let top = &sorted[sorted.len() - n_top..];
    let max = top.iter().map(|v| v.max(0.0)).sum::<f64>() / n_top as f64;
    let mean = x.iter().sum::<f64>() / n;
    let std = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    // _gini: shift to ≥ 0, add 1e-7, sort ascending, Σ(2i − n − 1)x_i / (n Σx_i)
    let mut g: Vec<f64> = p.clone();
    let mn = g.iter().cloned().fold(f64::INFINITY, f64::min);
    if mn < 0.0 {
        g.iter_mut().for_each(|v| *v -= mn);
    }
    g.iter_mut().for_each(|v| *v += 1e-7);
    g.sort_by(f64::total_cmp);
    let num: f64 = g
        .iter()
        .enumerate()
        .map(|(i, v)| (2.0 * (i as f64 + 1.0) - n - 1.0) * v)
        .sum();
    let gini = num / (n * g.iter().sum::<f64>());
    let entropy = -p.iter().map(|v| v * v.ln()).sum::<f64>();
    (max, std, gini, entropy)
}

#[test]
fn gini_entropy_std_and_max_are_hand_checked_on_a_three_by_two_score_matrix() {
    let dev = Device::Cpu;
    // cells × 1 dim: [1, 2, 3]; genes × 1 dim: [1, −1] → X = [[1, −1], [2, −2], [3, −3]]
    let e_cell = Tensor::from_vec(vec![1f32, 2., 3.], (3, 1), &dev).unwrap();
    let e_gene = Tensor::from_vec(vec![1f32, -1.], (2, 1), &dev).unwrap();
    let m = compare_entities(&e_cell, &e_gene, 2, 1.0).unwrap();
    let cols: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![-1.0, -2.0, -3.0]];
    for (g, x) in cols.iter().enumerate() {
        let (max, std, gini, entropy) = reference(x, 2, 1.0);
        assert!(
            approx(f64::from(m.max[g]), max, 1e-5),
            "max[{g}] {} vs {max}",
            m.max[g]
        );
        assert!(approx(f64::from(m.std[g]), std, 1e-5), "std[{g}]");
        assert!(
            approx(f64::from(m.gini[g]), gini, 1e-5),
            "gini[{g}] {} vs {gini}",
            m.gini[g]
        );
        assert!(
            approx(f64::from(m.entropy[g]), entropy, 1e-5),
            "entropy[{g}]"
        );
    }
    // Explicit numbers for gene 0, so the oracle itself is pinned:
    // std of [1,2,3] is 1; the top-2 clipped norms are [0, 3 − ln((e+e²+e³)/3)].
    assert!(approx(f64::from(m.std[0]), 1.0, 1e-6));
    let want_max = (3.0 - ((1f64.exp() + 2f64.exp() + 3f64.exp()) / 3.0).ln()).max(0.0) / 2.0;
    assert!(approx(f64::from(m.max[0]), want_max, 1e-5));
    let t = m.to_tensor().unwrap();
    assert_eq!(t.dims(), &[2, 4]);
    assert_eq!(EntityMetrics::COLUMNS, ["max", "std", "gini", "entropy"]);
}

#[test]
fn gini_is_zero_for_a_uniform_column_and_approaches_one_for_a_one_hot_column() {
    let dev = Device::Cpu;
    let n = 100usize;
    // cell r = (1, r == n−1 ? 1 : 0); gene 0 = (0, 0) → a flat column; gene 1 = (0, 60) → one hot cell.
    let mut cells = vec![0f32; 2 * n];
    for r in 0..n {
        cells[2 * r] = 1.0;
        if r == n - 1 {
            cells[2 * r + 1] = 1.0;
        }
    }
    let e_cell = Tensor::from_vec(cells, (n, 2), &dev).unwrap();
    let e_gene = Tensor::from_vec(vec![0f32, 0., 0., 60.], (2, 2), &dev).unwrap();
    let m = compare_entities(&e_cell, &e_gene, 50, 1.0).unwrap();
    assert!(
        f64::from(m.gini[0]) < 1e-6,
        "flat column gini {}",
        m.gini[0]
    );
    assert!(approx(f64::from(m.entropy[0]), (n as f64).ln(), 1e-5));
    assert!(f64::from(m.gini[1]) > 0.95, "one-hot gini {}", m.gini[1]);
    assert!(
        f64::from(m.entropy[1]) < 1e-3,
        "one-hot entropy {}",
        m.entropy[1]
    );
    assert!(f64::from(m.max[1]) > f64::from(m.max[0]));
}
