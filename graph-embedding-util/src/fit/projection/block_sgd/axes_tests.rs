use super::*;
use crate::fit::projection::block_sgd::{project_cells, Phase2Input};
use legume_numeric::candle::candle_core::Device;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// Deterministic frozen dictionary at a realistic magnitude (see block_sgd_tests).
fn dictionary(n_feat: usize, h: usize, scale: f32, shift: usize) -> (Vec<f32>, Vec<f32>) {
    let mut e = vec![0f32; n_feat * h];
    let mut b = vec![0f32; n_feat];
    for f in 0..n_feat {
        for k in 0..h {
            e[f * h + k] = (((((f + shift) * 7 + k * 13) % 11) as f32 / 11.0) - 0.5) * scale;
        }
        b[f] = ((((f + shift) * 5) % 7) as f32 / 7.0) - 0.3;
    }
    (e, b)
}

/// Noiseless Poisson rates of one cell on one axis: `exp(e_f·θ + β_f + c)`.
fn rates(e: &[f32], b: &[f32], h: usize, theta: &[f32], c: f32) -> (Vec<u32>, Vec<f32>) {
    let mut feats = Vec::with_capacity(b.len());
    let mut counts = Vec::with_capacity(b.len());
    for f in 0..b.len() {
        let ef = &e[f * h..(f + 1) * h];
        let s: f32 = ef.iter().zip(theta).map(|(a, t)| a * t).sum::<f32>() + b[f] + c;
        feats.push(f as u32);
        counts.push(s.exp());
    }
    (feats, counts)
}

const PLANTED: [[f32; 6]; 3] = [
    [0.8, -0.6, 0.4, 0.2, -0.3, 0.5],
    [-0.5, 0.7, -0.2, 0.6, 0.1, -0.4],
    [0.2, 0.1, -0.7, -0.3, 0.5, 0.2],
];
const DEPTHS: [[f32; 2]; 3] = [[0.3, 1.6], [0.5, 1.9], [-0.2, 1.3]];

/// Two axes with different dictionaries and different per-cell depths: the
/// shared latent comes back, and each axis's intercept is its own depth.
#[test]
fn two_axes_recover_theta_with_separate_intercepts() {
    let h = 6;
    let (e0, b0) = dictionary(220, h, 0.5, 0);
    let (e1, b1) = dictionary(340, h, 0.5, 3);
    let dicts = [
        AxisDict {
            label: "a0",
            feat: &e0,
            b_feat: &b0,
        },
        AxisDict {
            label: "a1",
            feat: &e1,
            b_feat: &b1,
        },
    ];
    let dev = Device::Cpu;
    let p = AxesProjector::new(&dicts, h, 1e-3, &dev).unwrap();
    let group = CellGroup {
        cells: (0..3).collect(),
        axes: vec![
            PLANTED
                .iter()
                .zip(&DEPTHS)
                .map(|(t, d)| rates(&e0, &b0, h, t, d[0]))
                .collect(),
            PLANTED
                .iter()
                .zip(&DEPTHS)
                .map(|(t, d)| rates(&e1, &b1, h, t, d[1]))
                .collect(),
        ],
    };
    let bar = indicatif::ProgressBar::hidden();
    let out = p.project_group(&group, None, &bar).unwrap();
    assert_eq!(out.intercepts.len(), 2);
    for (i, want) in PLANTED.iter().enumerate() {
        let c = cos(&out.theta[i * h..(i + 1) * h], want);
        assert!(c > 0.97, "cell {i} misaligned (cos={c:.3})");
        for (a, &planted) in DEPTHS[i].iter().enumerate() {
            let got = out.intercepts[a][i];
            assert!(
                (got - planted).abs() < 0.1,
                "cell {i} axis {a} intercept {got:.3} vs planted {planted:.3}"
            );
        }
    }
}

/// One axis through the per-axis engine is the single-partition cold solve.
#[test]
fn one_axis_matches_the_single_partition_solve() {
    let h = 5;
    let (e, b) = dictionary(160, h, 0.5, 0);
    let dev = Device::Cpu;
    let per_cell: Vec<(Vec<u32>, Vec<f32>)> = PLANTED
        .iter()
        .zip(&DEPTHS)
        .map(|(t, d)| rates(&e, &b, h, &t[..h], d[0]))
        .collect();
    let cells: Vec<(u32, &[u32], &[f32])> = per_cell
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let input = Phase2Input {
        feat: &e,
        b_feat: &b,
        h,
        n_cells: cells.len(),
        lambda: 1e-3,
        dev: &dev,
        label: "test",
        gauge_fix: false,
    };
    let single = project_cells(&input, &cells, None).unwrap();

    let dicts = [AxisDict {
        label: "a0",
        feat: &e,
        b_feat: &b,
    }];
    let p = AxesProjector::new(&dicts, h, 1e-3, &dev).unwrap();
    let group = CellGroup {
        cells: (0..3).collect(),
        axes: vec![per_cell.clone()],
    };
    let out = p
        .project_group(&group, None, &indicatif::ProgressBar::hidden())
        .unwrap();
    for i in 0..3 {
        let c = cos(
            &out.theta[i * h..(i + 1) * h],
            &single.theta[i * h..(i + 1) * h],
        );
        assert!(c > 0.999, "cell {i}: per-axis vs single-partition cos {c}");
        assert!((out.intercepts[0][i] - single.b_cell[i]).abs() < 1e-2);
    }
}

/// A cell with no counts on an axis keeps that intercept at the clamp floor and
/// is placed by the other axis alone.
#[test]
fn a_cell_empty_on_one_axis_keeps_that_intercept_at_the_floor() {
    let h = 5;
    let (e0, b0) = dictionary(160, h, 0.5, 0);
    let (e1, b1) = dictionary(120, h, 0.5, 2);
    let dev = Device::Cpu;
    let dicts = [
        AxisDict {
            label: "a0",
            feat: &e0,
            b_feat: &b0,
        },
        AxisDict {
            label: "a1",
            feat: &e1,
            b_feat: &b1,
        },
    ];
    let p = AxesProjector::new(&dicts, h, 1e-3, &dev).unwrap();
    let t = &PLANTED[0][..h];
    let group = CellGroup {
        cells: vec![0],
        axes: vec![
            vec![rates(&e0, &b0, h, t, 0.4)],
            vec![(Vec::new(), Vec::new())],
        ],
    };
    let out = p
        .project_group(&group, None, &indicatif::ProgressBar::hidden())
        .unwrap();
    assert!(cos(&out.theta[..h], t) > 0.97);
    assert_eq!(
        out.intercepts[1][0],
        -(crate::cell_projection::SCORE_CLAMP as f32)
    );
}

#[test]
fn an_axis_with_no_live_feature_is_refused() {
    let h = 3;
    let (e0, b0) = dictionary(10, h, 0.5, 0);
    let zeros = vec![0f32; 4 * h];
    let b1 = vec![0f32; 4];
    let dicts = [
        AxisDict {
            label: "genes",
            feat: &e0,
            b_feat: &b0,
        },
        AxisDict {
            label: "peaks",
            feat: &zeros,
            b_feat: &b1,
        },
    ];
    let err = AxesProjector::new(&dicts, h, 1.0, &Device::Cpu)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("peaks"), "{err}");
}

/// Started at the planted latent, the solve reaches the same answer as the
/// cold start in fewer steps, and each axis's intercept still lands on its depth.
#[test]
fn a_warm_start_reaches_the_cold_answer_in_fewer_steps() {
    let h = 6;
    let (e0, b0) = dictionary(220, h, 0.5, 0);
    let (e1, b1) = dictionary(340, h, 0.5, 3);
    let dicts = [
        AxisDict {
            label: "a0",
            feat: &e0,
            b_feat: &b0,
        },
        AxisDict {
            label: "a1",
            feat: &e1,
            b_feat: &b1,
        },
    ];
    let p = AxesProjector::new(&dicts, h, 1e-3, &Device::Cpu).unwrap();
    let group = CellGroup {
        cells: (0..3).collect(),
        axes: vec![
            PLANTED
                .iter()
                .zip(&DEPTHS)
                .map(|(t, d)| rates(&e0, &b0, h, t, d[0]))
                .collect(),
            PLANTED
                .iter()
                .zip(&DEPTHS)
                .map(|(t, d)| rates(&e1, &b1, h, t, d[1]))
                .collect(),
        ],
    };
    let bar = indicatif::ProgressBar::hidden();
    let cold = p.project_group(&group, None, &bar).unwrap();
    let init: Vec<f32> = PLANTED.iter().flatten().copied().collect();
    let warm = p.project_group(&group, Some(&init), &bar).unwrap();
    assert!(
        warm.steps < cold.steps,
        "warm {} steps, cold {} steps",
        warm.steps,
        cold.steps
    );
    for (i, depths) in DEPTHS.iter().enumerate() {
        let c = cos(
            &warm.theta[i * h..(i + 1) * h],
            &cold.theta[i * h..(i + 1) * h],
        );
        assert!(c > 0.999, "cell {i}: warm vs cold cos {c}");
        for (a, &planted) in depths.iter().enumerate() {
            let got = warm.intercepts[a][i];
            assert!(
                (got - planted).abs() < 0.1,
                "cell {i} axis {a} intercept {got:.3} vs planted {planted:.3}"
            );
        }
    }
}

#[test]
fn a_warm_start_of_the_wrong_size_is_refused() {
    let h = 3;
    let (e0, b0) = dictionary(40, h, 0.5, 0);
    let dicts = [AxisDict {
        label: "a0",
        feat: &e0,
        b_feat: &b0,
    }];
    let p = AxesProjector::new(&dicts, h, 1e-3, &Device::Cpu).unwrap();
    let group = CellGroup {
        cells: vec![0, 1],
        axes: vec![vec![
            rates(&e0, &b0, h, &PLANTED[0][..h], 0.2),
            rates(&e0, &b0, h, &PLANTED[1][..h], 0.2),
        ]],
    };
    let init = vec![0f32; h];
    let err = p
        .project_group(&group, Some(&init), &indicatif::ProgressBar::hidden())
        .expect_err("refused");
    assert!(err.to_string().contains("warm start"), "{err}");
}
