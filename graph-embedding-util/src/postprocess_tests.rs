//! Invariants of the co-embedding's per-feature rescale and its temperature
//! calibration. The claim these exist to pin is that `T` is calibrated and spent
//! in the SAME units: the rescale is applied inside both [`coembed_block`] and
//! the calibration path, and doing it in only one is shape-valid, silently wrong,
//! and invisible to every other test in the crate.

use super::*;
use candle_util::candle_core::{DType, Device};

const N: usize = 256;
const H: usize = 8;

/// Deterministic cells, plus features whose norms span three orders of magnitude
/// — the condition that breaks a single shared temperature on raw scores.
fn fixture(d: usize) -> (Tensor, Tensor) {
    let dev = Device::Cpu;
    let cells: Vec<f32> = (0..N * H)
        .map(|i| ((i * 37) % 23) as f32 * 0.1 - 1.1)
        .collect();
    let feats: Vec<f32> = (0..d * H)
        .map(|i| {
            let f = i / H;
            // norms spanning 1e-2 .. 1e1 across features
            let scale = 10f32.powf((f % 4) as f32 - 2.0);
            (((i * 17) % 19) as f32 * 0.1 - 0.9) * scale
        })
        .collect();
    (
        Tensor::from_vec(cells, (N, H), &dev).unwrap(),
        Tensor::from_vec(feats, (d, H), &dev).unwrap(),
    )
}

fn col_sds(t: &Tensor) -> Vec<f32> {
    let (n, b) = t.dims2().unwrap();
    let v = t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
    (0..b)
        .map(|j| {
            let col: Vec<f32> = (0..n).map(|i| v[i][j]).collect();
            let m = col.iter().sum::<f32>() / n as f32;
            (col.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n as f32).sqrt()
        })
        .collect()
}

/// Every feature column comes out at unit scale regardless of `‖e_f‖`. This is
/// the whole mechanism: it is what lets one `T` mean the same thing for a long
/// gene and a short one.
#[test]
fn every_feature_column_is_rescaled_to_unit_sd() {
    let (cells, feats) = fixture(32);
    let raw = cells
        .matmul(&feats.t().unwrap().contiguous().unwrap())
        .unwrap();

    let raw_sds = col_sds(&raw);
    let spread = raw_sds.iter().cloned().fold(f32::MIN, f32::max)
        / raw_sds.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        spread > 100.0,
        "fixture must actually exercise a wide norm spread, got {spread:.1}×"
    );

    for (j, sd) in col_sds(&standardize_columns(&raw).unwrap())
        .iter()
        .enumerate()
    {
        assert!(
            (sd - 1.0).abs() < 1e-3,
            "column {j} rescaled to sd {sd}, want 1.0"
        );
    }
}

/// A zero-embedding feature has a constant score column and zero SD. It must
/// divide by the floor and stay finite rather than emitting `NaN` that would
/// propagate into the written parquet.
#[test]
fn a_constant_column_stays_finite() {
    let dev = Device::Cpu;
    let (cells, _) = fixture(1);
    let zero_feat = Tensor::zeros((2, H), DType::F32, &dev).unwrap();
    let raw = cells
        .matmul(&zero_feat.t().unwrap().contiguous().unwrap())
        .unwrap();

    let out = standardize_columns(&raw).unwrap();
    assert!(
        out.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()),
        "a zero-embedding feature must not produce NaN/Inf"
    );
}

/// The end-to-end contract: the `T` returned by [`feature_coembedding`], applied
/// the way `coembed_block` applies it, actually lands near the requested
/// eff-cells target. This fails if calibration and application ever diverge in
/// units — the failure mode the shared helper exists to prevent.
#[test]
fn the_calibrated_temperature_hits_its_eff_target() {
    let (cells, feats) = fixture(96);
    let target = 40.0;
    let (co, t) = feature_coembedding(&cells, &feats, target).unwrap();
    assert_eq!(co.dims2().unwrap(), (96, H));

    let scores = standardize_columns(
        &cells
            .matmul(&feats.t().unwrap().contiguous().unwrap())
            .unwrap(),
    )
    .unwrap();
    let eff = eff_from_scores(&scores, f64::from(t)).unwrap();
    assert!(
        eff > target / 3.0 && eff < target * 3.0,
        "calibrated T={t} gives eff {eff:.1}, want within 3× of {target}"
    );
}

/// The clamp that keeps a big-cluster dataset off the far tail of the curve,
/// where the barycentre collapses toward the global centroid.
#[test]
fn the_eff_target_is_clamped_above_and_below() {
    // One giant cluster: median size 900 must not survive as the target.
    let huge: Vec<usize> = vec![0; 1800];
    assert_eq!(target_eff_from_labels(&huge, 1), MAX_TARGET_EFF);

    // All singletons: median size 1 must be lifted to the floor.
    let singletons: Vec<usize> = (0..64).collect();
    assert_eq!(target_eff_from_labels(&singletons, 64), MIN_TARGET_EFF);
}

//////////////////////////////////////////
// PIP-driven confidence shrinkage (§8) //
//////////////////////////////////////////

#[test]
fn shrinkage_scales_each_feature_row_by_its_own_weight() {
    let dev = candle_util::candle_core::Device::Cpu;
    let co = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &dev).unwrap();
    let out = shrink_by_confidence(&co, &[1.0, 0.5, 0.0]).unwrap();
    let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(got, vec![1.0, 2.0, 1.5, 2.0, 0.0, 0.0]);
}

/// The case measured on real data: a selection posterior estimated on far more
/// dims than the embedding's effective rank saturates, `max_h PIP` is ≈1 for
/// every feature, and the shrinkage becomes one constant. It must be reported as
/// degenerate rather than applied silently.
#[test]
fn a_saturated_posterior_is_reported_as_degenerate() {
    let saturated: Vec<f32> = (0..500).map(|i| 0.97 + (i % 4) as f32 * 0.005).collect();
    let s = confidence_spread(&saturated);
    assert!(
        s.is_degenerate(),
        "spread [{:.3}, {:.3}] should read as degenerate",
        s.min,
        s.max
    );
    assert_eq!(s.n_below_half, 0);
    assert_eq!(s.n, 500);
}

/// SIMBA's `si.tl.embed`: raw dot scores, a FIXED temperature, no per-feature
/// rescaling and no calibration. Checked against an f64 softmax average.
#[test]
fn fixed_t_coembedding_uses_raw_scores_and_matches_a_hand_computed_softmax_average() {
    use candle_util::candle_core::{Device, Tensor};
    let dev = Device::Cpu;
    let cells: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![-1.0, 0.5],
        vec![2.0, 2.0],
    ];
    let feats: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, -1.0], vec![0.3, 0.3]];
    let flat = |t: &[Vec<f64>]| -> Vec<f32> { t.iter().flatten().map(|&v| v as f32).collect() };
    let e_cell = Tensor::from_vec(flat(&cells), (4, 2), &dev).unwrap();
    let e_feat = Tensor::from_vec(flat(&feats), (3, 2), &dev).unwrap();
    let t = 0.5;
    let got = feature_coembedding_fixed_t(&e_cell, &e_feat, t)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    for (f, feat) in feats.iter().enumerate() {
        let scores: Vec<f64> = cells
            .iter()
            .map(|c| c.iter().zip(feat).map(|(a, b)| a * b).sum::<f64>() / t)
            .collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = scores.iter().map(|s| (s - m).exp()).sum();
        let p: Vec<f64> = scores.iter().map(|s| (s - m).exp() / z).collect();
        for h in 0..2 {
            let want: f64 = cells.iter().zip(&p).map(|(c, w)| c[h] * w).sum();
            assert!(
                (f64::from(got[f][h]) - want).abs() < 1e-5,
                "feature {f} dim {h}: {} vs {want}",
                got[f][h]
            );
        }
    }
    // T → ∞ is the cell centroid; T → 0 is the best-scoring cell.
    let hot = feature_coembedding_fixed_t(&e_cell, &e_feat, 1e6)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let centroid = [0.5f64, 0.875];
    for row in &hot {
        for h in 0..2 {
            assert!((f64::from(row[h]) - centroid[h]).abs() < 1e-4);
        }
    }
    let cold = feature_coembedding_fixed_t(&e_cell, &e_feat, 1e-3)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    // feature 0 = (1, 0) scores cell 3 = (2, 2) highest
    assert!(
        (f64::from(cold[0][0]) - 2.0).abs() < 1e-4 && (f64::from(cold[0][1]) - 2.0).abs() < 1e-4
    );
    // feature 1 = (0, −1) scores cell 0 = (1, 0) highest (0 vs −1, −0.5, −2)
    assert!((f64::from(cold[1][0]) - 1.0).abs() < 1e-4 && f64::from(cold[1][1]).abs() < 1e-4);
}
