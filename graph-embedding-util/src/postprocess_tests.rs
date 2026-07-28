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
