//! [`FusedTensorOps::clamped_exp_add_inplace`] must be indistinguishable from the op
//! chain it replaces.
//!
//! The bar is **bitwise** equality, not a tolerance: the fused loop performs the
//! same three scalar f32 operations in the same order as
//! `broadcast_add` → `minimum` → `exp`, so any difference is a bug in the kernel
//! rather than float reassociation.

use crate::traits::FusedTensorOps;
use candle_core::{DType, Device, Result, Tensor};

const CEILING: f64 = 30.0;

/// Deterministic spread of magnitudes, including values that clear the ceiling so
/// the guard is actually exercised.
fn ramp(n: usize, f: usize, scale: f32, shift: f32) -> Vec<f32> {
    (0..n * f)
        .map(|i| ((i % 97) as f32 / 97.0 - 0.5) * scale + shift)
        .collect()
}

/// Run both paths over an `[n, f]` receiver and an offset of `shape`, and assert
/// they agree bitwise. The reference is the chain the kernel stands in for.
fn assert_matches_op_chain(n: usize, f: usize, shape: (usize, usize)) -> Result<()> {
    let dev = Device::Cpu;
    let (n_off, f_off) = shape;
    let s = Tensor::from_vec(ramp(n, f, 80.0, 2.0), (n, f), &dev)?;
    let offset = Tensor::from_vec(ramp(n_off, f_off, 12.0, -3.0), shape, &dev)?;

    let want = s
        .broadcast_add(&offset)?
        .minimum(CEILING)?
        .exp()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let got = s
        .clamped_exp_add_inplace(&offset, CEILING)
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert_eq!(
        got, want,
        "offset {shape:?} on a [{n}, {f}] receiver diverged"
    );
    Ok(())
}

/// The three shapes `broadcast_add` accepts, each a distinct kernel branch.
#[test]
fn every_broadcast_shape_matches_the_op_chain() -> Result<()> {
    assert_matches_op_chain(41, 29, (41, 29))?; // per element
    assert_matches_op_chain(37, 53, (1, 53))?; // per column
    assert_matches_op_chain(37, 53, (37, 1))?; // per row
    Ok(())
}

/// Rows are batched into rayon tasks, so a receiver with far more rows than the
/// pool has workers is where a mis-indexed chunk would show up — and a thin panel
/// (`f` small) is the shape that makes the batching kick in.
#[test]
fn many_rows_and_thin_panels_stay_in_order() -> Result<()> {
    assert_matches_op_chain(1024, 7, (1024, 7))?;
    assert_matches_op_chain(1024, 7, (1, 7))?;
    assert_matches_op_chain(4096, 3, (4096, 1))?;
    Ok(())
}

/// The ceiling is the overflow guard — `exp` blows past f32 at 88 — so a run that
/// never reaches it would pass while testing nothing.
#[test]
fn the_ceiling_binds() -> Result<()> {
    let dev = Device::Cpu;
    let s = Tensor::from_vec(vec![100.0f32, 0.0, -100.0, 29.0], (1, 4), &dev)?;
    let offset = Tensor::zeros((1, 4), DType::F32, &dev)?;

    let got = s
        .clamped_exp_add_inplace(&offset, CEILING)
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert_eq!(got[0], 30f32.exp(), "over the ceiling was not clamped");
    assert_eq!(got[1], 1.0);
    assert_eq!(
        got[2],
        (-100f32).exp(),
        "underflow is the right answer here"
    );
    assert_eq!(got[3], 29f32.exp(), "under the ceiling was altered");
    assert!(got[0].is_finite(), "the guard let exp overflow");
    Ok(())
}

/// A shape the fused kernel does not handle must still compute the right thing —
/// the wrapper falls back to the op chain rather than refusing or, worse, walking
/// the wrong elements.
#[test]
fn unfusable_shapes_fall_back() -> Result<()> {
    let dev = Device::Cpu;
    let (n, f) = (8, 5);
    let s = Tensor::from_vec(ramp(n, f, 20.0, 1.0), (n, f), &dev)?;
    // `[F]` rather than `[1, F]`: rank 1, so the fused path declines it.
    let offset = Tensor::from_vec(ramp(1, f, 4.0, 0.5), f, &dev)?;

    let want = s
        .broadcast_add(&offset)?
        .minimum(CEILING)?
        .exp()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let got = s
        .clamped_exp_add_inplace(&offset, CEILING)
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert_eq!(got, want, "the fallback changed the answer");
    Ok(())
}
