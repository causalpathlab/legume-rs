//! The global-L2 gradient clip must never manufacture `NaN` parameters.
//!
//! Regression for the `masked-vae` divergence: one `Inf` gradient made the
//! global norm overflow, `scale` underflow to `0`, and the rescale evaluate
//! `Inf * 0 = NaN` — permanently poisoning the parameter. `clamp` does not
//! launder `NaN` back into its bounds, so every later forward was `NaN` and
//! the run still wrote a full set of all-`NaN` artifacts.

use candle_core::{Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use candle_util::vae::clip_and_step_dense;

/// `w` large enough that `(w * 1e20)^2` overflows f32 in the backward pass.
fn overflowing_grad(w: &Var) -> candle_core::backprop::GradStore {
    let scaled = (w.as_tensor() * 1e20f64).unwrap();
    let loss = scaled.sqr().unwrap().sum_all().unwrap();
    loss.backward().unwrap()
}

#[test]
fn non_finite_grad_norm_skips_the_step_and_leaves_params_finite() {
    let dev = Device::Cpu;
    let w = Var::from_tensor(&Tensor::from_vec(vec![3.0f32, -1.0], (2,), &dev).unwrap()).unwrap();
    let mut adam = AdamW::new(vec![w.clone()], ParamsAdamW::default()).unwrap();

    let grads = overflowing_grad(&w);
    let gnorm_sq = grads
        .get(&w)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        !gnorm_sq.is_finite(),
        "test premise: the gradient should overflow, got {gnorm_sq}"
    );

    let stepped = clip_and_step_dense(&mut adam, grads, 1.0).unwrap();
    assert!(!stepped, "a non-finite gradient norm must skip the step");

    let after = w.as_tensor().to_vec1::<f32>().unwrap();
    assert!(
        after.iter().all(|x| x.is_finite()),
        "params must survive a diverged step, got {after:?}"
    );
    assert_eq!(
        after,
        vec![3.0, -1.0],
        "a skipped step must not move params"
    );
}

#[test]
fn nan_grad_norm_skips_the_step() {
    // The overflow path yields an `Inf` norm; a genuine `NaN` gradient (norm
    // `NaN`) must be caught by the same guard — `NaN.is_finite()` is false.
    let dev = Device::Cpu;
    let w = Var::from_tensor(&Tensor::from_vec(vec![3.0f32, -1.0], (2,), &dev).unwrap()).unwrap();
    let mut adam = AdamW::new(vec![w.clone()], ParamsAdamW::default()).unwrap();

    // A `NaN` factor in the forward makes `dL/dw` `NaN` (not `Inf`).
    let nan = Tensor::new(f32::NAN, &dev).unwrap();
    let loss = w
        .as_tensor()
        .broadcast_mul(&nan)
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    assert!(
        grads.get(&w).unwrap().to_vec1::<f32>().unwrap()[0].is_nan(),
        "test premise: the gradient should be NaN"
    );

    assert!(
        !clip_and_step_dense(&mut adam, grads, 1.0).unwrap(),
        "a NaN gradient norm must skip the step"
    );
    assert_eq!(
        w.as_tensor().to_vec1::<f32>().unwrap(),
        vec![3.0, -1.0],
        "a skipped step must not move params"
    );
}

#[test]
fn finite_grad_still_steps_and_clips() {
    let dev = Device::Cpu;
    let w = Var::from_tensor(&Tensor::from_vec(vec![3.0f32, -1.0], (2,), &dev).unwrap()).unwrap();
    let mut adam = AdamW::new(vec![w.clone()], ParamsAdamW::default()).unwrap();

    let loss = w.as_tensor().sqr().unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();

    assert!(
        clip_and_step_dense(&mut adam, grads, 1.0).unwrap(),
        "a healthy gradient must still take the step"
    );
    let after = w.as_tensor().to_vec1::<f32>().unwrap();
    assert!(after.iter().all(|x| x.is_finite()));
    assert_ne!(after, vec![3.0, -1.0], "params should have moved");
}
