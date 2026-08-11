//! Growing a checkpoint must not change what the model computes.
//!
//! The two properties are in tension, which is the whole reason the fill rule
//! is per-tensor rather than "pad with zeros":
//!
//! - `β = softmax_g(α·ρᵀ)` must be **unchanged** at step 0, so `α`'s new
//!   columns are zero.
//! - The added subspace must still **receive gradient**, so `ρ`'s new columns
//!   are *not* zero. Zeroing both would make `∂β/∂α_new ∝ ρ_new = 0` and
//!   `∂β/∂ρ_new ∝ α_new = 0` — capacity that can never learn anything.
//!
//! A change that satisfies only the first is easy to make by accident, and
//! silently gives a model that trains as if the flag had done nothing.

use candle_core::{DType, Device, Tensor};
use candle_util::grow::{grow_tensor, GrowthDims, NEW_TOPIC_LOGIT_BIAS};

const K_OLD: usize = 4;
const K_NEW: usize = 7;
const H_OLD: usize = 12;
const H_NEW: usize = 18;
const D: usize = 40;

fn dims() -> GrowthDims {
    GrowthDims {
        k_old: K_OLD,
        k_new: K_NEW,
        h_old: H_OLD,
        h_new: H_NEW,
    }
}

/// A tensor whose entries are distinct and non-zero, so a value that survives
/// the copy is distinguishable from a value that was filled in.
fn ramp(shape: &[usize], dev: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
    Tensor::from_vec(v, shape, dev).expect("ramp")
}

fn to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .to_vec2()
        .expect("vec2")
}

#[test]
fn alpha_keeps_its_values_and_zeroes_the_new_embedding_columns() {
    let dev = Device::Cpu;
    let saved = ramp(&[K_OLD, H_OLD], &dev);
    let fresh = ramp(&[K_OLD, H_NEW], &dev).affine(-1.0, 0.0).expect("neg");

    let out = grow_tensor("dec_0.topic.embeddings", &fresh, &saved, &dims()).expect("grow");
    assert_eq!(out.dims(), &[K_OLD, H_NEW]);

    let (o, s) = (to_vec2(&out), to_vec2(&saved));
    for k in 0..K_OLD {
        for h in 0..H_OLD {
            assert_eq!(o[k][h], s[k][h], "checkpoint value lost at ({k},{h})");
        }
        for (h, &v) in o[k].iter().enumerate().skip(H_OLD) {
            assert_eq!(v, 0.0, "new alpha column ({k},{h}) must be zero");
        }
    }
}

/// The counterpart, and the one that is easy to get wrong: `ρ`'s new columns
/// must keep their random init or the added subspace is dead on arrival.
#[test]
fn rho_keeps_its_fresh_init_in_the_new_columns() {
    let dev = Device::Cpu;
    let saved = ramp(&[D, H_OLD], &dev);
    let fresh = ramp(&[D, H_NEW], &dev).affine(-1.0, 0.0).expect("neg");

    let out = grow_tensor("enc.feature.embeddings", &fresh, &saved, &dims()).expect("grow");
    let (o, f, s) = (to_vec2(&out), to_vec2(&fresh), to_vec2(&saved));
    for d in 0..D {
        for h in 0..H_OLD {
            assert_eq!(o[d][h], s[d][h], "checkpoint value lost at ({d},{h})");
        }
        for h in H_OLD..H_NEW {
            assert_eq!(
                o[d][h], f[d][h],
                "rho's new column ({d},{h}) must keep its init"
            );
            assert_ne!(o[d][h], 0.0, "a zeroed rho column is a dead subspace");
        }
    }
}

#[test]
fn added_topics_start_switched_off_at_the_encoder() {
    let dev = Device::Cpu;
    let l2 = 5usize;

    let w = grow_tensor(
        "enc.nn.enc.z.mean.weight",
        &ramp(&[K_NEW, l2], &dev),
        &ramp(&[K_OLD, l2], &dev),
        &dims(),
    )
    .expect("grow weight");
    for (k, row) in to_vec2(&w).iter().enumerate().skip(K_OLD) {
        assert!(
            row.iter().all(|&x| x == 0.0),
            "added topic {k} must start with zero weights, got {row:?}"
        );
    }
    let _ = l2;

    let b = grow_tensor(
        "enc.nn.enc.z.mean.bias",
        &ramp(&[K_NEW], &dev),
        &ramp(&[K_OLD], &dev),
        &dims(),
    )
    .expect("grow bias");
    let bv: Vec<f32> = b.to_vec1().expect("vec1");
    for (k, v) in bv.iter().enumerate().take(K_NEW).skip(K_OLD) {
        assert!(
            (f64::from(*v) - NEW_TOPIC_LOGIT_BIAS).abs() < 1e-6,
            "added topic {k} bias {v} should be the off-switch constant"
        );
    }
    // With zero weights the pre-activation IS the bias, so this is the mass the
    // topic starts with relative to a parent topic sitting at a logit of ~0.
    assert!(
        NEW_TOPIC_LOGIT_BIAS.exp() < 1e-3,
        "the off-switch must actually switch off"
    );
}

/// Growth must not become a way to paper over an architecture change: a
/// widened hidden layer also grows a tensor, and has to stay an error.
#[test]
fn an_unrelated_shape_change_is_rejected() {
    let dev = Device::Cpu;
    // 5 → 9 is neither the K growth (4 → 7) nor the H growth (12 → 18).
    let err = grow_tensor(
        "enc.nn.enc.fc.relu_linear_stack.1.weight",
        &ramp(&[9, 64], &dev),
        &ramp(&[5, 64], &dev),
        &dims(),
    )
    .expect_err("an unexplained growth must not be silently zero-padded");
    let msg = err.to_string();
    assert!(
        msg.contains("architecture change"),
        "should say what it is refusing: {msg}"
    );
}
