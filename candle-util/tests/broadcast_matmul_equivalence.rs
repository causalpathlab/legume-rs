//! Pins the batched-mat-vec rewrites against the broadcast forms they replaced.
//!
//! Several hot loops computed `Σ_t a[n,k,t]·b[n,t]` as
//! `a.broadcast_mul(&b.reshape((n,1,t))).sum(2)`. That layout puts a stride-0
//! dim BETWEEN two non-zero strides, and `candle_core::Layout::offsets_b`
//! strips only LEADING and TRAILING zero strides before requiring the rest to be
//! contiguous — so it returns `None` and the multiply runs a scalar
//! `StridedIndex` loop with no SIMD, materializing an `[N,K,T]` product that
//! backward multiplies again.
//!
//! The replacements are `matmul`. These tests assert the two agree, so the
//! speedup cannot silently change the math.

use candle_core::{DType, Device, Tensor};

fn deterministic(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u64).wrapping_mul(seed).wrapping_add(7919);
            ((x % 2000) as f32) / 1000.0 - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// The ETM decoders' `expected_counts`: `[N,K,T] × [N,T] → [N,K]`.
#[test]
fn etm_expected_counts_matmul_matches_broadcast() {
    let dev = Device::Cpu;
    let (n, k, t) = (64usize, 12usize, 24usize);
    let beta = Tensor::from_vec(deterministic(n * k * t, 3), (n, k, t), &dev).unwrap();
    let theta = Tensor::from_vec(deterministic(n * t, 11), (n, t), &dev).unwrap();

    let via_matmul = beta
        .matmul(&theta.reshape((n, t, 1)).unwrap())
        .unwrap()
        .squeeze(2)
        .unwrap();
    let via_broadcast = beta
        .broadcast_mul(&theta.reshape((n, 1, t)).unwrap())
        .unwrap()
        .sum(2)
        .unwrap();

    assert_eq!(via_matmul.dims(), &[n, k]);
    assert_eq!(via_matmul.dims(), via_broadcast.dims());
    assert!(max_abs_diff(&via_matmul, &via_broadcast) < 1e-4);
}

/// Attention pooling: `[N,K] weights × [N,K,H] content → [N,H]`.
#[test]
fn attention_pool_matmul_matches_broadcast() {
    let dev = Device::Cpu;
    let (n, k, h) = (48usize, 9usize, 16usize);
    let content = Tensor::from_vec(deterministic(n * k * h, 5), (n, k, h), &dev).unwrap();
    let attn = Tensor::from_vec(deterministic(n * k, 13), (n, k), &dev).unwrap();

    let via_matmul = attn
        .unsqueeze(1)
        .unwrap()
        .matmul(&content)
        .unwrap()
        .squeeze(1)
        .unwrap();
    let via_broadcast = content
        .broadcast_mul(&attn.unsqueeze(2).unwrap())
        .unwrap()
        .sum(1)
        .unwrap();

    assert_eq!(via_matmul.dims(), &[n, h]);
    assert!(max_abs_diff(&via_matmul, &via_broadcast) < 1e-4);
}

/// Gradients must match too — the rewrite changes the backward graph, not just
/// the forward value.
#[test]
fn matmul_rewrite_preserves_gradients() {
    let dev = Device::Cpu;
    let (n, k, t) = (16usize, 5usize, 8usize);
    let theta_data = deterministic(n * t, 17);

    let grad_of = |use_matmul: bool| {
        let beta = Tensor::from_vec(deterministic(n * k * t, 19), (n, k, t), &dev).unwrap();
        let theta = candle_core::Var::from_vec(theta_data.clone(), (n, t), &dev).unwrap();
        let out = if use_matmul {
            beta.matmul(&theta.reshape((n, t, 1)).unwrap())
                .unwrap()
                .squeeze(2)
                .unwrap()
        } else {
            beta.broadcast_mul(&theta.reshape((n, 1, t)).unwrap())
                .unwrap()
                .sum(2)
                .unwrap()
        };
        let loss = out.sqr().unwrap().sum_all().unwrap();
        loss.backward().unwrap().get(&theta).unwrap().clone()
    };

    assert!(max_abs_diff(&grad_of(true), &grad_of(false)) < 1e-3);
}
