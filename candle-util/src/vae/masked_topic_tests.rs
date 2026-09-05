//! Tests for the trainer's own stochastic and target-building pieces: the
//! per-epoch Poisson draw behind `MaskedTrainOpts::poisson_thin`, the seeded
//! context mask, the mask-rate schedule, the `[N, D]` target mask, the
//! per-level target table, and that visible genes are never scored.

use super::{
    draw_context_mask, mask_rate, poisson_draw, step_seed, target_mask_nd, LevelTarget,
    MaskSchedule, Mat,
};
use crate::decoder::masked_etm::{EmbeddedNbTopicDecoder, MaskedDenseTarget};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/////////////////////
// Poisson thinning //
/////////////////////

/// Zero rates must draw zero — a gene absent from a pseudobulk cannot acquire a
/// count — and every draw must be a non-negative integer.
#[test]
fn zero_rates_stay_zero_and_draws_are_counts() {
    let mut rates = Mat::zeros(50, 40);
    for i in 0..50 {
        for j in 0..40 {
            rates[(i, j)] = if (i + j) % 3 == 0 { 0.0 } else { 2.5 };
        }
    }
    let x = poisson_draw(&rates, 7);
    assert_eq!(x.shape(), rates.shape());
    for i in 0..50 {
        for j in 0..40 {
            let v = x[(i, j)];
            if rates[(i, j)] == 0.0 {
                assert_eq!(v, 0.0, "({i},{j}) drew {v} from a zero rate");
            }
            assert!(
                v >= 0.0 && v.fract() == 0.0,
                "({i},{j}) = {v} is not a count"
            );
        }
    }
}

/// The draw is unbiased: over enough entries its mean recovers the rate.
#[test]
fn draws_are_unbiased_around_the_rate() {
    for &rate in &[0.05f32, 1.0, 7.5] {
        let rates = Mat::from_element(400, 100, rate);
        let x = poisson_draw(&rates, 7);
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        // 40k draws; the standard error is sqrt(rate / 40k).
        let se = (rate / 40_000.0).sqrt();
        assert!(
            (mean - rate).abs() < 5.0 * se + 1e-3,
            "rate {rate}: mean {mean} is off by more than 5 SE ({se})"
        );
    }
}

/// Non-finite rates are treated as absent rather than panicking inside
/// `Poisson::new`.
#[test]
fn non_finite_rates_draw_zero() {
    let mut rates = Mat::from_element(4, 4, 1.0);
    rates[(0, 0)] = f32::NAN;
    rates[(1, 1)] = f32::INFINITY;
    let x = poisson_draw(&rates, 7);
    assert_eq!(x[(0, 0)], 0.0);
    assert_eq!(x[(1, 1)], 0.0);
}

/// The draw must be a function of its seed alone — not of the thread count,
/// and not of run-to-run OS entropy. This is the property that lets a
/// `--poisson-thin` result be replicated or bisected at all.
#[test]
fn the_draw_is_reproducible_from_its_seed() {
    let mut rates = Mat::zeros(200, 60);
    for i in 0..200 {
        for j in 0..60 {
            rates[(i, j)] = 0.5 + (i % 7) as f32;
        }
    }
    let a = poisson_draw(&rates, 11);
    let b = poisson_draw(&rates, 11);
    assert_eq!(a, b, "same seed must give the same draw");
    let c = poisson_draw(&rates, 12);
    assert_ne!(a, c, "a different seed must give a different draw");
}

//////////////////////////
// Seeded context mask   //
//////////////////////////

fn to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_vec2().unwrap()
}

/// A `[3, 4]` packed context with a pad slot in every row.
fn context_values() -> Tensor {
    #[rustfmt::skip]
    let v: Vec<f32> = vec![
        2.0, 5.0, 1.0, 0.0,
        7.0, 3.0, 0.0, 0.0,
        4.0, 6.0, 9.0, 0.0,
    ];
    Tensor::from_vec(v, (3, 4), &Device::Cpu).unwrap()
}

#[test]
fn context_mask_is_reproducible_from_its_seed() {
    let values = Tensor::from_vec(
        (0..2000)
            .map(|i| {
                if i % 5 == 0 {
                    0.0
                } else {
                    1.0 + (i % 7) as f32
                }
            })
            .collect(),
        (40, 50),
        &Device::Cpu,
    )
    .unwrap();
    let (vis_a, msk_a) = draw_context_mask(&values, 0.3, 99).unwrap();
    let (vis_b, msk_b) = draw_context_mask(&values, 0.3, 99).unwrap();
    assert_eq!(to_vec2(&vis_a), to_vec2(&vis_b), "same seed, same visible");
    assert_eq!(to_vec2(&msk_a), to_vec2(&msk_b), "same seed, same masked");
    let (vis_c, _) = draw_context_mask(&values, 0.3, 100).unwrap();
    assert_ne!(
        to_vec2(&vis_a),
        to_vec2(&vis_c),
        "a different seed must differ"
    );

    // visible + masked partitions exactly the real (value > 0) slots.
    let real = to_vec2(&values.gt(0.0).unwrap().to_dtype(DType::F32).unwrap());
    let (vis, msk) = (to_vec2(&vis_a), to_vec2(&msk_a));
    for n in 0..40 {
        for k in 0..50 {
            assert_eq!(vis[n][k] + msk[n][k], real[n][k], "slot ({n},{k})");
            assert!(vis[n][k] == 0.0 || vis[n][k] == 1.0);
        }
    }
    let masked_frac = msk.iter().flatten().sum::<f32>() / real.iter().flatten().sum::<f32>();
    assert!(
        (masked_frac - 0.3).abs() < 0.05,
        "masked fraction {masked_frac} is far from the rate"
    );
}

#[test]
fn pads_are_neither_visible_nor_masked() {
    let (vis, msk) = draw_context_mask(&context_values(), 0.5, 1).unwrap();
    let (vis, msk) = (to_vec2(&vis), to_vec2(&msk));
    for n in 0..3 {
        assert_eq!(vis[n][3], 0.0, "row {n} pad visible");
        assert_eq!(msk[n][3], 0.0, "row {n} pad masked");
    }
    assert_eq!(vis[1][2], 0.0);
    assert_eq!(msk[1][2], 0.0);
}

#[test]
fn uniform_schedule_rate_is_reproducible_and_in_range() {
    assert_eq!(mask_rate(MaskSchedule::Fixed, 0.3, 5), 0.3);
    let sched = MaskSchedule::Uniform { lo: 0.1, hi: 0.6 };
    let a = mask_rate(sched, 0.3, 5);
    let b = mask_rate(sched, 0.3, 5);
    assert_eq!(a, b, "same step seed, same rate");
    assert!((0.1..=0.6).contains(&a), "rate {a} outside [lo, hi]");
    let rates: Vec<f64> = (0..50).map(|s| mask_rate(sched, 0.3, s)).collect();
    assert!(
        rates.iter().any(|&r| (r - a).abs() > 1e-9),
        "rate never moves across steps"
    );
}

#[test]
fn step_seeds_are_distinct_across_epoch_level_and_minibatch() {
    let s = step_seed(42, 0, 0, 0);
    assert_ne!(s, step_seed(42, 1, 0, 0));
    assert_ne!(s, step_seed(42, 0, 1, 0));
    assert_ne!(s, step_seed(42, 0, 0, 1));
    assert_ne!(s, step_seed(43, 0, 0, 0));
    assert_eq!(s, step_seed(42, 0, 0, 0));
}

////////////////////////
// Targets and library //
////////////////////////

/// Three rows over eight genes with a three-slot context. Gene 0 is in no
/// row's context, so a pad (index 0, visible 0) must never touch it.
fn small_context() -> (Tensor, Tensor) {
    #[rustfmt::skip]
    let idx: Vec<u32> = vec![
        1, 4, 6,
        2, 5, 0,   // last slot is a pad
        3, 7, 1,
    ];
    #[rustfmt::skip]
    let vis: Vec<f32> = vec![
        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 1.0,
    ];
    (
        Tensor::from_vec(idx, (3, 3), &Device::Cpu).unwrap(),
        Tensor::from_vec(vis, (3, 3), &Device::Cpu).unwrap(),
    )
}

#[test]
fn targets_are_the_complement_of_the_visible_context_including_zeros() {
    let (idx, vis) = small_context();
    let mask = to_vec2(&target_mask_nd(&idx, &vis, 8).unwrap());
    #[rustfmt::skip]
    let expected: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    ];
    assert_eq!(mask, expected);
}

#[test]
fn level_target_library_is_the_full_row_total() {
    let mut rows = Mat::zeros(3, 8);
    for i in 0..3 {
        for j in 0..8 {
            rows[(i, j)] = if (i + j) % 3 == 0 {
                0.0
            } else {
                (i * 8 + j) as f32
            };
        }
    }
    let lt = LevelTarget::from_mat(&rows, &Device::Cpu).unwrap();
    let lib: Vec<f32> = lt.row_lib().flatten_all().unwrap().to_vec1().unwrap();
    for i in 0..3 {
        let total: f32 = (0..8).map(|j| rows[(i, j)]).sum::<f32>() + 1.0;
        assert!(
            (lib[i] - total).abs() < 1e-5,
            "row {i}: {} vs {total}",
            lib[i]
        );
    }
    let (vals, l) = lt
        .rows(&Tensor::new(&[2u32, 0, 2], &Device::Cpu).unwrap())
        .unwrap();
    assert_eq!(vals.dims(), &[3, 8]);
    assert_eq!(
        to_vec2(&vals)[0],
        (0..8).map(|j| rows[(2, j)]).collect::<Vec<_>>()
    );
    let l: Vec<f32> = l.flatten_all().unwrap().to_vec1().unwrap();
    assert!((l[1] - lib[0]).abs() < 1e-6);
}

/// Perturbing a count the encoder saw must not move the loss; perturbing a
/// scored gene must.
#[test]
fn visible_genes_are_never_scored() {
    const D: usize = 8;
    const K: usize = 2;
    let dev = Device::Cpu;
    let rho = Tensor::from_vec(
        (0..D * 3)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
            .collect(),
        (D, 3),
        &dev,
    )
    .unwrap();
    let mut ts = HashMap::new();
    ts.insert(
        "dec.topic.embeddings".to_string(),
        Tensor::from_vec(vec![0.5f32, -0.2, 0.1, -0.4, 0.3, 0.2], (K, 3), &dev).unwrap(),
    );
    ts.insert(
        "dec.log_phi".to_string(),
        Tensor::zeros((1, D), DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "dec.log_pi".to_string(),
        Tensor::full(-(D as f32).ln(), (1, D), &dev).unwrap(),
    );
    let vb = VarBuilder::from_tensors(ts, DType::F32, &dev);
    let dec = EmbeddedNbTopicDecoder::new(K, rho, vb.pp("dec")).unwrap();
    let full_kd = dec.full_logits_kd().unwrap();
    let log_theta = candle_nn::ops::log_softmax(
        &Tensor::from_vec(vec![0.2f32, -0.1, 0.4, 0.3, -0.5, 0.6], (3, K), &dev).unwrap(),
        1,
    )
    .unwrap();

    let (idx, vis) = small_context();
    let mask = target_mask_nd(&idx, &vis, D).unwrap();
    let mut rows = Mat::zeros(3, D);
    for i in 0..3 {
        for j in 0..D {
            rows[(i, j)] = ((i * 3 + j * 5) % 6) as f32;
        }
    }
    let score = |rows: &Mat| -> Vec<f32> {
        let lt = LevelTarget::from_mat(rows, &dev).unwrap();
        let (values, lib) = lt.rows(&Tensor::new(&[0u32, 1, 2], &dev).unwrap()).unwrap();
        let target = MaskedDenseTarget {
            values: &values,
            residual: None,
            lib: &lib,
            mask: &mask,
        };
        dec.impute_dense_nb(&log_theta, &target, &full_kd)
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let base = score(&rows);

    // Row 0 sees gene 1: change its count and only the library moves, which the
    // mask keeps out of the scored positions' likelihood up to the ℓ scale.
    let mut visible_perturbed = rows.clone();
    visible_perturbed[(0, 1)] += 3.0;
    let lt = LevelTarget::from_mat(&visible_perturbed, &dev).unwrap();
    let (values, _) = lt.rows(&Tensor::new(&[0u32, 1, 2], &dev).unwrap()).unwrap();
    let lib = LevelTarget::from_mat(&rows, &dev)
        .unwrap()
        .rows(&Tensor::new(&[0u32, 1, 2], &dev).unwrap())
        .unwrap()
        .1;
    let target = MaskedDenseTarget {
        values: &values,
        residual: None,
        lib: &lib,
        mask: &mask,
    };
    let same_lib: Vec<f32> = dec
        .impute_dense_nb(&log_theta, &target, &full_kd)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        (same_lib[0] - base[0]).abs() < 1e-6,
        "a visible gene's count reached the loss: {} vs {}",
        same_lib[0],
        base[0]
    );

    // Row 0 does not see gene 3: its count is scored.
    let mut target_perturbed = rows.clone();
    target_perturbed[(0, 3)] += 3.0;
    let moved = score(&target_perturbed);
    assert!(
        (moved[0] - base[0]).abs() > 1e-3,
        "a scored gene's count did not reach the loss"
    );
    assert!((moved[1] - base[1]).abs() < 1e-6, "row 1 must be untouched");
}
