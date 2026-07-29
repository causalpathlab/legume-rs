//! The batched kernel must be `n` copies of the audited scalar one.
//!
//! That is the load-bearing claim, and it is checkable exactly rather than
//! statistically: per item the RNG is consumed in the same order (`u`, then `φ`, then
//! one angle per shrinkage round), so with matched per-item seeds the batch and
//! [`super::super::elliptical_slice_step`] must agree bit-for-bit. Everything else
//! here guards a property that is easy to break silently in an active-set loop.

use super::*;
use crate::engine::elliptical_slice_step;
use nalgebra::DVector;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// A separable log-likelihood: item `i` is Gaussian about `mu[i]` with sd `s`.
/// Separable so "independent transitions" is a meaningful thing to check.
fn make_ll(mu: Vec<f32>, s: f32) -> impl Fn(usize, f32) -> f32 {
    move |i, x| {
        let z = (x - mu[i]) / s;
        -0.5 * z * z
    }
}

/// THE test: batch ≡ `n` scalar transitions, exactly.
///
/// If this drifts, no downstream number from the column sampler can be trusted,
/// because the scalar kernel is the only audited one.
#[test]
fn the_batch_is_n_scalar_transitions_bit_for_bit() {
    const N: usize = 64;
    let seeds: Vec<u64> = (0..N as u64).map(|i| 0xC0FF_EE00 ^ (i * 7 + 1)).collect();
    let mu: Vec<f32> = (0..N).map(|i| (i % 9) as f32 * 0.3 - 1.2).collect();
    let ll = make_ll(mu.clone(), 0.8);

    let cur: Vec<f32> = (0..N).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
    let nu: Vec<f32> = (0..N).map(|i| (i % 7) as f32 * 0.25 - 0.75).collect();
    let cur_ll: Vec<f32> = (0..N).map(|i| ll(i, cur[i])).collect();

    // Batched.
    let mut rngs: Vec<SmallRng> = seeds.iter().map(|&s| SmallRng::seed_from_u64(s)).collect();
    let step = elliptical_slice_batch(&cur, &nu, &cur_ll, &mut rngs, &mut |x, active, out| {
        for (slot, &i) in active.iter().enumerate() {
            out[slot] = ll(i as usize, x[slot]);
        }
    });

    // Scalar, one item at a time, each on its own stream with the same seed.
    for i in 0..N {
        let mut rng = SmallRng::seed_from_u64(seeds[i]);
        let c = DVector::from_vec(vec![cur[i]]);
        let p = DVector::from_vec(vec![nu[i]]);
        let (v, l) = elliptical_slice_step(&c, &p, &|x: &DVector<f32>| ll(i, x[0]), cur_ll[i], &mut rng);
        assert_eq!(
            step.value[i], v[0],
            "item {i}: batch value {} vs scalar {}",
            step.value[i], v[0]
        );
        assert_eq!(step.lnpdf[i], l, "item {i}: batch lnpdf disagrees with scalar");
    }
}

/// Results must not depend on how items were grouped into batches. The column
/// sampler tiles its anchors for cache locality, and a tile size is a performance
/// knob — if it changed the answer, reproducibility would silently be a function of
/// cache geometry.
#[test]
fn grouping_into_batches_changes_nothing() {
    const N: usize = 40;
    let seeds: Vec<u64> = (0..N as u64).map(|i| 0xABCD_0000 ^ (i * 13 + 5)).collect();
    let mu: Vec<f32> = (0..N).map(|i| (i % 11) as f32 * 0.2 - 1.0).collect();
    let ll = make_ll(mu, 0.7);
    let cur: Vec<f32> = (0..N).map(|i| (i % 6) as f32 * 0.15 - 0.4).collect();
    let nu: Vec<f32> = (0..N).map(|i| (i % 8) as f32 * 0.2 - 0.7).collect();
    let cur_ll: Vec<f32> = (0..N).map(|i| ll(i, cur[i])).collect();

    let one = {
        let mut rngs: Vec<SmallRng> = seeds.iter().map(|&s| SmallRng::seed_from_u64(s)).collect();
        elliptical_slice_batch(&cur, &nu, &cur_ll, &mut rngs, &mut |x, active, out| {
            for (slot, &i) in active.iter().enumerate() {
                out[slot] = ll(i as usize, x[slot]);
            }
        })
    };

    // Same items, four groups of ten, each with its own call.
    let mut split_value = vec![0.0f32; N];
    let mut split_lnpdf = vec![0.0f32; N];
    for chunk in 0..4 {
        let lo = chunk * 10;
        let hi = lo + 10;
        let mut rngs: Vec<SmallRng> = seeds[lo..hi]
            .iter()
            .map(|&s| SmallRng::seed_from_u64(s))
            .collect();
        let part = elliptical_slice_batch(
            &cur[lo..hi],
            &nu[lo..hi],
            &cur_ll[lo..hi],
            &mut rngs,
            // `active` is TILE-local here, so the global item is `lo + active[slot]`.
            &mut |x, active, out| {
                for (slot, &i) in active.iter().enumerate() {
                    out[slot] = ll(lo + i as usize, x[slot]);
                }
            },
        );
        split_value[lo..hi].copy_from_slice(&part.value);
        split_lnpdf[lo..hi].copy_from_slice(&part.lnpdf);
    }

    assert_eq!(one.value, split_value, "values depend on batch grouping");
    assert_eq!(one.lnpdf, split_lnpdf, "lnpdf depends on batch grouping");
}

/// The kernel must target the right distribution, not merely move. With prior
/// `N(0,1)` and likelihood `N(μ, s²)` the posterior is Gaussian with precision
/// `1 + 1/s²` — so at `μ = 2`, `s = 1` it is `N(1, 1/2)`, which is checkable.
#[test]
fn it_targets_the_analytic_posterior() {
    const N: usize = 256;
    const SWEEPS: usize = 600;
    const BURN: usize = 100;
    let mu = vec![2.0f32; N];
    let ll = make_ll(mu, 1.0);

    let mut rngs: Vec<SmallRng> = (0..N as u64)
        .map(|i| SmallRng::seed_from_u64(0x5EED ^ (i * 31 + 3)))
        .collect();
    let mut draw_rng = SmallRng::seed_from_u64(99);

    let mut cur = vec![0.0f32; N];
    let mut cur_ll: Vec<f32> = (0..N).map(|i| ll(i, cur[i])).collect();
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut kept = 0usize;

    for sweep in 0..SWEEPS {
        // ν from the PRIOR, which is what makes this ESS rather than a random walk.
        let nu: Vec<f32> = (0..N)
            .map(|_| {
                let g: f64 = StandardNormal.sample(&mut draw_rng);
                g as f32
            })
            .collect();
        let step = elliptical_slice_batch(&cur, &nu, &cur_ll, &mut rngs, &mut |x, active, out| {
            for (slot, &i) in active.iter().enumerate() {
                out[slot] = ll(i as usize, x[slot]);
            }
        });
        cur = step.value;
        cur_ll = step.lnpdf;
        if sweep >= BURN {
            for &v in &cur {
                sum += f64::from(v);
                sumsq += f64::from(v) * f64::from(v);
            }
            kept += N;
        }
    }

    let mean = sum / kept as f64;
    let var = sumsq / kept as f64 - mean * mean;
    // Pooled over 256 chains × 500 sweeps; loose enough not to be flaky, tight enough
    // that a wrong target (e.g. sampling the prior, N(0,1)) fails.
    assert!(
        (mean - 1.0).abs() < 0.05,
        "posterior mean {mean:.4} should be 1.0 — sampling the prior would give 0.0"
    );
    assert!(
        (var - 0.5).abs() < 0.05,
        "posterior variance {var:.4} should be 0.5 — sampling the prior would give 1.0"
    );
}

/// A likelihood no proposal can satisfy must fall back to the current value for every
/// item, and SAY SO. A fallback is this kernel's analogue of a rejected move; an
/// active-set loop that silently dropped the count would report a stalled sampler as
/// a healthy one.
#[test]
fn an_unsatisfiable_slice_falls_back_and_is_counted() {
    const N: usize = 16;
    let cur = vec![0.3f32; N];
    let nu = vec![1.0f32; N];
    // Current sits at 0; every other point is far worse, so no proposal clears
    // `hh = ln(U) + 0 < 0`.
    let cur_ll = vec![0.0f32; N];
    let mut rngs: Vec<SmallRng> = (0..N as u64)
        .map(|i| SmallRng::seed_from_u64(7 ^ i))
        .collect();

    let step = elliptical_slice_batch(&cur, &nu, &cur_ll, &mut rngs, &mut |_x, _active, out| {
        out.fill(f32::NEG_INFINITY);
    });

    assert_eq!(step.fallbacks, N, "every item should have fallen back");
    assert_eq!(step.value, cur, "a fallback must leave the value untouched");
    assert_eq!(step.lnpdf, cur_ll, "a fallback must leave the lnpdf untouched");
    assert!(
        step.rounds <= MAX_BRACKET_ITERS,
        "rounds {} exceeded the cap",
        step.rounds
    );
}

/// The active set must actually decay, or the batching buys nothing. With an easy
/// likelihood most items accept on the first round, so the whole transition should
/// take very few `lnpdf` calls regardless of `n`.
#[test]
fn an_easy_slice_retires_the_active_set_fast() {
    const N: usize = 512;
    let ll = make_ll(vec![0.0f32; N], 4.0); // broad, so almost anything clears
    let cur = vec![0.1f32; N];
    let nu: Vec<f32> = (0..N).map(|i| (i % 3) as f32 * 0.1 - 0.1).collect();
    let cur_ll: Vec<f32> = (0..N).map(|i| ll(i, cur[i])).collect();
    let mut rngs: Vec<SmallRng> = (0..N as u64)
        .map(|i| SmallRng::seed_from_u64(0x1234 ^ (i * 3 + 1)))
        .collect();

    let step = elliptical_slice_batch(&cur, &nu, &cur_ll, &mut rngs, &mut |x, active, out| {
        for (slot, &i) in active.iter().enumerate() {
            out[slot] = ll(i as usize, x[slot]);
        }
    });
    assert_eq!(step.fallbacks, 0, "a broad likelihood should not stall");
    assert!(
        step.rounds < 20,
        "took {} rounds for {N} items — the active set is not decaying, so the \
         batch is doing n scalar walks with extra bookkeeping",
        step.rounds
    );
}
