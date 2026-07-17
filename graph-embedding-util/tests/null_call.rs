//! Synthetic check for the empirical-Bayes null-gene call: planted null genes
//! (β at the init scale) vs signal genes (β with a much larger norm) must be
//! separated, the null scale σ̂² recovered, and the false-keep rate of null
//! genes held near the target FDR.

use graph_embedding_util::null_call::{ash_null_call, chi2_null_call};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[test]
fn ash_separates_signal_with_nhvg_null() {
    // Clean regime: isotropic init-scale null vs signal features with one large
    // (feature-specific) coordinate loading. The n-hvg presumed-null (bottom
    // n − n_sig by norm) should recover the null scale and the ash lfsr call
    // should recover the signal with FDR control.
    let h = 16;
    let n_null = 4000;
    let n_sig = 1000;
    let n = n_null + n_sig;
    let mut rng = StdRng::seed_from_u64(31);
    let init = Normal::new(0.0, 0.05).unwrap(); // isotropic null (init scale)

    let mut beta = vec![0f32; n * h];
    for g in 0..n {
        for k in 0..h {
            beta[g * h + k] = init.sample(&mut rng) as f32;
        }
        if g >= n_null {
            let d = g % h; // feature-specific signal axis
            let sign = if g % 2 == 0 { 1.0 } else { -1.0 };
            beta[g * h + d] += 0.5 * sign; // ~10× the init scale
        }
    }

    let call = ash_null_call(&beta, n, h, 0.05, n_sig);
    let recall = (n_null..n).filter(|&g| call.live[g]).count() as f64 / n_sig as f64;
    let leak = (0..n_null).filter(|&g| call.live[g]).count() as f64 / n_null as f64;
    assert!(recall > 0.9, "ash recall too low: {recall}");
    assert!(leak < 0.1, "ash null leak too high: {leak}");
}

#[test]
fn core_on_raw_stats() {
    // chi2_null_call directly on σ²·χ²_dof statistics: null bulk at σ²·dof,
    // a planted signal tail well above it.
    let mut rng = StdRng::seed_from_u64(3);
    let (dof, sigma2) = (16usize, 0.01_f64);
    let chi = rand_distr::ChiSquared::new(dof as f64).unwrap();
    let mut s: Vec<f64> = (0..900).map(|_| sigma2 * chi.sample(&mut rng)).collect();
    s.extend((0..100).map(|_| 40.0 * sigma2 * chi.sample(&mut rng))); // signal
    let call = chi2_null_call(&s, dof, 0.05);
    assert!(
        call.n_live >= 90 && call.n_live <= 200,
        "n_live {}",
        call.n_live
    );
    assert!(
        call.sigma2 > 0.33 * sigma2 && call.sigma2 < 3.0 * sigma2,
        "sigma2 {}",
        call.sigma2
    );
    // Planted null is a genuine χ²_dof, so ν̂ should recover most of `dof`.
    assert!(
        call.eff_dof > 9.0,
        "eff_dof {} (expected ≈ {})",
        call.eff_dof,
        dof
    );
}
