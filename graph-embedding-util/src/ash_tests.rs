use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RNormal};

#[test]
fn lfdr_separates_shifted_signal_from_null() {
    // This location mixture pins the null at the mean-0 atom and gives the
    // non-null components free means, so it calls null vs signal by *location*:
    // a feature the model never moved sits at 0, one it moved sits at a shifted
    // ± loading (a marker loads + or − on a given axis). The lfdr = P(z = null)
    // then separates them. Null N(0,1); the 2000-signal bulk is split ±4 so both
    // free-mean components must be discovered. (A zero-centred *scale* mixture is
    // the case this model deliberately does not target: with no location shift
    // there is nothing for the free means to grab, and lfdr cannot tell a wide
    // mean-0 signal from the null — that is what lfsr, not lfdr, is for.)
    let mut rng = StdRng::seed_from_u64(5);
    let se = 1.0_f64;
    let n_null = 8000;
    let n_sig = 2000;
    let noise = RNormal::new(0.0, se).unwrap();
    let pos = RNormal::new(4.0, se).unwrap();
    let neg = RNormal::new(-4.0, se).unwrap();

    let mut betahat = Vec::with_capacity(n_null + n_sig);
    for _ in 0..n_null {
        betahat.push(noise.sample(&mut rng));
    }
    for k in 0..n_sig {
        // Split the signal ± so the mixture must find both shifted components.
        betahat.push(if k % 2 == 0 {
            pos.sample(&mut rng)
        } else {
            neg.sample(&mut rng)
        });
    }

    // Seed s deliberately off (1.5× the truth): the empirical-null Gibbs must
    // recover the true null SD ≈ 1 despite the 2000 shifted-signal observations.
    let res = ash_normal(&betahat, 1.5, &AshOpts::default());
    let est_sd = res.null_var.sqrt();
    assert!(
        est_sd > 0.75 && est_sd < 1.4,
        "null SD not recovered: {est_sd}"
    );

    // The per-axis lfdr is a RANKING quantity — its absolute scale is deliberately
    // uncalibrated: with a strong signal bulk present the mean-0 null atom cedes
    // weight, so raw per-axis lfdr runs low even for nulls. Calibration is supplied
    // downstream by the `h·min_d` Bonferroni in `ash_null_call`, whose null-leak
    // control is covered in tests/null_call.rs. What this unit test pins is the
    // ranking + power: the shifted signal ranks well below the null, and every
    // clearly-shifted signal is confidently non-null.
    let null_lfdr: f64 = res.lfdr[..n_null].iter().sum::<f64>() / n_null as f64;
    let sig_lfdr: f64 = res.lfdr[n_null..].iter().sum::<f64>() / n_sig as f64;
    assert!(
        sig_lfdr < 0.5 * null_lfdr,
        "shifted signal not ranked below null: sig={sig_lfdr}, null={null_lfdr}"
    );

    // Clearly-shifted signals (|betahat| > 3) are confidently non-null (low lfdr).
    let strong: Vec<usize> = (n_null..n_null + n_sig)
        .filter(|&i| betahat[i].abs() > 3.0)
        .collect();
    let strong_live = strong.iter().filter(|&&i| res.lfdr[i] < 0.05).count();
    assert!(
        !strong.is_empty() && strong_live as f64 / strong.len() as f64 > 0.9,
        "strong signals not confidently live: {strong_live}/{}",
        strong.len()
    );
}

#[test]
fn all_null_calls_almost_nothing() {
    let mut rng = StdRng::seed_from_u64(9);
    let noise = RNormal::new(0.0, 1.0).unwrap();
    let betahat: Vec<f64> = (0..5000).map(|_| noise.sample(&mut rng)).collect();
    let res = ash_normal(&betahat, 1.5, &AshOpts::default());
    // The empirical-null Gibbs recovers the true null SD ≈ 1 from a 1.5× seed.
    let est_sd = res.null_var.sqrt();
    assert!(
        est_sd > 0.8 && est_sd < 1.3,
        "null SD not recovered on all-null: {est_sd}"
    );
    // No signal ⇒ almost nothing crosses lfdr ≤ 0.05.
    let called = res.lfdr.iter().filter(|&&l| l <= 0.05).count();
    assert!(
        called < betahat.len() / 20,
        "too many false non-null: {called}"
    );
}
