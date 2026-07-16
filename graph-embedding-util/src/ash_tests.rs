use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RNormal};

#[test]
fn lfsr_separates_null_from_signal() {
    let mut rng = StdRng::seed_from_u64(5);
    let se = 1.0_f64;
    let n_null = 8000;
    let n_sig = 2000;
    let noise = RNormal::new(0.0, se).unwrap();
    let sig = RNormal::new(0.0, 6.0).unwrap(); // signal ~ N(0, 36) + measurement noise

    let mut betahat = Vec::with_capacity(n_null + n_sig);
    for _ in 0..n_null {
        betahat.push(noise.sample(&mut rng));
    }
    for _ in 0..n_sig {
        betahat.push(sig.sample(&mut rng) + noise.sample(&mut rng));
    }

    // Seed s deliberately off (1.5× the truth): the empirical-null Gibbs must
    // recover the true null SD ≈ 1 despite the 2000 signal observations.
    let res = ash_normal(&betahat, 1.5, &AshOpts::default());
    let est_sd = res.null_var.sqrt();
    assert!(
        est_sd > 0.75 && est_sd < 1.4,
        "null SD not recovered: {est_sd}"
    );

    // Null observations keep an ambiguous sign ⇒ high lfsr (~0.5); signals lower.
    let null_lfsr: f64 = res.lfsr[..n_null].iter().sum::<f64>() / n_null as f64;
    let sig_lfsr: f64 = res.lfsr[n_null..].iter().sum::<f64>() / n_sig as f64;
    assert!(null_lfsr > 0.4, "null mean lfsr too low: {null_lfsr}");
    assert!(
        sig_lfsr < null_lfsr - 0.15,
        "signal lfsr not separated from null: sig={sig_lfsr}, null={null_lfsr}"
    );

    // Strong signals are confidently non-null (low lfsr).
    let strong: Vec<usize> = (n_null..n_null + n_sig)
        .filter(|&i| betahat[i].abs() > 10.0)
        .collect();
    let strong_live = strong.iter().filter(|&&i| res.lfsr[i] < 0.05).count();
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
    // No signal ⇒ almost nothing crosses lfsr ≤ 0.05.
    let called = res.lfsr.iter().filter(|&&l| l <= 0.05).count();
    assert!(
        called < betahat.len() / 20,
        "too many false non-null: {called}"
    );
}
