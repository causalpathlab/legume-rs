//! Synthetic check for the empirical-Bayes null-gene call: planted null genes
//! (β at the init scale) vs signal genes (β with a much larger norm) must be
//! separated, the null scale σ̂² recovered, and the false-keep rate of null
//! genes held near the target FDR.

use graph_embedding_util::null_call::{
    ash_null_call, chi2_null_call, embedding_lower_tail_call, embedding_null_call, NullCall,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[test]
fn separates_null_from_signal() {
    let h = 16;
    let sigma = 0.05_f64; // null per-coordinate std (= the init scale)
    let n_null = 800;
    let n_sig = 200;
    let n = n_null + n_sig;

    let mut rng = StdRng::seed_from_u64(7);
    let null_d = Normal::new(0.0, sigma).unwrap();
    let sig_d = Normal::new(0.0, sigma * 6.0).unwrap(); // ~36× the null variance

    let mut beta = vec![0f32; n * h];
    for g in 0..n {
        let d = if g < n_null { &null_d } else { &sig_d };
        for k in 0..h {
            beta[g * h + k] = d.sample(&mut rng) as f32;
        }
    }

    let call = embedding_null_call(&beta, n, h, 0.05);

    // Signal genes are recovered (high power).
    let sig_live = (n_null..n).filter(|&g| call.live[g]).count();
    assert!(
        sig_live as f64 / n_sig as f64 > 0.9,
        "signal recall too low: {sig_live}/{n_sig}"
    );

    // Null genes kept (false discoveries) stay near the 5% FDR target.
    let null_live = (0..n_null).filter(|&g| call.live[g]).count();
    assert!(
        null_live as f64 / n_null as f64 <= 0.1,
        "null leak too high: {null_live}/{n_null}"
    );

    // Null scale recovered within ~3× (the joint (σ̂², ν̂) fit carries a little
    // more variance than a fixed-dof scale-only estimate).
    let s2 = sigma * sigma;
    assert!(
        call.sigma2 > 0.33 * s2 && call.sigma2 < 3.0 * s2,
        "sigma2 off: {} vs {}",
        call.sigma2,
        s2
    );

    // Coordinates are independent here, so the effective dof must recover most
    // of the nominal H = 16 (the over-dispersion knob should NOT fire).
    assert!(
        call.eff_dof > 9.0 && call.eff_dof <= h as f64 + 1e-6,
        "eff_dof off (independent coords ⇒ ν≈h): {} vs {}",
        call.eff_dof,
        h
    );

    // Null proportion near the planted 0.8 (conservative side allowed: a
    // slightly high σ̂² skews π̂₀ up, which only makes the call more cautious).
    assert!(call.pi0 > 0.6 && call.pi0 <= 1.0, "pi0 off: {}", call.pi0);
}

#[test]
fn empty_input() {
    let call = embedding_null_call(&[], 0, 16, 0.05);
    assert_eq!(call.n_live, 0);
    assert!(call.live.is_empty());
}

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
fn ash_recovers_nondominant_signal_the_norm_call_misses() {
    // Collapse regime: a dominant, legitimate axis (coord 0) every feature loads
    // on (independent of signal) + a real signal on a non-dominant axis (coord 1)
    // carried by a minority. The χ²-norm call calibrates to coord 0 and misses
    // the coord-1 signal; the ash call (clean per-coord null from the n-hvg
    // presumed-null) recovers it.
    let h = 16;
    let n_null = 850;
    let n_sig = 150;
    let n = n_null + n_sig;
    let mut rng = StdRng::seed_from_u64(23);
    let dom = Normal::new(0.0, 2.0).unwrap(); // dominant axis, all features
    let noise = Normal::new(0.0, 0.02).unwrap(); // init-scale null on every coord

    let mut beta = vec![0f32; n * h];
    for g in 0..n {
        for k in 0..h {
            beta[g * h + k] = noise.sample(&mut rng) as f32;
        }
        beta[g * h] += dom.sample(&mut rng) as f32; // dominant, independent of signal
        if g >= n_null {
            let sign = if g % 2 == 0 { 1.0 } else { -1.0 };
            beta[g * h + 1] += 0.30 * sign; // non-dominant signal
        }
    }

    let ash = ash_null_call(&beta, n, h, 0.05, n_sig);
    let chi2 = embedding_null_call(&beta, n, h, 0.05);
    let recall = |c: &NullCall| (n_null..n).filter(|&g| c.live[g]).count() as f64 / n_sig as f64;

    // ash recovers the non-dominant signal; the norm call can't isolate it.
    assert!(recall(&ash) > 0.9, "ash recall too low: {}", recall(&ash));
    assert!(
        recall(&ash) - recall(&chi2) > 0.4,
        "ash should beat χ²-norm on non-dominant signal: ash={}, chi2={}",
        recall(&ash),
        recall(&chi2)
    );
}

#[test]
fn lower_tail_drops_empties_keeps_real() {
    // Mirror the cell-QC regime: a dominant real-cell mode (norm ≈ 13, heavy
    // log spread) plus a minority "empty" lower tail (norm ≈ 0.3) whose MAP
    // projection collapsed. The lower-tail call must drop (most of) the empties
    // and keep (essentially all of) the real cells.
    let mut rng = StdRng::seed_from_u64(11);
    let n_real = 9000;
    let n_empty = 1000;
    let real_d = Normal::new(13.0_f64.ln(), 0.42).unwrap(); // wide real mode
    let empty_d = Normal::new(0.30_f64.ln(), 0.30).unwrap(); // collapsed empties
    let mut nrm = vec![0f32; n_real + n_empty];
    for (i, slot) in nrm.iter_mut().enumerate() {
        let d = if i < n_real { &real_d } else { &empty_d };
        *slot = d.sample(&mut rng).exp() as f32;
    }

    let call = embedding_lower_tail_call(&nrm, 0.05);

    // Empties recovered (dropped) with high power.
    let empty_dropped = (n_real..n_real + n_empty).filter(|&i| call.drop[i]).count();
    assert!(
        empty_dropped as f64 / n_empty as f64 > 0.9,
        "empty recall too low: {empty_dropped}/{n_empty}"
    );
    // Real cells almost never dropped (the costly error).
    let real_dropped = (0..n_real).filter(|&i| call.drop[i]).count();
    assert!(
        real_dropped as f64 / n_real as f64 <= 0.02,
        "real-cell false-drop too high: {real_dropped}/{n_real}"
    );
    // Null mode sits at the real cells, not the empties.
    assert!(
        call.mu > 13.0_f64.ln() - 0.5 && call.mu < 13.0_f64.ln() + 0.5,
        "null μ̂ off (should sit on the real mode): {}",
        call.mu
    );
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
