//! Synthetic check for the empirical-Bayes null-gene call: planted null genes
//! (β at the init scale) vs signal genes (β with a much larger norm) must be
//! separated, the null scale σ̂² recovered, and the false-keep rate of null
//! genes held near the target FDR.

use graph_embedding_util::null_call::{
    chi2_null_call, embedding_lower_tail_call, embedding_null_call,
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
