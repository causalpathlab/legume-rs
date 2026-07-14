use super::*;

/// Deterministic LCG — the diagnostics must be testable without pulling an RNG dep in.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

#[test]
fn iid_chain_has_ess_near_n() {
    let mut rng = Lcg(42);
    let x: Vec<f32> = (0..4000).map(|_| rng.next_f32()).collect();
    let e = ess(&x);
    // An independent chain is worth ~all of its draws. The estimator is noisy, so this is a
    // band, not an equality — but it must not discount an uncorrelated chain by much.
    assert!(
        e > 0.6 * x.len() as f32,
        "iid chain should keep most of its draws, got ess={e} of {}",
        x.len()
    );
}

#[test]
fn autocorrelated_chain_has_ess_well_below_n() {
    // AR(1) with ρ = 0.9 ⇒ τ = (1+ρ)/(1−ρ) = 19 ⇒ ess ≈ n/19.
    let mut rng = Lcg(7);
    let rho = 0.9f32;
    let mut v = 0.0f32;
    let mut x = Vec::with_capacity(4000);
    for _ in 0..4000 {
        v = rho * v + rng.next_f32();
        x.push(v);
    }
    let e = ess(&x);
    let n = x.len() as f32;
    assert!(
        e < n / 5.0,
        "a strongly autocorrelated chain must be discounted, got ess={e} of {n}"
    );
    assert!(e >= 1.0, "ess stays positive, got {e}");
}

#[test]
fn constant_chain_does_not_divide_by_zero() {
    // The degenerate case the samplers actually produce: every draw on the same side, so the
    // sign-indicator chain is constant. Must return a usable divisor, not NaN/0.
    let e = ess(&vec![1.0f32; 500]);
    assert_eq!(e, 500.0);
    let short = ess(&[1.0, 2.0]);
    assert_eq!(short, 2.0, "too short to estimate ⇒ n");
}

#[test]
fn mcse_is_nonzero_at_p_zero() {
    // The whole point of the Jeffreys smoothing: an lfsr of 0 (no draw on the minority side)
    // must NOT report zero Monte-Carlo error — the plug-in √(p(1−p)/ess) would, and would
    // imply infinite confidence exactly at the top-ranked sites.
    let se = mcse_proportion(0.0, 1000.0);
    assert!(se > 0.0, "p=0 must still carry MC error, got {se}");
    assert!(
        se < 0.002,
        "…but it should be small at ess=1000, got {se}" // ~0.7/ess
    );

    // A borderline lfsr near a 0.1 threshold carries real error at ess=1000: ~0.0095, so the
    // called set genuinely can flip with the seed. This is the number the column exists to show.
    let se_border = mcse_proportion(0.1, 1000.0);
    assert!(
        se_border > 0.008 && se_border < 0.011,
        "lfsr=0.1 at ess=1000 ⇒ mcse≈0.0095, got {se_border}"
    );

    // Fewer effective draws ⇒ strictly more error.
    assert!(mcse_proportion(0.1, 100.0) > se_border);
}
