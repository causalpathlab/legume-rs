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

/////////////////
// Split-R̂    //
/////////////////

/// A stationary chain's segments are interchangeable, so R̂ sits at ~1.
#[test]
fn a_stationary_chain_has_rhat_near_one() {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = SmallRng::seed_from_u64(11);
    let x: Vec<f32> = (0..800)
        .map(|_| {
            let g: f64 = StandardNormal.sample(&mut rng);
            g as f32
        })
        .collect();
    let r = split_rhat(&x);
    assert!(
        (r - 1.0).abs() < 0.05,
        "iid draws should give R̂ ≈ 1, got {r}"
    );
}

/// THE case this exists for: a chain still drifting. The segments then sample different
/// regions, between-segment variance dominates, and R̂ must rise well past the 1.01
/// convention — otherwise a run that never left its initialization would report as
/// converged, which under SGD-xor-sampling is the whole output.
#[test]
fn a_drifting_chain_is_caught() {
    let x: Vec<f32> = (0..800).map(|i| i as f32 * 0.01).collect();
    let r = split_rhat(&x);
    assert!(
        r > 1.1,
        "a linear drift must not pass as converged, got R̂ {r}"
    );
}

/// A chain that never moved is pinned, not divergent — its segments agree exactly. Guards
/// the zero-within-variance branch, which would otherwise divide by zero and report a
/// stuck chain as infinitely bad rather than as suspiciously perfect.
#[test]
fn a_constant_chain_reports_one_not_a_division_by_zero() {
    let x = vec![0.7f32; 400];
    let r = split_rhat(&x);
    assert_eq!(r, 1.0, "a pinned chain should read 1.0, got {r}");
}

/// Segments that are each internally constant but disagree with each other are the one
/// case where zero within-variance means genuine non-convergence, and it must be reported
/// as such rather than silently becoming 1.0.
#[test]
fn constant_but_disagreeing_segments_are_not_converged() {
    let mut x = vec![0.0f32; 200];
    x.extend(vec![5.0f32; 200]);
    assert!(
        split_rhat(&x) > 1.1,
        "two constant halves at different levels are not converged"
    );
}

/// Too short to split ⇒ no claim, rather than a number computed from two points.
#[test]
fn a_chain_too_short_to_split_makes_no_claim() {
    assert_eq!(split_rhat(&[1.0, 2.0, 3.0]), 1.0);
    assert_eq!(split_rhat(&[]), 1.0);
}
