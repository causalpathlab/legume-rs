//! Planted-recovery for the conjugate hyperparameter draws.

use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

/// The half-Cauchy variance draw recovers a planted `σ²` from planted Gaussian
/// effects (its posterior mean sits near the truth for a well-populated group).
#[test]
fn half_cauchy_var_recovers_planted_variance() {
    let mut rng = StdRng::seed_from_u64(1);
    let sigma2_true = 0.36f64; // σ = 0.6
    let n = 400usize;
    // Plant effects x_i ~ N(0, σ²) and reduce to Σx².
    let sum_sq: f64 = (0..n)
        .map(|_| {
            let z: f64 = StandardNormal.sample(&mut rng);
            let x = z * sigma2_true.sqrt();
            x * x
        })
        .sum();

    // Gibbs over σ² (data fixed); collect after warmup.
    let mut hv = HalfCauchyVar::new(1.0);
    let (warmup, keep) = (200usize, 2000usize);
    let mut acc = 0.0f64;
    for i in 0..(warmup + keep) {
        let s2 = hv.sample(sum_sq, n, &mut rng);
        if i >= warmup {
            acc += s2;
        }
    }
    let post_mean = acc / keep as f64;
    let rel = (post_mean - sigma2_true).abs() / sigma2_true;
    assert!(
        rel < 0.2,
        "posterior σ² should recover the plant: got {post_mean:.4}, true {sigma2_true:.4}"
    );
}

/// A near-empty group does NOT collapse to zero — the half-Cauchy keeps the draw
/// finite and bounded away from 0 (the anti-`IG(ε,ε)` property).
#[test]
fn half_cauchy_var_no_collapse_on_tiny_group() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut hv = HalfCauchyVar::new(1.0);
    // One tiny effect, essentially no data.
    let (sum_sq, n) = (1e-6f64, 1usize);
    for _ in 0..500 {
        let s2 = hv.sample(sum_sq, n, &mut rng);
        assert!(
            s2.is_finite() && s2 > 0.0,
            "variance must stay finite/positive"
        );
        assert!(s2 < 1e6, "variance must stay bounded");
    }
}

/// Beta-Binomial sparsity recovers the planted null fraction under a weak prior.
#[test]
fn pi0_recovers_planted_sparsity() {
    let mut rng = StdRng::seed_from_u64(3);
    let (n_total, n_null) = (500usize, 350usize); // true π₀ = 0.70
                                                  // Weak, near-uniform prior so the data dominates.
    let (a, b) = (1.0, 1.0);
    let mut acc = 0.0f64;
    let draws = 2000;
    for _ in 0..draws {
        acc += sample_pi0(n_null, n_total, a, b, &mut rng);
    }
    let post_mean = acc / draws as f64;
    let truth = n_null as f64 / n_total as f64;
    assert!(
        (post_mean - truth).abs() < 0.03,
        "π₀ should recover the plant: got {post_mean:.3}, true {truth:.3}"
    );
}

///////////////////////////////////
// Truncated IBP ladder      //
///////////////////////////////

/// `π₀` INCREASES with the dim index for any α — equivalently the inclusion rates
/// decay. This is the structural property the independent per-dim `Beta(a,b)` cannot
/// express, and the whole reason for the prior: with ~34k features on a dim, an O(1)
/// Beta is swamped and every unused dim re-estimates the same rate, whereas a
/// monotone ladder cannot be outvoted.
#[test]
fn the_ibp_ladder_is_monotone_for_any_alpha() {
    for alpha in [0.1f64, 0.5, 1.0, 3.0, 10.0, 100.0] {
        let pi0 = ibp_pi0(alpha, 32);
        assert_eq!(pi0.len(), 32);
        for w in pi0.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "exclusion must not DECREASE with dim (α={alpha}): {pi0:?}"
            );
        }
        assert!(pi0.iter().all(|p| *p > 0.0 && *p < 1.0), "α={alpha}");
    }
}

/// α IS the expected number of dims a feature loads, and — the property that makes
/// `--embedding-dim` a truncation rather than a knob — it does not scale with `H`.
///
/// `Σ_h (α/(α+1))^{h+1}` is geometric with ratio `α/(α+1)`, so it converges to `α`.
/// Doubling `H` therefore cannot double how many dims the prior expects a feature to
/// use; it only extends a tail that is already negligible. Measured on BM1, that is
/// what separated this prior from the Beta: 16 -> 32 dims moved the active count
/// 10 -> 12 here, against 16 -> 32 for the unordered alternative.
#[test]
fn expected_dims_per_feature_is_alpha_and_does_not_scale_with_h() {
    for alpha in [0.5f64, 1.0, 2.0, 5.0] {
        let dims = |h: usize| -> f64 { ibp_pi0(alpha, h).iter().map(|p| 1.0 - p).sum() };
        let (d16, d32, d256) = (dims(16), dims(32), dims(256));
        // Up to the boundary clamp. `ibp_pi0` floors every rate off 0/1 so
        // `log_prior_odds` stays finite, which leaves each dim a residual `PI0_EPS` of
        // inclusion — so a long tail overshoots α by at most `H · PI0_EPS`. At H = 256
        // that is 0.026; at the H values anyone runs it is ~1e-3.
        let floor = 256.0 * PI0_EPS;
        assert!(
            d256 <= alpha + floor,
            "α={alpha}: the geometric sum must not exceed α beyond the clamp floor, \
             got {d256}"
        );
        assert!(
            (d256 - alpha).abs() < floor,
            "α={alpha}: at large H the sum must converge to α, got {d256}"
        );
        // The load-bearing one: doubling the truncation barely moves it — PROVIDED H is
        // large relative to α. The series has ratio α/(α+1), so a big α converges
        // slowly: at α = 5 the ratio is 0.833 and H = 16 still truncates ~5% of the
        // mass, which is why the bound below scales with α rather than being flat.
        let converged = alpha / (alpha + 1.0);
        let truncated_at_16 = alpha * converged.powi(16);
        assert!(
            (d32 - d16).abs() <= truncated_at_16 + 256.0 * PI0_EPS,
            "α={alpha}: H 16 -> 32 moved expected dims {d16} -> {d32}, more than the \
             {truncated_at_16} the truncation itself accounts for"
        );
    }
}

/// The H-invariance above is not unconditional: it holds once `H` is large relative to
/// `α`, and the docs should not promise more than that.
///
/// At the shipped default `α = 1` the ratio is 0.5, so 16 dims already carry
/// `1 − 2⁻¹⁶` of the mass and doubling H is invisible. At `α = 5` the ratio is 0.833
/// and 16 dims carry only ~95%, so H is still doing real work. Pinned so a future
/// default change has to confront it.
#[test]
fn h_invariance_needs_h_large_relative_to_alpha() {
    let dims = |alpha: f64, h: usize| -> f64 { ibp_pi0(alpha, h).iter().map(|p| 1.0 - p).sum() };

    // Default α: the geometry is fully converged by 16 dims (ratio 0.5, so the tail is
    // 2⁻¹⁶ ≈ 1.5e-5). What little the sum moves is the boundary CLAMP, not truncation —
    // 16 extra dims each floored at `PI0_EPS` inclusion — so the bound is that floor,
    // and the assertion is that the geometry contributes essentially nothing on top.
    let (lo, hi) = (dims(1.0, 16), dims(1.0, 32));
    let clamp_floor = 16.0 * PI0_EPS;
    assert!(
        (hi - lo) <= clamp_floor + 1e-4,
        "α=1: H 16 -> 32 should move only by the clamp floor {clamp_floor}, got \
         {lo} -> {hi}"
    );

    // Large α relative to H: the truncation still bites, by design.
    let (lo5, hi5) = (dims(5.0, 16), dims(5.0, 32));
    assert!(
        hi5 - lo5 > 0.1,
        "α=5 at H=16 must still be truncating — if this stops holding the ladder \
         changed shape: {lo5} -> {hi5}"
    );
}
