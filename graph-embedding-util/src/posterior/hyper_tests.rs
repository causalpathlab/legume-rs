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
// Truncated IBP stick-breaking //
///////////////////////////////////

/// `π₀` is monotonically INCREASING in the dim index, whatever the sticks are —
/// equivalently, the inclusion rates decrease. This is the structural property the
/// independent per-dim `Beta(a,b)` cannot express, and the reason for the change: it
/// cannot be outvoted by data, so surplus dims are squeezed off by construction.
#[test]
fn stick_breaking_pi0_is_monotone_whatever_the_sticks() {
    let mut rng = StdRng::seed_from_u64(7);
    for alpha in [0.5f64, 1.0, 3.0, 10.0] {
        let mut sb = StickBreaking::new(alpha, 16);
        // Drive it with arbitrary counts so the sticks are not at their init.
        let m: Vec<usize> = (0..16).map(|d| 500 - 25 * d).collect();
        let n = vec![1000usize; 16];
        for _ in 0..20 {
            sb.sample(&m, &n, &mut rng);
        }
        let pi0 = sb.pi0();
        for w in pi0.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "exclusion must not DECREASE with dim (α={alpha}): {pi0:?}"
            );
        }
    }
}

/// With no data the sticks sample their `Beta(α, 1)` prior, whose mean is `α/(α+1)`.
/// This is the check that the slice step targets the right density — everything else
/// rides on it, and a slice sampler that silently returns its starting point would
/// pass every monotonicity test above.
#[test]
fn sticks_recover_their_beta_prior_with_no_data() {
    let mut rng = StdRng::seed_from_u64(11);
    for alpha in [1.0f64, 4.0] {
        let h = 1usize;
        let mut sb = StickBreaking::new(alpha, h);
        // Zero eligible coordinates ⇒ the likelihood is flat ⇒ the prior is the target.
        let (m, n) = (vec![0usize; h], vec![0usize; h]);
        let mut acc = 0.0f64;
        let draws = 4000;
        for _ in 0..draws {
            sb.sample(&m, &n, &mut rng);
            acc += 1.0 - sb.pi0()[0]; // inclusion = v_0 at h = 1
        }
        let got = acc / draws as f64;
        let want = alpha / (alpha + 1.0);
        assert!(
            (got - want).abs() < 0.03,
            "α={alpha}: stick mean {got:.4} != Beta(α,1) mean {want:.4}"
        );
    }
}

/// The data moves the sticks: a dim with many inclusions holds a high rate, and the
/// tail collapses. This is the behaviour the flat per-dim `Beta` failed to produce on
/// BM1 (0.787–0.930 across 16 dims while only ~3.4 were supported).
#[test]
fn a_supported_head_and_an_empty_tail_separate() {
    let mut rng = StdRng::seed_from_u64(3);
    let h = 12usize;
    let mut sb = StickBreaking::new(3.0, h);
    // First 3 dims: 80% of genes included. Remaining 9: nothing.
    let n = vec![2000usize; h];
    let m: Vec<usize> = (0..h).map(|d| if d < 3 { 1600 } else { 0 }).collect();
    for _ in 0..200 {
        sb.sample(&m, &n, &mut rng);
    }
    let incl: Vec<f64> = sb.pi0().iter().map(|p| 1.0 - p).collect();
    assert!(
        incl[0] > 0.5,
        "a dim with 80% inclusion must keep a high rate, got {:.3}",
        incl[0]
    );
    assert!(
        incl[h - 1] < 0.05,
        "an empty tail dim must collapse, got {:.3}",
        incl[h - 1]
    );
    assert!(
        incl[2] > incl[h - 1] * 5.0,
        "head and tail must separate: {:.3} vs {:.3}",
        incl[2],
        incl[h - 1]
    );
}
