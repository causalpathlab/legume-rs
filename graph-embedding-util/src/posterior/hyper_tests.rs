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
