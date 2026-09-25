//! Propensity-adjusted exposure effect on simulated individuals.
//!
//! ```text
//!   V_i ~ N(0, 1),  X_i ~ Bernoulli(sigmoid(a V_i))
//!   y_di ~ Poisson( m_di * tau_d(X_i) * delta_di ),
//!   delta_di = exp(g(V_i)) * Gamma(phi, phi)
//! ```
//!
//! Every gene shares one confounding direction (`g` increasing in V), so the
//! unadjusted contrast is biased the same way for all of them and the bias
//! shows in the mean over genes. `g` is linear in V unless `nonlinear > 0`;
//! only the propensity is modelled, so either works.

use super::*;
use crate::stat::test_util::sigmoid;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma, Normal, Poisson};

const PHI: f32 = 10.0;

struct Sim {
    y: Mat,
    m: Mat,
    x: Vec<usize>,
    /// true P(X = x | V), individual x 2
    e: Mat,
}

/// `g(V) = 0.8 V + nonlinear * (V^2 - 1)`; the planted log effect is `psi`
/// for every gene; `slope` sets how strongly V drives the exposure.
fn simulate(n: usize, n_genes: usize, psi: f32, nonlinear: f32, slope: f32, seed: u64) -> Sim {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0f32, 1f32).unwrap();
    let eps = Gamma::new(PHI, 1.0 / PHI).unwrap();
    let v: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let p1: Vec<f32> = v.iter().map(|&z| sigmoid(slope * z)).collect();
    let x: Vec<usize> = p1
        .iter()
        .map(|&p| (rng.random::<f32>() < p) as usize)
        .collect();
    let m = Mat::from_fn(n_genes, n, |_, _| 20.0 + 40.0 * rng.random::<f32>());
    let mut y = Mat::zeros(n_genes, n);
    for d in 0..n_genes {
        for i in 0..n {
            let g = 0.8 * v[i] + nonlinear * (v[i] * v[i] - 1.0);
            let rate = (psi * x[i] as f32 + g).exp() * eps.sample(&mut rng) * m[(d, i)];
            y[(d, i)] = Poisson::new(rate).unwrap().sample(&mut rng);
        }
    }
    Sim {
        y,
        m,
        x,
        e: Mat::from_fn(n, 2, |i, l| if l == 1 { p1[i] } else { 1.0 - p1[i] }),
    }
}

/// With the propensity `e`, or without any (`None`).
fn fit(sim: &Sim, e: Option<&Mat>) -> ExposureEffect {
    estimate_exposure_effect(&sim.y, &sim.m, &sim.x, 2, e, false)
}

fn mean_psi(sim: &Sim, e: Option<&Mat>) -> f32 {
    fit(sim, e).psi.column(1).mean()
}

#[test]
fn without_confounders_psi_is_the_log_ratio_of_means() {
    let sim = simulate(100, 20, 0.3, 0.0, 1.5, 11);
    let eff = fit(&sim, None);
    for d in 0..sim.y.nrows() {
        let (mut s1, mut n1, mut s0, mut n0) = (0f64, 0f64, 0f64, 0f64);
        for i in 0..sim.x.len() {
            let l = f64::from(sim.y[(d, i)] / sim.m[(d, i)]);
            if sim.x[i] == 1 {
                s1 += l;
                n1 += 1.0;
            } else {
                s0 += l;
                n0 += 1.0;
            }
        }
        let want = ((s1 / n1) / (s0 / n0)).ln() as f32;
        assert!(
            (eff.psi[(d, 1)] - want).abs() < 1e-3,
            "gene {d}: psi {} vs log ratio of means {want}",
            eff.psi[(d, 1)]
        );
    }
}

#[test]
fn unadjusted_contrast_is_confounded() {
    let sim = simulate(200, 50, 0.0, 0.0, 1.5, 1);
    let psi = mean_psi(&sim, None);
    assert!(
        psi > 0.5,
        "unadjusted mean psi {psi}, expected a large positive bias"
    );
}

#[test]
fn the_propensity_removes_the_bias_and_recovers_a_planted_effect() {
    for &planted in &[0.0f32, 0.5] {
        let sim = simulate(200, 50, planted, 0.0, 1.5, 2);
        let psi = mean_psi(&sim, Some(&sim.e));
        assert!(
            (psi - planted).abs() < 0.1,
            "mean psi {psi}, planted {planted}"
        );
    }
}

#[test]
fn holds_when_expression_is_nonlinear_in_v() {
    // nonlinear g: a linear V outcome model is wrong; the propensity is right
    let reps: Vec<f32> = (0..10)
        .map(|seed| {
            let sim = simulate(300, 20, 0.0, 0.3, 1.5, 100 + seed);
            mean_psi(&sim, Some(&sim.e))
        })
        .collect();
    let mean = reps.iter().sum::<f32>() / reps.len() as f32;
    assert!(mean.abs() < 0.1, "mean psi over replicates {mean}");
    let sim = simulate(300, 20, 0.0, 0.3, 1.5, 100);
    let unadjusted = mean_psi(&sim, None);
    assert!(unadjusted > 0.5, "unadjusted {unadjusted}");
}

#[test]
fn extreme_propensities_need_no_clipping() {
    // V nearly determines X: fitted propensities reach 0 and 1
    let sim = simulate(200, 30, 0.5, 0.0, 6.0, 4);
    let eff = fit(&sim, Some(&sim.e));
    assert!(eff.psi.iter().all(|p| p.is_finite()));
    let psi = eff.psi.column(1).mean();
    assert!(
        (psi - 0.5).abs() < 0.2,
        "mean psi {psi} under near separation"
    );
}

#[test]
fn zero_counts_are_fine() {
    let mut sim = simulate(100, 10, 0.3, 0.0, 1.5, 5);
    for i in 0..sim.x.len() / 3 {
        sim.y[(0, i)] = 0.0;
    }
    sim.y.row_mut(1).fill(0.0);
    sim.y[(1, 0)] = 3.0;
    let eff = fit(&sim, Some(&sim.e));
    assert!(eff.psi.iter().all(|p| p.is_finite()));
    assert!(eff.tau.iter().all(|t| t.is_finite() && *t >= 0.0));
}

#[test]
fn delta_is_exposure_free_and_averages_to_one() {
    let sim = simulate(200, 20, 0.5, 0.0, 1.5, 5);
    let eff = fit(&sim, Some(&sim.e));
    for d in 0..sim.y.nrows() {
        let mean = eff.delta.row(d).mean();
        assert!((mean - 1.0).abs() < 1e-3, "gene {d}: mean delta {mean}");
        assert!(eff.tau[(d, 0)] > 0.0 && eff.tau[(d, 1)] > 0.0);
        let ratio = (eff.tau[(d, 1)] / eff.tau[(d, 0)]).ln();
        assert!((ratio - eff.psi[(d, 1)]).abs() < 1e-4);
    }
}

#[test]
fn parallel_blocks_match_the_serial_solution() {
    // more genes than one block, so the split is exercised
    let sim = simulate(60, 600, 0.3, 0.0, 1.5, 6);
    let a = estimate_exposure_effect(&sim.y, &sim.m, &sim.x, 2, Some(&sim.e), false);
    let b = estimate_exposure_effect(&sim.y, &sim.m, &sim.x, 2, Some(&sim.e), true);
    assert_eq!(a.psi, b.psi);
    assert_eq!(a.delta, b.delta);
}

/// Three levels, V drives both the level (softmax) and expression; the
/// planted log effects of levels 1 and 2 over level 0 are recovered.
#[test]
fn three_levels_recover_planted_effects_under_confounding() {
    let (n, n_genes) = (600, 40);
    let psi_true = [0.0f32, 0.4, -0.3];
    let mut rng = rand::rngs::StdRng::seed_from_u64(8);
    let normal = Normal::new(0f32, 1f32).unwrap();
    let eps = Gamma::new(PHI, 1.0 / PHI).unwrap();
    let v: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let e = Mat::from_fn(n, 3, |i, l| {
        let logits = [0.0, 1.2 * v[i], -1.2 * v[i]];
        let z: f32 = logits.iter().map(|t| t.exp()).sum();
        logits[l].exp() / z
    });
    let x: Vec<usize> = (0..n)
        .map(|i| {
            let r = rng.random::<f32>();
            if r < e[(i, 0)] {
                0
            } else if r < e[(i, 0)] + e[(i, 1)] {
                1
            } else {
                2
            }
        })
        .collect();
    let m = Mat::from_fn(n_genes, n, |_, _| 20.0 + 40.0 * rng.random::<f32>());
    let mut y = Mat::zeros(n_genes, n);
    for d in 0..n_genes {
        for i in 0..n {
            let rate = (psi_true[x[i]] + 0.8 * v[i]).exp() * eps.sample(&mut rng) * m[(d, i)];
            y[(d, i)] = Poisson::new(rate).unwrap().sample(&mut rng);
        }
    }
    let eff = estimate_exposure_effect(&y, &m, &x, 3, Some(&e), false);
    let plain = estimate_exposure_effect(&y, &m, &x, 3, None, false);
    for (l, &planted) in psi_true.iter().enumerate().skip(1) {
        let got = eff.psi.column(l).mean();
        let raw = plain.psi.column(l).mean();
        assert!(
            (got - planted).abs() < 0.1,
            "level {l}: psi {got}, planted {planted}"
        );
        assert!(
            (raw - planted).abs() > 0.3,
            "level {l}: unadjusted {raw} should be confounded"
        );
    }
}

#[test]
fn converges_with_few_individuals_and_strong_confounding() {
    // like the pipeline simulations: 20 individuals, V strongly drives X
    for seed in 0..5 {
        let sim = simulate(20, 50, 0.5, 0.0, 3.0, 200 + seed);
        let eff = fit(&sim, Some(&sim.e));
        assert!(
            eff.iterations < MAX_ITER,
            "seed {seed}: scoring hit the iteration cap ({} steps)",
            eff.iterations
        );
    }
}

#[test]
fn recovers_the_effect_with_few_individuals() {
    // the planted effect is not pulled toward the starting value: averaged
    // over replicate datasets, psi is centred on the truth
    let reps: Vec<f32> = (0..20)
        .map(|seed| {
            let sim = simulate(20, 50, 0.5, 0.0, 3.0, 300 + seed);
            mean_psi(&sim, Some(&sim.e))
        })
        .collect();
    let mean = reps.iter().sum::<f32>() / reps.len() as f32;
    assert!(
        (mean - 0.5).abs() < 0.1,
        "mean psi over replicates {mean}, planted 0.5"
    );
}

#[test]
fn weak_confounding_with_a_nearly_linear_propensity_keeps_the_effect() {
    // V barely drives X, so e(V) is nearly constant and nearly collinear with
    // the intercept; the effect must not collapse toward the starting value
    let reps: Vec<f32> = (0..20)
        .map(|seed| {
            let sim = simulate(20, 50, 0.5, 0.0, 0.2, 400 + seed);
            mean_psi(&sim, Some(&sim.e))
        })
        .collect();
    let mean = reps.iter().sum::<f32>() / reps.len() as f32;
    assert!(
        (mean - 0.5).abs() < 0.1,
        "mean psi over replicates {mean}, planted 0.5"
    );
}
