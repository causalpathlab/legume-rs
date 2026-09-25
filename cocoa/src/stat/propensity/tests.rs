//! Propensity fit and the conditional permutation sampler.

use super::*;
use crate::stat::test_util::sigmoid;
use rand_distr::{Distribution, Normal};

/// V ~ N(0, 1), X ~ Bernoulli(sigmoid(slope V)).
fn planted(n: usize, slope: f32, seed: u64) -> (Mat, Vec<usize>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0f32, 1f32).unwrap();
    let v = Mat::from_fn(n, 1, |_, _| normal.sample(&mut rng));
    let p: Vec<f32> = (0..n).map(|i| sigmoid(slope * v[(i, 0)])).collect();
    let x = p
        .iter()
        .map(|&pi| (rng.random::<f32>() < pi) as usize)
        .collect();
    (v, x, p)
}

#[test]
fn propensity_recovers_planted_logistic() {
    let (v, x, p) = planted(600, 1.5, 7);
    let prop = fit_propensity(&v, &x, 2).unwrap();
    let mae: f32 = (0..p.len())
        .map(|i| (prop.prob[(i, 1)] - p[i]).abs())
        .sum::<f32>()
        / p.len() as f32;
    assert!(mae < 0.06, "mean abs error {mae}");
    for i in 0..p.len() {
        let s = prop.prob[(i, 0)] + prop.prob[(i, 1)];
        assert!((s - 1.0).abs() < 1e-5);
        // clipped at max(0.01, 1 / n) = 0.01 for 600 labelled individuals
        assert!(prop.prob[(i, 1)] >= 0.0099);
    }
}

#[test]
fn unassigned_individuals_are_left_out_and_fixed() {
    let (v, mut x, _) = planted(40, 1.0, 3);
    x[0] = 2;
    x[5] = 2;
    let prop = fit_propensity(&v, &x, 2).unwrap();
    for draw in prop.conditional_permutations(&x, 50, 10, 1) {
        assert_eq!(draw[0], 2);
        assert_eq!(draw[5], 2);
    }
}

#[test]
fn flat_propensity_gives_uniform_relabeling_with_fixed_arm_sizes() {
    let n = 8;
    let v = Mat::zeros(n, 1);
    let x: Vec<usize> = (0..n).map(|i| (i >= n / 2) as usize).collect();
    let prop = fit_propensity(&v, &x, 2).unwrap();
    let draws = prop.conditional_permutations(&x, 4000, SWAP_SWEEPS, 11);
    let mut freq = vec![0f32; n];
    for d in &draws {
        assert_eq!(d.iter().filter(|&&g| g == 1).count(), n / 2);
        for i in 0..n {
            freq[i] += d[i] as f32;
        }
    }
    for f in freq {
        let f = f / draws.len() as f32;
        assert!((f - 0.5).abs() < 0.04, "label-1 frequency {f}, want 0.5");
    }
}

#[test]
fn relabelings_follow_the_propensity() {
    // Individuals with high P(X = 1 | V) should keep label 1 more often.
    let (v, x, _) = planted(60, 2.0, 5);
    let prop = fit_propensity(&v, &x, 2).unwrap();
    let draws = prop.conditional_permutations(&x, 500, SWAP_SWEEPS, 2);
    let n = x.len();
    let mut freq = vec![0f32; n];
    for d in &draws {
        for i in 0..n {
            freq[i] += d[i] as f32 / draws.len() as f32;
        }
    }
    let p1: Vec<f32> = (0..n).map(|i| prop.prob[(i, 1)]).collect();
    let (mf, mp) = (
        freq.iter().sum::<f32>() / n as f32,
        p1.iter().sum::<f32>() / n as f32,
    );
    let cov: f32 = (0..n).map(|i| (freq[i] - mf) * (p1[i] - mp)).sum();
    let sf: f32 = freq.iter().map(|f| (f - mf).powi(2)).sum::<f32>().sqrt();
    let sp: f32 = p1.iter().map(|p| (p - mp).powi(2)).sum::<f32>().sqrt();
    let r = cov / (sf * sp);
    assert!(r > 0.8, "corr(label frequency, propensity) = {r}");
}
