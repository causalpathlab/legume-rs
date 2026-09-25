//! Individual-level propensity P(X | V) and the conditional permutation
//! built on it.
//!
//! The exposure is assigned to individuals, so the propensity lives at the
//! individual level and is shared by every gene, topic, and cell of that
//! individual. A uniform shuffle of exposure labels is the special case of a
//! constant propensity; drawing relabelings from the fitted propensity keeps
//! the permutation null centred when V drives exposure (Berrett, Wang,
//! Barber & Samworth 2020, "The conditional permutation test for independence
//! while controlling for confounders", JRSS-B).

use crate::common::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

#[cfg(test)]
mod tests;

/// Ridge on the slopes of the multinomial logistic fit (covariates are
/// standardized first, so this is a unit-scale N(0, 1/RIDGE) prior).
const RIDGE: f64 = 1.0;
/// A light ridge on the intercepts only keeps them finite under separation.
const RIDGE_INTERCEPT: f64 = 1e-4;
const MAX_NEWTON: usize = 100;
const NEWTON_TOL: f64 = 1e-8;
/// Pairwise-swap sweeps from the observed labels to the hub, and from the
/// hub to each draw.
pub const SWAP_SWEEPS: usize = 50;

/// Fitted P(X = x | V_i) for every individual, over the assigned groups.
pub struct Propensity {
    /// individual x group, rows sum to one after clipping (for the
    /// conditional permutation sampler)
    pub prob: Mat,
    /// individual x group, unclipped fitted probabilities (for the effect
    /// estimator, where no-overlap individuals drop out on their own)
    pub prob_raw: Mat,
    /// per-group effective sample size `(sum w)^2 / sum w^2` of the
    /// inverse-propensity weights `w_i = 1 / pi(x_i | V_i)`
    pub ess: Vec<f32>,
}

/// Multinomial logistic regression of `x` (in `0..n_groups`; values
/// `>= n_groups` are unassigned and left out) on standardized `v`, fit by
/// ridge-penalized Newton with step halving, then clipped to
/// `[clip, 1 - clip]` with `clip = max(0.01, 1 / n_assigned)`.
pub fn fit_propensity(v: &Mat, x: &[usize], n_groups: usize) -> anyhow::Result<Propensity> {
    let n = v.nrows();
    anyhow::ensure!(
        x.len() == n,
        "exposure has {} entries, V has {} rows",
        x.len(),
        n
    );
    anyhow::ensure!(n_groups >= 2, "propensity needs at least two groups");
    let assigned: Vec<bool> = x.iter().map(|&g| g < n_groups).collect();
    let n_assigned = assigned.iter().filter(|&&a| a).count();
    anyhow::ensure!(
        n_assigned >= 2,
        "propensity needs at least two labelled individuals"
    );

    // design: intercept + covariates
    let q = v.ncols() + 1;
    let design = |i: usize| -> DVector<f64> {
        DVector::from_fn(q, |c, _| {
            if c == 0 {
                1.0
            } else {
                f64::from(v[(i, c - 1)])
            }
        })
    };
    let rows: Vec<usize> = (0..n).filter(|&i| assigned[i]).collect();
    let feats: Vec<DVector<f64>> = (0..n).map(design).collect();

    // parameters: (n_groups - 1) x q, group 0 is the reference
    let k = n_groups - 1;
    let dim = k * q;
    let penalty = DVector::from_fn(dim, |j, _| if j % q == 0 { RIDGE_INTERCEPT } else { RIDGE });

    let probs_of = |theta: &DVector<f64>, f: &DVector<f64>| -> Vec<f64> {
        let mut eta = vec![0f64; n_groups];
        for g in 1..n_groups {
            eta[g] = (0..q).map(|c| theta[(g - 1) * q + c] * f[c]).sum();
        }
        let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = eta.iter().map(|e| (e - m).exp()).sum();
        eta.iter().map(|e| (e - m).exp() / z).collect()
    };
    let objective = |theta: &DVector<f64>| -> f64 {
        let ll: f64 = rows
            .iter()
            .map(|&i| probs_of(theta, &feats[i])[x[i]].max(1e-300).ln())
            .sum();
        ll - 0.5
            * theta
                .iter()
                .zip(penalty.iter())
                .map(|(t, p)| p * t * t)
                .sum::<f64>()
    };

    let mut theta = DVector::<f64>::zeros(dim);
    let mut obj = objective(&theta);
    for _ in 0..MAX_NEWTON {
        let mut grad = DVector::<f64>::zeros(dim);
        let mut hess = DMatrix::<f64>::zeros(dim, dim);
        for &i in &rows {
            let f = &feats[i];
            let p = probs_of(&theta, f);
            for a in 1..n_groups {
                let r = (x[i] == a) as u8 as f64 - p[a];
                for c in 0..q {
                    grad[(a - 1) * q + c] += r * f[c];
                }
                for b in 1..n_groups {
                    let w = p[a] * ((a == b) as u8 as f64 - p[b]);
                    for c in 0..q {
                        for e in 0..q {
                            hess[((a - 1) * q + c, (b - 1) * q + e)] += w * f[c] * f[e];
                        }
                    }
                }
            }
        }
        for j in 0..dim {
            grad[j] -= penalty[j] * theta[j];
            hess[(j, j)] += penalty[j];
        }
        let step = match hess.clone().cholesky() {
            Some(ch) => ch.solve(&grad),
            None => break,
        };
        let mut t = 1.0;
        let mut improved = false;
        for _ in 0..30 {
            let cand = &theta + &step * t;
            let cand_obj = objective(&cand);
            if cand_obj >= obj - 1e-12 {
                let gain = cand_obj - obj;
                theta = cand;
                obj = cand_obj;
                improved = gain > NEWTON_TOL * (1.0 + obj.abs());
                break;
            }
            t *= 0.5;
        }
        if !improved {
            break;
        }
    }

    let clip = (1.0 / n_assigned as f32)
        .max(0.01)
        .min(0.5 / n_groups as f32);
    info!(
        "propensity clipped to [{:.3}, {:.3}] for the permutation sampler",
        clip,
        1.0 - clip
    );
    let mut prob = Mat::zeros(n, n_groups);
    let mut prob_raw = Mat::zeros(n, n_groups);
    for i in 0..n {
        let p = probs_of(&theta, &feats[i]);
        for g in 0..n_groups {
            prob_raw[(i, g)] = p[g] as f32;
        }
        let clipped: Vec<f32> = p
            .iter()
            .map(|&v| (v as f32).clamp(clip, 1.0 - clip))
            .collect();
        let s: f32 = clipped.iter().sum();
        for g in 0..n_groups {
            prob[(i, g)] = clipped[g] / s;
        }
    }

    let ess = (0..n_groups)
        .map(|g| {
            let w: Vec<f32> = rows
                .iter()
                .filter(|&&i| x[i] == g)
                .map(|&i| 1.0 / prob[(i, g)])
                .collect();
            let s: f32 = w.iter().sum();
            let s2: f32 = w.iter().map(|v| v * v).sum();
            if s2 > 0.0 {
                s * s / s2
            } else {
                0.0
            }
        })
        .collect();

    Ok(Propensity {
        prob,
        prob_raw,
        ess,
    })
}

impl Propensity {
    /// Probability that individual `i` carries label `g`.
    fn p(&self, i: usize, g: usize) -> f64 {
        f64::from(self.prob[(i, g)])
    }

    /// One sweep of the pairwise-swap Gibbs kernel over the labelled
    /// individuals: pair them up at random, and swap each pair's labels with
    /// the conditional probability of the swapped configuration. It leaves
    /// `prod_i pi(label_i | V_i)`, restricted to permutations of the labels,
    /// invariant, and is reversible.
    fn sweep(&self, labels: &mut [usize], idx: &mut [usize], rng: &mut StdRng) {
        idx.shuffle(rng);
        for &[i, j] in idx.as_chunks::<2>().0 {
            let (a, b) = (labels[i], labels[j]);
            if a == b {
                continue;
            }
            let keep = self.p(i, a) * self.p(j, b);
            let swap = self.p(i, b) * self.p(j, a);
            if rng.random::<f64>() * (keep + swap) < swap {
                labels[i] = b;
                labels[j] = a;
            }
        }
    }

    /// `n_draws` relabelings of `x`, exchangeable with `x` under the null
    /// X independent of Y given V when the propensity is right. Star-shaped
    /// sampler: run the reversible kernel from the observed labels to a hub,
    /// then independently from the hub to each draw. Unassigned individuals
    /// keep their label; group sizes are preserved.
    pub fn conditional_permutations(
        &self,
        x: &[usize],
        n_draws: usize,
        sweeps: usize,
        seed: u64,
    ) -> Vec<Vec<usize>> {
        let mut rng = StdRng::seed_from_u64(seed);
        // unassigned individuals (no exposure label) keep their label
        let n_groups = self.prob.ncols();
        let mut idx: Vec<usize> = (0..x.len()).filter(|&i| x[i] < n_groups).collect();
        let mut hub = x.to_vec();
        for _ in 0..sweeps {
            self.sweep(&mut hub, &mut idx, &mut rng);
        }
        (0..n_draws)
            .map(|_| {
                let mut draw = hub.clone();
                for _ in 0..sweeps {
                    self.sweep(&mut draw, &mut idx, &mut rng);
                }
                draw
            })
            .collect()
    }
}
