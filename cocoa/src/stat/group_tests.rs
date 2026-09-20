//! Tests for the group-level exposure model.
//!
//! Generative model (single topic):
//!
//! ```text
//!   y1(d,i,p) ~ Poisson( tau(d, x(i)) * delta(d,i) * mu(d,p) * n(i,p) )
//!   y0(d,p)   ~ Poisson( gamma(d,p) * mu(d,p) * n(p) )
//!   delta(d,i) ~ Gamma(phi, phi)     mean 1 within every exposure group
//! ```
//!
//! `tau` is the average exposure effect (gene x group), `delta` the
//! individual effect with exposure removed, `phi` the between-individual
//! dispersion. Unlike the noise-free fixtures in `tests.rs`, these are
//! sampled: the group model has to recover a planted average effect and the
//! between-individual spread from noisy per-individual totals. Inference on
//! the contrast is by exposure-label permutation in `run_diff`, not tested here.

use super::*;
use matrix_param::traits::Inference;
use matrix_util::hypothesis::mean;
use matrix_util::utils::median;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

const N_PB: usize = 6;
const N_PER_GROUP: usize = 8;
const PHI_TRUE: f32 = 8.0;

struct GroupSim {
    stat: CocoaStat,
    indv_to_group: Vec<usize>,
}

fn uniform(rng: &mut rand::rngs::StdRng, lo: f32, hi: f32) -> f32 {
    Uniform::new(lo, hi).unwrap().sample(rng)
}

/// Sample one topic's sufficient statistics; `beta[d]` is the planted log
/// fold of group 1 over group 0 for gene `d`.
fn simulate(n_genes: usize, beta: &[f32], seed: u64) -> GroupSim {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_indv = 2 * N_PER_GROUP;
    let indv_to_group: Vec<usize> = (0..n_indv).map(|i| i / N_PER_GROUP).collect();

    let mut size_ip = Mat::zeros(n_indv, N_PB);
    for i in 0..n_indv {
        for p in 0..N_PB {
            size_ip[(i, p)] = uniform(&mut rng, 20.0, 80.0);
        }
    }
    let mut size_p = DVec::zeros(N_PB);
    for p in 0..N_PB {
        size_p[p] = size_ip.column(p).sum();
    }

    let delta_dist = Gamma::new(PHI_TRUE, 1.0 / PHI_TRUE).unwrap();

    let mut y1_dp = Mat::zeros(n_genes, N_PB);
    let mut y0_dp = Mat::zeros(n_genes, N_PB);
    let mut y1_di = Mat::zeros(n_genes, n_indv);

    for d in 0..n_genes {
        let tau_x = [1.0f32, beta[d].exp()];
        let delta: Vec<f32> = (0..n_indv).map(|_| delta_dist.sample(&mut rng)).collect();
        for p in 0..N_PB {
            let mu = uniform(&mut rng, 3.0, 15.0);
            let gamma = uniform(&mut rng, 0.5, 2.0);
            for i in 0..n_indv {
                let lambda = tau_x[indv_to_group[i]] * delta[i] * mu * size_ip[(i, p)];
                let y = Poisson::new(lambda).unwrap().sample(&mut rng);
                y1_dp[(d, p)] += y;
                y1_di[(d, i)] += y;
            }
            y0_dp[(d, p)] = Poisson::new(gamma * mu * size_p[p])
                .unwrap()
                .sample(&mut rng);
        }
    }

    let mut stat = CocoaStat::new(
        CocoaStatArgs {
            n_genes,
            n_topics: 1,
            n_indv,
            n_samples: N_PB,
        },
        Some(60),
        Some((1e-2, 1e-2)),
    );
    stat.y1_stat_mut(0).copy_from(&y1_dp);
    stat.y0_stat_mut(0).copy_from(&y0_dp);
    stat.indv_y1_stat_mut(0).copy_from(&y1_di);
    stat.size_stat_mut(0).copy_from(&size_p);
    stat.indv_size_stat_mut(0).copy_from(&size_ip);

    GroupSim {
        stat,
        indv_to_group,
    }
}

#[test]
fn group_effect_recovers_planted_log_fold() {
    let n_genes = 200;
    let beta: Vec<f32> = (0..n_genes)
        .map(|d| if d % 2 == 0 { 0.7 } else { 0.0 })
        .collect();
    let sim = simulate(n_genes, &beta, 1);
    let params = sim
        .stat
        .estimate_group_parameters(&sim.indv_to_group, 2)
        .unwrap();
    let contrast = compute_group_contrast(&params, 1, 0);

    let planted: Vec<f32> = (0..n_genes)
        .filter(|d| d % 2 == 0)
        .map(|d| contrast[d])
        .collect();
    let null: Vec<f32> = (0..n_genes)
        .filter(|d| d % 2 == 1)
        .map(|d| contrast[d])
        .collect();
    let (m1, m0) = (mean(&planted), mean(&null));
    assert!(
        (m1 - 0.7).abs() < 0.1,
        "planted mean contrast {m1}, want 0.7"
    );
    assert!(m0.abs() < 0.1, "null mean contrast {m0}, want 0");
}

#[test]
fn individual_delta_averages_to_one_within_each_group() {
    let n_genes = 100;
    let beta = vec![0.7f32; n_genes];
    let sim = simulate(n_genes, &beta, 2);
    let params = sim
        .stat
        .estimate_group_parameters(&sim.indv_to_group, 2)
        .unwrap();
    let delta = params[0].indv_delta.posterior_mean();
    // Groups are contiguous blocks of N_PER_GROUP individuals by construction.
    for x in 0..2 {
        let m = delta.columns(x * N_PER_GROUP, N_PER_GROUP).mean();
        assert!((m - 1.0).abs() < 0.1, "group {x}: mean delta {m}, want 1");
    }
}

#[test]
fn dispersion_recovers_between_individual_spread() {
    let n_genes = 150;
    let beta = vec![0.0f32; n_genes];
    let sim = simulate(n_genes, &beta, 3);
    let params = sim
        .stat
        .estimate_group_parameters(&sim.indv_to_group, 2)
        .unwrap();
    let med = median(params[0].dispersion.as_slice());
    assert!(
        med > PHI_TRUE / 2.5 && med < PHI_TRUE * 2.5,
        "median dispersion {med}, simulated {PHI_TRUE}"
    );
}

#[test]
fn group_contrast_is_antisymmetric_in_group_labels() {
    let n_genes = 20;
    let beta = vec![0.5f32; n_genes];
    let sim = simulate(n_genes, &beta, 6);
    let params = sim
        .stat
        .estimate_group_parameters(&sim.indv_to_group, 2)
        .unwrap();
    let a = compute_group_contrast(&params, 1, 0);
    let b = compute_group_contrast(&params, 0, 1);
    for d in 0..n_genes {
        assert!((a[d] + b[d]).abs() < 1e-5);
    }
}

#[test]
fn gene_blocks_do_not_change_the_contrast() {
    let n_genes = 150;
    let beta: Vec<f32> = (0..n_genes)
        .map(|d| if d % 3 == 0 { 0.6 } else { 0.0 })
        .collect();
    let sim = simulate(n_genes, &beta, 7);
    let whole = sim
        .stat
        .estimate_group_parameters_blocked(&sim.indv_to_group, 2, n_genes)
        .unwrap();
    let blocked = sim
        .stat
        .estimate_group_parameters_blocked(&sim.indv_to_group, 2, 32)
        .unwrap();
    let a = compute_group_contrast(&whole, 1, 0);
    let b = compute_group_contrast(&blocked, 1, 0);
    let worst = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    // Blocks only change the dispersion shrinkage center; the contrast must
    // agree to well inside the between-individual noise.
    assert!(worst < 0.05, "block fit differs from whole fit by {worst}");
    assert_eq!(blocked[0].dispersion.len(), n_genes);
}
