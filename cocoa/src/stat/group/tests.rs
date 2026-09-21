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
use legume_numeric::param::traits::Inference;
use legume_numeric::matrix::hypothesis::mean;
use legume_numeric::matrix::utils::median;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma, Normal, Poisson, Uniform};

const N_PB: usize = 6;
const N_PER_GROUP: usize = 8;
const PHI_TRUE: f32 = 8.0;

struct GroupSim {
    stat: CocoaStat,
    indv_to_group: Vec<usize>,
    /// Planted between-individual dispersion per gene.
    phi_true: Vec<f32>,
}

/// Largest |a[i] - b[i]|.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// How the planted dispersion varies across genes.
#[derive(Clone, Copy)]
enum PhiSpec {
    Const(f32),
    /// `log phi_d = a + b * log_base_d + N(0, sd^2)`
    Trend {
        a: f32,
        b: f32,
        sd: f32,
    },
}

/// Gene-level layout of a simulated topic.
#[derive(Clone, Copy)]
struct SimSpec {
    /// `log_base_d ~ N(base_log_mean, base_log_sd^2)`; the per-pseudobulk rate is
    /// `exp(log_base_d) * U(3, 15)`.
    base_log_mean: f32,
    base_log_sd: f32,
    phi: PhiSpec,
}

impl Default for SimSpec {
    /// Today's layout: every gene at the same baseline, one shared phi.
    fn default() -> Self {
        Self {
            base_log_mean: 0.0,
            base_log_sd: 0.0,
            phi: PhiSpec::Const(PHI_TRUE),
        }
    }
}

fn uniform(rng: &mut rand::rngs::StdRng, lo: f32, hi: f32) -> f32 {
    Uniform::new(lo, hi).unwrap().sample(rng)
}

fn normal(rng: &mut rand::rngs::StdRng, mean: f32, sd: f32) -> f32 {
    if sd > 0.0 {
        Normal::new(mean, sd).unwrap().sample(rng)
    } else {
        mean
    }
}

/// Sample one topic's sufficient statistics; `beta[d]` is the planted log
/// fold of group 1 over group 0 for gene `d`.
fn simulate(n_genes: usize, beta: &[f32], seed: u64) -> GroupSim {
    simulate_with(n_genes, beta, seed, SimSpec::default())
}

fn simulate_with(n_genes: usize, beta: &[f32], seed: u64, spec: SimSpec) -> GroupSim {
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

    let mut y1_dp = Mat::zeros(n_genes, N_PB);
    let mut y0_dp = Mat::zeros(n_genes, N_PB);
    let mut y1_di = Mat::zeros(n_genes, n_indv);
    let mut phi_true = Vec::with_capacity(n_genes);

    for d in 0..n_genes {
        let tau_x = [1.0f32, beta[d].exp()];
        let log_base = normal(&mut rng, spec.base_log_mean, spec.base_log_sd);
        let mu: Vec<f32> = (0..N_PB)
            .map(|_| log_base.exp() * uniform(&mut rng, 3.0, 15.0))
            .collect();
        let gamma: Vec<f32> = (0..N_PB).map(|_| uniform(&mut rng, 0.5, 2.0)).collect();
        let phi = match spec.phi {
            PhiSpec::Const(v) => v,
            PhiSpec::Trend { a, b, sd } => (a + b * log_base + normal(&mut rng, 0.0, sd))
                .exp()
                .clamp(0.05, 500.0),
        };
        let delta_dist = Gamma::new(phi, 1.0 / phi).unwrap();
        let delta: Vec<f32> = (0..n_indv).map(|_| delta_dist.sample(&mut rng)).collect();
        for p in 0..N_PB {
            for i in 0..n_indv {
                let lambda = tau_x[indv_to_group[i]] * delta[i] * mu[p] * size_ip[(i, p)];
                let y = Poisson::new(lambda).unwrap().sample(&mut rng);
                y1_dp[(d, p)] += y;
                y1_di[(d, i)] += y;
            }
            y0_dp[(d, p)] = Poisson::new(gamma[p] * mu[p] * size_p[p])
                .unwrap()
                .sample(&mut rng);
        }
        phi_true.push(phi);
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
        phi_true,
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
    // Every update is row-separable and the trend is global, so block size
    // only leaves floating-point noise.
    let worst = max_abs_diff(&a, &b);
    assert!(worst < 1e-3, "block fit differs from whole fit by {worst}");
    assert_eq!(blocked[0].dispersion.len(), n_genes);
    let worst_phi = whole[0]
        .dispersion
        .iter()
        .zip(blocked[0].dispersion.iter())
        .map(|(w, b)| (w - b).abs() / w)
        .fold(0f32, f32::max);
    assert!(
        worst_phi < 1e-3,
        "block phi differs from whole phi by {worst_phi} relative"
    );
}

/// The delta update through the row prior is the hand-folded update it
/// replaced: adding phi_d to both statistics of a zero-prior Gamma matrix.
#[test]
fn delta_step_with_row_prior_equals_hand_folded_prior() {
    let sim = simulate(5, &[0.0; 5], 3);
    let y1_di = sim.stat.indv_y1_stat(0).clone();
    let lambda_di = y1_di.map(|y| 0.9 * y + 1.0);
    let phi = DVec::from_column_slice(&[0.5, 2.0, 8.0, 30.0, 300.0]);
    let (n_genes, n_indv) = y1_di.shape();

    let mut folded = GammaMatrix::new((n_genes, n_indv), 0.0, 0.0);
    let mut num = y1_di.clone();
    let mut den = lambda_di.clone();
    for d in 0..n_genes {
        num.row_mut(d).add_scalar_mut(phi[d]);
        den.row_mut(d).add_scalar_mut(phi[d]);
    }
    folded.update_stat(&num, &den);
    folded.calibrate();

    let mut row = GammaMatrix::with_row_prior((n_genes, n_indv), &phi, &phi);
    row.update_stat(&y1_di, &lambda_di);
    row.calibrate();

    let planes = [
        (folded.posterior_mean(), row.posterior_mean()),
        (folded.posterior_sd(), row.posterior_sd()),
        (folded.posterior_log_mean(), row.posterior_log_mean()),
        (folded.posterior_log_sd(), row.posterior_log_sd()),
    ];
    for (f, r) in planes {
        for (x, y) in f.iter().zip(r.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} != {y}");
        }
    }
}

/// Cox-Reid adjustment: with two fitted group means per gene and only four
/// individuals per group, the plug-in profile MLE overstates phi; the adjusted
/// profile is closer to the truth.
#[test]
fn cox_reid_profile_is_less_biased_than_unadjusted() {
    let (n_genes, n_indv, phi) = (400usize, 8usize, 8.0f32);
    let group: Vec<usize> = (0..n_indv).map(|i| i / 4).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(21);
    let delta_dist = Gamma::new(phi, 1.0 / phi).unwrap();
    let mut y = Mat::zeros(n_genes, n_indv);
    let mut lambda = Mat::zeros(n_genes, n_indv);
    for d in 0..n_genes {
        let m: Vec<f32> = (0..n_indv)
            .map(|_| uniform(&mut rng, 200.0, 800.0))
            .collect();
        for i in 0..n_indv {
            let dl: f32 = delta_dist.sample(&mut rng);
            y[(d, i)] = Poisson::new(m[i] * dl).unwrap().sample(&mut rng);
        }
        for x in 0..2 {
            let (sy, sm): (f32, f32) = (0..n_indv)
                .filter(|&i| group[i] == x)
                .map(|i| (y[(d, i)], m[i]))
                .fold((0.0, 0.0), |acc, (a, b)| (acc.0 + a, acc.1 + b));
            for i in (0..n_indv).filter(|&i| group[i] == x) {
                lambda[(d, i)] = sy / sm * m[i];
            }
        }
    }
    let layout = ActiveLayout::new(&vec![true; n_indv], &group, 2);
    let ml = profile_dispersion(&y, &lambda, &layout, false);
    let cr = profile_dispersion(&y, &lambda, &layout, true);
    let med = |e: &[GeneEvidence]| {
        let v: Vec<f32> = e
            .iter()
            .map(|g| g.log_phi_hat)
            .filter(|x| x.is_finite())
            .map(|x| x.exp())
            .collect();
        median(&v)
    };
    let (m_ml, m_cr) = (med(&ml), med(&cr));
    assert!(
        (m_cr - phi).abs() < (m_ml - phi).abs(),
        "CR median {m_cr} not closer to {phi} than ML median {m_ml}"
    );
    assert!(
        m_cr > phi / 1.6 && m_cr < phi * 1.6,
        "CR median {m_cr} far from {phi}"
    );
}

/// The global trend recovers a planted slope of log phi on log baseline.
#[test]
fn dispersion_trend_recovers_planted_slope() {
    let spec = SimSpec {
        base_log_mean: 8f32.ln(),
        base_log_sd: 1.0,
        phi: PhiSpec::Trend {
            a: 8f32.ln(),
            b: 0.4,
            sd: 0.3,
        },
    };
    let n_genes = 400;
    let sim = simulate_with(n_genes, &vec![0.0; n_genes], 22, spec);
    let params = sim
        .stat
        .estimate_group_parameters(&sim.indv_to_group, 2)
        .unwrap();
    let prior = params[0].dispersion_prior.expect("fitted prior");
    assert!(
        (prior.b - 0.4).abs() < 0.15,
        "slope {} vs planted 0.4",
        prior.b
    );
    // The trend at each gene's mean tracks the planted phi up to the planted
    // scatter (sd 0.3) plus the profile noise.
    let mean_abs_err = (0..n_genes)
        .map(|d| (prior.log_phi_trend(params[0].log_mean[d]) - sim.phi_true[d].ln()).abs())
        .sum::<f32>()
        / n_genes as f32;
    assert!(mean_abs_err < 0.5, "trend mean abs error {mean_abs_err}");
}

/// With the dispersion prior in place, the contrast matches an oracle fit run
/// with the true phi fixed.
#[test]
fn eb_dispersion_leaves_contrast_near_oracle() {
    let spec = SimSpec {
        base_log_mean: 8f32.ln(),
        base_log_sd: 1.0,
        phi: PhiSpec::Trend {
            a: 8f32.ln(),
            b: 0.4,
            sd: 0.3,
        },
    };
    let n_genes = 300;
    let beta: Vec<f32> = (0..n_genes)
        .map(|d| if d % 3 == 0 { 0.6 } else { 0.0 })
        .collect();
    let sim = simulate_with(n_genes, &beta, 25, spec);
    let eb = sim
        .stat
        .estimate_group_parameters(&sim.indv_to_group, 2)
        .unwrap();
    let phi_true = DVec::from_column_slice(&sim.phi_true);
    let oracle = sim
        .stat
        .fit_with_dispersion(&sim.indv_to_group, 2, GENE_BLOCK, vec![phi_true])
        .unwrap();
    let a = compute_group_contrast(&eb, 1, 0);
    let b = compute_group_contrast(&oracle, 1, 0);
    let worst = max_abs_diff(&a, &b);
    assert!(
        worst < 0.05,
        "trend-phi contrast differs from oracle by {worst}"
    );
    let planted = mean(
        &(0..n_genes)
            .filter(|d| d % 3 == 0)
            .map(|d| a[d])
            .collect::<Vec<_>>(),
    );
    let null = mean(
        &(0..n_genes)
            .filter(|d| d % 3 != 0)
            .map(|d| a[d])
            .collect::<Vec<_>>(),
    );
    assert!((planted - 0.6).abs() < 0.1, "planted mean {planted}");
    assert!(null.abs() < 0.1, "null mean {null}");
}

/// How much the exposure contrast depends on phi at all: a single global phi
/// for every gene versus the gene-wise posterior, on a sim with a planted
/// trend and scatter. The contrast is a pooled ratio in which delta enters
/// only through its denominator weights, so it should move little.
#[test]
fn global_phi_moves_the_contrast_little() {
    let spec = SimSpec {
        base_log_mean: 8f32.ln(),
        base_log_sd: 1.0,
        phi: PhiSpec::Trend {
            a: 8f32.ln(),
            b: 0.4,
            sd: 0.8,
        },
    };
    let n_genes = 300;
    let beta: Vec<f32> = (0..n_genes)
        .map(|d| if d % 3 == 0 { 0.6 } else { 0.0 })
        .collect();
    let sim = simulate_with(n_genes, &beta, 26, spec);
    let eb = sim
        .stat
        .estimate_group_parameters(&sim.indv_to_group, 2)
        .unwrap();
    let global = DVec::from_element(n_genes, median(&sim.phi_true));
    let one_phi = sim
        .stat
        .fit_with_dispersion(&sim.indv_to_group, 2, GENE_BLOCK, vec![global])
        .unwrap();
    let a = compute_group_contrast(&eb, 1, 0);
    let b = compute_group_contrast(&one_phi, 1, 0);
    let worst = max_abs_diff(&a, &b);
    let r = {
        let (ma, mb) = (mean(&a), mean(&b));
        let num: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum();
        let da: f32 = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
        let db: f32 = b.iter().map(|y| (y - mb).powi(2)).sum::<f32>().sqrt();
        num / (da * db)
    };
    assert!(worst < 0.1, "contrast moved by {worst} under a global phi");
    assert!(r > 0.99, "contrast correlation {r} under a global phi");
}

/// Trend-only dispersion: every gene's phi is the trend evaluated at its own
/// log mean, nothing gene-specific on top.
#[test]
fn dispersion_equals_trend_at_each_gene() {
    let spec = SimSpec {
        base_log_mean: 8f32.ln(),
        base_log_sd: 1.0,
        phi: PhiSpec::Trend {
            a: 8f32.ln(),
            b: 0.4,
            sd: 0.5,
        },
    };
    let n_genes = 200;
    let sim = simulate_with(n_genes, &vec![0.0; n_genes], 28, spec);
    let params = sim
        .stat
        .estimate_group_parameters(&sim.indv_to_group, 2)
        .unwrap();
    let prior = params[0].dispersion_prior.expect("fitted prior");
    for d in 0..n_genes {
        let want = prior.log_phi_trend(params[0].log_mean[d]).exp();
        let got = params[0].dispersion[d];
        assert!(
            (got - want).abs() < 1e-4 * want,
            "gene {d}: phi {got} != trend {want}"
        );
    }
}

/// The same comparison at low counts, where delta's prior dominates the data
/// and phi sets the weights in tau's denominator: this is the regime where a
/// wrong phi could move the calls.
#[test]
fn global_phi_moves_the_contrast_little_at_low_counts() {
    let spec = SimSpec {
        base_log_mean: 0.02f32.ln(),
        base_log_sd: 1.0,
        phi: PhiSpec::Trend {
            a: 8f32.ln(),
            b: 0.4,
            sd: 0.8,
        },
    };
    let n_genes = 300;
    let beta: Vec<f32> = (0..n_genes)
        .map(|d| if d % 3 == 0 { 0.6 } else { 0.0 })
        .collect();
    let sim = simulate_with(n_genes, &beta, 29, spec);
    let trend = sim
        .stat
        .estimate_group_parameters(&sim.indv_to_group, 2)
        .unwrap();
    let global = DVec::from_element(n_genes, median(&sim.phi_true));
    let one_phi = sim
        .stat
        .fit_with_dispersion(&sim.indv_to_group, 2, GENE_BLOCK, vec![global])
        .unwrap();
    let phi_true = DVec::from_column_slice(&sim.phi_true);
    let oracle = sim
        .stat
        .fit_with_dispersion(&sim.indv_to_group, 2, GENE_BLOCK, vec![phi_true])
        .unwrap();
    let a = compute_group_contrast(&trend, 1, 0);
    let b = compute_group_contrast(&one_phi, 1, 0);
    let c = compute_group_contrast(&oracle, 1, 0);
    let counts_per_indv = sim.stat.indv_y1_stat(0).sum() / (n_genes * 2 * N_PER_GROUP) as f32;
    let planted = mean(
        &(0..n_genes)
            .filter(|d| d % 3 == 0)
            .map(|d| a[d])
            .collect::<Vec<_>>(),
    );
    // Measured: about 0.14 at this depth (0.02 at 600 counts). A wrong phi
    // does move sparse genes' estimates; this guards against it getting worse.
    let vs_oracle = max_abs_diff(&a, &c);
    assert!(
        vs_oracle < 0.25,
        "at {counts_per_indv:.1} counts per gene per individual: trend vs oracle {vs_oracle:.3}, \
         trend vs global {:.3}, planted mean {planted:.3}",
        max_abs_diff(&a, &b)
    );
}
