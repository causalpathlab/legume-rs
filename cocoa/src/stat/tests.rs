//! Recovery tests for the COCOA variational decomposition.
//!
//! The generative model the coordinate updates in [`super::CocoaStat`] invert:
//!
//! ```text
//!   y1(d,p) = mu(d,p) * sum_i tau(d,i) * n(i,p)     treated arm
//!   y0(d,p) = mu(d,p) * gamma(d,p) * n(p)           matched arm
//!   y1(d,i) = tau(d,i) * sum_p mu(d,p) * n(i,p)     per-individual treated
//! ```
//!
//! `mu` is the shared baseline, `gamma` the matched residual, `tau` the
//! individual exposure effect.
//!
//! Note what these tests deliberately do NOT assert: that `mu`, `gamma` and
//! `tau` are recovered individually. They cannot be. Scaling
//! `mu -> c*mu`, `tau -> tau/c`, `gamma -> gamma/c` leaves all three
//! observation equations unchanged, so the whole one-parameter family is a
//! fixed point of the updates -- `scale_family_is_a_fixed_point` pins that
//! down as a property rather than leaving it as a latent surprise. What is
//! identifiable, and what the pipeline actually consumes, is the
//! reconstruction and the *contrast* between exposure groups.

use super::*;
use matrix_param::traits::Inference;

////////////////////
// Test fixtures  //
////////////////////

const N_GENES: usize = 6;
const N_INDV: usize = 4;
const N_PB: usize = 5;

/// Deterministic positive values, spread over roughly an order of
/// magnitude so a bug that collapses everything to a constant shows up.
fn spread(seed: usize, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            // cheap deterministic hash -> [0,1)
            let h = ((seed * 2_654_435_761 + i * 40_503) % 1000) as f32 / 1000.0;
            lo + h * (hi - lo)
        })
        .collect()
}

fn mat_from(seed: usize, nrows: usize, ncols: usize, lo: f32, hi: f32) -> Mat {
    Mat::from_vec(nrows, ncols, spread(seed, nrows * ncols, lo, hi))
}

/// Ground-truth parameters plus the sufficient statistics they imply.
struct Sim {
    mu: Mat,      // genes x pseudobulk
    gamma: Mat,   // genes x pseudobulk
    tau: Mat,     // genes x individuals
    size_ip: Mat, // individuals x pseudobulk
    size_p: DVec, // pseudobulk
    y1_dp: Mat,
    y0_dp: Mat,
    y1_di: Mat,
}

impl Sim {
    /// Build noise-free statistics from the given parameters. Noise-free is
    /// the point: the truth is then an *exact* fixed point of the updates, so
    /// any residual is the estimator's error and not sampling error.
    fn new(mu: Mat, gamma: Mat, tau: Mat, size_ip: Mat) -> Self {
        let n_genes = mu.nrows();
        let n_pb = mu.ncols();
        let n_indv = tau.ncols();

        // size_p(p) is the total over individuals, matching how
        // `collapse_cocoa_data` accumulates the two size statistics.
        let mut size_p = DVec::zeros(n_pb);
        for p in 0..n_pb {
            size_p[p] = (0..n_indv).map(|i| size_ip[(i, p)]).sum();
        }

        let mut y1_dp = Mat::zeros(n_genes, n_pb);
        let mut y0_dp = Mat::zeros(n_genes, n_pb);
        let mut y1_di = Mat::zeros(n_genes, n_indv);

        for d in 0..n_genes {
            for p in 0..n_pb {
                let exposure: f32 = (0..n_indv).map(|i| tau[(d, i)] * size_ip[(i, p)]).sum();
                y1_dp[(d, p)] = mu[(d, p)] * exposure;
                y0_dp[(d, p)] = mu[(d, p)] * gamma[(d, p)] * size_p[p];
            }
            for i in 0..n_indv {
                let base: f32 = (0..n_pb).map(|p| mu[(d, p)] * size_ip[(i, p)]).sum();
                y1_di[(d, i)] = tau[(d, i)] * base;
            }
        }

        Self {
            mu,
            gamma,
            tau,
            size_ip,
            size_p,
            y1_dp,
            y0_dp,
            y1_di,
        }
    }

    fn default_params() -> Self {
        // Counts are scaled up so the Gamma prior is negligible next to the
        // evidence; the estimator is then testable as a fixed point.
        let mu = mat_from(1, N_GENES, N_PB, 2.0, 20.0);
        let gamma = mat_from(2, N_GENES, N_PB, 0.5, 3.0);
        let tau = mat_from(3, N_GENES, N_INDV, 0.5, 4.0);
        let size_ip = mat_from(4, N_INDV, N_PB, 10.0, 60.0);
        Self::new(mu, gamma, tau, size_ip)
    }

    /// Load the simulated statistics into a single-topic `CocoaStat`.
    fn into_stat(self, n_opt_iter: usize) -> (CocoaStat, Self) {
        let mut stat = CocoaStat::new(
            CocoaStatArgs {
                n_genes: self.y1_dp.nrows(),
                n_topics: 1,
                n_indv: self.y1_di.ncols(),
                n_samples: self.y1_dp.ncols(),
            },
            Some(n_opt_iter),
            // Weak prior: we are testing the likelihood's fixed point, not
            // the shrinkage.
            Some((1e-4, 1e-4)),
        );
        stat.y1_stat_mut(0).copy_from(&self.y1_dp);
        stat.y0_stat_mut(0).copy_from(&self.y0_dp);
        stat.indv_y1_stat_mut(0).copy_from(&self.y1_di);
        stat.size_stat_mut(0).copy_from(&self.size_p);
        stat.indv_size_stat_mut(0).copy_from(&self.size_ip);
        (stat, self)
    }
}

/// Largest relative deviation between two same-shaped matrices.
fn max_rel_err(actual: &Mat, expected: &Mat) -> f32 {
    assert_eq!(actual.shape(), expected.shape());
    let mut worst = 0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let denom = e.abs().max(1e-6);
        worst = worst.max((a - e).abs() / denom);
    }
    worst
}

/// Reconstruct the three observation equations from fitted parameters.
fn reconstruct(out: &CocoaGammaOut, size_ip: &Mat, size_p: &DVec) -> (Mat, Mat, Mat) {
    let mu = out.shared.posterior_mean();
    let gamma = out.residual.posterior_mean();
    let tau = out.exposure.posterior_mean();

    let n_genes = mu.nrows();
    let n_pb = mu.ncols();
    let n_indv = tau.ncols();

    let mut y1_dp = Mat::zeros(n_genes, n_pb);
    let mut y0_dp = Mat::zeros(n_genes, n_pb);
    let mut y1_di = Mat::zeros(n_genes, n_indv);

    for d in 0..n_genes {
        for p in 0..n_pb {
            let exposure: f32 = (0..n_indv).map(|i| tau[(d, i)] * size_ip[(i, p)]).sum();
            y1_dp[(d, p)] = mu[(d, p)] * exposure;
            y0_dp[(d, p)] = mu[(d, p)] * gamma[(d, p)] * size_p[p];
        }
        for i in 0..n_indv {
            let base: f32 = (0..n_pb).map(|p| mu[(d, p)] * size_ip[(i, p)]).sum();
            y1_di[(d, i)] = tau[(d, i)] * base;
        }
    }
    (y1_dp, y0_dp, y1_di)
}

//////////////////////////////////
// Recovery of the decomposition //
//////////////////////////////////

#[test]
fn reconstructs_noise_free_statistics() {
    let (stat, sim) = Sim::default_params().into_stat(200);
    let out = stat.optimize_each_topic(0).expect("optimization failed");

    let (y1_dp, y0_dp, y1_di) = reconstruct(&out, &sim.size_ip, &sim.size_p);

    assert!(
        max_rel_err(&y1_dp, &sim.y1_dp) < 1e-2,
        "treated arm not reconstructed: max rel err {}",
        max_rel_err(&y1_dp, &sim.y1_dp)
    );
    assert!(
        max_rel_err(&y0_dp, &sim.y0_dp) < 1e-2,
        "matched arm not reconstructed: max rel err {}",
        max_rel_err(&y0_dp, &sim.y0_dp)
    );
    assert!(
        max_rel_err(&y1_di, &sim.y1_di) < 1e-2,
        "per-individual treated arm not reconstructed: max rel err {}",
        max_rel_err(&y1_di, &sim.y1_di)
    );
}

#[test]
fn more_iterations_improve_the_reconstruction() {
    // One iteration started from the truth must not move the reconstruction:
    // this separates "the updates are correct" from "the updates happen to
    // converge somewhere reasonable".
    let (stat, sim) = Sim::default_params().into_stat(1);
    let out = stat.optimize_each_topic(0).expect("optimization failed");
    let (y1_dp, _, _) = reconstruct(&out, &sim.size_ip, &sim.size_p);

    // A single sweep from the default init won't be exact, but 200 sweeps
    // must be a strict improvement over 1 -- the estimator has to be moving
    // toward the truth, not oscillating around it.
    let (stat_long, sim_long) = Sim::default_params().into_stat(200);
    let out_long = stat_long
        .optimize_each_topic(0)
        .expect("optimization failed");
    let (y1_long, _, _) = reconstruct(&out_long, &sim_long.size_ip, &sim_long.size_p);

    let err_1 = max_rel_err(&y1_dp, &sim.y1_dp);
    let err_200 = max_rel_err(&y1_long, &sim_long.y1_dp);
    assert!(
        err_200 < err_1,
        "more iterations did not help: 1 sweep {err_1}, 200 sweeps {err_200}"
    );
}

#[test]
fn scale_family_is_a_fixed_point() {
    // mu -> c*mu, tau -> tau/c, gamma -> gamma/c generates identical
    // statistics. Documenting this as a test keeps anyone from writing an
    // assertion that fitted `mu` equals simulated `mu`.
    let base = Sim::default_params();
    let c = 3.5f32;
    let scaled = Sim::new(
        base.mu.clone() * c,
        base.gamma.clone() / c,
        base.tau.clone() / c,
        base.size_ip.clone(),
    );

    assert!(max_rel_err(&scaled.y1_dp, &base.y1_dp) < 1e-4);
    assert!(max_rel_err(&scaled.y0_dp, &base.y0_dp) < 1e-4);
    assert!(max_rel_err(&scaled.y1_di, &base.y1_di) < 1e-4);
}

#[test]
fn estimate_parameters_matches_per_topic_optimization() {
    // The parallel `estimate_parameters` over topics must agree with the
    // per-topic entry point it fans out to.
    let sim_a = Sim::default_params();
    let sim_b = Sim::new(
        mat_from(11, N_GENES, N_PB, 1.0, 8.0),
        mat_from(12, N_GENES, N_PB, 0.5, 2.0),
        mat_from(13, N_GENES, N_INDV, 0.5, 3.0),
        mat_from(14, N_INDV, N_PB, 5.0, 40.0),
    );

    let mut stat = CocoaStat::new(
        CocoaStatArgs {
            n_genes: N_GENES,
            n_topics: 2,
            n_indv: N_INDV,
            n_samples: N_PB,
        },
        Some(50),
        Some((1e-4, 1e-4)),
    );
    for (k, sim) in [&sim_a, &sim_b].iter().enumerate() {
        stat.y1_stat_mut(k).copy_from(&sim.y1_dp);
        stat.y0_stat_mut(k).copy_from(&sim.y0_dp);
        stat.indv_y1_stat_mut(k).copy_from(&sim.y1_di);
        stat.size_stat_mut(k).copy_from(&sim.size_p);
        stat.indv_size_stat_mut(k).copy_from(&sim.size_ip);
    }

    assert_eq!(stat.num_topics(), 2);
    let all = stat.estimate_parameters().expect("estimation failed");
    assert_eq!(all.len(), 2);

    for (k, fanned_out) in all.iter().enumerate() {
        let one = stat.optimize_each_topic(k).expect("optimization failed");
        assert!(
            max_rel_err(
                fanned_out.shared.posterior_mean(),
                one.shared.posterior_mean()
            ) < 1e-5,
            "topic {k}: estimate_parameters disagrees with optimize_each_topic"
        );
    }
}

///////////////////////
// Exposure contrast //
///////////////////////

#[test]
fn exposure_contrast_recovers_a_planted_log_fold() {
    // Individuals 0,1 are unexposed; 2,3 exposed. Genes 0 and 1 carry a
    // planted log-fold in tau; the rest carry none.
    let exposure_assignment = vec![0usize, 0, 1, 1];
    let planted = [0.9f32, -0.6, 0.0, 0.0, 0.0, 0.0];

    let mu = mat_from(21, N_GENES, N_PB, 4.0, 12.0);
    let gamma = mat_from(22, N_GENES, N_PB, 0.8, 2.0);
    let size_ip = mat_from(23, N_INDV, N_PB, 20.0, 50.0);

    let base_tau = spread(24, N_GENES, 1.0, 3.0);
    let mut tau = Mat::zeros(N_GENES, N_INDV);
    for d in 0..N_GENES {
        for i in 0..N_INDV {
            let lift = if exposure_assignment[i] == 1 {
                planted[d].exp()
            } else {
                1.0
            };
            tau[(d, i)] = base_tau[d] * lift;
        }
    }

    let (stat, sim) = Sim::new(mu, gamma, tau, size_ip).into_stat(200);
    let _ = sim;
    let params = stat.estimate_parameters().expect("estimation failed");
    let contrast = compute_exposure_contrast(&params, &exposure_assignment);

    assert_eq!(contrast.len(), N_GENES);
    for d in 0..N_GENES {
        assert!(
            (contrast[d] - planted[d]).abs() < 0.05,
            "gene {d}: contrast {} vs planted {}",
            contrast[d],
            planted[d]
        );
    }
}

#[test]
fn exposure_contrast_is_antisymmetric_in_group_labels() {
    // Swapping which group is "exposed" must flip the sign of every entry.
    let planted = [0.7f32, -0.4, 0.0, 0.0, 0.0, 0.0];
    let mu = mat_from(31, N_GENES, N_PB, 4.0, 12.0);
    let gamma = mat_from(32, N_GENES, N_PB, 0.8, 2.0);
    let size_ip = mat_from(33, N_INDV, N_PB, 20.0, 50.0);
    let base_tau = spread(34, N_GENES, 1.0, 3.0);

    let assignment = vec![0usize, 0, 1, 1];
    let mut tau = Mat::zeros(N_GENES, N_INDV);
    for d in 0..N_GENES {
        for i in 0..N_INDV {
            let lift = if assignment[i] == 1 {
                planted[d].exp()
            } else {
                1.0
            };
            tau[(d, i)] = base_tau[d] * lift;
        }
    }

    let (stat, _) = Sim::new(mu, gamma, tau, size_ip).into_stat(120);
    let params = stat.estimate_parameters().expect("estimation failed");

    let forward = compute_exposure_contrast(&params, &assignment);
    let flipped_assignment = vec![1usize, 1, 0, 0];
    let reversed = compute_exposure_contrast(&params, &flipped_assignment);

    for d in 0..N_GENES {
        assert!(
            (forward[d] + reversed[d]).abs() < 1e-4,
            "gene {d} not antisymmetric: {} vs {}",
            forward[d],
            reversed[d]
        );
    }
}

//////////////////
// z_to_pvalue  //
//////////////////

#[test]
fn z_to_pvalue_matches_the_two_sided_normal_tail() {
    assert!((z_to_pvalue(0.0) - 1.0).abs() < 1e-6, "z=0 must give p=1");
    assert!(
        (z_to_pvalue(1.959_964) - 0.05).abs() < 1e-3,
        "z=1.96 must give p~0.05"
    );
    assert!(
        (z_to_pvalue(2.575_829) - 0.01).abs() < 1e-3,
        "z=2.58 must give p~0.01"
    );
    // Two-sided, so the sign of z cannot matter.
    assert!((z_to_pvalue(1.5) - z_to_pvalue(-1.5)).abs() < 1e-7);
    // Monotone decreasing in |z|.
    assert!(z_to_pvalue(3.0) < z_to_pvalue(2.0));
    assert!(z_to_pvalue(2.0) < z_to_pvalue(1.0));
}

//////////////////////////////////////
// Residual collider stratification //
//////////////////////////////////////

#[test]
fn removing_the_exposure_effect_equalizes_group_means() {
    // Two individuals per group, three cells each. Group 1 has topic 0
    // inflated by exp(0.8); the adjustment must remove exactly that.
    let n_topics = 2;
    let cell_to_individual = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let individual_exposure_group = vec![0usize, 0, 1, 1];
    let lift = 0.8f32;

    let n_cells = cell_to_individual.len();
    let mut props = Mat::zeros(n_cells, n_topics);
    for (j, &i) in cell_to_individual.iter().enumerate() {
        // Within-individual variation that the adjustment must preserve.
        let jitter = 1.0 + 0.05 * ((j % 3) as f32);
        let group_lift = if individual_exposure_group[i] == 1 {
            lift.exp()
        } else {
            1.0
        };
        props[(j, 0)] = 0.4 * jitter * group_lift;
        props[(j, 1)] = 0.6 * jitter;
    }

    let before = super::average_topic_log_proportions_per_individual(
        &props,
        &cell_to_individual,
        individual_exposure_group.len(),
    );
    let (before_groups, _) =
        super::average_topic_logits_per_exposure_group(&before, &individual_exposure_group);
    let gap_before = before_groups[(1, 0)] - before_groups[(0, 0)];
    assert!(
        (gap_before - lift).abs() < 1e-4,
        "fixture did not plant the intended gap: {gap_before}"
    );

    let max_shift = remove_exposure_effect_from_topic_proportions(
        &mut props,
        &cell_to_individual,
        &individual_exposure_group,
    );

    // Balanced groups, so each sits half the gap from the grand mean.
    assert!(
        (max_shift[0] - lift / 2.0).abs() < 1e-4,
        "reported shift {} != half the planted gap",
        max_shift[0]
    );
    assert!(max_shift[1] < 1e-4, "untouched topic reported a shift");

    let after = super::average_topic_log_proportions_per_individual(
        &props,
        &cell_to_individual,
        individual_exposure_group.len(),
    );
    let (after_groups, _) =
        super::average_topic_logits_per_exposure_group(&after, &individual_exposure_group);
    let gap_after = after_groups[(1, 0)] - after_groups[(0, 0)];
    assert!(
        gap_after.abs() < 1e-4,
        "exposure gap survived the adjustment: {gap_after}"
    );
}

#[test]
fn removing_the_exposure_effect_preserves_within_individual_variation() {
    // The adjustment scales every cell in a group by one constant per topic,
    // so ratios between cells of the same individual must be untouched.
    let cell_to_individual = vec![0, 0, 1, 1];
    let individual_exposure_group = vec![0usize, 1];
    let mut props = Mat::from_row_slice(4, 2, &[0.2, 0.8, 0.5, 0.5, 0.3, 0.7, 0.6, 0.4]);
    let ratio_before = props[(0, 0)] / props[(1, 0)];

    remove_exposure_effect_from_topic_proportions(
        &mut props,
        &cell_to_individual,
        &individual_exposure_group,
    );

    let ratio_after = props[(0, 0)] / props[(1, 0)];
    assert!(
        (ratio_before - ratio_after).abs() < 1e-5,
        "within-individual ratio changed: {ratio_before} -> {ratio_after}"
    );
}

#[test]
fn cells_with_no_matching_individual_are_left_alone() {
    // `cell_to_individual` may point past the end for unmatched cells; those
    // must be skipped rather than panicking or being silently rescaled.
    let cell_to_individual = vec![0, 1, 99];
    let individual_exposure_group = vec![0usize, 1];
    let mut props = Mat::from_row_slice(3, 1, &[0.3, 0.7, 0.5]);

    remove_exposure_effect_from_topic_proportions(
        &mut props,
        &cell_to_individual,
        &individual_exposure_group,
    );

    assert!(
        (props[(2, 0)] - 0.5).abs() < 1e-7,
        "unmatched cell was modified: {}",
        props[(2, 0)]
    );
}
