//! Unit tests for the trajectory (velocity) cell-state generator.

use super::{sample_trajectory_topics, traj_theta};
use rand::SeedableRng;

/// θ(t, b) is always a valid proportion (non-negative, sums to 1).
#[test]
fn traj_theta_is_valid_proportion() {
    let k = 6;
    for &(t, b) in &[
        (0.0, 0),
        (0.25, 0),
        (0.5, 1),
        (0.75, 0),
        (0.75, 1),
        (1.0, 3),
    ] {
        let th = traj_theta(t, b, k);
        assert_eq!(th.len(), k);
        assert!(th.iter().all(|&v| v >= 0.0), "negative mass at t={t}");
        let s: f32 = th.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "θ sums to {s} at t={t}");
    }
}

/// The path is monotone: the root builds topic-1 mass as t→0.5, and beyond the
/// bifurcation each branch builds its terminal-vertex (2+b) mass as t→1.
#[test]
fn traj_theta_progresses_along_path() {
    let k = 6;
    // Root: topic 1 grows with t.
    assert!(traj_theta(0.1, 0, k)[1] < traj_theta(0.4, 0, k)[1]);
    // Branch b=0 → vertex 2, b=1 → vertex 3: terminal mass grows toward t=1.
    assert!(traj_theta(0.6, 0, k)[2] < traj_theta(0.9, 0, k)[2]);
    assert!(traj_theta(0.6, 1, k)[3] < traj_theta(0.9, 1, k)[3]);
    // Each branch peaks on its own vertex at the terminus.
    let e0 = traj_theta(1.0, 0, k);
    let e1 = traj_theta(1.0, 1, k);
    assert!(e0[2] > e0[3], "branch 0 should peak on vertex 2");
    assert!(e1[3] > e1[2], "branch 1 should peak on vertex 3");
}

/// The look-ahead (nascent) state leads the mature state along the path, so the
/// mature→nascent contrast is a forward velocity; at the terminus it saturates
/// (future clamps to θ(1), so velocity shrinks).
#[test]
fn future_leads_current_and_saturates_at_terminus() {
    let k = 6;
    let n = 400;
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let traj = sample_trajectory_topics(k, n, 2, 0.2, &mut rng).unwrap();

    assert_eq!(traj.pseudotime.len(), n);
    assert!(traj.pseudotime.iter().all(|&t| (0.0..=1.0).contains(&t)));
    assert!(traj.branch.iter().all(|&b| b < 2));

    // Velocity magnitude in topic space = ‖θ_future − θ‖. It should be non-trivial
    // for mid-trajectory cells and ~0 for terminal cells (look-ahead saturated).
    let mut mid_mag = 0.0f32;
    let mut mid_n = 0;
    let mut term_mag = 0.0f32;
    let mut term_n = 0;
    for j in 0..n {
        let d: f32 = (0..k)
            .map(|kk| {
                let x = traj.theta_future[(kk, j)] - traj.theta[(kk, j)];
                x * x
            })
            .sum::<f32>()
            .sqrt();
        if traj.pseudotime[j] > 0.9 {
            term_mag += d;
            term_n += 1;
        } else if traj.pseudotime[j] < 0.7 {
            mid_mag += d;
            mid_n += 1;
        }
    }
    let mid = mid_mag / mid_n.max(1) as f32;
    let term = term_mag / term_n.max(1) as f32;
    assert!(mid > 0.05, "mid-trajectory velocity too small ({mid})");
    assert!(
        term < mid,
        "terminal velocity ({term}) should fade below mid ({mid})"
    );
}
