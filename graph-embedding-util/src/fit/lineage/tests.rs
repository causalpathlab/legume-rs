//! Unit tests for the fixed pb-lineage graph builder.

use super::{build_pb_lineage, smooth_pb_velocity, PbLineageLevel};
use crate::fit::projection::PbLevelVelocity;

/// Velocity smoothing flips a sign-corrupted node back to agree with its neighbours,
/// leaving θ untouched.
#[test]
fn smoothing_repairs_corrupted_sign() {
    let h = 2;
    // Four nodes on a line at x=0..3, velocity +x, but node 2 is CORRUPTED to −x. Its
    // +x neighbours should outvote the corrupted self after smoothing.
    let theta = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
    let delta = vec![
        1.0, 0.0, // n0 +x
        1.0, 0.0, // n1 +x
        -1.0, 0.0, // n2 CORRUPTED −x
        1.0, 0.0, // n3 +x
    ];
    let vel = PbLevelVelocity {
        n_pb: 4,
        theta,
        delta,
    };
    let sm = smooth_pb_velocity(&vel, h, 3);

    let d2x = sm.delta[2 * h];
    assert!(d2x > 0.0, "node 2 sign not repaired (δx = {d2x})");
    assert_eq!(sm.theta, vel.theta, "θ must be untouched");
}

/// Three pb nodes on a line at x = 0, 1, 2 (signal in dim 0), each with velocity
/// pointing +x. Every retained edge must run from a lower x to a higher x
/// (velocity-forward), and no backward edge may appear.
#[test]
fn edges_are_velocity_forward() {
    let h = 2;
    // θ_pb: positions along dim 0.
    let theta = vec![
        0.0, 0.0, /*n0*/ 1.0, 0.0, /*n1*/ 2.0, 0.0, /*n2*/
    ];
    // δ_pb: all point +dim0.
    let delta = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let lvl = PbLevelVelocity {
        n_pb: 3,
        theta: theta.clone(),
        delta,
    };
    let out = build_pb_lineage(&[lvl], h, 2);
    let g: &PbLineageLevel = &out[0];
    assert!(!g.edges.is_empty(), "expected forward edges");
    for &(i, j, w) in &g.edges {
        let xi = theta[i as usize * h];
        let xj = theta[j as usize * h];
        assert!(xj > xi, "edge {i}->{j} not forward (x {xi} -> {xj})");
        assert!(w > 0.0);
    }
    // Unit velocity rows.
    for i in 0..3 {
        let v = &g.velocity[i * h..(i + 1) * h];
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "velocity row {i} not unit ({n})");
    }
}

/// A node with ~zero velocity has no defined orientation, so it emits no
/// outgoing edges (and its unit-velocity row stays zero).
#[test]
fn zero_velocity_node_has_no_outgoing_edges() {
    let h = 2;
    let theta = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
    // Node 1 has zero velocity; 0 and 2 point +x.
    let delta = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let lvl = PbLevelVelocity {
        n_pb: 3,
        theta,
        delta,
    };
    let out = build_pb_lineage(&[lvl], h, 2);
    let g = &out[0];
    assert!(
        g.edges.iter().all(|&(i, _, _)| i != 1),
        "zero-velocity node 1 should emit no outgoing edges"
    );
    let v1 = &g.velocity[h..2 * h];
    assert_eq!(v1, &[0.0, 0.0]);
}
