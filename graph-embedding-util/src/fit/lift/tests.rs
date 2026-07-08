//! Unit tests for the phase-2 cell-lineage lift (cell-lift).

use super::{dense_to_edges, lift_cells, pb_trajectory};
use crate::fit::projection::PbLevelVelocity;

/// A linear chain 0→1→2→3 in a 1-D latent (θ = 0,1,2,3), velocity all +1. The
/// integrated pseudotime must be monotone increasing along the chain and the
/// single sink (node 3) the only terminal.
#[test]
fn linear_chain_pseudotime_is_monotone() {
    let h = 1;
    let vel = PbLevelVelocity {
        n_pb: 4,
        theta: vec![0.0, 1.0, 2.0, 3.0],
        delta: vec![1.0, 1.0, 1.0, 1.0],
    };
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
    let traj = pb_trajectory(&vel, &edges, h, 1.0);

    assert!(
        traj.tau[0] < traj.tau[1] && traj.tau[1] < traj.tau[2] && traj.tau[2] < traj.tau[3],
        "pseudotime should increase along the chain: {:?}",
        traj.tau
    );
    assert!(
        (traj.tau[0] - 0.0).abs() < 1e-4,
        "root τ≈0 ({})",
        traj.tau[0]
    );
    assert!(
        (traj.tau[3] - 1.0).abs() < 1e-4,
        "leaf τ≈1 ({})",
        traj.tau[3]
    );
    assert_eq!(traj.terminals, vec![3], "only node 3 is a sink");
    assert_eq!(traj.roots, vec![0], "node 0 is the only source/root");
}

/// Two disjoint forward chains (0→1→2 and 3→4→5) are two separate lineages, so the
/// root stage must return BOTH sources — a velocity DAG can have several origins.
#[test]
fn multiple_roots_for_disjoint_lineages() {
    let h = 1;
    let vel = PbLevelVelocity {
        n_pb: 6,
        theta: vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
        delta: vec![1.0; 6],
    };
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0)];
    let traj = pb_trajectory(&vel, &edges, h, 1.0);
    let mut roots = traj.roots.clone();
    roots.sort_unstable();
    assert_eq!(roots, vec![0, 3], "both chain heads are roots");
}

/// A redundant (k=2) forward chain with ONE edge flipped backward (2→3 given as 3→2).
/// Node 3 still has a forward in-edge (1→3), so it is NOT a spurious source — the root
/// stays node 0, and the root-distance re-orientation flips the bad edge back, so the
/// integrated pseudotime is monotone again.
#[test]
fn reorientation_repairs_a_flipped_edge() {
    let h = 1;
    let vel = PbLevelVelocity {
        n_pb: 5,
        theta: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        delta: vec![1.0; 5],
    };
    // Forward chain + skip edges, but 2→3 is CORRUPTED to 3→2.
    let edges = vec![
        (0, 1, 1.0),
        (1, 2, 1.0),
        (3, 2, 1.0), // flipped (should be 2→3)
        (3, 4, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
        (2, 4, 1.0),
    ];
    let traj = pb_trajectory(&vel, &edges, h, 1.0);
    assert_eq!(traj.roots, vec![0], "node 0 remains the sole root");
    assert!(
        (0..4).all(|i| traj.tau[i] < traj.tau[i + 1]),
        "pseudotime should be monotone after flip repair: {:?}",
        traj.tau
    );
}

/// A bifurcation 0→1, 1→2, 1→3 with leaves 2,3. Both leaves are terminals; the
/// root's fate splits between them; each leaf is one-hot on itself.
#[test]
fn bifurcation_has_two_fates() {
    let h = 1;
    let vel = PbLevelVelocity {
        n_pb: 4,
        theta: vec![0.0, 1.0, 2.0, 2.0],
        delta: vec![1.0, 1.0, 1.0, 1.0],
    };
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)];
    let traj = pb_trajectory(&vel, &edges, h, 1.0);

    assert_eq!(traj.terminals, vec![2, 3], "leaves 2 and 3 are terminals");
    let k = 2;
    // Leaf 2 one-hot on column 0, leaf 3 one-hot on column 1.
    assert!((traj.fate[2 * k] - 1.0).abs() < 1e-4);
    assert!((traj.fate[3 * k + 1] - 1.0).abs() < 1e-4);
    // Root reaches both leaves (non-degenerate split).
    assert!(
        traj.fate[0] > 0.1 && traj.fate[1] > 0.1,
        "root fate splits: {:?}",
        &traj.fate[0..2]
    );
}

/// Cells sitting exactly on the pb landmarks inherit their pseudotime; a cell at
/// the root end is earlier than a cell at the leaf end.
#[test]
fn cell_lift_orders_cells_along_chain() {
    let h = 1;
    let vel = PbLevelVelocity {
        n_pb: 4,
        theta: vec![0.0, 1.0, 2.0, 3.0],
        delta: vec![1.0, 1.0, 1.0, 1.0],
    };
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
    let traj = pb_trajectory(&vel, &edges, h, 1.0);

    // Three cells: near root (θ=0.1), middle (θ=1.5), near leaf (θ=2.9).
    let theta_c = vec![0.1f32, 1.5, 2.9];
    let lin = lift_cells(&theta_c, 3, &vel, &traj, h, 0);
    assert!(
        lin.tau[0] < lin.tau[1] && lin.tau[1] < lin.tau[2],
        "lifted cell pseudotime should track the chain: {:?}",
        lin.tau
    );
    // Ambiguity is in [0,1]; a cell right on a landmark is less ambiguous than one
    // wedged between two.
    assert!(lin.ambiguity.iter().all(|&a| (0.0..=1.0).contains(&a)));
}

/// `dense_to_edges` keeps only strictly-positive entries (forward mass); zeros and
/// negatives (a backward-masked learnable `W`) carry no edge.
#[test]
fn dense_to_edges_keeps_forward_mass_only() {
    // 3×3: 0→1 = 0.5, 1→2 = 0.3, 2→0 = −0.2 (dropped), diagonal 0.
    let w = vec![0.0, 0.5, 0.0, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0];
    let edges = dense_to_edges(&w, 3);
    assert_eq!(edges, vec![(0, 1, 0.5), (1, 2, 0.3)]);
}
