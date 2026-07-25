//! Diagnostics behave on well-mixed vs stalled chains.

use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// An i.i.d. chain mixes: high ESS, no stuck steps.
#[test]
fn well_mixed_chain_reads_healthy() {
    let mut rng = StdRng::seed_from_u64(1);
    let (t, d) = (1000usize, 3usize);
    let draws: Vec<Vec<f32>> = (0..t)
        .map(|_| {
            (0..d)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect()
        })
        .collect();
    let diag = chain_diagnostics(&draws);
    assert_eq!(diag.stuck_fraction, 0.0, "i.i.d. draws are never identical");
    assert!(
        diag.min_ess > 0.5 * t as f32,
        "i.i.d. ESS should be a large fraction of T, got {}",
        diag.min_ess
    );
}

/// A stalled chain (fallback to current every step) is caught by stuck_fraction.
#[test]
fn stalled_chain_is_visible() {
    let frozen = vec![0.3f32, -1.2, 0.7];
    let draws: Vec<Vec<f32>> = (0..500).map(|_| frozen.clone()).collect();
    let diag = chain_diagnostics(&draws);
    assert_eq!(
        diag.stuck_fraction, 1.0,
        "every adjacent pair identical ⇒ stuck_fraction == 1"
    );
}

/// `worst_case` surfaces the stalled node even amid healthy ones.
#[test]
fn worst_case_surfaces_the_bad_node() {
    let healthy = ChainDiag {
        min_ess: 800.0,
        stuck_fraction: 0.0,
    };
    let bad = ChainDiag {
        min_ess: 3.0,
        stuck_fraction: 0.9,
    };
    let w = worst_case(&[healthy, bad, healthy]);
    assert_eq!(w.min_ess, 3.0);
    assert_eq!(w.stuck_fraction, 0.9);
}

/// A too-short chain returns a well-formed (non-panicking) summary.
#[test]
fn short_chain_is_well_formed() {
    let diag = chain_diagnostics(&[vec![1.0, 2.0]]);
    assert_eq!(diag.min_ess, 1.0);
    assert_eq!(diag.stuck_fraction, 0.0);
}
