//! Cheap structural checks on the per-dim block Gibbs. The expensive claim — that
//! a MULTI-DIM planted truth is recovered on dims the single-effect gate has to
//! trade off against each other — is an integration test, in
//! `tests/posterior_dim_block.rs`, so it can run in release.

use super::*;

const H: usize = 4;
const N_CELL: usize = 24;

/// A frozen side with deterministic, non-degenerate rows.
fn side_buffers() -> (Vec<f32>, Vec<f32>) {
    let mut e = vec![0.0f32; N_CELL * H];
    for c in 0..N_CELL {
        for d in 0..H {
            // Coprime strides so no two dims come out collinear.
            e[c * H + d] = ((c * (d + 2) + d) % 7) as f32 * 0.25 - 0.75;
        }
    }
    let b = vec![-0.4f32; N_CELL];
    (e, b)
}

fn run(pos: &[Vec<(u32, f32)>], seed: u64) -> DimBlockResult {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let all: Vec<u32> = (0..N_CELL as u32).collect();
    let nodes: Vec<NodeTerm> = pos.iter().map(|p| NodeTerm::new(p, &all, 1.0)).collect();
    let mut cfg = DimBlockConfig::new(60, 20, seed);
    cfg.transitions_per_dim = 1;
    dim_block(&nodes, &side, &cfg)
}

#[test]
fn tables_are_gene_major_and_fully_populated() {
    let pos = vec![vec![(0u32, 3.0f32), (5, 2.0)], vec![(1, 4.0), (7, 1.0)]];
    let res = run(&pos, 11);

    assert_eq!(res.h, H);
    assert_eq!(res.pip.len(), pos.len() * H);
    assert_eq!(res.mean_beta.len(), pos.len() * H);
    assert_eq!(res.n_kept, 40, "60 sweeps minus 20 burn-in");
    // Per-dim hypers and diagnostics are one per DIM, not one global scalar —
    // the whole point of Tier 2.
    assert_eq!(res.sigma2.len(), H);
    assert_eq!(res.pi0.len(), H);
    assert_eq!(res.sigma_diag.len(), H);
    assert_eq!(res.pi0_diag.len(), H);

    for g in 0..pos.len() {
        assert_eq!(res.pip_row(g).len(), H);
        assert_eq!(res.beta_row(g).len(), H);
        // The row accessor must agree with the flat gene-major layout.
        assert_eq!(res.pip_row(g), &res.pip[g * H..(g + 1) * H]);
    }
    assert!(
        res.pip.iter().all(|p| (0.0..=1.0).contains(p)),
        "every PIP is a probability"
    );
    assert!(
        res.mean_beta.iter().all(|b| b.is_finite()),
        "no non-finite loading"
    );
    assert!(
        res.sigma2.iter().all(|s| s.is_finite() && *s > 0.0),
        "every per-dim slab variance is a positive, finite scale: {:?}",
        res.sigma2
    );
}

/// A gene with counts picks up inclusion mass. Deliberately NOT asserting
/// `row_sum > 1` here: on this tiny fixture that is not guaranteed, and the real
/// "a row is not a simplex" claim needs a planted multi-dim truth — it is asserted
/// where it belongs, in `tests/posterior_dim_block.rs`. An earlier version of this
/// test asserted `row_sum <= H`, which is implied by every PIP being a probability
/// and so could never fail, including for a genuine simplex.
#[test]
fn a_gene_with_counts_gains_inclusion_mass() {
    // Counts spread over every cell, so several dims can plausibly be on.
    let pos = vec![(0..N_CELL as u32).map(|c| (c, 2.0f32)).collect::<Vec<_>>()];
    let res = run(&pos, 23);
    let row_sum: f32 = res.pip_row(0).iter().sum();
    assert!(
        row_sum > 0.0,
        "a gene with counts must have some inclusion mass, got {row_sum}"
    );
}

/// A gene with no counts has a flat likelihood, so its inclusion is decided by the
/// prior alone and its loading must not drift away from the slab mean. Guards
/// against an empty anchor being reported as a confident selection — over half the
/// anchors on a real annotation have no counts.
#[test]
fn an_empty_gene_reverts_to_the_prior() {
    let pos = vec![Vec::new(), vec![(2u32, 5.0f32), (9, 4.0)]];
    let res = run(&pos, 37);

    for (d, &p) in res.pip_row(0).iter().enumerate() {
        assert!(
            p < 0.5,
            "empty gene dim {d} must fall to the null-biased prior, got PIP {p}"
        );
    }
    assert!(
        res.beta_row(0).iter().all(|b| b.is_finite()),
        "an empty gene's loading stays finite"
    );
}

/// Same seed ⇒ same answer. The per-gene streams are derived from `(seed, gene,
/// sweep)` with no shared mutable state, so a rayon reschedule must not change the
/// result — otherwise no A/B against this sampler is trustworthy.
#[test]
fn the_same_seed_reproduces_the_run() {
    let pos = vec![
        vec![(0u32, 3.0f32), (4, 2.0), (11, 1.0)],
        vec![(1, 5.0), (6, 2.0)],
        Vec::new(),
    ];
    let a = run(&pos, 101);
    let b = run(&pos, 101);
    assert_eq!(a.pip, b.pip, "PIP table is seed-reproducible");
    assert_eq!(
        a.mean_beta, b.mean_beta,
        "loading table is seed-reproducible"
    );
    assert_eq!(a.pi0, b.pi0, "per-dim sparsity is seed-reproducible");
}
