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
    run_with(pos, seed, |c| c)
}

fn run_with(
    pos: &[Vec<(u32, f32)>],
    seed: u64,
    tweak: impl FnOnce(DimBlockConfig) -> DimBlockConfig,
) -> DimBlockResult {
    let (e, b) = side_buffers();
    let side = FrozenSide { e: &e, b: &b, h: H };
    let all: Vec<u32> = (0..N_CELL as u32).collect();
    let nodes: Vec<NodeTerm> = pos.iter().map(|p| NodeTerm::new(p, &all, 1.0)).collect();
    let mut cfg = DimBlockConfig::new(60, 20, seed);
    cfg.transitions_per_dim = 1;
    dim_block(&nodes, &side, &tweak(cfg))
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
/// prior alone and its loading must not drift away from the slab mean. Guards against
/// an empty anchor being reported as a MEASUREMENT — over half the anchors on a real
/// annotation have no counts.
///
/// Asserted against the sampled per-dim prior (`1 − π₀ₕ`) rather than a fixed
/// threshold, which is what "reverts to the prior" actually means and is the only form
/// that survives a change of prior. The previous version asserted `PIP < 0.5` on every
/// dim — true under an unordered `Beta(9,1)`, where every dim sits near 0.1, but false
/// under stick-breaking, where the LEADING dim's rate is `α/(α+1)` before the data
/// speaks. That is a real consequence, not a test artifact: an empty anchor now tracks
/// the population inclusion rate of each dim, which on dim 0 is a coin flip.
#[test]
fn an_empty_gene_reverts_to_the_prior() {
    let pos = vec![Vec::new(), vec![(2u32, 5.0f32), (9, 4.0)]];
    let res = run(&pos, 37);

    for (d, &p) in res.pip_row(0).iter().enumerate() {
        let prior_incl = 1.0 - res.pi0[d] as f32;
        assert!(
            (p - prior_incl).abs() < 0.25,
            "empty gene dim {d}: PIP {p} must track the prior {prior_incl}, not the data"
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

/// REGRESSION (review finding): a block driven one sweep at a time must resume
/// from the previous sweep's `z`, or the `z[d]` branch never runs, every β is
/// redrawn from its prior, and `init_beta` is silently discarded.
///
/// The check is that the OUTPUT depends on the warm start at all. With the bug,
/// two very different `init_beta` values fed through the same seed produced
/// byte-identical results, because both were overwritten by the same prior draw.
#[test]
fn a_resumed_block_actually_reads_its_warm_start() {
    let pos = vec![
        (0..N_CELL as u32).map(|c| (c, 2.0f32)).collect::<Vec<_>>(),
        vec![(1u32, 5.0f32), (6, 2.0)],
    ];
    let on = vec![true; pos.len() * H];
    let run_warm = |scale: f32| {
        let init: Vec<f32> = (0..pos.len() * H)
            .map(|i| scale * (1.0 + i as f32))
            .collect();
        run_with(&pos, 404, |c| {
            c.with_init_beta(init.clone()).with_init_z(on.clone())
        })
    };

    let small = run_warm(0.01);
    let large = run_warm(3.0);
    assert_ne!(
        small.mean_beta, large.mean_beta,
        "warm start is being discarded — the loadings do not depend on init_beta"
    );
}

/// With selection off every coordinate stays included, so no PIP is below 1 and
/// nothing is written back as an exact zero. The old `beta_b = 1e12` trick could
/// not achieve this: the prior odds floor at ~27.6 nats, which a real likelihood
/// difference swamps.
#[test]
fn selection_off_keeps_every_coordinate_included() {
    let pos = vec![
        (0..N_CELL as u32).map(|c| (c, 4.0f32)).collect::<Vec<_>>(),
        vec![(2u32, 9.0f32), (7, 1.0)],
        Vec::new(),
    ];
    let res = run_with(&pos, 12, |c| c.without_selection());
    assert!(
        res.pip.iter().all(|&p| (p - 1.0).abs() < 1e-6),
        "every coordinate must stay included; min PIP was {:?}",
        res.pip.iter().cloned().fold(f32::INFINITY, f32::min)
    );
    assert!(res.final_z.iter().all(|&z| z));
}

/// A vetoed coordinate was never eligible, so it must not be counted as an
/// observed null in the per-dim sparsity reduction — otherwise the nesting
/// constraint is reported back as if the data had produced it.
#[test]
fn vetoed_coordinates_do_not_inflate_pi0() {
    let pos: Vec<Vec<(u32, f32)>> = (0..40)
        .map(|g| {
            vec![
                ((g % N_CELL) as u32, 6.0f32),
                (((g + 5) % N_CELL) as u32, 3.0),
            ]
        })
        .collect();
    // Dim 0 is allowed for every anchor; dim 1 is vetoed for 90% of them.
    let mut allowed = vec![true; pos.len() * H];
    for (g, a) in allowed.chunks_exact_mut(H).enumerate() {
        if g % 10 != 0 {
            a[1] = false;
        }
    }
    let res = run_with(&pos, 31, |c| c.with_z_allowed(allowed.clone()));

    // If vetoed coordinates were folded in as nulls, pi0 on dim 1 would be pinned
    // near 0.9 by construction regardless of the data.
    assert!(
        res.pi0[1] < 0.88,
        "pi0 on the vetoed dim reads {:.3} — the constraint is masquerading as data",
        res.pi0[1]
    );
}

/// An alternating driver calls a block one sweep at a time, so the slab variance
/// only becomes a chain if the driver carries it. Without the carry every entry
/// restarts `σ₀²` at `half_cauchy_scale²`, the `β` draw always uses the PRIOR
/// width, and the value sampled at the end of the sweep is thrown away.
///
/// The data here has loadings far larger than the prior scale, so an adapting
/// slab has to grow. Break the fix — drop `with_init_hyper` from the loop — and
/// the carried run collapses onto the pinned one and this fails.
#[test]
fn slab_variance_only_chains_across_blocks_when_it_is_carried() {
    let h = 2;
    let n_anchors = 24;
    let n_obs = 40;
    // Frozen side with a big signal on both dims, so `Σβ²` pulls σ₀² well above 1.
    let e: Vec<f32> = (0..n_obs * h)
        .map(|i| if (i / h) % 2 == 0 { 3.0 } else { -3.0 })
        .collect();
    let b = vec![0.0f32; n_obs];
    let side = FrozenSide { e: &e, b: &b, h };
    let partition: Vec<u32> = (0..n_obs as u32).collect();
    let pos: Vec<Vec<(u32, f32)>> = (0..n_anchors)
        .map(|a| {
            (0..n_obs as u32)
                .map(|o| {
                    (
                        o,
                        if (o as usize + a).is_multiple_of(2) {
                            40.0
                        } else {
                            1.0
                        },
                    )
                })
                .collect()
        })
        .collect();
    let nodes: Vec<NodeTerm> = pos
        .iter()
        .map(|p| NodeTerm::new(p, &partition, 1.0))
        .collect();

    let run = |carry: bool| -> Vec<f64> {
        let mut hyper: Option<HyperState> = None;
        let mut last = Vec::new();
        for sweep in 0..24u64 {
            let mut cfg = DimBlockConfig::new(1, 0, 1234 + sweep).quiet();
            if carry {
                if let Some(hs) = hyper.take() {
                    cfg = cfg.with_init_hyper(hs);
                }
            }
            let out = dim_block(&nodes, &side, &cfg);
            hyper = Some(out.final_hyper.clone());
            last = out.final_hyper.sigma2.clone();
        }
        last
    };

    let carried = run(true);
    let pinned = run(false);

    // The pinned run re-enters at the prior every sweep, so whatever it reports is
    // a one-shot draw off a state generated under σ = 1.
    let carried_max = carried.iter().cloned().fold(0.0f64, f64::max);
    let pinned_max = pinned.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        carried_max > pinned_max,
        "carried σ₀² should adapt upward on strong loadings: carried {carried:?} vs pinned {pinned:?}"
    );
}

/// A block that owns all its sweeps must be untouched by the new plumbing: with
/// no `init_hyper` the result has to be exactly what it always was.
#[test]
fn omitting_the_hyper_state_reproduces_the_old_behaviour() {
    let h = 2;
    let e: Vec<f32> = (0..20 * h).map(|i| (i % 5) as f32 - 2.0).collect();
    let b = vec![0.0f32; 20];
    let side = FrozenSide { e: &e, b: &b, h };
    let partition: Vec<u32> = (0..20u32).collect();
    let pos: Vec<Vec<(u32, f32)>> = (0..8)
        .map(|a| {
            (0..20u32)
                .map(|o| (o, 1.0 + ((o + a) % 3) as f32))
                .collect()
        })
        .collect();
    let nodes: Vec<NodeTerm> = pos
        .iter()
        .map(|p| NodeTerm::new(p, &partition, 1.0))
        .collect();

    let cfg = DimBlockConfig::new(12, 4, 99).quiet();
    let a = dim_block(&nodes, &side, &cfg);
    let b2 = dim_block(&nodes, &side, &cfg);
    assert_eq!(a.pip, b2.pip, "same config, same seed, same answer");
    assert_eq!(a.final_hyper.sigma2.len(), h);
    assert_eq!(a.final_hyper.vars.len(), h);
    assert_eq!(a.final_hyper.pi0.len(), h);
}
