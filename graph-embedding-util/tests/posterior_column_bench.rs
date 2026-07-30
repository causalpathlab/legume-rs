//! Portable timing / distribution harness for the per-dim block sampler.
//!
//! Deliberately written against the PUBLIC api only (`dim_block_multi`,
//! `DimBlockConfig`, `FrozenSide`, `NodeTerm`) and touching no field added by the
//! column rewrite, so the same file can be dropped into an older checkout and run
//! there. That is what makes a before/after ratio measurable at all rather than
//! estimated.
//!
//! Ignored by default — it is a measurement, so it would be flaky as an assertion and
//! slow as a unit test. Run it deliberately:
//!
//! ```text
//! GEU_BENCH_SEED=1 cargo test -p graph-embedding-util --release \
//!   --test posterior_column_bench -- --ignored --nocapture
//! ```

use graph_embedding_util::posterior::dim_block::{dim_block_multi, DimBlockConfig};
use graph_embedding_util::posterior::lnpdf::{FrozenSide, NodeTerm};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

/// Phase-1 shapes, scaled down so a run finishes in seconds while keeping the ratios
/// that matter: `h` well past 1, a slate at the production default, and an anchor count
/// large enough to span many tiles.
const H: usize = 32;
const K: usize = 1024;
const N_ANCHORS: usize = 4000;
const N_POS: usize = 120;
const SWEEPS: usize = 12;
const BURNIN: usize = 4;

fn env_seed() -> u64 {
    std::env::var("GEU_BENCH_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

#[test]
#[ignore = "timing measurement; run explicitly with --ignored --nocapture"]
fn column_pass_bench() {
    let seed = env_seed();
    let mut rng = SmallRng::seed_from_u64(seed ^ 0xBEEF);

    // Frozen side: `K` rows at `H` dims, small enough that no score saturates.
    let mut e = vec![0.0f32; K * H];
    for v in &mut e {
        *v = (rng.random::<f32>() - 0.5) * 0.5;
    }
    let b: Vec<f32> = (0..K).map(|_| (rng.random::<f32>() - 0.5) * 0.3).collect();
    let side = FrozenSide { e: &e, b: &b, h: H };

    // Per-anchor edge lists over the slate, with a realistic spread of totals.
    let pos: Vec<Vec<(u32, f32)>> = (0..N_ANCHORS)
        .map(|_| {
            let n = 1 + rng.random_range(0..N_POS);
            (0..n)
                .map(|_| (rng.random_range(0..K as u32), 1.0 + rng.random_range(0..5) as f32))
                .collect()
        })
        .collect();
    let partition: Vec<u32> = (0..K as u32).collect();
    let anchors: Vec<Vec<NodeTerm>> = pos
        .iter()
        .map(|p| vec![NodeTerm::new(p, &partition, 1.0)])
        .collect();

    let mut cfg = DimBlockConfig::new(SWEEPS, BURNIN, seed);
    cfg.transitions_per_dim = 1;
    cfg.show_progress = false;

    let t0 = Instant::now();
    let res = dim_block_multi(&anchors, &side, &cfg);
    let dt = t0.elapsed();

    let units = SWEEPS * N_ANCHORS * H;
    let mean_pip: f64 = res.pip.iter().map(|&p| f64::from(p)).sum::<f64>() / res.pip.len() as f64;
    println!(
        "column bench seed={seed} h={H} slate={K} anchors={N_ANCHORS} sweeps={SWEEPS}\n  \
         wall {dt:?}  ({:.1} ns per anchor-dim-sweep)\n  \
         mean pip {mean_pip:.4}  n_kept {}\n  \
         sigma2 {:?}\n  pi0 {:?}",
        dt.as_nanos() as f64 / units as f64,
        res.n_kept,
        res.sigma2.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        res.pi0.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
    );

    // Not an assertion about speed — just that the run produced a usable posterior, so
    // a timing number is not being read off a degenerate chain.
    assert!(res.n_kept > 0, "nothing retained");
    assert!(
        res.pip.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
        "PIP table is not a probability table"
    );
    assert!(
        res.sigma2.iter().all(|s| s.is_finite() && *s > 0.0),
        "slab variances degenerate: {:?}",
        res.sigma2
    );
}
