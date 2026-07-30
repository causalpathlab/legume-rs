//! What one `faba gem --posterior` sweep costs, at a shape taken from a real run.
//!
//! [`posterior_column_bench`] measures the column pass in the abstract. This one asks a
//! narrower question: given the shapes a gem posterior run actually reports, how long is
//! one outer sweep, and therefore how long is the whole request? gem's sweep is three
//! blocks, and the gene side of it carries THREE term-passes, not one:
//!
//! ```text
//!   beta | delta, pb   n_genes anchors, 2 terms (spliced + unspliced)
//!   delta | beta       n_genes anchors, 1 term  (unspliced)
//!   pb | beta, delta   n_pb    anchors, 1 term  (over the ROW axis)
//! ```
//!
//! `senna bge` runs two blocks with one term each, so at equal `h`, gene count and slate
//! the gene side of gem is ~3x bge's. That factor is structural — it is what sampling a
//! second gate over a second track costs — so it belongs in a measurement, not in a
//! guess about why a run is slow.
//!
//! Everything is env-overridable so a shape can be dialled in without a recompile:
//!
//! ```text
//! GEU_BENCH_H=128 GEU_BENCH_K=1460 GEU_BENCH_ANCHORS=4000 GEU_BENCH_TERMS=2 \
//!   cargo test -p graph-embedding-util --release \
//!   --test posterior_gem_sweep_bench -- --ignored --nocapture
//! ```
//!
//! Cost is linear in the anchor count (anchors are conditionally independent given the
//! frozen side, which is the whole basis of the column batching), so a run at a reduced
//! `GEU_BENCH_ANCHORS` extrapolates to the real one by ratio. The harness prints that
//! extrapolation for `GEU_BENCH_REAL_ANCHORS` so the arithmetic is not left to the reader.

use graph_embedding_util::posterior::dim_block::{dim_block_multi, DimBlockConfig};
use graph_embedding_util::posterior::lnpdf::{FrozenSide, NodeTerm};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "timing measurement; run explicitly with --ignored --nocapture"]
fn gem_sweep_bench() {
    let h = env_usize("GEU_BENCH_H", 128);
    let k = env_usize("GEU_BENCH_K", 1460);
    let n_anchors = env_usize("GEU_BENCH_ANCHORS", 4000);
    let n_terms = env_usize("GEU_BENCH_TERMS", 2);
    // Mean positives per anchor. On the reported run it is 38.5M edges / 34179 genes
    // ~= 1128 over a 1460-wide pb axis: pseudobulk columns are dense, not sparse.
    let n_pos = env_usize("GEU_BENCH_POS", 1128);
    let real_anchors = env_usize("GEU_BENCH_REAL_ANCHORS", 34179);
    let real_sweeps = env_usize("GEU_BENCH_REAL_SWEEPS", 750);
    let seed = env_usize("GEU_BENCH_SEED", 1) as u64;

    let mut rng = SmallRng::seed_from_u64(seed ^ 0xBEEF);
    let mut e = vec![0.0f32; k * h];
    for v in &mut e {
        *v = (rng.random::<f32>() - 0.5) * 0.5;
    }
    let b: Vec<f32> = (0..k).map(|_| (rng.random::<f32>() - 0.5) * 0.3).collect();
    let side = FrozenSide { e: &e, b: &b, h };

    let pos: Vec<Vec<(u32, f32)>> = (0..n_anchors)
        .map(|_| {
            let n = 1 + rng.random_range(0..n_pos.max(2));
            (0..n)
                .map(|_| {
                    (
                        rng.random_range(0..k as u32),
                        1.0 + rng.random_range(0..5) as f32,
                    )
                })
                .collect()
        })
        .collect();
    let partition: Vec<u32> = (0..k as u32).collect();
    let anchors: Vec<Vec<NodeTerm>> = pos
        .iter()
        .map(|p| vec![NodeTerm::new(p, &partition, 1.0); n_terms.max(1)])
        .collect();

    // ONE inner sweep, no burn-in: an outer gem sweep runs each block for exactly one.
    let mut cfg = DimBlockConfig::new(1, 0, seed);
    cfg.transitions_per_dim = 1;
    cfg.show_progress = false;

    let t0 = Instant::now();
    let res = dim_block_multi(&anchors, &side, &cfg);
    let dt = t0.elapsed();
    assert!(
        res.n_kept > 0,
        "nothing retained — timing a degenerate chain"
    );

    let secs = dt.as_secs_f64();
    let per_anchor = secs / n_anchors as f64;
    // The gene side of a gem sweep = this block (2 terms) + the delta block (1 term).
    // Scale by term count so one measurement covers both.
    let gene_side = per_anchor * real_anchors as f64 * (3.0 / n_terms.max(1) as f64);
    println!(
        "gem sweep bench  h={h} slate={k} anchors={n_anchors} terms={n_terms} pos~{n_pos}\n  \
         block wall {dt:?}  ({:.1} ns per anchor-dim, {:.2} ms per 1000 anchors)\n  \
         --> extrapolated to {real_anchors} genes:\n      \
         this block {:.1} s;  full gene side (beta 2-term + delta 1-term) {:.1} s/sweep\n      \
         {real_sweeps} sweeps = {:.2} h  (gene side only; the pb block adds to this)\n  \
         mean pip {:.4}",
        dt.as_nanos() as f64 / (n_anchors * h) as f64,
        secs * 1e3 / (n_anchors as f64 / 1000.0),
        per_anchor * real_anchors as f64,
        gene_side,
        gene_side * real_sweeps as f64 / 3600.0,
        res.pip.iter().map(|&p| f64::from(p)).sum::<f64>() / res.pip.len() as f64,
    );
}
