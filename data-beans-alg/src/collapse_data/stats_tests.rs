use super::*;
use matrix_param::traits::Inference;
use nalgebra::{DMatrix, DVector};

/// Build a small but non-trivial multi-batch `CollapsedStat`.
fn toy_stat(num_genes: usize, num_samples: usize, num_batches: usize) -> CollapsedStat {
    let mut s = CollapsedStat::new(num_genes, num_samples, num_batches);
    // Deterministic pseudo-random fills; keep everything strictly
    // positive so the Gamma updates are well-defined.
    let f = |a: usize, b: usize| -> f32 { 1.0 + ((a * 7 + b * 13) % 11) as f32 };
    s.observed_sum_ds = DMatrix::from_fn(num_genes, num_samples, &f);
    s.imputed_sum_ds = DMatrix::from_fn(num_genes, num_samples, |g, c| 0.5 * f(g + 1, c + 2));
    s.residual_sum_ds = DMatrix::from_fn(num_genes, num_samples, |g, c| 0.3 * f(g + 2, c + 1));
    s.size_s = DVector::from_fn(num_samples, |c, _| 2.0 + (c % 3) as f32);
    s.observed_sum_db = DMatrix::from_fn(num_genes, num_batches, |g, b| f(g, b) + 0.7);
    s.n_bs = DMatrix::from_fn(num_batches, num_samples, |b, c| 1.0 + ((b + c) % 4) as f32);
    s
}

fn assert_mat_close(a: &DMatrix<f32>, b: &DMatrix<f32>, tag: &str) {
    assert_eq!(a.shape(), b.shape(), "{tag}: shape mismatch");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() <= 1e-5 * (1.0 + x.abs().max(y.abs())),
            "{tag}: {x} vs {y}"
        );
    }
}

/// The whole point of gene-blocking: a per-row-block fit reassembled by
/// `vconcat` must equal the single-shot fit, because every update is
/// separable across feature rows.
#[test]
fn blocked_optimize_matches_single_block() {
    let (g, k, b) = (10usize, 4usize, 2usize);
    let stat = toy_stat(g, k, b);
    let hyper = (1.0, 1.0);
    let iters = 25;

    let full = optimize_block(&stat, hyper, iters, CalibrateTarget::All, None).unwrap();

    // Split rows into uneven blocks and reassemble.
    let ranges = [(0usize, 3usize), (3, 4), (7, 3)];
    let mut mu_obs = Vec::new();
    let mut mu_adj = Vec::new();
    let mut mu_res = Vec::new();
    let mut gam = Vec::new();
    let mut del = Vec::new();
    for (r0, nr) in ranges {
        let sub = stat.select_rows(r0, nr);
        let out = optimize_block(&sub, hyper, iters, CalibrateTarget::All, None).unwrap();
        mu_obs.push(out.mu_observed);
        mu_adj.push(out.mu_adjusted.unwrap());
        mu_res.push(out.mu_residual.unwrap());
        gam.push(out.gamma.unwrap());
        del.push(out.delta.unwrap());
    }
    let blk_obs = GammaMatrix::vconcat(mu_obs, true);
    let blk_adj = GammaMatrix::vconcat(mu_adj, true);
    let blk_res = GammaMatrix::vconcat(mu_res, true);
    let blk_gam = GammaMatrix::vconcat(gam, true);
    let blk_del = GammaMatrix::vconcat(del, true);

    assert_mat_close(
        full.mu_observed.posterior_mean(),
        blk_obs.posterior_mean(),
        "mu_obs mean",
    );
    assert_mat_close(
        full.mu_adjusted.as_ref().unwrap().posterior_mean(),
        blk_adj.posterior_mean(),
        "mu_adj mean",
    );
    assert_mat_close(
        full.mu_residual.as_ref().unwrap().posterior_mean(),
        blk_res.posterior_mean(),
        "mu_resid mean",
    );
    assert_mat_close(
        full.gamma.as_ref().unwrap().posterior_mean(),
        blk_gam.posterior_mean(),
        "gamma mean",
    );
    assert_mat_close(
        full.delta.as_ref().unwrap().posterior_mean(),
        blk_del.posterior_mean(),
        "delta mean",
    );
    // sd / log planes too (All target).
    assert_mat_close(
        full.mu_adjusted.as_ref().unwrap().posterior_log_mean(),
        blk_adj.posterior_log_mean(),
        "mu_adj log_mean",
    );
}

/// MeanOnly drops each mean's per-column prior baseline (unobserved cells
/// → exactly 0), so triplet-ization is sparse; `All` keeps the baseline.
#[test]
fn mean_only_sparsifies_unobserved_cells() {
    let (g, k, b) = (4usize, 3usize, 2usize);
    let mut stat = CollapsedStat::new(g, k, b);
    stat.observed_sum_ds[(0, 0)] = 5.0; // observed support
    stat.observed_sum_ds[(1, 1)] = 3.0;
    stat.imputed_sum_ds[(2, 2)] = 2.0; // imputed-only support
    stat.size_s = DVector::from_element(k, 10.0);
    stat.n_bs = DMatrix::from_element(b, k, 5.0);
    stat.observed_sum_db.fill(1.0); // so δ is well-defined

    let out = optimize_block(&stat, (1.0, 1.0), 10, CalibrateTarget::MeanOnly, None).unwrap();
    let adj = out.mu_adjusted.unwrap();
    let m = adj.posterior_mean();
    // support of mu_adjusted = (observed ∪ imputed) > 0
    assert!(m[(0, 0)] > 0.0);
    assert!(m[(1, 1)] > 0.0);
    assert!(m[(2, 2)] > 0.0);
    // unobserved & unimputed cells → exactly 0 (baseline dropped)
    assert_eq!(m[(3, 0)], 0.0);
    assert_eq!(m[(0, 1)], 0.0);

    // All keeps the prior baseline (nonzero everywhere).
    let out_all = optimize_block(&stat, (1.0, 1.0), 10, CalibrateTarget::All, None).unwrap();
    let ma = out_all.mu_adjusted.unwrap();
    assert!(
        ma.posterior_mean()[(3, 0)] > 0.0,
        "All path must keep the prior baseline"
    );
}

/// MeanOnly: `vconcat(.., false)` keeps the assembled means but drops the
/// sufficient-stat planes, and the means still match the calibrated fit.
#[test]
fn mean_only_vconcat_drops_stats_keeps_means() {
    let (g, k, b) = (9usize, 3usize, 2usize);
    let stat = toy_stat(g, k, b);
    let hyper = (1.0, 1.0);
    let iters = 20;

    let reference = optimize_block(&stat, hyper, iters, CalibrateTarget::All, None).unwrap();

    let mut blocks = Vec::new();
    for (r0, nr) in [(0usize, 5usize), (5, 4)] {
        let sub = stat.select_rows(r0, nr);
        let mut out = optimize_block(&sub, hyper, iters, CalibrateTarget::MeanOnly, None).unwrap();
        out.release_stats();
        blocks.push(out.mu_observed);
    }
    let assembled = GammaMatrix::vconcat(blocks, false);

    // means equal the All-target reference …
    assert_mat_close(
        reference.mu_observed.posterior_mean(),
        assembled.posterior_mean(),
        "mu_obs mean (mean-only)",
    );
    // … but the stat planes were dropped (empty), proving the memory win.
    assert_eq!(
        assembled.posterior_sd().nrows(),
        0,
        "sd should be empty under MeanOnly"
    );
}
