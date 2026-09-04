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
    s.size_s = DVector::from_fn(num_samples, |c, _| 2.0 + (c % 3) as f32);
    s.observed_sum_db = DMatrix::from_fn(num_genes, num_batches, |g, b| f(g, b) + 0.7);
    s.n_bs = DMatrix::from_fn(num_batches, num_samples, |b, c| 1.0 + ((b + c) % 4) as f32);
    s.matched_bs = DMatrix::from_fn(num_batches, num_samples, |b, c| {
        0.5 + ((b + 2 * c) % 3) as f32
    });
    s
}

/// Two batch-pure pseudobulks of ONE cell type: pb 0 in batch 0 (frame `a`),
/// pb 1 in batch 1 (frame `b`), each matched to the other. `n` cells each, so the
/// unit priors are negligible.
fn two_pure_pbs(a: &[f32], b: &[f32], n: f32) -> CollapsedStat {
    let g = a.len();
    let mut s = CollapsedStat::new(g, 2, 2);
    for i in 0..g {
        s.observed_sum_ds[(i, 0)] = a[i] * n;
        s.imputed_sum_ds[(i, 0)] = b[i] * n;
        s.observed_sum_ds[(i, 1)] = b[i] * n;
        s.imputed_sum_ds[(i, 1)] = a[i] * n;
        s.observed_sum_db[(i, 0)] = a[i] * n;
        s.observed_sum_db[(i, 1)] = b[i] * n;
    }
    s.size_s = DVector::from_element(2, n);
    s.n_bs = DMatrix::from_row_slice(2, 2, &[n, 0.0, 0.0, n]);
    // pb 0's counterfactual comes from batch 1 and vice versa
    s.matched_bs = DMatrix::from_row_slice(2, 2, &[0.0, n, n, 0.0]);
    s
}

fn assert_rel(got: f32, want: f32, tol: f32, tag: &str) {
    assert!(
        (got / want - 1.0).abs() < tol,
        "{tag}: got {got}, want {want} (rel tol {tol})"
    );
}

/// Pooled adjustment: both pseudobulks land in ONE frame — the per-gene
/// geometric mean of the two batch frames — with `δ` carrying the whole
/// batch ratio, normalised to geometric mean 1 across batches. The per-pb
/// outputs keep their documented meaning: `mu_residual` is the own-batch fold
/// `E[y]/E[μ]`, `gamma` the counterfactual's `E[ŷ]/E[μ]`.
#[test]
#[allow(clippy::needless_range_loop)] // `g` indexes four parallel tables
fn pooled_two_pure_batches_share_one_frame() {
    let a = [8.0f32, 2.0, 12.0, 1.0];
    let b = [2.0f32, 8.0, 3.0, 4.0];
    let stat = two_pure_pbs(&a, &b, 2000.0);
    let out = optimize_block(&stat, (1.0, 1.0), 60, CalibrateTarget::All, None).unwrap();
    let mu = out.mu_adjusted.as_ref().unwrap().posterior_mean();
    let delta = out.delta.as_ref().unwrap().posterior_mean();
    let resid = out.mu_residual.as_ref().unwrap().posterior_mean();
    let gamma = out.gamma.as_ref().unwrap().posterior_mean();
    for g in 0..a.len() {
        let common = (a[g] * b[g]).sqrt();
        assert_rel(mu[(g, 0)], common, 0.02, &format!("gene {g} mu pb0"));
        assert_rel(mu[(g, 1)], common, 0.02, &format!("gene {g} mu pb1"));
        assert_rel(
            delta[(g, 0)] / delta[(g, 1)],
            a[g] / b[g],
            0.02,
            &format!("gene {g} delta ratio"),
        );
        assert_rel(
            delta[(g, 0)] * delta[(g, 1)],
            1.0,
            0.02,
            &format!("gene {g} delta geometric mean"),
        );
        assert_rel(
            resid[(g, 0)],
            delta[(g, 0)],
            0.02,
            &format!("gene {g} mu_residual pb0 = own delta"),
        );
        assert_rel(
            resid[(g, 1)],
            delta[(g, 1)],
            0.02,
            &format!("gene {g} mu_residual pb1 = own delta"),
        );
        assert_rel(
            gamma[(g, 0)],
            delta[(g, 1)],
            0.02,
            &format!("gene {g} gamma pb0 = source delta"),
        );
    }
}

/// Anchored: the anchor batch's frame is the frame. Its pseudobulk self-matches
/// (counterfactual = its own profile), the other batch is matched to it. Then
/// `δ_anchor = 1`, both `μ` equal the anchor profile, and the new batch's `δ` is
/// its fold relative to the anchor.
#[test]
fn anchored_frame_is_the_anchor_batch() {
    let a = [8.0f32, 2.0, 12.0, 1.0];
    let b = [2.0f32, 8.0, 3.0, 4.0];
    let n = 2000.0;
    let mut stat = two_pure_pbs(&a, &b, n);
    // pb 1 (batch 1) is the anchor: it self-matches.
    for (g, &bg) in b.iter().enumerate() {
        stat.imputed_sum_ds[(g, 1)] = bg * n;
    }
    stat.matched_bs = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, n, n]);
    stat.anchor_batches = vec![1];
    let out = optimize_block(&stat, (1.0, 1.0), 60, CalibrateTarget::All, None).unwrap();
    let mu = out.mu_adjusted.as_ref().unwrap().posterior_mean();
    let delta = out.delta.as_ref().unwrap().posterior_mean();
    for g in 0..a.len() {
        assert_rel(delta[(g, 1)], 1.0, 0.02, &format!("gene {g} anchor delta"));
        assert_rel(
            delta[(g, 0)],
            a[g] / b[g],
            0.03,
            &format!("gene {g} new-batch delta"),
        );
        assert_rel(
            mu[(g, 0)],
            b[g],
            0.03,
            &format!("gene {g} mu pb0 in the anchor frame"),
        );
        assert_rel(mu[(g, 1)], b[g], 0.02, &format!("gene {g} mu pb1"));
    }
}

/// Coarsening adds the counterfactual source mass like every other per-sample
/// statistic, and carries the anchor set.
#[test]
fn merge_stat_sums_matched_mass_and_keeps_anchors() {
    let mut fine = toy_stat(3, 4, 2);
    fine.anchor_batches = vec![0];
    let coarse = merge_stat(&fine, &[0, 1, 0, 1], 2);
    for b in 0..2 {
        assert_eq!(
            coarse.matched_bs[(b, 0)],
            fine.matched_bs[(b, 0)] + fine.matched_bs[(b, 2)]
        );
        assert_eq!(
            coarse.matched_bs[(b, 1)],
            fine.matched_bs[(b, 1)] + fine.matched_bs[(b, 3)]
        );
    }
    assert_eq!(coarse.anchor_batches, vec![0]);
    let sub = fine.select_rows(1, 2);
    assert_eq!(sub.matched_bs, fine.matched_bs);
    assert_eq!(sub.anchor_batches, vec![0]);
    let cols = fine.select_columns(&[3, 1]);
    assert_eq!(cols.matched_bs.column(0), fine.matched_bs.column(3));
    assert_eq!(cols.anchor_batches, vec![0]);
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
        out.retain(StatRetention::None);
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

/////////////////////////////////////////////////
// Shape-only retention for posterior jitter  //
/////////////////////////////////////////////////

/// Under `StatRetention::Shape` the two mean parameters keep enough to be
/// resampled — and draw exactly what a fully retained fit draws at the same
/// seed — while the residual, `γ` and `δ` planes nothing resamples are dropped.
/// Under `None` nothing is resamplable at all.
#[test]
fn shape_retention_keeps_the_draw_and_drops_the_rest() {
    let stat = toy_stat(12, 5, 2);
    let hyper = (1.0, 1.0);
    let full = optimize_with(
        &stat,
        hyper,
        20,
        "full",
        CalibrateTarget::MeanOnly,
        StatRetention::Full,
    )
    .unwrap();
    let shape = optimize_with(
        &stat,
        hyper,
        20,
        "shape",
        CalibrateTarget::MeanOnly,
        StatRetention::Shape,
    )
    .unwrap();
    let none = optimize_with(
        &stat,
        hyper,
        20,
        "none",
        CalibrateTarget::MeanOnly,
        StatRetention::None,
    )
    .unwrap();

    assert!(shape.mu_observed.has_shape_stat());
    assert!(shape.mu_adjusted.as_ref().unwrap().has_shape_stat());
    for p in [&shape.mu_residual, &shape.gamma, &shape.delta] {
        assert!(
            !p.as_ref().unwrap().has_shape_stat(),
            "only the mean parameters are resampled; the rest must be released"
        );
    }
    assert!(!none.mu_observed.has_shape_stat());
    assert!(none.mu_observed.posterior_sample_seeded(1).is_err());

    let a = full.mu_observed.posterior_sample_seeded(3).unwrap();
    let b = shape.mu_observed.posterior_sample_seeded(3).unwrap();
    let rel = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / x.abs().max(1e-6))
        .fold(0.0f32, f32::max);
    assert!(
        rel < 1e-5,
        "shape-only draw must match the full draw: {rel}"
    );
    let a = full
        .mu_adjusted
        .as_ref()
        .unwrap()
        .posterior_sample_seeded(4)
        .unwrap();
    let b = shape
        .mu_adjusted
        .as_ref()
        .unwrap()
        .posterior_sample_seeded(4)
        .unwrap();
    assert_mat_close(&a, &b, "mu_adjusted shape-only draw");
}
