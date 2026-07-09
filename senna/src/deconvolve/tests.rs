//! Deterministic recovery tests for the deconvolution Gibbs sampler.

use super::anchors::AnchorPrior;
use super::args::SamplerConfig;
use super::gibbs::run_gibbs;
use super::source::EmbeddingSource;
use crate::embed_common::Mat;
use crate::run_manifest::RunKind;

/// Build a synthetic problem: known ρ, anchors, and abundances w*, with bulk
/// counts = Σ_c w*_c μ_{g,c} at large depth (near-noiseless).
fn synthetic() -> (EmbeddingSource, AnchorPrior, Mat, Mat) {
    let (d, h, c, s) = (48usize, 4usize, 3usize, 5usize);

    // ρ: deterministic pseudo-random rows.
    let rho = Mat::from_fn(d, h, |g, j| (((g * 7 + j * 13) % 11) as f32 / 11.0) - 0.5);
    let gene_offset =
        crate::embed_common::DVec::from_fn(d, |g, _| (((g * 5) % 7) as f32 / 7.0) - 0.3);

    // Well-separated anchors.
    let anchors = Mat::from_fn(c, h, |ct, j| if j == ct { 0.9 } else { -0.2 });

    // True abundances (distinct per sample).
    let w_true = Mat::from_fn(s, c, |si, ct| {
        20.0 + 60.0 * (((si + 2 * ct) % 4) as f32) + 5.0
    });

    // μ and bulk counts (scaled to large depth).
    let sc = &rho * anchors.transpose();
    let mu = Mat::from_fn(d, c, |g, ct| (sc[(g, ct)] + gene_offset[g]).exp());
    let bulk = Mat::from_fn(d, s, |g, si| {
        let mut y = 0f32;
        for ct in 0..c {
            y += w_true[(si, ct)] * mu[(g, ct)];
        }
        y.round()
    });

    let names: Vec<Box<str>> = (0..c).map(|ct| format!("T{ct}").into_boxed_str()).collect();
    let feature_names: Vec<Box<str>> = (0..d).map(|g| format!("g{g}").into_boxed_str()).collect();

    let src = EmbeddingSource {
        rho: rho.clone(),
        anchor_emb: rho,
        gene_offset,
        feature_names,
        h,
        kind: RunKind::Bge,
        exact: true,
    };
    // Anchors pinned near truth (tiny prior spread) so we isolate fraction recovery.
    let prior = AnchorPrior {
        mean: anchors,
        names,
        sigma: vec![1e-4; c],
        chol: None,
    };
    (src, prior, bulk, w_true)
}

fn cfg_tau(tau: f32) -> SamplerConfig {
    SamplerConfig {
        warmup: 150,
        draws: 200,
        thin: 1,
        seed: 1,
        a0: 1.0,
        b0: 1.0,
        project_ridge: 1.0,
        init_iters: 50,
        // Perfectly-specified synthetic reference → test the core estimator in
        // the Poisson limit (r fixed large).
        nb_r: 1e4,
        tau,
    }
}

fn cfg() -> SamplerConfig {
    cfg_tau(1.0)
}

#[test]
fn recovers_fractions() {
    let (src, prior, bulk, w_true) = synthetic();
    let (s, c) = (bulk.ncols(), prior.mean.nrows());
    let init_w = Mat::from_element(s, c, 1.0); // neutral start
    let res = run_gibbs(&src, &bulk, &prior, &init_w, &cfg()).unwrap();

    // True count-fractions.
    let mut f_true = Mat::zeros(s, c);
    for si in 0..s {
        let tot: f32 = (0..c).map(|ct| w_true[(si, ct)]).sum();
        for ct in 0..c {
            f_true[(si, ct)] = w_true[(si, ct)] / tot;
        }
    }

    // Correlation between recovered and true fractions across all (s,c).
    let n = (s * c) as f32;
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f32, 0f32, 0f32, 0f32, 0f32);
    let mut max_abs = 0f32;
    for si in 0..s {
        for ct in 0..c {
            let x = res.fractions_mean[(si, ct)];
            let y = f_true[(si, ct)];
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            max_abs = max_abs.max((x - y).abs());
        }
    }
    let corr = (sxy - sx * sy / n) / (((sxx - sx * sx / n) * (syy - sy * sy / n)).sqrt() + 1e-9);
    assert!(corr > 0.9, "fraction recovery corr too low: {corr:.3}");
    assert!(
        max_abs < 0.15,
        "fraction max abs error too high: {max_abs:.3}"
    );
}

#[test]
fn tempering_widens_posterior() {
    // Lower τ (fewer effective counts) must widen the fraction posterior.
    let (src, prior, bulk, _) = synthetic();
    let (s, c) = (bulk.ncols(), prior.mean.nrows());
    let init_w = Mat::from_element(s, c, 1.0);
    let mean_sd = |cfg| {
        let res = run_gibbs(&src, &bulk, &prior, &init_w, &cfg).unwrap();
        res.fractions_sd.iter().sum::<f32>() / (s * c) as f32
    };
    let tight = mean_sd(cfg_tau(1.0));
    let loose = mean_sd(cfg_tau(0.02));
    assert!(
        loose > 2.0 * tight,
        "tempering did not widen posterior: sd(τ=1)={tight:.5}, sd(τ=0.02)={loose:.5}"
    );
}

#[test]
fn expression_conserves_counts() {
    let (src, prior, bulk, _) = synthetic();
    let (s, c) = (bulk.ncols(), prior.mean.nrows());
    let d = bulk.nrows();
    let init_w = Mat::from_element(s, c, 1.0);
    let res = run_gibbs(&src, &bulk, &prior, &init_w, &cfg()).unwrap();

    // Σ_c E[Z_{s,c,g}] ≈ y_{s,g}: the gene split apportions all observed counts.
    for si in 0..s {
        for g in (0..d).step_by(7) {
            let split: f32 = (0..c).map(|ct| res.expression[ct][(g, si)]).sum();
            let y = bulk[(g, si)];
            assert!(
                (split - y).abs() <= 0.02 * y + 1.0,
                "count conservation violated at (g={g}, s={si}): split={split:.2} y={y:.2}"
            );
        }
    }
}
