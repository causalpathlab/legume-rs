//! Deterministic recovery tests for the deconvolution Gibbs sampler.

use super::anchors::AnchorPrior;
use super::args::SamplerConfig;
use super::gibbs::{finalize, run_chain, AnchorSampler, ComponentTable, PosteriorAccum};
use super::monitor::Monitor;
use super::reference::{ComponentAxis, FractionUnits, Reference};
use super::result::split_rhat_ess;
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
        cell_embedding_paths: Vec::new(),
        data_files: Vec::new(),
        annotation_path: None,
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
        expression_every: 1,
    }
}

fn cfg() -> SamplerConfig {
    cfg_tau(1.0)
}

/// Run the low-rank sampler end to end on the synthetic problem.
fn run_lowrank(
    src: &EmbeddingSource,
    prior: &AnchorPrior,
    bulk: &Mat,
    cfg: &SamplerConfig,
) -> super::result::DeconvResult {
    let (s, c) = (bulk.ncols(), prior.mean.nrows());
    let mut reference = Reference::low_rank(src, prior, bulk.nrows());
    let init_w = Mat::from_element(s, c, 1.0); // neutral start
    let anchors = AnchorSampler::new(src, prior, cfg.seed);
    drive(&mut reference, bulk, &init_w, cfg, Some(anchors))
}

/// Run one chain and finalize it, the shape both modes use in tests.
fn drive(
    reference: &mut Reference,
    bulk: &Mat,
    init_w: &Mat,
    cfg: &SamplerConfig,
    anchors: Option<AnchorSampler<'_>>,
) -> super::result::DeconvResult {
    let mut monitor = Monitor::silent();
    let mut accum = PosteriorAccum::new(bulk.ncols(), reference.n_types(), reference.n_genes);
    run_chain(
        reference,
        bulk,
        init_w,
        cfg,
        anchors,
        &mut monitor,
        &mut accum,
    )
    .unwrap();
    let rates = reference.type_rates();
    finalize(
        &accum,
        bulk,
        &rates,
        ComponentTable {
            coords: reference.coords.clone(),
            names: reference.comp_names.clone(),
            axis: reference.axis,
            units: reference.units,
        },
        reference.celltype_names.clone(),
    )
    .unwrap()
}

/// Pearson correlation and max absolute deviation between two `s×c` matrices.
fn agreement(a: &Mat, b: &Mat) -> (f32, f32) {
    let n = (a.nrows() * a.ncols()) as f32;
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f32, 0f32, 0f32, 0f32, 0f32);
    let mut max_abs = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        max_abs = max_abs.max((x - y).abs());
    }
    let corr = (sxy - sx * sy / n) / (((sxx - sx * sx / n) * (syy - sy * sy / n)).sqrt() + 1e-9);
    (corr, max_abs)
}

/// Row-normalise abundances into fractions.
fn to_fractions(w: &Mat) -> Mat {
    let (s, c) = (w.nrows(), w.ncols());
    let mut out = Mat::zeros(s, c);
    for si in 0..s {
        let tot: f32 = (0..c).map(|ct| w[(si, ct)]).sum();
        for ct in 0..c {
            out[(si, ct)] = w[(si, ct)] / tot;
        }
    }
    out
}

////////////////////////
// Low-rank reference //
////////////////////////

#[test]
fn recovers_fractions() {
    let (src, prior, bulk, w_true) = synthetic();
    let res = run_lowrank(&src, &prior, &bulk, &cfg());
    let (corr, max_abs) = agreement(&res.fractions_mean, &to_fractions(&w_true));
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
    let mean_sd = |cfg: SamplerConfig| {
        let res = run_lowrank(&src, &prior, &bulk, &cfg);
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
    let res = run_lowrank(&src, &prior, &bulk, &cfg());

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

/////////////////////////
// Archetype reference //
/////////////////////////

/// `R` normalised empirical profiles mapped onto `C` types, with a bulk built as
/// an exact non-negative mixture. `dup` repeats every archetype, which must not
/// change the reported fractions.
fn archetype_problem(dup: bool) -> (Reference, Mat, Mat) {
    let (d, c, s) = (48usize, 3usize, 4usize);
    let base = 4usize;
    let r = if dup { base * 2 } else { base };

    // Each base archetype concentrates on its own quarter of the gene panel.
    let profile = |m: usize, g: usize| -> f32 {
        let lobe = m % base;
        let peak = (lobe * d) / base;
        let dist = (g as i32 - peak as i32).unsigned_abs() as f32;
        (1.0 + 6.0 * (-dist / 4.0).exp()).max(1e-3)
    };
    let mut xbar = Mat::from_fn(d, r, |g, m| profile(m, g));
    for m in 0..r {
        let total: f32 = (0..d).map(|g| xbar[(g, m)]).sum();
        for g in 0..d {
            xbar[(g, m)] /= total;
        }
    }

    // Archetypes 0,1 → type 0; 2 → type 1; 3 → type 2 (repeated when duplicated).
    let readout = Mat::from_fn(r, c, |m, ct| {
        let want = match m % base {
            0 | 1 => 0,
            2 => 1,
            _ => 2,
        };
        if ct == want {
            1.0
        } else {
            0.0
        }
    });

    let u_true = Mat::from_fn(s, r, |si, m| {
        let lobe = m % base;
        let w = 200.0 + 400.0 * (((si + 2 * lobe) % 4) as f32);
        if dup {
            w / 2.0
        } else {
            w
        }
    });
    let bulk = Mat::from_fn(d, s, |g, si| {
        let mut y = 0f32;
        for m in 0..r {
            y += u_true[(si, m)] * xbar[(g, m)];
        }
        y.round()
    });

    let mut mu_gm = vec![0f32; d * r];
    for g in 0..d {
        for m in 0..r {
            mu_gm[g * r + m] = xbar[(g, m)];
        }
    }
    let reference = Reference {
        mu_gm,
        n_genes: d,
        n_comp: r,
        readout,
        coords: Mat::zeros(r, 2),
        comp_names: (0..r).map(|m| format!("a{m}").into_boxed_str()).collect(),
        celltype_names: (0..c).map(|ct| format!("T{ct}").into_boxed_str()).collect(),
        units: FractionUnits::Mrna,
        axis: ComponentAxis::Archetype,
    };

    // True fractions at the type level.
    let mut w_type = Mat::zeros(s, c);
    for si in 0..s {
        for m in 0..r {
            for ct in 0..c {
                w_type[(si, ct)] += u_true[(si, m)] * reference.readout[(m, ct)];
            }
        }
    }
    (reference, bulk, to_fractions(&w_type))
}

fn run_archetype(reference: &mut Reference, bulk: &Mat, a0: f32) -> super::result::DeconvResult {
    let mut cfg = cfg();
    cfg.a0 = a0;
    let init_w = Mat::from_fn(bulk.ncols(), reference.n_comp, |si, _| {
        bulk.column(si).iter().sum::<f32>() / reference.n_comp as f32
    });
    drive(reference, bulk, &init_w, &cfg, None)
}

#[test]
fn archetype_recovers_fractions() {
    let (mut reference, bulk, f_true) = archetype_problem(false);
    let res = run_archetype(&mut reference, &bulk, 1.0);
    let (corr, max_abs) = agreement(&res.fractions_mean, &f_true);
    assert!(corr > 0.95, "archetype recovery corr too low: {corr:.3}");
    assert!(max_abs < 0.05, "archetype max abs error: {max_abs:.3}");
}

#[test]
fn archetype_readout_is_split_invariant() {
    // Duplicating every archetype splits the abundance mass but must leave the
    // reported per-type fractions alone: the readout marginalises over
    // components, so R is a resolution knob, not a modelling choice.
    // A weak per-component prior keeps the a0 mass from scaling with R.
    let a0 = 1e-3;
    let (mut plain, bulk_a, _) = archetype_problem(false);
    let (mut doubled, bulk_b, _) = archetype_problem(true);
    assert_eq!(bulk_a, bulk_b, "duplication changed the simulated bulk");

    let one = run_archetype(&mut plain, &bulk_a, a0);
    let two = run_archetype(&mut doubled, &bulk_b, a0);
    let (corr, max_abs) = agreement(&one.fractions_mean, &two.fractions_mean);
    assert!(
        corr > 0.99 && max_abs < 0.02,
        "splitting archetypes moved the fractions: corr={corr:.4}, max|Δ|={max_abs:.4}"
    );
}

#[test]
fn archetype_expression_conserves_counts() {
    let (mut reference, bulk, _) = archetype_problem(false);
    let c = reference.n_types();
    let res = run_archetype(&mut reference, &bulk, 1.0);
    for si in 0..bulk.ncols() {
        for g in (0..bulk.nrows()).step_by(5) {
            let split: f32 = (0..c).map(|ct| res.expression[ct][(g, si)]).sum();
            let y = bulk[(g, si)];
            assert!(
                (split - y).abs() <= 0.02 * y + 1.0,
                "count conservation violated at (g={g}, s={si}): split={split:.2} y={y:.2}"
            );
        }
    }
}

///////////////////////////
// Convergence diagnostics //
///////////////////////////

#[test]
fn diagnostics_guard_and_wiring() {
    // The estimators themselves are mcmc-util's and tested there; what is ours
    // is the short-chain guard and that a drifting chain reaches them.
    assert!(
        split_rhat_ess(&[0.1, 0.2, 0.3]).is_none(),
        "too few draws should yield no diagnostic"
    );
    let drifting: Vec<f32> = (0..400).map(|t| t as f32 / 400.0).collect();
    let (rhat, ess) = split_rhat_ess(&drifting).expect("enough draws");
    assert!(rhat > 1.05, "split-R̂ missed a drifting chain: {rhat:.3}");
    assert!(ess > 0.0, "ESS should be positive, got {ess}");
}
