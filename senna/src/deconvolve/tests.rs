//! Deterministic recovery tests for the deconvolution Gibbs sampler.

use super::args::SamplerConfig;
use super::gibbs::{finalize, run_chain, ComponentTable, PosteriorAccum};
use super::monitor::Monitor;
use super::reference::Reference;
use super::result::split_rhat_ess;
use crate::embed_common::Mat;

fn cfg_tau(tau: f32) -> SamplerConfig {
    SamplerConfig {
        warmup: 150,
        draws: 200,
        thin: 1,
        seed: 1,
        a0: 1.0,
        b0: 1.0,
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

/// Run one chain and finalize it.
fn drive(
    reference: &mut Reference,
    bulk: &Mat,
    init_w: &Mat,
    cfg: &SamplerConfig,
) -> super::result::DeconvResult {
    let mut monitor = Monitor::silent();
    let mut accum = PosteriorAccum::new(
        bulk.ncols(),
        reference.n_types(),
        reference.n_genes(),
        reference.n_comp(),
    );
    run_chain(reference, bulk, init_w, cfg, &mut monitor, &mut accum).unwrap();
    finalize(
        accum,
        bulk,
        ComponentTable {
            coords: reference.coords.clone(),
            names: reference.comp_names.clone(),
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
        readout,
        coords: Mat::zeros(r, 2),
        comp_names: (0..r).map(|m| format!("a{m}").into_boxed_str()).collect(),
        n_cells: vec![1.0; r],
        celltype_names: (0..c).map(|ct| format!("T{ct}").into_boxed_str()).collect(),
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
    let init_w = reference.init_abundances(bulk);
    drive(reference, bulk, &init_w, &cfg)
}

#[test]
fn tempering_widens_posterior() {
    // Lower τ (fewer effective counts) must widen the fraction posterior.
    let mean_sd = |tau: f32| {
        let (mut reference, bulk, _) = archetype_problem(false);
        let mut cfg = cfg_tau(tau);
        cfg.a0 = 1.0;
        let init_w = reference.init_abundances(&bulk);
        let res = drive(&mut reference, &bulk, &init_w, &cfg);
        res.fractions_sd.iter().sum::<f32>() / res.fractions_sd.len() as f32
    };
    let tight = mean_sd(1.0);
    let loose = mean_sd(0.02);
    assert!(
        loose > 2.0 * tight,
        "tempering did not widen posterior: sd(τ=1)={tight:.5}, sd(τ=0.02)={loose:.5}"
    );
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
            let split: f32 = (0..c).map(|ct| res.expression.get(ct, g, si)).sum();
            let y = bulk[(g, si)];
            assert!(
                (split - y).abs() <= 0.02 * y + 1.0,
                "count conservation violated at (g={g}, s={si}): split={split:.2} y={y:.2}"
            );
        }
    }
}

/////////////////////
// Pooled chains   //
/////////////////////

/// Run `n_chains` chains over the same reference into one accumulator, the way
/// `deconvolve::run` pools over archetype granularities.
fn drive_pooled(
    reference: &mut Reference,
    bulk: &Mat,
    init_w: &Mat,
    cfg: &SamplerConfig,
    n_chains: usize,
) -> super::result::DeconvResult {
    let mut monitor = Monitor::silent();
    // Each chain claims its own slice of the pooled component axis; here every
    // chain reuses the same reference, so the axis is `n_chains` copies of it.
    let mut accum = PosteriorAccum::new(
        bulk.ncols(),
        reference.n_types(),
        reference.n_genes(),
        reference.n_comp() * n_chains,
    );
    for chain in 0..n_chains {
        let mut chain_cfg = SamplerConfig { ..*cfg };
        chain_cfg.seed = cfg
            .seed
            .wrapping_add((chain as u64).wrapping_mul(super::SEED_STRIDE));
        run_chain(
            reference,
            bulk,
            init_w,
            &chain_cfg,
            &mut monitor,
            &mut accum,
        )
        .unwrap();
    }
    finalize(
        accum,
        bulk,
        ComponentTable {
            coords: reference.coords.clone(),
            names: reference.comp_names.clone(),
        },
        reference.celltype_names.clone(),
    )
    .unwrap()
}

#[test]
fn pooling_chains_agrees_with_one_long_chain() {
    // Pooling three chains over the same reference must land where a single
    // chain does; only the interval should differ. This is what makes averaging
    // over the archetype partition meaningful — any bias here would be read as
    // partition sensitivity.
    let (mut reference, bulk, _) = archetype_problem(false);
    let init_w = reference.init_abundances(&bulk);
    let one = drive_pooled(&mut reference, &bulk, &init_w, &cfg(), 1);
    let three = drive_pooled(&mut reference, &bulk, &init_w, &cfg(), 3);

    let (corr, max_abs) = agreement(&one.fractions_mean, &three.fractions_mean);
    assert!(
        corr > 0.99 && max_abs < 0.02,
        "pooled mean drifted from the single-chain mean: corr={corr:.4}, max|Δ|={max_abs:.4}"
    );
    assert_eq!(one.n_chains, 1, "chain count not recorded");
    assert_eq!(three.n_chains, 3, "chain count not recorded");
}

#[test]
fn pooled_chains_report_between_chain_diagnostics() {
    // With several chains the diagnostic must come from the between-chain
    // comparison, not from splitting a concatenated trace whose chain
    // boundaries would read as drift.
    let (mut reference, bulk, _) = archetype_problem(false);
    let init_w = reference.init_abundances(&bulk);
    let res = drive_pooled(&mut reference, &bulk, &init_w, &cfg(), 3);
    assert!(
        !res.convergence.is_empty(),
        "pooled run reported no convergence diagnostics"
    );
    // Independent chains over one reference should agree.
    let worst = res.convergence.iter().fold(0f32, |a, cv| a.max(cv.rhat));
    assert!(
        worst.is_finite() && worst < 1.5,
        "chains over the same reference disagree: worst R-hat {worst:.3}"
    );
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
