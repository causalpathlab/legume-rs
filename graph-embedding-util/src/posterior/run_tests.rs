//! Flag resolution between `--posterior` (which posterior) and `--mcmc` /
//! `--jitter` (how long).
//!
//! Parsed through clap rather than by building [`PosteriorArgs`] by hand, so the
//! `alias` and `value_enum` spellings — which live in the attributes — are
//! covered too. This is the single home for these now: both `senna bge` and
//! `faba gem` flatten the same struct, so the table cannot hold on one CLI and
//! not the other.

use super::*;
use crate::posterior::{ChainDiag, Samplers};
use clap::Parser;

#[derive(Parser, Debug)]
struct Harness {
    #[command(flatten)]
    posterior: PosteriorArgs,
}

const SEED: u64 = 7;

fn plan_of(extra: &[&str]) -> anyhow::Result<Option<PosteriorPlan>> {
    let mut argv = vec!["prog"];
    argv.extend_from_slice(extra);
    Harness::try_parse_from(argv)?.posterior.resolve(SEED)
}

/// Off is `None`, not a `Some(Off)` every caller has to re-test.
#[test]
fn no_flags_is_off() {
    assert!(plan_of(&[]).unwrap().is_none());
}

#[test]
fn explicit_off_is_off() {
    assert!(plan_of(&["--posterior", "off"]).unwrap().is_none());
}

#[test]
fn mcmc_alone_implies_both() {
    let plan = plan_of(&["--mcmc", "500"]).unwrap().unwrap();
    assert_eq!(plan.mode, PosteriorMode::Both);
    assert_eq!(plan.n_samples, 500);
}

/// `--jitter` is a clap alias on the same argument, so it must resolve to the
/// identical plan, not merely a similar one.
#[test]
fn jitter_is_an_alias_for_mcmc() {
    assert_eq!(
        plan_of(&["--jitter", "500"]).unwrap(),
        plan_of(&["--mcmc", "500"]).unwrap()
    );
}

#[test]
fn posterior_alone_takes_the_default_length() {
    let plan = plan_of(&["--posterior", "gate"]).unwrap().unwrap();
    assert_eq!(plan.mode, PosteriorMode::Gate);
    assert_eq!(plan.n_samples, DEFAULT_SAMPLES);
}

/// Per-dim is its own mode, reachable by name and NOT implied by `both` — it
/// costs `H` likelihood passes per anchor per sweep, so it must stay opt-in.
#[test]
fn perdim_is_opt_in_and_not_part_of_both() {
    let plan = plan_of(&["--posterior", "perdim", "--mcmc", "300"])
        .unwrap()
        .unwrap();
    assert_eq!(plan.mode, PosteriorMode::PerDim);
    assert_eq!(plan.n_samples, 300);
    // Self-contained: it learns its own per-dim hypers, so it neither runs nor
    // needs the Tier-1 globals or the gate.
    assert_eq!(
        plan.samplers,
        Samplers {
            gate: false,
            hyper: false,
            per_dim: true
        }
    );

    let both = plan_of(&["--mcmc", "300"]).unwrap().unwrap().samplers;
    assert!(
        !both.per_dim,
        "`both` must not silently pull in the per-dim sampler"
    );
    assert!(both.gate && both.hyper);
}

/// `all` is the ONLY mode that puts the gate and the per-dim sampler on one fit.
/// That comparison has to be same-fit: the basis rotates between runs (cross-seed
/// ARI 0.0365 against a same-seed floor of 1.0000), so a dim index from one fit is
/// not the dim index from another. Before `all` existed the comparison the
/// `posterior_dim_block` integration test calls the real test was unreachable from
/// the CLI at all.
#[test]
fn all_is_the_only_mode_that_runs_gate_and_perdim_together() {
    let s = plan_of(&["--posterior", "all", "--mcmc", "50"])
        .unwrap()
        .unwrap()
        .samplers;
    assert_eq!(
        s,
        Samplers {
            gate: true,
            hyper: true,
            per_dim: true
        }
    );
    for m in ["gate", "hyper", "both", "perdim"] {
        let o = plan_of(&["--posterior", m, "--mcmc", "50"])
            .unwrap()
            .unwrap()
            .samplers;
        assert!(
            !(o.gate && o.per_dim),
            "`{m}` must not pair gate with per-dim; only `all` does"
        );
    }
}

#[test]
fn posterior_narrows_the_mode_and_mcmc_sets_the_length() {
    let plan = plan_of(&["--posterior", "hyper", "--mcmc", "500"])
        .unwrap()
        .unwrap();
    assert_eq!(plan.mode, PosteriorMode::Hyper);
    assert_eq!(plan.n_samples, 500);
}

/// Explicitly off AND explicitly on: ambiguous intent, so it is an error rather
/// than a silent win for either side.
#[test]
fn off_with_mcmc_is_an_error() {
    let msg = plan_of(&["--posterior", "off", "--mcmc", "500"])
        .unwrap_err()
        .to_string();
    assert!(msg.contains("--posterior off"), "unexpected: {msg}");
    assert!(msg.contains("--mcmc 500"), "unexpected: {msg}");
}

#[test]
fn zero_draws_is_an_error() {
    assert!(plan_of(&["--mcmc", "0"]).is_err());
}

#[test]
fn coverage_flags_reach_the_plan() {
    let plan = plan_of(&[
        "--mcmc",
        "10",
        "--posterior-genes",
        "400",
        "--posterior-partition",
        "64",
    ])
    .unwrap()
    .unwrap();
    assert_eq!(plan.n_genes, 400);
    assert_eq!(plan.n_partition, 64);
}

#[test]
fn partition_defaults_to_the_documented_value() {
    let plan = plan_of(&["--mcmc", "10"]).unwrap().unwrap();
    assert_eq!(plan.n_partition, DEFAULT_PARTITION);
}

/// The caller's seed reaches the plan, so a reproducible fit gives a
/// reproducible posterior.
#[test]
fn seed_is_carried_from_the_caller() {
    assert_eq!(plan_of(&["--mcmc", "10"]).unwrap().unwrap().seed, SEED);
}

/// The guard exists to refuse a funnel-degenerate scale; check both directions
/// so a future threshold change cannot silently invert it.
#[test]
fn sigma2_adoption_respects_the_ess_bar() {
    let mk = |mean: f64, ess: f32| HyperSsResult {
        sigma2_chain: vec![],
        pi0_chain: vec![],
        sigma2_mean: mean,
        pi0_mean: 0.5,
        inclusion_prob: vec![],
        sigma_diag: ChainDiag {
            min_ess: ess,
            stuck_fraction: 0.0,
        },
        pi0_diag: ChainDiag {
            min_ess: 100.0,
            stuck_fraction: 0.0,
        },
    };
    assert_eq!(mk(1.5, SIGMA2_MIN_ESS).trustworthy_sigma2(), Some(1.5));
    assert!(mk(1.5, SIGMA2_MIN_ESS - 1.0).trustworthy_sigma2().is_none());
    // A well-mixed chain at a degenerate value is still refused.
    assert!(mk(0.0, 100.0).trustworthy_sigma2().is_none());
    assert!(mk(f64::NAN, 100.0).trustworthy_sigma2().is_none());
}
