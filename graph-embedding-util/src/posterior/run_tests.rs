//! Flag resolution for `--posterior [N]`.
//!
//! Parsed through clap rather than by building [`PosteriorArgs`] by hand, so the
//! aliases and the optional-value spelling — which live in the attributes — are
//! covered too. This is the single home for these: both `senna bge` and
//! `senna gem` flatten the same struct, so the table cannot hold on one CLI and
//! not the other.

use super::*;
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

/// Absent is `None`, not a `Some(..)` every caller has to re-test.
#[test]
fn no_flag_is_off() {
    assert!(plan_of(&[]).unwrap().is_none());
}

/// Bare `--posterior` takes the documented default rather than erroring, so the
/// common case needs no number.
#[test]
fn bare_posterior_uses_the_default_length() {
    let plan = plan_of(&["--posterior"]).unwrap().unwrap();
    assert_eq!(plan.n_samples, DEFAULT_SAMPLES);
}

#[test]
fn posterior_takes_an_explicit_draw_count() {
    assert_eq!(
        plan_of(&["--posterior", "500"]).unwrap().unwrap().n_samples,
        500
    );
}

/// `--mcmc` / `--jitter` are clap aliases on the same argument, so they must
/// resolve to the IDENTICAL plan, not merely a similar one.
#[test]
fn mcmc_and_jitter_are_aliases() {
    let want = plan_of(&["--posterior", "500"]).unwrap();
    assert_eq!(plan_of(&["--mcmc", "500"]).unwrap(), want);
    assert_eq!(plan_of(&["--jitter", "500"]).unwrap(), want);
}

#[test]
fn zero_draws_is_an_error() {
    assert!(plan_of(&["--posterior", "0"]).is_err());
}

/// The caller's seed reaches the plan, so a reproducible fit gives a
/// reproducible posterior.
#[test]
fn seed_is_carried_from_the_caller() {
    assert_eq!(plan_of(&["--posterior", "10"]).unwrap().unwrap().seed, SEED);
}

/// The truncated IBP is the DEFAULT prior, so a bare `--posterior` must already
/// carry a concentration. A regression here is silent: the sampler falls back to
/// the unordered Beta and reports a full, plausible, entirely flat `π₀`.
#[test]
fn the_default_prior_is_the_truncated_ibp() {
    let plan = plan_of(&["--posterior"]).unwrap().unwrap();
    assert_eq!(
        plan.stick_alpha,
        Some(crate::posterior::dim_block::DEFAULT_STICK_ALPHA),
        "bare --posterior must default to stick-breaking"
    );
}

/// `--no-stick-breaking` is the opt-out, and it must reach the plan as `None`
/// rather than as some sentinel the sampler then has to re-interpret.
#[test]
fn no_stick_breaking_falls_back_to_the_independent_beta() {
    let plan = plan_of(&["--posterior", "--no-stick-breaking"])
        .unwrap()
        .unwrap();
    assert!(plan.stick_alpha.is_none());
}

#[test]
fn an_explicit_concentration_is_carried() {
    let plan = plan_of(&["--posterior", "--stick-alpha", "2.5"])
        .unwrap()
        .unwrap();
    assert_eq!(plan.stick_alpha, Some(2.5));
}

/// A non-positive or non-finite concentration is not a Beta at all — reject it
/// rather than letting `Beta::new` panic deep inside a sweep.
#[test]
fn a_non_positive_concentration_is_an_error() {
    assert!(plan_of(&["--posterior", "--stick-alpha", "0"]).is_err());
    assert!(plan_of(&["--posterior", "--stick-alpha", "-1"]).is_err());
    assert!(plan_of(&["--posterior", "--stick-alpha", "nan"]).is_err());
}
