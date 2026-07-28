//! Flag resolution for `--posterior [N]`.
//!
//! Parsed through clap rather than by building [`PosteriorArgs`] by hand, so the
//! aliases and the optional-value spelling — which live in the attributes — are
//! covered too. This is the single home for these: both `senna bge` and
//! `faba gem` flatten the same struct, so the table cannot hold on one CLI and
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
