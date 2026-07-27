//! Flag resolution between `--posterior` (which posterior) and `--mcmc` /
//! `--jitter` (how long). Parsed through clap rather than by constructing
//! `BgeArgs` by hand, so the `alias` and the `value_enum` spellings are covered
//! too — those live in the attributes, where a hand-built struct would not
//! exercise them.

use super::*;
use clap::Parser;

/// Minimal parser wrapper: `BgeArgs` is `#[derive(Args)]`, not a `Parser`.
#[derive(Parser, Debug)]
struct Harness {
    #[command(flatten)]
    bge: BgeArgs,
}

/// Parse a bge command line, given only the posterior-related flags. The
/// positional input and `-o` are always required.
fn plan_of(extra: &[&str]) -> anyhow::Result<PosteriorPlan> {
    let mut argv = vec!["senna-bge", "counts.zarr", "-o", "out"];
    argv.extend_from_slice(extra);
    let h = Harness::try_parse_from(argv)?;
    resolve(&h.bge)
}

#[test]
fn no_flags_is_off() {
    let plan = plan_of(&[]).unwrap();
    assert_eq!(plan.mode, PosteriorMode::Off);
}

#[test]
fn mcmc_alone_implies_both() {
    let plan = plan_of(&["--mcmc", "500"]).unwrap();
    assert_eq!(plan.mode, PosteriorMode::Both);
    assert_eq!(plan.n_samples, 500);
}

/// `--jitter` is a clap alias on the same argument, so it must not merely behave
/// similarly — it must resolve to the identical plan.
#[test]
fn jitter_is_an_alias_for_mcmc() {
    assert_eq!(
        plan_of(&["--jitter", "500"]).unwrap(),
        plan_of(&["--mcmc", "500"]).unwrap()
    );
}

#[test]
fn posterior_alone_takes_the_default_length() {
    let plan = plan_of(&["--posterior", "gate"]).unwrap();
    assert_eq!(plan.mode, PosteriorMode::Gate);
    assert_eq!(plan.n_samples, DEFAULT_SAMPLES);
}

#[test]
fn posterior_narrows_the_mode_and_mcmc_sets_the_length() {
    let plan = plan_of(&["--posterior", "hyper", "--mcmc", "500"]).unwrap();
    assert_eq!(plan.mode, PosteriorMode::Hyper);
    assert_eq!(plan.n_samples, 500);
}

/// Explicitly off AND explicitly on: the intent is genuinely ambiguous, so it is
/// an error rather than a silent win for either side.
#[test]
fn off_with_mcmc_is_an_error() {
    let err = plan_of(&["--posterior", "off", "--mcmc", "500"]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("--posterior off"), "unexpected message: {msg}");
    assert!(msg.contains("--mcmc 500"), "unexpected message: {msg}");
}

#[test]
fn zero_draws_is_an_error() {
    assert!(plan_of(&["--mcmc", "0"]).is_err());
}

/// A zero-draw request is only contradictory when a posterior was actually
/// asked for; the default (off) path must not trip the same check.
#[test]
fn zero_draws_is_fine_when_off() {
    assert_eq!(plan_of(&[]).unwrap().n_samples, DEFAULT_SAMPLES);
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
    .unwrap();
    assert_eq!(plan.n_genes, 400);
    assert_eq!(plan.n_partition, 64);
}

#[test]
fn partition_defaults_to_the_documented_value() {
    assert_eq!(plan_of(&[]).unwrap().n_partition, DEFAULT_PARTITION);
}
