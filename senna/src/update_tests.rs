//! What `senna update` decides before it dispatches: whether to substitute the
//! parent's carried pseudobulks, and what it does when it cannot.

use super::UpdateArgs;
use clap::Parser;

#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    args: UpdateArgs,
}

fn parse(extra: &[&str]) -> Result<UpdateArgs, clap::Error> {
    let base = ["senna-update", "new.zarr", "--model", "m", "-o", "out"];
    Cli::try_parse_from(base.iter().copied().chain(extra.iter().copied())).map(|c| c.args)
}

/// Carrying the pseudobulks forward is what makes a round cost the new data
/// rather than the whole history, so it is on unless asked otherwise.
#[test]
fn the_carried_reference_is_used_unless_refused() {
    let a = parse(&[]).expect("bare update parses");
    assert!(!a.no_pb_reference, "substitution is the default");
    assert!(!a.use_pb_reference, "and is not an explicit request");

    let a = parse(&["--no-pb-reference"]).expect("opt-out parses");
    assert!(a.no_pb_reference);
}

/// The two spellings are mutually exclusive: one forces the substitution, the
/// other forces the exact re-collapse, and asking for both is a contradiction
/// rather than a precedence puzzle.
#[test]
fn forcing_both_ways_at_once_is_refused() {
    assert!(parse(&["--use-pb-reference", "--no-pb-reference"]).is_err());
}

/// `--use-pb-reference` survives as an explicit request. It is redundant for
/// the behaviour, but it is what turns a silent fallback into an error when the
/// parent carries nothing or the new data has no batch labels.
#[test]
fn the_explicit_request_still_parses() {
    let a = parse(&["--use-pb-reference"]).expect("legacy scripts keep working");
    assert!(a.use_pb_reference);
    assert!(!a.no_pb_reference);
}
