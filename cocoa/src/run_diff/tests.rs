//! The `diff` command line carries no ablation switches for its own method:
//! topic residualization, the likelihood pseudobulk refine, and raw counts
//! into the model are the method, not options.

use super::*;

fn parse(extra: &[&str]) -> Result<DiffArgs, clap::Error> {
    let required = [
        "cocoa-diff",
        "data.zarr",
        "-i",
        "indv.gz",
        "-e",
        "exposure.gz",
        "-o",
        "out",
    ];
    DiffArgs::try_parse_from(required.iter().chain(extra).copied())
}

#[test]
fn required_arguments_parse() {
    parse(&[]).expect("minimal diff invocation parses");
}

#[test]
fn removed_switches_are_rejected() {
    for flag in [
        "--no-refine",
        "--pb-refine-sweeps",
        "--no-residualize-topics",
        "--no-adjust-housekeeping",
    ] {
        assert!(parse(&[flag]).is_err(), "{flag} should not parse");
    }
}
