//! The `diff` command line carries no ablation switches for its own method:
//! topic residualization and the multilevel pseudobulk refine are the method,
//! not options.

use super::*;

const REQUIRED: [&str; 8] = [
    "cocoa-diff",
    "data.zarr",
    "-i",
    "indv.gz",
    "-e",
    "exposure.gz",
    "-o",
    "out",
];

fn parse(extra: &[&str]) -> Result<DiffArgs, clap::Error> {
    let argv: Vec<&str> = REQUIRED.iter().chain(extra).cloned().collect();
    DiffArgs::try_parse_from(argv)
}

#[test]
fn required_arguments_parse() {
    parse(&[]).expect("minimal diff invocation parses");
}

#[test]
fn hash_only_partition_is_not_an_option() {
    assert!(parse(&["--no-refine"]).is_err());
}

#[test]
fn skipping_topic_residualization_is_not_an_option() {
    assert!(parse(&["--no-residualize-topics"]).is_err());
}
