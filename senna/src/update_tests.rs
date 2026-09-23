//! What `senna update` decides before it dispatches: whether to substitute the
//! parent's carried pseudobulks, and what it does when it cannot.

use super::{carried_reference_among, multiome_in_args, recorded_paths, UpdateArgs};
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
/// rather than the whole history, so it is on unless refused.
/// `--use-pb-reference` is redundant for the behaviour, but it turns a silent
/// fallback into an error when the parent carries nothing; asking for both is
/// a contradiction, not a precedence puzzle.
#[test]
fn the_carried_reference_is_used_unless_refused() {
    let a = parse(&[]).expect("bare update parses");
    assert!(!a.no_pb_reference && !a.use_pb_reference);
    let a = parse(&["--no-pb-reference"]).expect("opt-out parses");
    assert!(a.no_pb_reference);
    let a = parse(&["--use-pb-reference"]).expect("explicit request parses");
    assert!(a.use_pb_reference && !a.no_pb_reference);
    assert!(parse(&["--use-pb-reference", "--no-pb-reference"]).is_err());
}

/// The recorded fit arguments say whether the parent loads under multiome
/// alignment: a flag on the topic families, a suffix list on bge, and absent
/// on a run that predates the option.
#[test]
fn a_multiome_parent_is_read_off_its_recorded_arguments() {
    assert!(multiome_in_args(&serde_json::json!({ "multiome": true })));
    assert!(!multiome_in_args(&serde_json::json!({ "multiome": false })));
    assert!(multiome_in_args(
        &serde_json::json!({ "multiome": ["rna", "atac"] })
    ));
    assert!(!multiome_in_args(&serde_json::json!({ "multiome": [] })));
    assert!(!multiome_in_args(&serde_json::json!({ "epochs": 10 })));
}

/// A lineage that substituted once holds the carried reference among its
/// inputs, under the name `update` gave it.
#[test]
fn a_substituted_lineage_is_recognised_by_its_carried_reference() {
    let plain: Vec<Box<str>> = vec!["a.zarr".into(), "b.zarr".into()];
    assert_eq!(carried_reference_among(&plain), None);
    let substituted: Vec<Box<str>> = vec!["c.zarr".into(), "runs/r1.pb_reference.zarr.zip".into()];
    assert_eq!(
        carried_reference_among(&substituted),
        Some("runs/r1.pb_reference.zarr.zip")
    );
}

/// Manifests store data paths relative to the run directory; they resolve
/// there, never against the caller's cwd, and absolute paths pass through.
#[test]
fn recorded_paths_resolve_against_the_run_directory() {
    let run = std::path::Path::new("/runs/r1");
    let got = recorded_paths(&["rna.zarr.zip".into(), "/abs/atac.zarr".into()], run);
    assert_eq!(
        got,
        vec![
            Box::from("/runs/r1/rna.zarr.zip"),
            Box::from("/abs/atac.zarr")
        ]
    );
}
