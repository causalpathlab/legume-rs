//! `senna vae` takes the same row-name rule as the masked family, so both
//! families can be pinned to one gene axis for a comparison. A fit recorded
//! before the flag existed replays under auto-detection, exactly what those
//! runs got.

use super::VaeArgs;
use crate::masked_topic::FeatureNameKindArg;
use crate::run_manifest::{RunKind, RunManifest, TrainArgsRecord};

#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    args: VaeArgs,
}

fn parse(argv: &[&str]) -> VaeArgs {
    use clap::Parser;
    Cli::try_parse_from(std::iter::once("senna-vae").chain(argv.iter().copied()))
        .expect("parse")
        .args
}

#[test]
fn the_row_name_rule_is_a_flag_and_defaults_to_auto() {
    let a = parse(&["d.zarr", "-o", "out"]);
    assert!(matches!(a.feature_name_kind, FeatureNameKindArg::Auto));
    let a = parse(&["d.zarr", "-o", "out", "--feature-name-kind", "exact"]);
    assert!(matches!(a.feature_name_kind, FeatureNameKindArg::Exact));
}

#[test]
fn a_fit_recorded_before_the_flag_replays_under_auto() {
    let mut m = RunManifest::new(RunKind::Vae, "old");
    m.train_args = Some(TrainArgsRecord {
        senna_version: "0.14.2".into(),
        args: serde_json::json!({ "data_files": ["d.zarr"], "out": "old", "n_latent": 7 }),
    });
    let a: VaeArgs = m.train_args_as("old").expect("older record must replay");
    assert_eq!(a.n_latent, 7);
    assert!(matches!(a.feature_name_kind, FeatureNameKindArg::Auto));
}

/// `--max-coarse-features` moved into a shared flattened struct. The recorded
/// key sits at the top level of the fit arguments, so a manifest written
/// before the move has to replay onto the new shape unchanged: `senna update`
/// reads these back and would otherwise silently re-collapse at the default.
#[test]
fn a_coarsening_recorded_before_the_shared_struct_still_replays() {
    let recorded = serde_json::json!({
        "data_files": ["a.zarr"],
        "out": "old",
        "max_coarse_features": 250,
    });
    let args: super::VaeArgs = serde_json::from_value(recorded).expect("an old manifest replays");
    assert_eq!(args.coarsening.max_coarse_features, 250);
    assert_eq!(
        args.coarsening.cap().map(std::num::NonZeroUsize::get),
        Some(250)
    );

    let off =
        serde_json::json!({ "data_files": ["a.zarr"], "out": "old", "max_coarse_features": 0 });
    let args: super::VaeArgs = serde_json::from_value(off).expect("replays");
    assert!(args.coarsening.cap().is_none(), "0 means every feature");
}
