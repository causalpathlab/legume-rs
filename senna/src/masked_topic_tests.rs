//! The masked family's argument surface.

use super::MaskedTopicArgs;
use clap::CommandFactory;

#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    args: MaskedTopicArgs,
}

/// `senna predict` tells a residual-trained model that `--adj-method batch`
/// removes its null mismatch, so the flag has to be where a user can find it.
#[test]
fn adj_method_is_listed_in_help() {
    let cmd = Cli::command();
    let arg = cmd
        .get_arguments()
        .find(|a| a.get_id() == "adj_method")
        .expect("the masked family accepts --adj-method");
    assert!(!arg.is_hide_set(), "--adj-method must not be hidden from --help");
}

/// The source run's gene names are aligned under the rule this run's own files
/// were loaded under, so `--feature-name-kind` means one thing per command.
#[test]
fn the_source_axis_is_aligned_under_the_run_name_rule() {
    use auxiliary_data::feature_names::FeatureNameKind;
    use clap::Parser;
    let parse = |extra: &[&str]| {
        let base = ["senna-masked-topic", "d.zarr", "-o", "out"];
        Cli::try_parse_from(base.iter().copied().chain(extra.iter().copied()))
            .expect("parses")
            .args
    };
    assert_eq!(parse(&["--feature-name-kind", "exact"]).axis_opts().kind, FeatureNameKind::Exact);
    assert!(matches!(parse(&[]).axis_opts().kind, FeatureNameKind::Gene { .. }));
}

/// The masked objective is the regularizer; `masked-vae` no longer weighs a KL.
mod no_kl {
    use super::{Cli, MaskedTopicArgs};
    use crate::run_manifest::{RunKind, RunManifest, TrainArgsRecord};
    use clap::Parser;

    #[test]
    fn kl_weight_is_not_a_flag_any_more() {
        let r = Cli::try_parse_from(["senna-masked-vae", "d.zarr", "-o", "out", "--kl-weight", "0.1"]);
        assert!(r.is_err(), "--kl-weight must be rejected, not silently ignored");
    }

    #[test]
    fn a_fit_recorded_with_a_kl_weight_still_replays() {
        let mut m = RunManifest::new(RunKind::MaskedVae, "old");
        m.train_args = Some(TrainArgsRecord {
            senna_version: "0.14.2".into(),
            args: serde_json::json!({ "data_files": ["d.zarr"], "out": "old", "kl_weight": 1.0, "n_latent_topics": 9 }),
        });
        let a: MaskedTopicArgs = m.train_args_as("old").expect("a recorded kl_weight is ignored");
        assert_eq!(a.n_latent_topics, 9);
    }
}
