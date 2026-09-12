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
    assert!(
        !arg.is_hide_set(),
        "--adj-method must not be hidden from --help"
    );
}

/// The masked objective is the regularizer; `masked-vae` no longer weighs a KL.
mod no_kl {
    use super::{Cli, MaskedTopicArgs};
    use crate::run_manifest::{RunKind, RunManifest, TrainArgsRecord};
    use clap::Parser;

    #[test]
    fn kl_weight_is_not_a_flag_any_more() {
        let r = Cli::try_parse_from([
            "senna-masked-vae",
            "d.zarr",
            "-o",
            "out",
            "--kl-weight",
            "0.1",
        ]);
        assert!(
            r.is_err(),
            "--kl-weight must be rejected, not silently ignored"
        );
    }

    #[test]
    fn a_fit_recorded_with_a_kl_weight_still_replays() {
        let mut m = RunManifest::new(RunKind::MaskedVae, "old");
        m.train_args = Some(TrainArgsRecord {
            senna_version: "0.14.2".into(),
            args: serde_json::json!({ "data_files": ["d.zarr"], "out": "old", "kl_weight": 1.0, "n_latent_topics": 9 }),
        });
        let a: MaskedTopicArgs = m
            .train_args_as("old")
            .expect("a recorded kl_weight is ignored");
        assert_eq!(a.n_latent_topics, 9);
    }
}

/// The window is gone: the encoder reads every gene, so the flags that sized
/// and populated a context window have nothing left to do.
mod window_free {
    use super::{Cli, MaskedTopicArgs};
    use crate::run_manifest::{RunKind, RunManifest, TrainArgsRecord};
    use clap::Parser;

    fn parse(extra: &[&str]) -> MaskedTopicArgs {
        let mut argv = vec!["senna-masked-vae", "d.zarr", "-o", "out"];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(argv).expect("parses").args
    }

    /// `--context-size` is not a flag any more. Silently accepting it would let
    /// a user believe they had changed what the encoder reads.
    #[test]
    fn context_size_is_not_a_flag_any_more() {
        let r = Cli::try_parse_from([
            "senna-masked-vae",
            "d.zarr",
            "-o",
            "out",
            "--context-size",
            "1000",
        ]);
        assert!(
            r.is_err(),
            "--context-size must be rejected, not silently ignored"
        );
    }

    /// An old run manifest still carries `context_size`, and `senna update` has
    /// to be able to replay it. The value no longer applies; the record must
    /// still deserialise.
    #[test]
    fn a_fit_recorded_with_a_context_size_still_replays() {
        let mut m = RunManifest::new(RunKind::MaskedVae, "old");
        m.train_args = Some(TrainArgsRecord {
            senna_version: "0.15.5".into(),
            args: serde_json::json!({
                "data_files": ["d.zarr"], "out": "old",
                "context_size": 1000, "n_latent_topics": 9
            }),
        });
        let a: MaskedTopicArgs = m
            .train_args_as("old")
            .expect("a recorded context_size still parses");
        assert_eq!(a.n_latent_topics, 9);
        assert_eq!(
            a.recorded_context_size(),
            Some(1000),
            "the recorded window must survive the read so the run can say it no longer applies"
        );
    }

    /// Gene modules pool a cell by membership over its CONTEXT SLOTS. With no
    /// context there are no slots, so the branch has no dense counterpart —
    /// refuse at argument validation rather than at the first forward.
    #[test]
    fn gene_modules_are_refused_and_the_message_names_both_ideas() {
        let args = parse(&["--gene-modules", "64"]);
        let msg = args
            .validate()
            .expect_err("modules and the window-free encoder do not compose")
            .to_string();
        for needle in ["--gene-modules", "context"] {
            assert!(
                msg.contains(needle),
                "the message must name {needle}; got: {msg}"
            );
        }
        assert!(parse(&["--gene-modules", "0"]).validate().is_ok());
    }

    /// The query head attends from each query gene over the genes the encoder
    /// read. Window-free that set is the whole gene axis, so its `[N, Q, D]`
    /// attention cannot be formed at any `--query-extra`. Refuse by name.
    #[test]
    fn the_query_decoder_is_refused_and_the_message_names_both_flags() {
        let args = parse(&["--query-decoder"]);
        let msg = args
            .validate()
            .expect_err("the query head needs a bounded read set")
            .to_string();
        for needle in ["--query-decoder", "--query-extra"] {
            assert!(
                msg.contains(needle),
                "the message must name {needle}; got: {msg}"
            );
        }
        // Off, it costs nothing and nothing is refused.
        assert!(parse(&[]).validate().is_ok());
    }
}
