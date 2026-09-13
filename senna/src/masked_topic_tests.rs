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

    /// Nothing is refused once the query head is unwired.
    #[test]
    fn a_plain_window_free_run_validates() {
        assert!(parse(&[]).validate().is_ok());
    }
}

/// The query decoder is not wired into the masked family any more. The library
/// module stays (for re-wiring later), but nothing reaches it from a command.
mod query_unwired {
    use super::{Cli, MaskedTopicArgs};
    use crate::run_manifest::{RunKind, RunManifest, TrainArgsRecord};
    use clap::Parser;

    /// None of the four flags is a flag any more. Silently accepting one would
    /// let a user believe they had turned a head on.
    #[test]
    fn the_query_flags_are_not_flags_any_more() {
        for flag in [
            vec!["--query-decoder"],
            vec!["--query-rank", "32"],
            vec!["--query-extra", "64"],
            vec!["--query-penalty", "1.0"],
        ] {
            let mut argv = vec!["senna-masked-vae", "d.zarr", "-o", "out"];
            argv.extend_from_slice(&flag);
            assert!(
                Cli::try_parse_from(argv).is_err(),
                "{flag:?} must be rejected, not silently ignored"
            );
        }
    }

    /// An old run manifest carries all four, and `senna update` has to be able
    /// to replay it. The values no longer apply; the record must still
    /// deserialise, and the run says so once — the way a recorded
    /// `--context-size` is handled.
    #[test]
    fn a_fit_recorded_with_the_query_flags_still_replays() {
        let mut m = RunManifest::new(RunKind::MaskedVae, "old");
        m.train_args = Some(TrainArgsRecord {
            senna_version: "0.15.5".into(),
            args: serde_json::json!({
                "data_files": ["d.zarr"], "out": "old", "n_latent_topics": 9,
                "query_decoder": true, "query_rank": 32,
                "query_extra": 128, "query_penalty": 1.0
            }),
        });
        let a: MaskedTopicArgs = m
            .train_args_as("old")
            .expect("a recorded query-decoder setting still parses");
        assert_eq!(a.n_latent_topics, 9);
        let named = a.recorded_query_flags();
        for needle in [
            "--query-decoder",
            "--query-rank",
            "--query-extra",
            "--query-penalty",
        ] {
            assert!(
                named.contains(&needle),
                "the replay must name {needle}; got: {named:?}"
            );
        }
        // A manifest this build wrote has none of them and says nothing.
        let mut m2 = RunManifest::new(RunKind::MaskedVae, "new");
        m2.train_args = Some(TrainArgsRecord {
            senna_version: "0.15.6".into(),
            args: serde_json::json!({ "data_files": ["d.zarr"], "out": "new" }),
        });
        let b: MaskedTopicArgs = m2.train_args_as("new").expect("parses");
        assert!(b.recorded_query_flags().is_empty());
    }
}

/// The mask rate is the model, so the command line owns its bounds.
///
/// A rate of 0 hides nothing and a rate of 1 hides everything; the loader used
/// to clamp both back to a one-gene draw, which answered a question nobody
/// asked. `senna gem-encoder` already refuses the same flag by name — this is
/// the masked family catching up, in the OPEN interval the draw actually needs.
mod mask_fraction_bounds {
    use super::{Cli, MaskedTopicArgs};
    use clap::Parser;

    fn parse(extra: &[&str]) -> MaskedTopicArgs {
        let mut argv = vec!["senna-masked-vae", "d.zarr", "-o", "out"];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(argv).expect("parses").args
    }

    #[test]
    fn the_degenerate_rates_are_refused_by_name() {
        for bad in ["0.0", "1.0"] {
            let msg = parse(&["--mask-fraction", bad])
                .validate()
                .expect_err("a degenerate mask rate must be refused")
                .to_string();
            for needle in ["--mask-fraction", "(0, 1)"] {
                assert!(
                    msg.contains(needle),
                    "the message for {bad} must name {needle}; got: {msg}"
                );
            }
        }
        assert!(parse(&["--mask-fraction", "0.4"]).validate().is_ok());
    }

    /// The uniform schedule draws a rate per row, so ITS bounds are the ones
    /// that have to land inside the interval.
    #[test]
    fn the_uniform_schedule_bounds_are_refused_by_name() {
        let uniform = |lo: &str, hi: &str| {
            parse(&[
                "--mask-schedule",
                "uniform",
                "--mask-rate-lo",
                lo,
                "--mask-rate-hi",
                hi,
            ])
            .validate()
        };
        for (lo, hi, flag) in [
            ("0.0", "0.6", "--mask-rate-lo"),
            ("0.1", "1.0", "--mask-rate-hi"),
        ] {
            let msg = uniform(lo, hi)
                .expect_err("a degenerate uniform bound must be refused")
                .to_string();
            assert!(
                msg.contains(flag) && msg.contains("(0, 1)"),
                "the message must name {flag} and the interval; got: {msg}"
            );
        }
        assert!(uniform("0.1", "0.6").is_ok());
    }
}
