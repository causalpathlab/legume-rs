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
    use clap::Parser;
    use senna::run_manifest::{RunKind, RunManifest, TrainArgsRecord};

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
    use clap::Parser;
    use senna::run_manifest::{RunKind, RunManifest, TrainArgsRecord};

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
    use clap::Parser;
    use senna::run_manifest::{RunKind, RunManifest, TrainArgsRecord};

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
/// asked. This refuses the flag by name instead, in the OPEN interval the
/// draw actually needs.
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

////////////////////////////////////////////////////////////////
// --lora-feature-embedding end to end on a planted data set  //
////////////////////////////////////////////////////////////////

use super::fit_masked_topic_model;
use clap::Parser;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use legume_numeric::candle::candle_core;
use legume_numeric::matrix::traits::IoOps;
use senna::embed_common::Mat;
use std::path::Path;

fn parse_masked(argv: &[&str]) -> MaskedTopicArgs {
    let mut full = vec!["masked-topic"];
    full.extend_from_slice(argv);
    Cli::try_parse_from(full).expect("valid argv").args
}

/// Two groups of cells, each with its own block of high genes and sparse
/// background elsewhere, large enough for the pseudobulk tree to build.
fn planted_zarr(dir: &Path) -> String {
    let (n_genes, n_cells) = (60usize, 200usize);
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..n_cells {
        let grp = usize::from(c >= n_cells / 2);
        for g in 0..n_genes {
            let own = usize::from(g >= n_genes / 2) == grp;
            let x = if own {
                3 + (c + g) % 4
            } else if (c * 7 + g) % 5 == 0 {
                1
            } else {
                0
            };
            if x > 0 {
                triplets.push((g as u64, c as u64, x as f32));
            }
        }
    }
    let nnz = triplets.len();
    let path = dir.join("planted.zarr").to_string_lossy().into_owned();
    let mut b = create_sparse_from_triplets(
        &triplets,
        (n_genes, n_cells, nnz),
        Some(&path),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("backend");
    b.register_row_names_vec(
        &(0..n_genes)
            .map(|g| format!("GENE{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..n_cells)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    path
}

/// The safetensors header's tensor names.
fn checkpoint_keys(prefix: &str) -> Vec<String> {
    let bytes = std::fs::read(format!("{prefix}.safetensors")).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let mut keys: Vec<String> = hdr
        .as_object()
        .unwrap()
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .cloned()
        .collect();
    keys.sort();
    keys
}

/// Anchored to an earlier run's table, a second run writes that table plus a
/// residual of exactly the given rank, takes H from the table, and saves a
/// checkpoint with one feature table equal to what it wrote — no factor
/// tensors survive the fold, so `predict` reads a plain model.
#[test]
fn masked_topic_anchors_rho_with_a_low_rank_residual_and_folds_it_before_saving() {
    let dir = tempfile::tempdir().unwrap();
    let data = planted_zarr(dir.path());
    let first = dir.path().join("first").to_string_lossy().into_owned();
    let second = dir.path().join("second").to_string_lossy().into_owned();
    let common = [
        "-t",
        "3",
        "-i",
        "3",
        "--gene-modules",
        "0",
        "--minibatch-size",
        "50",
    ];
    let mut argv = vec![data.as_str(), "-o", &first, "--embedding-dim", "8"];
    argv.extend_from_slice(&common);
    fit_masked_topic_model(&parse_masked(&argv)).unwrap();
    let mut argv = vec![
        data.as_str(),
        "-o",
        &second,
        "--embedding-dim",
        "auto",
        "--lora-feature-embedding",
        &first,
        "--lora-rank",
        "2",
        "--lora-lr-ratio",
        "4",
        "--seed",
        "11",
    ];
    argv.extend_from_slice(&common);
    fit_masked_topic_model(&parse_masked(&argv)).unwrap();

    let a = Mat::from_parquet(&format!("{first}.feature_embedding.parquet")).unwrap();
    let b = Mat::from_parquet(&format!("{second}.feature_embedding.parquet")).unwrap();
    assert_eq!(a.rows, b.rows);
    assert_eq!(b.mat.ncols(), 8, "H taken from the table");
    let sv = (&b.mat - &a.mat).singular_values();
    assert!(sv[0] > 1e-6, "the residual never moved");
    assert!(sv[2] <= 1e-4 * sv[0], "the residual is not rank 2: {sv}");

    let keys = checkpoint_keys(&second);
    assert!(keys.iter().any(|k| k == "enc.feature.embeddings"));
    assert!(
        !keys.iter().any(|k| k.contains("lora")),
        "factor tensors survived the fold: {keys:?}"
    );
    let saved =
        candle_core::safetensors::load(format!("{second}.safetensors"), &candle_core::Device::Cpu)
            .unwrap();
    let table: Vec<f32> = saved["enc.feature.embeddings"]
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let written: Vec<f32> = (0..b.mat.nrows())
        .flat_map(|g| b.mat.row(g).iter().copied().collect::<Vec<_>>())
        .collect();
    for (x, y) in table.iter().zip(&written) {
        assert!((x - y).abs() < 1e-5, "checkpoint {x} vs written {y}");
    }
}

/// The LoRA settings are checked against the width the table fixes, as on every
/// other engine: a residual as wide as the table is no low-rank residual.
#[test]
fn a_lora_rank_as_wide_as_the_table_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let data = planted_zarr(dir.path());
    let first = dir.path().join("first").to_string_lossy().into_owned();
    let second = dir.path().join("second").to_string_lossy().into_owned();
    let common = [
        "-t",
        "3",
        "-i",
        "1",
        "--gene-modules",
        "0",
        "--minibatch-size",
        "50",
    ];
    let mut argv = vec![data.as_str(), "-o", &first, "--embedding-dim", "4"];
    argv.extend_from_slice(&common);
    fit_masked_topic_model(&parse_masked(&argv)).unwrap();
    let mut argv = vec![
        data.as_str(),
        "-o",
        &second,
        "--embedding-dim",
        "auto",
        "--lora-feature-embedding",
        &first,
        "--lora-rank",
        "4",
    ];
    argv.extend_from_slice(&common);
    let err = fit_masked_topic_model(&parse_masked(&argv))
        .expect_err("rank 4 on a width-4 table must be refused");
    assert!(err.to_string().contains("rank"), "{err}");
}

/// A pinned table wider than the data: its unmatched rows come out after the
/// data's genes, unchanged, with a types table over every row; the model's
/// own per-gene tables stay on the data's axis.
#[test]
fn masked_topic_carries_the_unmatched_rows_of_a_pinned_table_through() {
    use crate::feature_preset::test_support::{assert_carried, widen};
    let dir = tempfile::tempdir().unwrap();
    let data = planted_zarr(dir.path());
    let first = dir.path().join("first").to_string_lossy().into_owned();
    let plus = dir.path().join("plus").to_string_lossy().into_owned();
    let second = dir.path().join("second").to_string_lossy().into_owned();
    let common = [
        "-t",
        "3",
        "-i",
        "2",
        "--gene-modules",
        "0",
        "--minibatch-size",
        "50",
    ];
    let mut argv = vec![data.as_str(), "-o", &first, "--embedding-dim", "8"];
    argv.extend_from_slice(&common);
    fit_masked_topic_model(&parse_masked(&argv)).unwrap();
    let extra = widen(&format!("{first}.feature_embedding.parquet"), &plus);
    let mut argv = vec![
        data.as_str(),
        "-o",
        &second,
        "--embedding-dim",
        "auto",
        "--freeze-feature-embedding",
        &plus,
    ];
    argv.extend_from_slice(&common);
    fit_masked_topic_model(&parse_masked(&argv)).unwrap();
    let a = Mat::from_parquet(&format!("{first}.feature_embedding.parquet")).unwrap();
    assert_carried(
        &second,
        &format!("{second}.feature_embedding.parquet"),
        a.rows.len(),
        &extra,
    );
    let d = Mat::from_parquet(&format!("{second}.dictionary.parquet")).unwrap();
    assert_eq!(d.rows, a.rows, "the dictionary stays on the data's genes");
}
