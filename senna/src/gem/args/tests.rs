use super::GemArgs;
use clap::{CommandFactory, Parser};

/// `GemArgs` is an `Args` group, not a `Parser`; wrap it to parse standalone.
#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: GemArgs,
}

fn parse(argv: &[&str]) -> GemArgs {
    let mut full = vec!["senna-gem"];
    full.extend_from_slice(argv);
    Cli::try_parse_from(full).expect("parse").args
}

fn try_parse(argv: &[&str]) -> Result<GemArgs, clap::Error> {
    let mut full = vec!["senna-gem"];
    full.extend_from_slice(argv);
    Cli::try_parse_from(full).map(|c| c.args)
}

#[test]
fn parses_genes_and_repeated_modality_flags() {
    let a = parse(&[
        "a.zarr.zip",
        "b.zarr.zip",
        "--modality",
        "x_m6a.zarr.zip,y_m6a.zarr.zip",
        "--modality",
        "z_apa.zarr.zip",
        "-o",
        "out",
    ]);
    assert_eq!(a.genes.len(), 2);
    assert_eq!(&*a.genes[0], "a.zarr.zip");
    assert_eq!(&*a.genes[1], "b.zarr.zip");
    let mods: Vec<&str> = a.modality_files.iter().map(AsRef::as_ref).collect();
    assert_eq!(
        mods,
        vec!["x_m6a.zarr.zip", "y_m6a.zarr.zip", "z_apa.zarr.zip"]
    );
}

#[test]
fn defaults() {
    let a = parse(&["a.zarr.zip", "-o", "out"]);
    assert_eq!(a.phase1_cells_per_pb, 16);
    assert_eq!(a.collapse.proj_dim, 50);
    assert_eq!(a.collapse.knn_cells, 10);
    assert_eq!(a.collapse.iter_opt, 30);
    assert_eq!(a.seed, 1);
    assert_eq!(a.hvg.n_hvg, 5000);
    assert_eq!(a.offset_l2, 1.0);
}

/// Every flag the driver's composite β-sharing engine (or gem's old
/// collapse/runtime surface) used to own, and must fail to parse now that
/// `GemArgs` mirrors `BgeArgs`.
#[test]
fn dropped_flags_fail_to_parse() {
    let dropped = [
        "--genes",
        "--delta-l2",
        "--feature-embedding-l2",
        "--nce-objective",
        "--num-opt-iter",
        "--knn-pb",
        "--max-grad-norm",
        "--gpu-mem-fraction",
        "--lineage-dag",
        "--lineage-smooth",
        "--dense-dag",
        "--sequential-velocity",
        "--markers",
        "--batches-per-epoch",
        "--no-preload-data",
        "--threads",
        "--feature-name-delim",
        "--feature-name-exact",
    ];
    for flag in dropped {
        let err = try_parse(&["a.zarr.zip", "-o", "out", flag]);
        assert!(err.is_err(), "{flag} must fail to parse (dropped)");
    }
}

#[test]
fn serde_round_trip_is_stable() {
    let a = parse(&["a.zarr.zip", "-o", "out", "--offset-l2", "2.5"]);
    let v1 = serde_json::to_value(&a).expect("serialize");
    let a2: GemArgs = serde_json::from_value(v1.clone()).expect("deserialize");
    let v2 = serde_json::to_value(&a2).expect("serialize again");
    assert_eq!(v1, v2, "a serde round trip must reproduce the same record");
}

#[test]
fn a_record_missing_offset_l2_replays_with_the_clap_default() {
    let a = parse(&["a.zarr.zip", "-o", "out"]);
    let mut v = serde_json::to_value(&a).expect("serialize");
    v.as_object_mut()
        .expect("object")
        .remove("offset_l2")
        .expect("offset_l2 was present before removal");
    let replayed: GemArgs = serde_json::from_value(v).expect("deserialize without offset_l2");
    assert_eq!(replayed.offset_l2, 1.0);
}

/// Every flag `gem/args.rs` declares directly (excludes the flattened shared
/// groups `hvg` / `collapse` / `qc` / `modules`, whose help text belongs to
/// their own source files and is governed there, not here).
const GEM_OWN_ARG_IDS: [&str; 21] = [
    "genes",
    "modality_files",
    "batch_files",
    "genes_sample_strip",
    "embedding_dim",
    "phase1_cells_per_pb",
    "modules_per_unit",
    "skip_etm",
    "num_topics",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "block_size",
    "preload_data",
    "offset_l2",
    "offset_rank",
    "seed",
    "device",
    "device_no",
    "out",
];

#[test]
fn gems_own_flags_have_no_em_dash_and_wrap_under_100_cols() {
    let cmd = Cli::command();
    for id in GEM_OWN_ARG_IDS {
        let arg = cmd
            .get_arguments()
            .find(|a| a.get_id() == id)
            .unwrap_or_else(|| panic!("gem/args.rs must declare --{id}"));
        for text in [arg.get_help(), arg.get_long_help()].into_iter().flatten() {
            let s = text.to_string();
            assert!(
                !s.contains('\u{2014}'),
                "{id}: em dash (U+2014) in help: {s:?}"
            );
            for line in s.lines() {
                assert!(
                    line.chars().count() <= 100,
                    "{id}: help line over 100 columns: {line:?}"
                );
            }
        }
    }
}

/// Sibling to `gems_own_flags_have_no_em_dash_and_wrap_under_100_cols`, but for
/// the `gem` subcommand's own `about`/`long_about` (`senna/src/main.rs`'s
/// `Commands::Gem` block), which lives outside `GemArgs` entirely and so isn't
/// covered by that test.
#[test]
fn gem_subcommand_about_has_no_em_dash_and_wraps_under_100_cols() {
    let cmd = crate::Cli::command();
    let gem = cmd
        .find_subcommand("gem")
        .expect("main.rs must declare the `gem` subcommand");
    for text in [gem.get_about(), gem.get_long_about()]
        .into_iter()
        .flatten()
    {
        let s = text.to_string();
        assert!(
            !s.contains('\u{2014}'),
            "gem subcommand about/long_about: em dash (U+2014): {s:?}"
        );
        for line in s.lines() {
            assert!(
                line.chars().count() <= 100,
                "gem subcommand about/long_about: line over 100 columns: {line:?}"
            );
        }
    }
}

/// `render_long_help()` runs cleanly end to end AND every flattened group
/// actually reaches the render: one distinctive flag from `GemArgs` itself
/// plus one from each of `HvgCliArgs`, `refine_weighting::CollapseArgs`,
/// `QcArgs` and `ge::FeatureModuleArgs`, so a group dropped from the flatten (or
/// renamed out from under this test) fails here instead of only showing up
/// as a missing flag in `senna gem --help`. The full text, shared groups
/// included, is still read by hand per the task's help-review step.
#[test]
fn full_help_renders() {
    let help = Cli::command().render_long_help().to_string();
    for flag in [
        "--offset-l2",          // GemArgs
        "--genes-sample-strip", // GemArgs
        "--embedding-dim",      // GemArgs
        "--n-hvg",              // HvgCliArgs
        "--num-levels",         // refine_weighting::CollapseArgs
        "--no-qc",              // QcArgs
        "--feature-modules",    // ge::FeatureModuleArgs
    ] {
        assert!(help.contains(flag), "gem --help is missing {flag}");
    }
}

/// The feature-embedding triple, `--embedding-dim auto` and `--offset-rank`
/// parse on gem exactly as on bge; the rank has its own default and is never
/// read off the embedding dimension.
#[test]
fn the_feature_embedding_triple_the_auto_dim_and_the_offset_rank_parse() {
    use graph_embedding_util::{EmbeddingDim, LoraSpec, PresetMode};
    let a = parse(&["a.zarr", "-o", "out"]);
    assert_eq!(a.offset_rank, LoraSpec::default().rank);
    assert_eq!(a.embedding_dim, EmbeddingDim::Fixed(128));
    assert!(a.feature_embedding.resolve().unwrap().is_none());

    let a = parse(&[
        "a.zarr",
        "-o",
        "out",
        "--freeze-feature-embedding",
        "prev",
        "--embedding-dim",
        "auto",
        "--offset-rank",
        "3",
    ]);
    assert_eq!(a.offset_rank, 3);
    assert_eq!(a.embedding_dim, EmbeddingDim::Auto);
    assert!(matches!(
        a.feature_embedding.resolve().unwrap(),
        Some(("prev", PresetMode::Freeze))
    ));

    let a = parse(&[
        "a.zarr",
        "-o",
        "out",
        "--lora-feature-embedding",
        "prev",
        "--lora-rank",
        "4",
    ]);
    assert!(matches!(
        a.feature_embedding.resolve().unwrap(),
        Some(("prev", PresetMode::Lora(s))) if s.rank == 4
    ));
    assert!(try_parse(&[
        "a.zarr",
        "-o",
        "out",
        "--freeze-feature-embedding",
        "p",
        "--init-feature-embedding",
        "q",
    ])
    .is_err());
}
