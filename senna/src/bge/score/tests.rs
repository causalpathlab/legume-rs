//! `BgeEmbedding::open`'s gem-only additions: the row axis becomes a
//! `TrackPlan`, and the manifest's `track_encoders` slots resolve to numeric
//! track ids by matching their recorded name against that RE-DERIVED plan,
//! never a stored id (which could drift if the axis's own track order ever
//! changed).
//!
//! Runs a real, tiny gem fit rather than hand-building parquet files, so the
//! fixture tracks the real writer's schema instead of a hand-typed guess at
//! it. `QueryProjector::Tracks::place` itself (the actual encode_edges +
//! polish path) is covered by `predict::tests`'s end-to-end gem test; what is
//! specific to `open` and worth its own unit coverage is the path-resolution
//! step tested here.

use super::*;
use crate::gem::args::GemArgs;
use crate::gem::run::run_gem_embedding;
use crate::gem::test_fixtures::{genes_file, m6a_file};
use crate::run_manifest::{RunManifest, TrackEncoderSlot};
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: GemArgs,
}

/// A tiny mixed-axis (count + m6a) gem fit, real end to end so every file
/// `BgeEmbedding::open` reads back carries the real writer's own schema.
/// GENE1 carries both count channels and both m6a channels; GENE2 only
/// `count/spliced` — four tracks total: `count/spliced`, `count/unspliced`,
/// `m6a/methylated`, `m6a/unmethylated`.
fn fit_gem(dir: &std::path::Path) -> String {
    let genes = genes_file(dir);
    let m6a = m6a_file(dir);
    let out = dir.join("run").to_string_lossy().into_owned();
    let cli = Cli::try_parse_from([
        "senna-gem",
        &genes,
        "--modality",
        &m6a,
        "--epochs",
        "2",
        "--skip-etm",
        "--no-emit-pb-reference",
        "--embedding-dim",
        "4",
        "--offset-rank",
        "2",
        "--phase1-cells-per-pb",
        "0",
        "-o",
        &out,
    ])
    .expect("GemArgs parses");
    run_gem_embedding(&cli.args).expect("gem run must succeed");
    out
}

#[test]
fn track_encoder_slots_resolve_by_name_against_the_rederived_axis() {
    let dir = tempfile::tempdir().unwrap();
    let out = fit_gem(dir.path());

    let model = BgeEmbedding::open(&out).expect("open a gem run");
    let plan = model
        .tracks
        .as_ref()
        .expect("a gem run must carry a TrackPlan");
    assert_eq!(
        plan.tracks.len(),
        4,
        "count/spliced, count/unspliced, m6a/methylated, m6a/unmethylated"
    );

    // Only `count/unspliced` (track 1) ever gets a distilled encoder: m6a
    // tracks are not COUNT tracks (only the count modality's channels are),
    // so phase 2 never trains one for them, and the manifest never records
    // one either.
    assert_eq!(
        model.track_encoders.len(),
        1,
        "m6a tracks must carry no cell encoder: {:?}",
        model.track_encoders
    );
    let (id, path) = &model.track_encoders[0];
    assert_eq!(*id, 1, "count/unspliced must resolve to track 1");
    assert!(
        path.ends_with("run.cell_encoder.count.unspliced.safetensors"),
        "{path}"
    );
}

/// The manifest is a compatibility surface a hand-edited or drifted copy can
/// disagree with the axis it sits beside; naming a track the re-derived plan
/// does not have must be a clear, named error, not a panic or a silent
/// misplacement onto the wrong track.
#[test]
fn an_unknown_track_encoder_name_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let out = fit_gem(dir.path());

    let manifest_path = format!("{out}.senna.json");
    let (mut m, _dir) = RunManifest::load(std::path::Path::new(&manifest_path)).unwrap();
    m.outputs.track_encoders.push(TrackEncoderSlot {
        track: "atoi/edited".into(),
        path: "run.cell_encoder.atoi.edited.safetensors".into(),
    });
    m.save(std::path::Path::new(&manifest_path)).unwrap();

    let err = match BgeEmbedding::open(&out) {
        Ok(_) => panic!("an unknown track-encoder name must not open"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("atoi/edited"), "{err}");
}
