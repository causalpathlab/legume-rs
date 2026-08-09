//! The run manifest carries the fit configuration `senna update` replays.
//!
//! The risk this guards is not the writer but the **readers**: every downstream
//! command loads a manifest, mutates one section and saves it back. If `train_args`
//! failed to survive that round trip, a run would silently lose the ability to
//! be updated the first time anything touched its manifest — `senna layout`,
//! `senna clustering`, `senna plot`. Nothing else would look wrong.

use senna::run_manifest::{record_train_args, RunKind, RunManifest, TrainArgsRecord};
use std::path::PathBuf;
use tempfile::TempDir;

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
struct FakeArgs {
    epochs: usize,
    out: String,
    layers: Vec<usize>,
}

fn fake() -> FakeArgs {
    FakeArgs {
        epochs: 1000,
        out: "run-a".into(),
        layers: vec![128, 1024, 128],
    }
}

/// A scratch manifest path in a per-test temp dir. The `TempDir` is returned
/// because dropping it deletes the directory — bind it for the test body.
fn scratch() -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create scratch dir");
    let path = dir.path().join("run.senna.json");
    (dir, path)
}

/// The regression guard: load → mutate an unrelated section → save → load.
/// This is exactly what `layout` / `clustering` / `annotate` do.
#[test]
fn train_args_survives_an_unrelated_mutation() {
    let (_dir, path) = scratch();
    let mut m = RunManifest::new(RunKind::Itopic, "run-b");
    m.train_args = Some(record_train_args(&fake()).expect("record"));
    m.save(&path).expect("save");

    let (mut reloaded, _) = RunManifest::load(&path).expect("load");
    reloaded.defaults.colour_by = Some("cluster".into());
    reloaded.save(&path).expect("re-save");

    let (again, _) = RunManifest::load(&path).expect("load again");
    let back: FakeArgs = again
        .train_args_as("run-b")
        .expect("fit configuration must survive a downstream command touching the manifest");
    assert_eq!(back, fake());
}

/// Manifests written before `train_args` existed must still load.
#[test]
fn manifest_without_train_args_still_loads() {
    let (_dir, path) = scratch();
    std::fs::write(
        &path,
        r#"{"version":1,"kind":"topic","prefix":"old","data":{"input":["a.zarr"]}}"#,
    )
    .expect("write legacy manifest");

    let (m, _) = RunManifest::load(&path).expect("a pre-train_args manifest must still load");
    assert_eq!(m.kind, RunKind::Topic);
    assert_eq!(m.data.input, vec!["a.zarr".to_string()]);
    assert!(m.train_args.is_none());
}

/// The property that keeps `update` usable across senna versions.
///
/// A fit recorded before a flag existed must replay with that flag's **declared
/// default**, not with `Default::default()`. The difference is `epochs: 1000`
/// versus `epochs: 0` — the second trains nothing and reports success.
#[test]
fn a_field_added_later_replays_with_its_clap_default() {
    use senna::embed_common::{clap_defaults, Args};

    /// Stands in for a fit struct that has since gained `--epochs`.
    ///
    /// It deliberately mixes the argument shapes the real ones use — a
    /// required positional-ish value, a boolean flag (which accepts no value,
    /// and once made `clap_defaults` panic when handed a placeholder), a `Vec`,
    /// and an `Option`.
    #[derive(Args, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
    #[serde(default)]
    struct Fit {
        #[arg(long, required = true)]
        out: String,
        #[arg(long, default_value_t = 1000)]
        epochs: usize,
        #[arg(long, default_value_t = 0.01)]
        learning_rate: f32,
        #[arg(long)]
        ignore_batch: bool,
        #[arg(long, value_delimiter = ',', default_values_t = vec![128, 1024])]
        encoder_layers: Vec<usize>,
        #[arg(long)]
        init_from: Option<String>,
    }
    impl Default for Fit {
        fn default() -> Self {
            clap_defaults()
        }
    }

    // Relaxing `required` must not panic, and declared defaults must come
    // through rather than the type's zero.
    let d = Fit::default();
    assert_eq!(d.epochs, 1000, "clap's default_value_t, not 0");
    assert!((d.learning_rate - 0.01).abs() < 1e-9);
    assert!(!d.ignore_batch);
    assert_eq!(d.encoder_layers, vec![128, 1024]);
    assert_eq!(d.init_from, None);

    // A record written before `epochs` and `learning_rate` existed.
    let mut m = RunManifest::new(RunKind::Topic, "run-d");
    m.train_args = Some(TrainArgsRecord {
        senna_version: "0.0.1-old".into(),
        args: serde_json::json!({ "out": "run-d" }),
    });

    let replayed: Fit = m.train_args_as("run-d").expect("older record must replay");
    assert_eq!(replayed.out, "run-d");
    assert_eq!(replayed.epochs, 1000, "must not train for zero epochs");
    assert!((replayed.learning_rate - 0.01).abs() < 1e-9);
}
