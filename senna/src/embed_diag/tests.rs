//! `collect_geometry` against a manifest planted on disk, written through the
//! same parquet writer senna's runs use, so the read path is the real one.

use super::*;
use crate::run_manifest::{default_path, RunKind, RunManifest};
use graph_embedding_util::embedding_col_names;
use std::path::Path;

/// Write `m` as `{prefix}.{suffix}` the way `save_embedding` does, and return
/// the basename the manifest should record.
fn plant_table(prefix: &str, suffix: &str, m: &DMatrix<f32>) -> String {
    let path = format!("{prefix}.{suffix}");
    let rows: Vec<Box<str>> = (0..m.nrows()).map(|i| format!("r{i}").into()).collect();
    let cols = embedding_col_names(m.ncols());
    m.to_parquet_with_names(&path, (Some(&rows), Some("cell")), Some(&cols))
        .expect("write fixture parquet");
    Path::new(&path)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

/// Sixteen rows on four balanced ± axes: full rank, no common mode.
fn balanced() -> DMatrix<f32> {
    let h = 4;
    let n = 16;
    DMatrix::from_fn(n, h, |i, j| {
        if j == i % h {
            if (i / h).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            }
        } else {
            0.0
        }
    })
}

/// Ten rows along one direction: rank one.
fn rank_one() -> DMatrix<f32> {
    let dir = [0.5f32, -0.5, 0.5, -0.5];
    DMatrix::from_fn(10, 4, |i, j| ((i as f32) - 5.0) * dir[j])
}

#[test]
fn every_recorded_table_is_measured_in_report_order() {
    let dir = tempfile::tempdir().expect("tmp");
    let prefix = dir.path().join("run").to_string_lossy().into_owned();

    let cells = balanced();
    let genes = rank_one();
    let mut manifest = RunManifest::new(RunKind::Bge, &prefix);
    manifest.outputs.cell_embedding = Some(plant_table(&prefix, "cell_embedding.parquet", &cells));
    manifest.outputs.feature_loading =
        Some(plant_table(&prefix, "feature_loading.parquet", &genes));
    // No module dictionary recorded: it must simply be absent from the report.
    manifest
        .save(Path::new(&default_path(&prefix)))
        .expect("save manifest");

    let rows = collect_geometry(&prefix).expect("collect");
    let names: Vec<&str> = rows.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["cell_embedding", "feature_loading"]);

    // f32 through parquet is lossless, so the readout must equal a direct
    // measurement of the same matrix exactly — not approximately.
    assert_eq!(rows[0].1, embedding_geometry(&cells));
    assert_eq!(rows[1].1, embedding_geometry(&genes));
    assert!(
        rows[0].1.eff_rank_raw > 3.9 && rows[1].1.eff_rank_raw < 1.05,
        "the planted geometry must survive the round trip: {rows:?}"
    );
}

#[test]
fn the_manifest_path_and_the_prefix_resolve_to_the_same_report() {
    let dir = tempfile::tempdir().expect("tmp");
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let mut manifest = RunManifest::new(RunKind::Bge, &prefix);
    manifest.outputs.cell_embedding =
        Some(plant_table(&prefix, "cell_embedding.parquet", &balanced()));
    let manifest_path = default_path(&prefix);
    manifest.save(Path::new(&manifest_path)).expect("save");

    let by_prefix = collect_geometry(&prefix).expect("prefix");
    let by_path = collect_geometry(&manifest_path).expect("path");
    assert_eq!(by_prefix, by_path);
}

/// A run that records none of the measurable tables is a wrong `--from`, and
/// the error must say what it looked for.
#[test]
fn a_manifest_with_nothing_measurable_is_an_error_naming_the_tables() {
    let dir = tempfile::tempdir().expect("tmp");
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    RunManifest::new(RunKind::Bge, &prefix)
        .save(Path::new(&default_path(&prefix)))
        .expect("save");

    let err = collect_geometry(&prefix).expect_err("nothing to measure");
    let msg = err.to_string();
    for t in TABLES {
        assert!(msg.contains(t), "error must name `{t}`: {msg}");
    }
}
