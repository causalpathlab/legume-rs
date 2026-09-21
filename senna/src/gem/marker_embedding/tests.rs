//! [`load_marker_feature_embedding_from`] against a manifest planted on disk.

use super::*;
use crate::run_manifest::{default_path, load_for, RunKind, RunManifest};
use legume_numeric::matrix::traits::IoOps;
use std::path::Path;

fn load_for_prefix(prefix: &str) -> MatWithNames<DMatrix<f32>> {
    let (manifest, dir) = load_for(prefix).expect("load manifest");
    load_marker_feature_embedding_from(&manifest, &dir, prefix).expect("load feature embedding")
}

fn load_err(prefix: &str) -> String {
    let (manifest, dir) = load_for(prefix).expect("load manifest");
    load_marker_feature_embedding_from(&manifest, &dir, prefix)
        .err()
        .expect("expected error")
        .to_string()
}

/// Plant `{prefix}.feature_coembedding.parquet` with the given row names and
/// return its basename, the way a manifest records it.
fn plant_feature_coembedding(prefix: &str, rows: &[&str]) -> String {
    let path = format!("{prefix}.feature_coembedding.parquet");
    let mat = DMatrix::<f32>::from_fn(rows.len(), 2, |i, j| (i * 2 + j) as f32);
    let row_names: Vec<Box<str>> = rows.iter().map(|&s| s.into()).collect();
    let cols: Vec<Box<str>> = vec!["h0".into(), "h1".into()];
    mat.to_parquet_with_names(&path, (Some(&row_names), Some("feature")), Some(&cols))
        .expect("write fixture parquet");
    Path::new(&path)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

fn write_manifest(prefix: &str, kind: RunKind, coembedding_basename: String) {
    let mut m = RunManifest::new(kind, prefix);
    m.outputs.feature_coembedding = Some(coembedding_basename);
    m.save(Path::new(&default_path(prefix)))
        .expect("save manifest");
}

/// A `gem`-kind run carries two rows per gene; the marker loader must keep only
/// the spliced one and re-key it by gene, dropping any other track.
#[test]
fn gem_kind_keeps_only_the_spliced_row_per_gene() {
    let dir = tempfile::tempdir().expect("tmp");
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let rows = [
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE1/m6a/methylated",
    ];
    let basename = plant_feature_coembedding(&prefix, &rows);
    write_manifest(&prefix, RunKind::Gem, basename);

    let feat = load_for_prefix(&prefix);
    assert_eq!(feat.rows.as_slice(), [Box::<str>::from("GENE1")].as_slice());
    assert_eq!(feat.mat.nrows(), 1);
    assert_eq!(feat.mat.ncols(), 2);
}

/// A non-`gem` kind's feature embedding is already gene-keyed; it must pass
/// through unchanged, row-for-row and value-for-value.
#[test]
fn non_gem_kind_passes_the_table_through_untouched() {
    let dir = tempfile::tempdir().expect("tmp");
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let rows = [
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE1/m6a/methylated",
    ];
    let basename = plant_feature_coembedding(&prefix, &rows);
    write_manifest(&prefix, RunKind::Bge, basename);

    let feat = load_for_prefix(&prefix);
    let expect_rows: Vec<Box<str>> = rows.iter().map(|&s| s.into()).collect();
    assert_eq!(feat.rows, expect_rows);
    assert_eq!(feat.mat.nrows(), 3);
    assert_eq!(feat.mat.ncols(), 2);
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(feat.mat[(i, j)], (i * 2 + j) as f32);
        }
    }
}

/// A kind that co-embeds but recorded no co-embed (an interrupted run) is
/// refused: its ρ is off the cell manifold. A kind that never co-embeds
/// (`fne`) matches on its ρ, which is the only gene table it has.
#[test]
fn a_coembedding_kind_without_its_coembed_is_refused_and_fne_uses_rho() {
    let dir = tempfile::tempdir().expect("tmp");
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let rho = format!("{prefix}.feature_embedding.parquet");
    let mat = DMatrix::<f32>::from_fn(2, 2, |i, j| (i * 2 + j) as f32);
    let rows: Vec<Box<str>> = vec!["GENE1".into(), "GENE2".into()];
    mat.to_parquet_with_names(&rho, (Some(&rows), Some("feature")), None)
        .expect("write ρ");
    let basename = "run.feature_embedding.parquet".to_string();

    let mut m = RunManifest::new(RunKind::Bge, &prefix);
    m.outputs.feature_embedding = Some(basename.clone());
    m.save(Path::new(&default_path(&prefix))).expect("save");
    let err = load_err(&prefix);
    assert!(err.contains("feature_coembedding"), "{err}");

    let mut m = RunManifest::new(RunKind::Fne, &prefix);
    m.outputs.feature_embedding = Some(basename);
    m.save(Path::new(&default_path(&prefix))).expect("save");
    let feat = load_for_prefix(&prefix);
    assert_eq!(feat.rows, rows);
}
