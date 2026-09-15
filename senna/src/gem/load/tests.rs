use super::{load_gem_data, resolve_inputs};
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use nalgebra::DMatrix;

fn boxes(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&s| s.into()).collect()
}

/// Write a tiny synthetic zarr backend at `dir/{stem}.zarr` with the given
/// row and column names, one small positive integer per cell (so nothing is
/// near-empty). Mirrors `bge::driver::tests::synthetic_backend`.
fn synth(dir: &std::path::Path, stem: &str, rows: &[&str], cols: &[&str]) -> Box<str> {
    let path = dir.join(format!("{stem}.zarr"));
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_fn(rows.len(), cols.len(), |r, c| {
        (((r * 7 + c * 11 + 3) % 9) + 1) as f32
    });
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr))
        .expect("create synthetic backend");
    b.register_row_names_vec(&boxes(rows));
    b.register_column_names_vec(&boxes(cols));
    path
}

#[test]
fn resolve_inputs_matches_by_default_suffix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(
        dir.path(),
        "s1_genes",
        &["GENE1/count/spliced"],
        &["C1", "C2"],
    );
    let m6a = synth(
        dir.path(),
        "s1_m6a",
        &["GENE1/m6a/methylated", "GENE1/m6a/unmethylated"],
        &["C1", "C2"],
    );

    let inputs = resolve_inputs(&[genes], &[m6a], "").expect("resolve_inputs");
    assert_eq!(inputs.files.len(), 2);
    // Both files match the same sample id: "s1" twice.
    assert_eq!(&*inputs.sample_ids[0], "s1");
    assert_eq!(&*inputs.sample_ids[1], "s1");
    assert_eq!(inputs.modality_of_file[0], None);
    assert_eq!(inputs.modality_of_file[1].as_deref(), Some("m6a"));
}

#[test]
fn resolve_inputs_honours_an_explicit_strip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes_dir = dir.path().join("genes");
    let mods_dir = dir.path().join("mods");
    std::fs::create_dir_all(&genes_dir).unwrap();
    std::fs::create_dir_all(&mods_dir).unwrap();
    // Same basename in different directories; an explicit strip applies
    // uniformly to every file (not the type-specific `_genes` / `_{modality}`
    // default), so both resolve to the same sample id.
    let genes = synth(
        &genes_dir,
        "sample1_batch",
        &["GENE1/count/spliced"],
        &["C1"],
    );
    let m6a = synth(
        &mods_dir,
        "sample1_batch",
        &["GENE1/m6a/methylated", "GENE1/m6a/unmethylated"],
        &["C1"],
    );

    let inputs = resolve_inputs(&[genes], &[m6a], "_batch").expect("resolve_inputs");
    assert_eq!(&*inputs.sample_ids[0], "sample1");
    assert_eq!(&*inputs.sample_ids[1], "sample1");
}

#[test]
fn a_sample_id_mismatch_names_both_id_sets() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(dir.path(), "s1_genes", &["GENE1/count/spliced"], &["C1"]);
    let m6a = synth(
        dir.path(),
        "s2_m6a",
        &["GENE1/m6a/methylated", "GENE1/m6a/unmethylated"],
        &["C1"],
    );

    let err = resolve_inputs(&[genes], &[m6a], "")
        .unwrap_err()
        .to_string();
    assert!(err.contains("s2"), "{err}");
    assert!(err.contains("s1"), "{err}");
}

#[test]
fn a_gene_file_with_an_m6a_row_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(
        dir.path(),
        "s1_genes",
        &["GENE1/count/spliced", "GENE1/m6a/methylated"],
        &["C1"],
    );

    let err = resolve_inputs(&[genes], &[], "").unwrap_err().to_string();
    assert!(err.contains("count"), "{err}");
    assert!(err.contains("m6a"), "{err}");
}

#[test]
fn end_to_end_union_axis_and_track_plan() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cells = ["C1", "C2", "C3", "C4", "C5", "C6"];
    let genes = synth(
        dir.path(),
        "S1_genes",
        &[
            "GENE1/count/spliced",
            "GENE1/count/unspliced",
            "GENE2/count/spliced",
        ],
        &cells,
    );
    let m6a = synth(
        dir.path(),
        "S1_m6a",
        &["GENE1/m6a/methylated", "GENE1/m6a/unmethylated"],
        &cells,
    );

    let inputs = resolve_inputs(&[genes], &[m6a], "").expect("resolve_inputs");
    let (unified, plan) = load_gem_data(&inputs, None, false).expect("load_gem_data");

    assert_eq!(
        unified.n_features(),
        5,
        "union feature axis: 3 count + 2 m6a rows"
    );
    assert_eq!(unified.n_cells(), 6);
    let mut barcodes: Vec<String> = unified.barcodes.iter().map(ToString::to_string).collect();
    barcodes.sort();
    let mut expected: Vec<String> = cells.iter().map(|c| format!("{c}@S1")).collect();
    expected.sort();
    assert_eq!(barcodes, expected, "cells merge as `{{barcode}}@S1`");

    // 4 tracks: count/spliced (0), count/unspliced (1), and one track per
    // m6a channel (2 = methylated, 3 = unmethylated) — NOT 3, because
    // `ge::fit::TrackSpec::validate` hard-rejects a gene repeated within one
    // track ("a track holds at most one row per gene"), and GENE1 has a row
    // on both m6a channels; they cannot share a track id.
    assert_eq!(plan.tracks.len(), 4);
    assert_eq!(plan.to_ge().count_tracks(), vec![0, 1]);
    plan.to_ge()
        .validate(unified.n_features())
        .expect("TrackSpec::validate");
}

/// `per_file_barcode_suffix` (the `@sample` tagging) must not be skipped just
/// because `--batch-files` was also given: under `Union` alignment the batch
/// file names one label per already-MERGED cell, not per input file, so the
/// two are independent and both apply together. Same two-file fixture as
/// `end_to_end_union_axis_and_track_plan`.
#[test]
fn sample_tagging_still_applies_with_explicit_batch_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cells = ["C1", "C2", "C3", "C4", "C5", "C6"];
    let genes = synth(
        dir.path(),
        "S1_genes",
        &[
            "GENE1/count/spliced",
            "GENE1/count/unspliced",
            "GENE2/count/spliced",
        ],
        &cells,
    );
    let m6a = synth(
        dir.path(),
        "S1_m6a",
        &["GENE1/m6a/methylated", "GENE1/m6a/unmethylated"],
        &cells,
    );

    // `Union` alignment wants exactly one --batch-files file, one label per
    // UNIFIED cell (post-merge), not per input file.
    let batch_path = dir.path().join("batches.txt");
    std::fs::write(&batch_path, "b0\nb0\nb0\nb1\nb1\nb1\n").expect("write batch file");
    let batch_path: Box<str> = batch_path.to_string_lossy().into_owned().into();
    let batch_files = [batch_path];

    let inputs = resolve_inputs(&[genes], &[m6a], "").expect("resolve_inputs");
    let (unified, _plan) = load_gem_data(&inputs, Some(&batch_files), false)
        .expect("load_gem_data with --batch-files");

    assert_eq!(unified.n_cells(), 6);
    let mut barcodes: Vec<String> = unified.barcodes.iter().map(ToString::to_string).collect();
    barcodes.sort();
    let mut expected: Vec<String> = cells.iter().map(|c| format!("{c}@S1")).collect();
    expected.sort();
    assert_eq!(
        barcodes, expected,
        "cells still merge as `{{barcode}}@S1` when --batch-files is also given"
    );
    assert_eq!(unified.n_batches(), 2, "both batch labels round-trip");
}
