//! End-to-end regression guard for `run_gem_embedding`, on the same two-file
//! (genes + one modality) synthetic fixture `gem::load::tests` builds: GENE1
//! carries both channels of `count` and of `m6a`; GENE2 carries only
//! `count/spliced`, so it exercises the contrast skip rule for real. A tiny
//! fit (`--epochs 2 --phase1-cells-per-pb 0 --embedding-dim 4`) must:
//! - write bge's own output set, plus `feature_contrast.parquet` /
//!   `feature_contrast_bias.parquet` and a SECOND cell encoder
//!   (`cell_encoder.count.unspliced.safetensors`, for the axis's second
//!   count track);
//! - tag the manifest `RunKind::Gem`;
//! - keep succeeding with `--n-hvg 0` (selection off) and with no
//!   `--modality` file at all (genes-only, writing only the `count`
//!   contrast row).

use super::run_gem_embedding;
use crate::embed_common::*;
use crate::gem::args::GemArgs;
use crate::run_manifest::{self, RunKind};
use clap::Parser;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use matrix_util::parquet::read_parquet_string_columns_by_name;
use nalgebra::DMatrix;

fn boxes(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&s| s.into()).collect()
}

/// Mirrors `gem::load::tests::synth`: a tiny synthetic zarr backend at
/// `dir/{stem}.zarr`, one small positive integer per cell so nothing is
/// near-empty.
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

const CELLS: [&str; 6] = ["C1", "C2", "C3", "C4", "C5", "C6"];

/// The genes file every test here shares: GENE1 has both count channels,
/// GENE2 only `spliced` (no `unspliced` anywhere for GENE2).
fn genes_file(dir: &std::path::Path) -> Box<str> {
    synth(
        dir,
        "S1_genes",
        &[
            "GENE1/count/spliced",
            "GENE1/count/unspliced",
            "GENE2/count/spliced",
        ],
        &CELLS,
    )
}

fn m6a_file(dir: &std::path::Path) -> Box<str> {
    synth(
        dir,
        "S1_m6a",
        &["GENE1/m6a/methylated", "GENE1/m6a/unmethylated"],
        &CELLS,
    )
}

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: GemArgs,
}

/// The output-file suffixes `fit_embed_family` writes on the plain (no ETM,
/// no pb reference) path, shared with `bge::driver::tests`'s own guard: see
/// that test's comment for why gene modules never populate
/// `module_membership`/`module_dictionary` here.
const BGE_PARQUET_SUFFIXES: [&str; 8] = [
    "cell_embedding",
    "cell_bias",
    "feature_embedding",
    "feature_loading",
    "feature_bias",
    "dictionary",
    "pb_embedding",
    "pb_batch",
];

#[test]
fn gem_fits_at_defaults_and_writes_a_gem_manifest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = genes_file(dir.path());
    let m6a = m6a_file(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();

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
        "--phase1-cells-per-pb",
        "0",
        "-o",
        &out,
    ])
    .expect("GemArgs parses at defaults plus the tiny-fit knobs");

    run_gem_embedding(&cli.args).expect("run_gem_embedding must succeed at gem's defaults");

    let mut actual: Vec<String> = std::fs::read_dir(dir.path())
        .expect("read the run directory")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        // Exclude the two input fixtures; every real output is `run.<suffix>`.
        .filter(|name| name.starts_with("run."))
        .collect();
    actual.sort();

    let mut expected: Vec<String> = BGE_PARQUET_SUFFIXES
        .into_iter()
        .chain(["feature_contrast", "feature_contrast_bias"])
        .map(|suffix| format!("run.{suffix}.parquet"))
        .collect();
    expected.push("run.cell_encoder.safetensors".into());
    // The axis's second count track (count/unspliced) gets its own encoder
    // file, namespaced by `encoder_suffix_for`.
    expected.push("run.cell_encoder.count.unspliced.safetensors".into());
    expected.push("run.senna.json".into());
    expected.sort();

    assert_eq!(
        actual, expected,
        "gem's output set must be bge's set plus the contrast tables and the \
         second (count/unspliced) cell encoder"
    );

    let (manifest, _dir) = run_manifest::load_for(&out).expect("load the manifest back");
    assert_eq!(manifest.kind, RunKind::Gem, "manifest kind must be gem");

    // feature_loading rows are exactly the feature axis rows (the union of
    // the two input files, 5 rows).
    let loading =
        Mat::from_parquet_with_row_names(&format!("{out}.feature_loading.parquet"), Some(0))
            .expect("read feature_loading back");
    let mut rows = loading.rows;
    rows.sort();
    let mut expected_rows = boxes(&[
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE2/count/spliced",
        "GENE1/m6a/methylated",
        "GENE1/m6a/unmethylated",
    ]);
    expected_rows.sort();
    assert_eq!(
        rows, expected_rows,
        "feature_loading rows must be the feature axis rows"
    );

    // The contrast table: GENE1/count and GENE1/m6a, GENE2 skipped on both
    // (no unspliced row, no m6a row at all).
    let contrast_cols = read_parquet_string_columns_by_name(
        &format!("{out}.feature_contrast.parquet"),
        &["feature", "modality", "gene"],
    )
    .expect("read feature_contrast string columns");
    let mut features = contrast_cols[0].clone();
    features.sort();
    assert_eq!(
        features,
        boxes(&["GENE1/count", "GENE1/m6a"]),
        "contrast rows: GENE1 on both modalities, GENE2 skipped on both"
    );
}

#[test]
fn n_hvg_zero_run_succeeds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = genes_file(dir.path());
    let m6a = m6a_file(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();

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
        "--phase1-cells-per-pb",
        "0",
        "--n-hvg",
        "0",
        "-o",
        &out,
    ])
    .expect("GemArgs parses with --n-hvg 0");

    run_gem_embedding(&cli.args).expect("run_gem_embedding must succeed with --n-hvg 0");
    let (manifest, _dir) = run_manifest::load_for(&out).expect("load the manifest back");
    assert_eq!(manifest.kind, RunKind::Gem);
}

#[test]
fn genes_only_run_writes_only_the_count_contrast_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = genes_file(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();

    // No --modality at all: the axis carries only the two count tracks.
    let cli = Cli::try_parse_from([
        "senna-gem",
        &genes,
        "--epochs",
        "2",
        "--skip-etm",
        "--no-emit-pb-reference",
        "--embedding-dim",
        "4",
        "--phase1-cells-per-pb",
        "0",
        "-o",
        &out,
    ])
    .expect("GemArgs parses with genes only");

    run_gem_embedding(&cli.args).expect("run_gem_embedding must succeed with genes only");

    let contrast_path = format!("{out}.feature_contrast.parquet");
    let cols =
        read_parquet_string_columns_by_name(&contrast_path, &["feature", "modality", "gene"])
            .expect("read feature_contrast string columns");
    assert_eq!(
        cols[0],
        boxes(&["GENE1/count"]),
        "genes-only: exactly one contrast row, from the count track alone"
    );
    assert_eq!(cols[1], boxes(&["count"]));
    assert_eq!(cols[2], boxes(&["GENE1"]));

    // No modality data was ever loaded, so there is no second-track encoder
    // beyond the base + count/unspliced pair; a stray m6a encoder file would
    // mean a modality leaked in from nowhere.
    let bad = format!("{out}.cell_encoder.m6a.methylated.safetensors");
    assert!(
        !std::path::Path::new(&bad).exists(),
        "genes-only run must not write an m6a encoder"
    );
}

/// A spliced-only axis (no `count/unspliced` row anywhere, no `--modality`
/// file) has exactly one track, so no modality's two contrast channels are
/// ever both present: the contrast table has zero rows. The files must still
/// exist, correctly typed (`h0..h{H-1}` / `bias`), not be skipped, since a
/// later manifest reader expects them to be there.
#[test]
fn spliced_only_run_writes_an_empty_but_present_contrast_pair() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(
        dir.path(),
        "S1_genes",
        &["GENE1/count/spliced", "GENE2/count/spliced"],
        &CELLS,
    );
    let out = dir.path().join("run").to_string_lossy().into_owned();

    let cli = Cli::try_parse_from([
        "senna-gem",
        &genes,
        "--epochs",
        "2",
        "--skip-etm",
        "--no-emit-pb-reference",
        "--embedding-dim",
        "4",
        "--phase1-cells-per-pb",
        "0",
        "-o",
        &out,
    ])
    .expect("GemArgs parses with a spliced-only axis");

    run_gem_embedding(&cli.args).expect("run_gem_embedding must succeed on a spliced-only axis");

    let contrast_path = format!("{out}.feature_contrast.parquet");
    let bias_path = format!("{out}.feature_contrast_bias.parquet");
    assert!(
        std::path::Path::new(&contrast_path).exists(),
        "feature_contrast.parquet must exist even with zero rows"
    );
    assert!(
        std::path::Path::new(&bias_path).exists(),
        "feature_contrast_bias.parquet must exist even with zero rows"
    );

    let delta = Mat::from_parquet_with_row_names(&contrast_path, Some(0))
        .expect("read the empty feature_contrast.parquet back");
    assert_eq!(delta.rows.len(), 0);
    assert_eq!(
        delta.cols,
        vec![
            Box::from("h0"),
            Box::from("h1"),
            Box::from("h2"),
            Box::from("h3")
        ],
        "column schema must still be h0..h3 (embedding-dim 4) with zero rows"
    );
    assert_eq!(delta.mat.nrows(), 0);
    assert_eq!(delta.mat.ncols(), 4);

    let bias = Mat::from_parquet_with_row_names(&bias_path, Some(0))
        .expect("read the empty feature_contrast_bias.parquet back");
    assert_eq!(bias.rows.len(), 0);
    assert_eq!(bias.cols, vec![Box::from("bias")]);

    // `Mat::from_parquet_with_row_names` only reads NUMERIC columns (it skips
    // the string ones entirely), so the assertions above say nothing about
    // `feature`/`modality`/`gene` actually existing on either file. Read
    // them back explicitly: the columns must be present (the call itself
    // errors if a named column is missing from the schema) and, since there
    // are zero contrast rows, empty.
    let contrast_str_cols =
        read_parquet_string_columns_by_name(&contrast_path, &["feature", "modality", "gene"])
            .expect("feature_contrast.parquet must still carry feature/modality/gene columns");
    assert_eq!(contrast_str_cols.len(), 3);
    assert!(
        contrast_str_cols.iter().all(Vec::is_empty),
        "feature_contrast.parquet's feature/modality/gene columns must be present but empty: \
         {contrast_str_cols:?}"
    );

    let bias_str_cols =
        read_parquet_string_columns_by_name(&bias_path, &["feature", "modality", "gene"])
            .expect("feature_contrast_bias.parquet must still carry feature/modality/gene columns");
    assert_eq!(bias_str_cols.len(), 3);
    assert!(
        bias_str_cols.iter().all(Vec::is_empty),
        "feature_contrast_bias.parquet's feature/modality/gene columns must be present but \
         empty: {bias_str_cols:?}"
    );
}
