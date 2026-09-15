//! Smoke test for `run_gem_embedding`: a real (tiny) fit at DEFAULTS, over the
//! same two-file (genes + one modality) synthetic fixture `gem::load::tests`
//! builds, must actually complete and write a manifest tagged `RunKind::Gem`.
//! Task 5b extends this into the multi-track / contrast-output regression
//! guard.

use super::run_gem_embedding;
use crate::gem::args::GemArgs;
use crate::run_manifest::{self, RunKind};
use clap::Parser;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
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

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: GemArgs,
}

#[test]
fn gem_fits_at_defaults_and_writes_a_gem_manifest() {
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

    let (manifest, _dir) = run_manifest::load_for(&out).expect("load the manifest back");
    assert_eq!(manifest.kind, RunKind::Gem, "manifest kind must be gem");
}
