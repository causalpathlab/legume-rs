//! Regression guard for the shared bge driver: `senna bge` must keep writing
//! the same output-file set and manifest `kind` once its body moves into
//! [`super::fit_embed_family`]. Written before the extraction (see the task
//! report for the honest RED/GREEN ordering — the crate did not compile at
//! all until the extraction was finished, so "RED" here is a compile
//! failure, not a failing assertion).

use clap::Parser;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use nalgebra::DMatrix;

use crate::bge::{fit_bge, BgeArgs};
use crate::run_manifest::{self, RunKind};

const N_GENES: usize = 12;
const N_CELLS: usize = 40;

/// A tiny synthetic count backend: `GENE1..GENE12` rows × `C1..C40` columns,
/// one batch, every entry a small positive integer so no cell or gene is
/// empty (near-empty cell QC would otherwise drop it before it reaches the
/// fit).
fn synthetic_backend(dir: &std::path::Path) -> Box<str> {
    let path = dir.join("fixture.zarr");
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_fn(N_GENES, N_CELLS, |g, c| {
        (((g * 7 + c * 11 + 3) % 9) + 1) as f32
    });
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr))
        .expect("create synthetic backend");
    let genes: Vec<Box<str>> = (1..=N_GENES).map(|i| format!("GENE{i}").into()).collect();
    let cells: Vec<Box<str>> = (1..=N_CELLS).map(|i| format!("C{i}").into()).collect();
    b.register_row_names_vec(&genes);
    b.register_column_names_vec(&cells);
    path
}

/// `BgeArgs` is an `Args` group, not a `Parser`; wrap it to parse standalone
/// (matches the pattern already used for other arg-group tests, e.g.
/// `masked_topic_tests.rs`).
#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: BgeArgs,
}

fn parse(data: &str, out: &str) -> BgeArgs {
    Cli::try_parse_from([
        "senna-bge",
        data,
        "-o",
        out,
        "--epochs",
        "2",
        "--skip-etm",
        "--no-emit-pb-reference",
        "--embedding-dim",
        "4",
    ])
    .expect("BgeArgs parses")
    .args
}

#[test]
fn bge_writes_its_documented_output_set_and_manifest_kind() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = synthetic_backend(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();

    let args = parse(&data, &out);
    fit_bge(&args).expect("fit_bge on the synthetic fixture");

    // The exact suffix set `fit_bge` writes for `--skip-etm
    // --no-emit-pb-reference`, confirmed by running this test against the
    // pre-extraction `fit_bge` (see the task report for the RED/GREEN
    // evidence): the plain hier phase 1 composes its dictionary through an
    // internal hard gene partition but never builds the *learned* module
    // layer (`model.modules` stays `None` on this path — see
    // `FitConfig::gene_modules`'s doc), so `write_module_tables` is a no-op
    // regardless of `--gene-modules`; no ETM tables (`--skip-etm`); no
    // `pb_reference.zarr` (`--no-emit-pb-reference`).
    let mut actual: Vec<String> = std::fs::read_dir(dir.path())
        .expect("read the run directory")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        // Exclude the input fixture (`fixture.zarr`); every real output is
        // named `run.<suffix>`.
        .filter(|name| name.starts_with("run."))
        .collect();
    actual.sort();

    let mut expected: Vec<String> = [
        "cell_embedding",
        "cell_bias",
        "feature_embedding",
        "feature_loading",
        "feature_bias",
        "dictionary",
        "pb_embedding",
        "pb_batch",
    ]
    .into_iter()
    .map(|suffix| format!("run.{suffix}.parquet"))
    .collect();
    expected.push("run.cell_encoder.safetensors".into());
    expected.push("run.senna.json".into());
    expected.sort();

    assert_eq!(
        actual, expected,
        "fit_bge's output-file set changed under --skip-etm --no-emit-pb-reference"
    );

    let (manifest, _dir) = run_manifest::load_for(&out).expect("load the manifest back");
    assert_eq!(manifest.kind, RunKind::Bge, "manifest kind must be bge");
}
