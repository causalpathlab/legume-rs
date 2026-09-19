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

use super::{run_gem_embedding, validate_offset_rank};
use senna::embed_common::*;
use crate::gem::args::GemArgs;
use crate::gem::test_fixtures::{boxes, genes_file, m6a_file, synth, CELLS};
use senna::run_manifest::{self, RunKind};
use clap::Parser;
use matrix_util::parquet::read_parquet_string_columns_by_name;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: GemArgs,
}

/// The output-file suffixes `fit_embed_family` writes on the plain (no ETM,
/// no pb reference) path, shared with `bge::driver::tests`'s own guard: see
/// that test's comment for why gene modules never populate
/// `module_membership`/`module_dictionary` here.
const BGE_PARQUET_SUFFIXES: [&str; 7] = [
    "cell_embedding",
    "cell_bias",
    "feature_embedding",
    "feature_coembedding",
    "feature_bias",
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
        "--offset-rank",
        "2",
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

    // feature_embedding rows are exactly the feature axis rows (the union of
    // the two input files, 5 rows).
    let loading =
        Mat::from_parquet_with_row_names(&format!("{out}.feature_embedding.parquet"), Some(0))
            .expect("read feature_embedding back");
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
        "feature_embedding rows must be the feature axis rows"
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
        "--offset-rank",
        "2",
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
        "--offset-rank",
        "2",
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

/// `--mixture-batch` and an explicit `--emit-pb-reference` both ride on the
/// shared `CollapseArgs`, but gem's collapse path carries neither a mixture
/// role nor a consumer for a carried pb reference (`senna update` cannot
/// continue a gem run). Before this test's fix, `validate_args` never called
/// `CollapseArgs::reject_pb_reference`, so both flags parsed and were then
/// silently discarded. A plain invocation (neither flag) must still succeed.
#[test]
fn mixture_batch_and_emit_pb_reference_are_refused_but_a_plain_run_still_passes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = genes_file(dir.path());

    let mixture_out = dir.path().join("mixture").to_string_lossy().into_owned();
    let cli = Cli::try_parse_from([
        "senna-gem",
        &genes,
        "--mixture-batch",
        "batchA",
        "-o",
        &mixture_out,
    ])
    .expect("GemArgs parses --mixture-batch");
    let err =
        run_gem_embedding(&cli.args).expect_err("--mixture-batch must be refused for a gem run");
    assert!(
        err.to_string()
            .contains("--mixture-batch has no effect on `gem`"),
        "unexpected error message: {err}"
    );

    let emit_out = dir.path().join("emit").to_string_lossy().into_owned();
    let cli = Cli::try_parse_from(["senna-gem", &genes, "--emit-pb-reference", "-o", &emit_out])
        .expect("GemArgs parses --emit-pb-reference");
    let err = run_gem_embedding(&cli.args)
        .expect_err("--emit-pb-reference must be refused for a gem run");
    assert!(
        err.to_string()
            .contains("--emit-pb-reference has no effect on `gem`"),
        "unexpected error message: {err}"
    );

    // Neither flag: the guard must not fire on the ordinary path.
    let plain_out = dir.path().join("plain").to_string_lossy().into_owned();
    let cli = Cli::try_parse_from([
        "senna-gem",
        &genes,
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
        &plain_out,
    ])
    .expect("GemArgs parses a plain invocation");
    run_gem_embedding(&cli.args).expect("a plain gem run must still succeed");
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
        "S1_count",
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
        "--offset-rank",
        "2",
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

/// `--offset-rank` is its own number, checked against the resolved H: the
/// refusal names both flags, and rank H is legal.
#[test]
fn the_offset_rank_is_checked_against_the_resolved_h() {
    for (rank, h) in [(0usize, 4usize), (5, 4)] {
        let e = validate_offset_rank(rank, h).unwrap_err().to_string();
        assert!(
            e.contains("--offset-rank") && e.contains("--embedding-dim"),
            "{e}"
        );
    }
    validate_offset_rank(1, 4).unwrap();
    validate_offset_rank(4, 4).unwrap();
}

fn tiny_fit(genes: &str, m6a: Option<&str>, out: &str, extra: &[&str]) -> GemArgs {
    let mut argv = vec!["senna-gem", genes];
    if let Some(m6a) = m6a {
        argv.extend_from_slice(&["--modality", m6a]);
    }
    argv.extend_from_slice(&[
        "--epochs",
        "2",
        "--skip-etm",
        "--no-emit-pb-reference",
        "--phase1-cells-per-pb",
        "0",
        "--offset-rank",
        "2",
        "-o",
        out,
    ]);
    argv.extend_from_slice(extra);
    Cli::try_parse_from(argv).expect("GemArgs parses").args
}

fn row_of(t: &matrix_util::traits::MatWithNames<Mat>, name: &str) -> Vec<f32> {
    let i = t
        .rows
        .iter()
        .position(|n| n.as_ref() == name)
        .unwrap_or_else(|| panic!("no row {name} among {:?}", t.rows));
    t.mat.row(i).iter().copied().collect()
}

/// A plain gene table (bare names, as `senna bge` writes) as the frozen base:
/// the matched genes' `count/spliced` rows come out verbatim, their unspliced
/// rows train as offsets on them, `--embedding-dim auto` takes the table's
/// width, and a gene the data lacks is carried through under its lifted name
/// with a type for every row. Four genes with both channels in two modules,
/// so the unspliced track has softmaxes to learn from (a track with one gene
/// per module scores one-entry softmaxes and can move nothing).
#[test]
fn a_bare_gene_table_pins_the_spliced_rows_and_is_carried_in_the_row_grammar() {
    use matrix_util::traits::IoOps;
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = synth(
        dir.path(),
        "S1_count",
        &[
            "GENE1/count/spliced",
            "GENE1/count/unspliced",
            "GENE2/count/spliced",
            "GENE2/count/unspliced",
            "GENE3/count/spliced",
            "GENE3/count/unspliced",
            "GENE4/count/spliced",
            "GENE4/count/unspliced",
        ],
        &CELLS,
    );
    let plus = dir.path().join("plus").to_string_lossy().into_owned();
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let h = 4;
    let names = boxes(&["GENE1", "GENE2", "GENE3", "GENE4", "EXTRA1"]);
    let table = Mat::from_fn(5, h, |i, k| (i as f32 + 1.0) * 0.25 - k as f32 * 0.1);
    table
        .to_parquet_with_names(
            &format!("{plus}.feature_embedding.parquet"),
            (Some(&names), Some("gene")),
            None,
        )
        .unwrap();
    let args = tiny_fit(
        &genes,
        None,
        &out,
        &[
            "--embedding-dim",
            "auto",
            "--freeze-feature-embedding",
            &plus,
            "--feature-modules",
            "2",
        ],
    );
    run_gem_embedding(&args).expect("a gem run on a frozen bare-gene table");
    let loading = Mat::from_parquet(&format!("{out}.feature_embedding.parquet")).unwrap();
    assert_eq!(loading.mat.ncols(), h, "auto takes the table's width");
    let given = |i: usize| -> Vec<f32> { table.row(i).iter().copied().collect() };
    for (i, g) in ["GENE1", "GENE2", "GENE3", "GENE4"].iter().enumerate() {
        assert_eq!(
            row_of(&loading, &format!("{g}/count/spliced")),
            given(i),
            "{g}"
        );
    }
    assert!(
        (0..4).any(|i| {
            let g = i + 1;
            row_of(&loading, &format!("GENE{g}/count/unspliced")) != given(i)
        }),
        "the unspliced rows are offsets on the pinned rows"
    );
    assert_eq!(
        row_of(&loading, "EXTRA1/count/spliced"),
        given(4),
        "carried under its lifted name"
    );
    assert_eq!(loading.rows.len(), 8 + 1);
    let types = auxiliary_data::feature_types::read_feature_types(&out)
        .unwrap()
        .expect("a type for every row");
    assert_eq!(types.len(), 9);
    assert!(std::path::Path::new(&format!("{out}.feature_contrast.parquet")).exists());
}

/// An earlier gem table as the frozen base (a gem-to-gem chain): every given
/// row comes out as given on every track, the offsets included; the table's
/// rows beyond this axis are carried through (a gene under its lifted name,
/// a term as it is); and under lora the rows move.
#[test]
fn an_earlier_gem_table_pins_its_track_rows_and_lora_moves_them() {
    use crate::feature_preset::test_support::{widen, EXTRA};
    use matrix_util::traits::IoOps;
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = genes_file(dir.path());
    let m6a = m6a_file(dir.path());
    let first = dir.path().join("first").to_string_lossy().into_owned();
    let plus = dir.path().join("plus").to_string_lossy().into_owned();
    let second = dir.path().join("second").to_string_lossy().into_owned();
    let third = dir.path().join("third").to_string_lossy().into_owned();
    run_gem_embedding(&tiny_fit(
        &genes,
        Some(&m6a),
        &first,
        &["--embedding-dim", "4"],
    ))
    .expect("first run");
    let extra = widen(&format!("{first}.feature_embedding.parquet"), &plus);

    run_gem_embedding(&tiny_fit(
        &genes,
        Some(&m6a),
        &second,
        &[
            "--embedding-dim",
            "auto",
            "--freeze-feature-embedding",
            &plus,
        ],
    ))
    .expect("second run, frozen to the first");
    let a = Mat::from_parquet(&format!("{first}.feature_embedding.parquet")).unwrap();
    let b = Mat::from_parquet(&format!("{second}.feature_embedding.parquet")).unwrap();
    for (i, name) in a.rows.iter().enumerate() {
        for (k, got) in row_of(&b, name).into_iter().enumerate() {
            assert!(
                (got - a.mat[(i, k)]).abs() < 1e-6,
                "{name}[{k}]: {got} vs given {}",
                a.mat[(i, k)]
            );
        }
    }
    assert_eq!(b.rows.len(), a.rows.len() + EXTRA.len());
    let lifted = format!("{}/count/spliced", EXTRA[0].0);
    assert_eq!(
        row_of(&b, &lifted),
        extra.row(0).iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(
        row_of(&b, EXTRA[1].0),
        extra.row(1).iter().copied().collect::<Vec<_>>(),
        "a term keeps its name"
    );
    let types = auxiliary_data::feature_types::read_feature_types(&second)
        .unwrap()
        .expect("types");
    assert!(types
        .iter()
        .any(|(n, t)| n.as_ref() == EXTRA[1].0 && t.as_ref() == EXTRA[1].1));

    run_gem_embedding(&tiny_fit(
        &genes,
        Some(&m6a),
        &third,
        &[
            "--embedding-dim",
            "auto",
            "--lora-feature-embedding",
            &plus,
            "--lora-rank",
            "1",
        ],
    ))
    .expect("third run, lora on the first");
    let c = Mat::from_parquet(&format!("{third}.feature_embedding.parquet")).unwrap();
    assert_eq!(c.rows.len(), a.rows.len() + EXTRA.len());
    assert!(
        a.rows.iter().enumerate().any(|(i, name)| {
            let got = row_of(&c, name);
            (0..4).any(|k| (got[k] - a.mat[(i, k)]).abs() > 1e-6)
        }),
        "under lora the rows move"
    );
}
