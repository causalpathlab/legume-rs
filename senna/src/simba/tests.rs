//! End to end: `senna simba` on a small planted matrix writes bge-shaped
//! artifacts and a manifest that downstream commands can open.

use super::{fit_simba, SimbaArgs};
use crate::embed_common::Mat;
use crate::run_manifest::{CellSpace, RunKind, RunManifest};
use clap::Parser;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use matrix_util::traits::IoOps;
use std::path::Path;

/// Parse any `#[derive(Args)]` struct from a bare argv, as the CLI would.
#[derive(Parser)]
struct Wrap<A: clap::Args> {
    #[command(flatten)]
    args: A,
}

fn parse_args<A: clap::Args>(argv: &[&str]) -> A {
    Wrap::<A>::parse_from(argv).args
}

/// Two planted groups: cells 0..20 express genes 0..10, cells 20..40 express
/// genes 10..20; the off-block is sparse and low. Returns the on-disk path
/// and the total nnz.
fn planted_zarr(dir: &Path) -> (String, usize) {
    let (p, nnz, _) = planted_zarr_with(dir, false);
    (p, nnz)
}

/// As [`planted_zarr`], plus (with `near_empty`) two extra cells carrying a
/// single count each; also returns the nnz per gene.
fn planted_zarr_with(dir: &Path, near_empty: bool) -> (String, usize, Vec<usize>) {
    planted_zarr_named(dir, "planted.zarr", 20, near_empty)
}

/// The same cells measured on the first `n_genes` of the planted genes only
/// (group 0's block is genes 0..10, so a half panel sees one group's markers).
fn half_panel_zarr(dir: &Path) -> String {
    planted_zarr_named(dir, "half.zarr", 10, false).0
}

fn planted_zarr_named(
    dir: &Path,
    name: &str,
    n_genes: usize,
    near_empty: bool,
) -> (String, usize, Vec<usize>) {
    let n_cells = 40usize + 2 * usize::from(near_empty);
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    if near_empty {
        triplets.push((0, 40, 1.0));
        triplets.push((n_genes as u64 - 1, 41, 1.0));
    }
    for c in 0..40 {
        let grp = usize::from(c >= 20);
        for g in 0..n_genes {
            let own = usize::from(g >= 10) == grp;
            let x = if own {
                3 + (c + g) % 4
            } else if (c * 7 + g) % 5 == 0 {
                1
            } else {
                0
            };
            if x > 0 {
                triplets.push((g as u64, c as u64, x as f32));
            }
        }
    }
    let nnz = triplets.len();
    let mut per_gene = vec![0usize; n_genes];
    for t in &triplets {
        per_gene[t.0 as usize] += 1;
    }
    let path = dir.join(name).to_string_lossy().into_owned();
    let mut b = create_sparse_from_triplets(
        &triplets,
        (n_genes, n_cells, nnz),
        Some(&path),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("backend");
    b.register_row_names_vec(
        &(0..n_genes)
            .map(|g| format!("GENE{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..n_cells)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    (path, nnz, per_gene)
}

fn run(args: &[&str]) -> SimbaArgs {
    let mut argv = vec!["simba"];
    argv.extend_from_slice(args);
    let parsed: SimbaArgs = parse_args(&argv);
    fit_simba(&parsed).expect("simba run");
    parsed
}

/// A whole-gene, no-QC run at `epochs`, written to `{dir}/run`.
/// `FAST` is short enough to smoke-test; the projection tests need the two
/// planted groups to have separated, hence the longer variant.
fn planted_run(dir: &Path, epochs: &str) -> (String, String) {
    let (data, _nnz) = planted_zarr(dir);
    let out = dir.join("run").to_string_lossy().into_owned();
    let mut argv = vec![data.as_str(), "--out", &out, "--n-hvg", "0", "--no-qc"];
    argv.extend_from_slice(&FAST_EXCEPT_EPOCHS);
    argv.extend_from_slice(&["--epochs", epochs]);
    run(&argv);
    (data, out)
}

fn fast_run(dir: &Path) -> (String, String) {
    planted_run(dir, FAST_EPOCHS)
}

fn trained_run(dir: &Path) -> (String, String) {
    planted_run(dir, "20")
}

const FAST_EPOCHS: &str = "3";
const FAST_EXCEPT_EPOCHS: [&str; 12] = [
    "--embedding-dim",
    "8",
    "--batch-size",
    "50",
    "--num-batch-negs",
    "10",
    "--num-uniform-negs",
    "10",
    "--weight-decay",
    "0",
    "--eval-fraction",
    "0",
];
const FAST: [&str; 14] = [
    FAST_EXCEPT_EPOCHS[0],
    FAST_EXCEPT_EPOCHS[1],
    "--epochs",
    FAST_EPOCHS,
    FAST_EXCEPT_EPOCHS[2],
    FAST_EXCEPT_EPOCHS[3],
    FAST_EXCEPT_EPOCHS[4],
    FAST_EXCEPT_EPOCHS[5],
    FAST_EXCEPT_EPOCHS[6],
    FAST_EXCEPT_EPOCHS[7],
    FAST_EXCEPT_EPOCHS[8],
    FAST_EXCEPT_EPOCHS[9],
    FAST_EXCEPT_EPOCHS[10],
    FAST_EXCEPT_EPOCHS[11],
];

/// Barcodes are written as `barcode@batch`; strip the batch.
fn bare(r: &str) -> String {
    r.split('@').next().unwrap_or(r).to_string()
}

/// The default HVG selection hard-subsets the gene axis: the raw table, the
/// co-embed and the bins all describe exactly the selected genes.
#[test]
fn the_hvg_selection_hard_subsets_the_embedded_genes() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _nnz, per_gene) = planted_zarr_with(dir.path(), false);
    let out = dir.path().join("hvg").to_string_lossy().into_owned();
    let mut argv = vec![data.as_str(), "--out", &out, "--n-hvg", "10", "--no-qc"];
    argv.extend_from_slice(&FAST);
    run(&argv);
    let genes = Mat::from_parquet(&format!("{out}.feature_loading.parquet")).unwrap();
    assert_eq!(genes.mat.nrows(), 10);
    let all: Vec<String> = (0..20).map(|g| format!("GENE{g}")).collect();
    assert!(genes.rows.iter().all(|r| all.contains(&r.to_string())));
    let coembed = Mat::from_parquet(&format!("{out}.feature_embedding.parquet")).unwrap();
    assert_eq!(coembed.rows, genes.rows);
    let bins = Mat::from_parquet(&format!("{out}.simba_bins.parquet")).unwrap();
    let n_edges: f32 = (0..bins.mat.nrows()).map(|r| bins.mat[(r, 4)]).sum();
    let want: usize = genes
        .rows
        .iter()
        .map(|r| per_gene[r.trim_start_matches("GENE").parse::<usize>().unwrap()])
        .sum();
    assert_eq!(
        n_edges as usize, want,
        "edges come from the selected genes only"
    );
}

/// Default cell QC drops the near-empty cells from every per-cell output and
/// keeps the barcodes aligned with the rows.
#[test]
fn cell_qc_filters_the_outputs_and_keeps_barcodes_aligned() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _nnz, _) = planted_zarr_with(dir.path(), true);
    let out = dir.path().join("qc").to_string_lossy().into_owned();
    let mut argv = vec![data.as_str(), "--out", &out, "--n-hvg", "0"];
    argv.extend_from_slice(&FAST);
    run(&argv);
    let cells = Mat::from_parquet(&format!("{out}.cell_embedding.parquet")).unwrap();
    let rows: Vec<String> = cells.rows.iter().map(|r| bare(r)).collect();
    assert!(
        !rows.iter().any(|r| r == "c40" || r == "c41"),
        "near-empty cells dropped: {rows:?}"
    );
    assert_eq!(cells.mat.nrows(), 40, "every planted cell kept");
    assert!(cells.mat.iter().all(|v| v.is_finite()));
    for c in 0..40 {
        assert!(rows.contains(&format!("c{c}")), "cell c{c} present");
    }
    let coembed = Mat::from_parquet(&format!("{out}.feature_embedding.parquet")).unwrap();
    assert!(coembed.mat.iter().all(|v| v.is_finite()));
}

#[test]
fn simba_writes_bge_shaped_artifacts_and_a_manifest_downstream_commands_can_open() {
    let dir = tempfile::tempdir().unwrap();
    let (data, nnz) = planted_zarr(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let mut argv = vec![data.as_str(), "--out", &out, "--n-hvg", "0", "--no-qc"];
    argv.extend_from_slice(&FAST_EXCEPT_EPOCHS);
    argv.extend_from_slice(&["--epochs", "5"]);
    run(&argv);

    for suffix in ["cell_embedding", "feature_loading", "feature_embedding"] {
        let m = Mat::from_parquet(&format!("{out}.{suffix}.parquet")).expect(suffix);
        assert_eq!(m.mat.ncols(), 8, "{suffix} has h0..h7");
        assert!(m.mat.iter().all(|v| v.is_finite()), "{suffix} is finite");
        assert_eq!(m.cols[0].as_ref(), "h0");
        assert_eq!(m.cols[7].as_ref(), "h7");
    }
    let cells = Mat::from_parquet(&format!("{out}.cell_embedding.parquet")).unwrap();
    assert_eq!(cells.mat.nrows(), 40);
    assert_eq!(cells.rows.len(), 40);
    let genes = Mat::from_parquet(&format!("{out}.feature_loading.parquet")).unwrap();
    assert_eq!(genes.mat.nrows(), 20);
    assert_eq!(genes.rows[0].as_ref(), "GENE0");
    assert_eq!(genes.rows[19].as_ref(), "GENE19");
    let coembed = Mat::from_parquet(&format!("{out}.feature_embedding.parquet")).unwrap();
    assert_eq!(coembed.rows, genes.rows, "co-embed rows are the same genes");

    let scores = Mat::from_parquet(&format!("{out}.feature_scores.parquet")).unwrap();
    assert_eq!(scores.mat.nrows(), 20);
    let cols: Vec<&str> = scores.cols.iter().map(AsRef::as_ref).collect();
    assert_eq!(cols, ["max", "std", "gini", "entropy"]);
    assert!(scores.mat.iter().all(|v| v.is_finite()));

    let bins = Mat::from_parquet(&format!("{out}.simba_bins.parquet")).unwrap();
    let cols: Vec<&str> = bins.cols.iter().map(AsRef::as_ref).collect();
    assert_eq!(cols, ["level", "lower", "upper", "weight", "n_edges"]);
    assert!(bins.mat.nrows() >= 1 && bins.mat.nrows() <= 5);
    let n_edges: f32 = (0..bins.mat.nrows()).map(|r| bins.mat[(r, 4)]).sum();
    assert_eq!(n_edges as usize, nnz, "every nonzero entry is one edge");
    let weights: Vec<f32> = (0..bins.mat.nrows()).map(|r| bins.mat[(r, 3)]).collect();
    assert!(weights.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(weights[0], 1.0);

    let (m, _dir) = RunManifest::load(Path::new(&format!("{out}.senna.json"))).unwrap();
    assert_eq!(m.kind, RunKind::Simba);
    assert_eq!(m.kind.cell_space(), CellSpace::Embedding);
    let geometry = m.outputs.geometry_latent().expect("an embedding table");
    assert!(geometry.ends_with("cell_embedding.parquet"), "{geometry}");
}

//////////////////////////////////////////////////////////////////
// Downstream consumers: the bge query path opens a simba run.  //
//////////////////////////////////////////////////////////////////

/// Cell × cell cosine table of a cell × H matrix.
fn cosines(z: &Mat) -> Mat {
    crate::geometry::similarity::compute_cosine_similarity(&z.transpose())
}

/// Mean cosine within the two planted groups minus the mean across them.
fn group_margin(z: &Mat) -> f32 {
    let c = cosines(z);
    let (mut within, mut across, mut nw, mut na) = (0.0f32, 0.0f32, 0usize, 0usize);
    for i in 0..40 {
        for j in (i + 1)..40 {
            if (i < 20) == (j < 20) {
                within += c[(i, j)];
                nw += 1;
            } else {
                across += c[(i, j)];
                na += 1;
            }
        }
    }
    within / nw as f32 - across / na as f32
}

/// Mean over cells of the cosine between a cell's row in `a` and in `b`.
fn mean_paired_cosine(a: &Mat, b: &Mat) -> f32 {
    let n = a.nrows();
    let mut stacked = Mat::zeros(2 * n, a.ncols());
    stacked.view_mut((0, 0), (n, a.ncols())).copy_from(a);
    stacked.view_mut((n, 0), (n, a.ncols())).copy_from(b);
    let c = cosines(&stacked);
    (0..n).map(|i| c[(i, n + i)]).sum::<f32>() / n as f32
}

/// The simba gene table is a bge-shaped frozen side with no gene bias: the
/// score is a pure dot product, so the reader supplies zeros.
#[test]
fn bge_embedding_opens_a_simba_run_with_a_zero_gene_bias() {
    let dir = tempfile::tempdir().unwrap();
    let (_data, out) = fast_run(dir.path());
    let model = crate::bge::score::BgeEmbedding::open(&out).expect("opens a simba run");
    assert_eq!(model.h, 8);
    let genes = Mat::from_parquet(&format!("{out}.feature_loading.parquet")).unwrap();
    assert_eq!(model.gene_names, genes.rows);
    assert_eq!(model.b_feat.len(), 20);
    assert!(model.b_feat.iter().all(|&b| b == 0.0), "no gene bias");
    assert!(model.modules.is_none());
    // The manifest path works as well as the prefix.
    let via_manifest = crate::bge::score::BgeEmbedding::open(&format!("{out}.senna.json")).unwrap();
    assert_eq!(via_manifest.gene_names, model.gene_names);
}

/// The zero-bias default is simba's alone: a bge run whose bias table is
/// missing is still refused, because there the bias is half the model.
#[test]
fn open_still_refuses_a_bge_run_without_its_bias() {
    let dir = tempfile::tempdir().unwrap();
    let (_data, _out) = fast_run(dir.path());
    let bgeish = dir.path().join("bgeish").to_string_lossy().into_owned();
    let mut m = RunManifest::new(RunKind::Bge, &bgeish);
    m.outputs.feature_loading = Some("run.feature_loading.parquet".into());
    m.save(Path::new(&format!("{bgeish}.senna.json"))).unwrap();
    let err = match crate::bge::score::BgeEmbedding::open(&bgeish) {
        Ok(_) => panic!("a bge run without its bias must not open"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("feature_bias"), "{err}");
}

/// `predict` places new cells against the frozen gene table and lands the
/// training cells back near their trained positions, with the planted groups
/// still apart.
#[test]
fn predict_projects_query_cells_onto_the_frozen_gene_table() {
    let dir = tempfile::tempdir().unwrap();
    let (data, out) = trained_run(dir.path());
    let pout = dir.path().join("pred").to_string_lossy().into_owned();
    let args: crate::predict::PredictArgs =
        parse_args(&["predict", &data, "--model", &out, "-o", &pout]);
    crate::predict::predict_model(&args).expect("predict on a simba run");

    let z = Mat::from_parquet(&format!("{pout}.latent.parquet")).unwrap();
    assert_eq!((z.mat.nrows(), z.mat.ncols()), (40, 8));
    assert!(z.mat.iter().all(|v| v.is_finite()));
    let trained = Mat::from_parquet(&format!("{out}.cell_embedding.parquet")).unwrap();
    let trained_rows: Vec<String> = trained.rows.iter().map(|r| bare(r)).collect();
    let pred_rows: Vec<String> = z.rows.iter().map(|r| bare(r)).collect();
    assert_eq!(pred_rows, trained_rows, "same cells in the same order");

    let margin = group_margin(&z.mat);
    assert!(margin > 0.2, "projected groups separate: margin {margin}");
    let mean_cos = mean_paired_cosine(&z.mat, &trained.mat);
    assert!(
        mean_cos > 0.5,
        "projection lands near the trained cells: mean cosine {mean_cos}"
    );

    let pred = Mat::from_parquet(&format!("{pout}.predictive.parquet")).unwrap();
    let cols: Vec<&str> = pred.cols.iter().map(AsRef::as_ref).collect();
    assert_eq!(&cols[..3], ["llik", "total", "llik_per_count"]);
    for i in 0..40 {
        assert!(pred.mat[(i, 0)].is_finite() && pred.mat[(i, 0)] <= 0.0);
        assert!(pred.mat[(i, 1)] > 0.0);
    }
}

/// Refinement is an encoder step; a simba run has none to refine.
#[test]
fn predict_on_a_simba_run_refuses_refinement() {
    let dir = tempfile::tempdir().unwrap();
    let (data, out) = fast_run(dir.path());
    let pout = dir.path().join("pred").to_string_lossy().into_owned();
    let args: crate::predict::PredictArgs = parse_args(&[
        "predict",
        &data,
        "--model",
        &out,
        "-o",
        &pout,
        "--refine-steps",
        "5",
    ]);
    let err = crate::predict::predict_model(&args).unwrap_err();
    assert!(err.to_string().contains("simba"), "{err}");
}

/// Coverage is measured against the trained gene set; a half panel scores by
/// default and is refused only under an explicit floor.
#[test]
fn predict_gates_on_coverage_of_the_trained_genes() {
    let dir = tempfile::tempdir().unwrap();
    let (_data, out) = trained_run(dir.path());
    let half = half_panel_zarr(dir.path());
    let pout = dir.path().join("pred").to_string_lossy().into_owned();
    let args: crate::predict::PredictArgs =
        parse_args(&["predict", &half, "--model", &out, "-o", &pout]);
    crate::predict::predict_model(&args).expect("half the genes still score");
    let z = Mat::from_parquet(&format!("{pout}.latent.parquet")).unwrap();
    assert_eq!((z.mat.nrows(), z.mat.ncols()), (40, 8));

    let args: crate::predict::PredictArgs = parse_args(&[
        "predict",
        &half,
        "--model",
        &out,
        "-o",
        &pout,
        "--min-gene-overlap",
        "0.9",
    ]);
    assert!(crate::predict::predict_model(&args).is_err());
}

/// `impute` matches query cells to reference cells by cosine in the simba
/// space; each planted group is filled in from its own block.
#[test]
fn impute_on_a_simba_run_recovers_each_groups_marker_block() {
    let dir = tempfile::tempdir().unwrap();
    let (data, out) = trained_run(dir.path());
    let iout = dir.path().join("imp").to_string_lossy().into_owned();
    let args: crate::impute::ImputeArgs =
        parse_args(&["impute", &data, "--model", &out, "-o", &iout, "--knn", "5"]);
    crate::impute::impute_model(&args).expect("impute on a simba run");
    let imputed = Mat::from_parquet(&format!("{iout}.imputed.parquet")).unwrap();
    assert_eq!((imputed.mat.nrows(), imputed.mat.ncols()), (40, 20));
    let block_mean = |cells: std::ops::Range<usize>, genes: std::ops::Range<usize>| -> f32 {
        let n = (cells.len() * genes.len()) as f32;
        imputed
            .mat
            .view((cells.start, genes.start), (cells.len(), genes.len()))
            .sum()
            / n
    };
    assert!(block_mean(0..20, 0..10) > block_mean(0..20, 10..20));
    assert!(block_mean(20..40, 10..20) > block_mean(20..40, 0..10));
}

/// `predict --bulk` orients a table against the simba gene table.
#[test]
fn bulk_gene_axis_is_the_simba_gene_table() {
    let dir = tempfile::tempdir().unwrap();
    let (_data, out) = fast_run(dir.path());
    let axis = crate::predict::bulk::model_gene_axis(RunKind::Simba, &out).unwrap();
    let genes = Mat::from_parquet(&format!("{out}.feature_loading.parquet")).unwrap();
    assert_eq!(axis, genes.rows);
}

/// With no topic latent, `plot-topic` shows the cell embedding as a per-cell
/// softmax over its axes, and the gene table stands in for the dictionary.
#[test]
fn plot_topic_falls_back_to_the_cell_embedding() {
    let dir = tempfile::tempdir().unwrap();
    let (_data, out) = fast_run(dir.path());
    let pout = dir.path().join("pt").to_string_lossy().into_owned();
    let manifest = format!("{out}.senna.json");
    let args: crate::postprocess::PlotTopicArgs =
        parse_args(&["plot-topic", "--from", &manifest, "-o", &pout]);
    crate::postprocess::fit_plot_topic(&args).expect("plot-topic on a simba run");
    assert!(Path::new(&format!("{pout}.plots/struct/all.pdf")).is_file());
    assert!(Path::new(&format!("{pout}.plots/dict/hinton.pdf")).is_file());
}

/// A QC-filtered run has fewer latent rows than its data files have cells;
/// the structure plot groups by the rows' own `@batch` suffix instead.
#[test]
fn plot_topic_groups_a_qc_filtered_run_by_the_rows_batch_suffix() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _nnz, _) = planted_zarr_with(dir.path(), true);
    let out = dir.path().join("qc").to_string_lossy().into_owned();
    let mut argv = vec![data.as_str(), "--out", &out, "--n-hvg", "0"];
    argv.extend_from_slice(&FAST);
    run(&argv);
    let cells = Mat::from_parquet(&format!("{out}.cell_embedding.parquet")).unwrap();
    assert_eq!(cells.mat.nrows(), 40, "QC dropped the two near-empty cells");
    let pout = dir.path().join("pt").to_string_lossy().into_owned();
    let manifest = format!("{out}.senna.json");
    let args: crate::postprocess::PlotTopicArgs =
        parse_args(&["plot-topic", "--from", &manifest, "-o", &pout]);
    crate::postprocess::fit_plot_topic(&args).expect("plot-topic on a QC-filtered run");
    assert!(Path::new(&format!("{pout}.plots/struct/all.pdf")).is_file());
}

/// `update` re-fits a simba run on the union of the recorded and the new
/// data with the recorded configuration (no checkpoint, so not a warm start).
#[test]
fn update_refits_a_simba_run_on_the_union() {
    let dir = tempfile::tempdir().unwrap();
    let (data, out) = trained_run(dir.path());
    let second = dir.path().join("second");
    std::fs::create_dir_all(&second).unwrap();
    let (data2, _nnz) = planted_zarr(&second);
    let out2 = dir.path().join("run_v2").to_string_lossy().into_owned();
    let args: crate::update::UpdateArgs = parse_args(&[
        "update", &data2, "--model", &out, "-o", &out2, "--epochs", "2",
    ]);
    crate::update::run_update(&args).expect("update on a simba run");
    let (m, _dir) = RunManifest::load(Path::new(&format!("{out2}.senna.json"))).unwrap();
    assert_eq!(m.kind, RunKind::Simba);
    assert_eq!(
        m.data.input.len(),
        2,
        "recorded input followed by the new one"
    );
    assert!(m.data.input[0].ends_with(Path::new(&data).file_name().unwrap().to_str().unwrap()));
    let cells = Mat::from_parquet(&format!("{out2}.cell_embedding.parquet")).unwrap();
    assert_eq!((cells.mat.nrows(), cells.mat.ncols()), (80, 8));
    let args: SimbaArgs = m.train_args_as(&out2).unwrap();
    assert_eq!(args.epochs, 2, "the per-round epoch override is recorded");
}

/// `deconvolve` takes only the gene axis, the width and the cell embedding
/// from its source, all of which a simba run has.
#[test]
fn deconvolve_source_loads_a_simba_run() {
    let dir = tempfile::tempdir().unwrap();
    let (_data, out) = fast_run(dir.path());
    let src = crate::deconvolve::source::EmbeddingSource::load(&format!("{out}.senna.json"))
        .expect("a simba run is a deconvolve source");
    assert_eq!(src.kind, RunKind::Simba);
    assert_eq!(src.h, 8);
    assert_eq!(src.feature_names.len(), 20);
    assert_eq!(src.cell_embedding_paths.len(), 1);
    assert!(src.cell_embedding_paths[0].ends_with("run.cell_embedding.parquet"));
}
