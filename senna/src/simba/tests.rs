//! End to end: `senna simba` on a small planted matrix writes bge-shaped
//! artifacts and a manifest that downstream commands can open.

use super::{fit_simba, SimbaArgs};
use crate::embed_common::Mat;
use crate::run_manifest::{CellSpace, RunKind, RunManifest};
use clap::Parser;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use matrix_util::traits::IoOps;
use std::path::Path;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    args: SimbaArgs,
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
    let (n_cells, n_genes) = (40usize + 2 * usize::from(near_empty), 20usize);
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    if near_empty {
        triplets.push((0, 40, 1.0));
        triplets.push((19, 41, 1.0));
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
    let path = dir.join("planted.zarr").to_string_lossy().into_owned();
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
    let cli = Cli::parse_from(argv);
    fit_simba(&cli.args).expect("simba run");
    cli.args
}

const FAST: [&str; 14] = [
    "--embedding-dim",
    "8",
    "--epochs",
    "3",
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
    let cli = Cli::parse_from([
        "simba",
        &data,
        "--out",
        &out,
        "--embedding-dim",
        "8",
        "--epochs",
        "5",
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
        "--n-hvg",
        "0",
        "--no-qc",
    ]);
    fit_simba(&cli.args).expect("simba run");

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
