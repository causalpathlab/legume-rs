//! End-to-end checks of `faba qc` and `faba qc-report` on a synthetic output
//! directory: a count matrix, an m6A site matrix, a gene-level m6A matrix and
//! an m6A site table, all in the producers' layout. Drives the built binary.

use std::process::Command;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use data_beans::sparse_io::{
    create_sparse_from_triplets_owned, open_sparse_matrix, SparseIoBackend,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

fn faba() -> Command {
    Command::new(env!("CARGO_BIN_EXE_faba"))
}

/// Write `{dir}/{name}.zarr` from named triplets.
fn write_matrix(
    dir: &str,
    name: &str,
    rows: &[&str],
    cols: &[&str],
    entries: &[(usize, usize, f32)],
) {
    let path = format!("{dir}/{name}.zarr");
    let triplets: Vec<(u64, u64, f32)> = entries
        .iter()
        .map(|&(r, c, x)| (r as u64, c as u64, x))
        .collect();
    let mut data = create_sparse_from_triplets_owned(
        triplets,
        (rows.len(), cols.len(), entries.len()),
        Some(&path),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    let rows: Vec<Box<str>> = rows.iter().map(|s| (*s).into()).collect();
    let cols: Vec<Box<str>> = cols.iter().map(|s| (*s).into()).collect();
    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);
}

struct Site {
    chr: &'static str,
    gene: &'static str,
    pos: i64,
    pv: f32,
    log_odds: Option<f32>,
    cov: u64,
    conv: u64,
    ctl_cov: u64,
    ctl_conv: u64,
}

/// The m6A site table with exactly the columns `faba qc` reads.
fn write_sites(path: &str, sites: &[Site]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("chr", DataType::Utf8, false),
        Field::new("gene", DataType::Utf8, false),
        Field::new("primary_pos", DataType::Int64, false),
        Field::new("conversion_pos", DataType::Int64, true),
        Field::new("pv", DataType::Float32, false),
        Field::new("log_odds", DataType::Float32, true),
        Field::new("coverage", DataType::UInt64, false),
        Field::new("converted", DataType::UInt64, false),
        Field::new("control_coverage", DataType::UInt64, false),
        Field::new("control_converted", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                sites.iter().map(|s| s.chr).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                sites.iter().map(|s| s.gene).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                sites.iter().map(|s| s.pos - 1).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Int64Array::from(
                sites.iter().map(|s| Some(s.pos)).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float32Array::from(
                sites.iter().map(|s| s.pv).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float32Array::from(
                sites.iter().map(|s| s.log_odds).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                sites.iter().map(|s| s.cov).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                sites.iter().map(|s| s.conv).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                sites.iter().map(|s| s.ctl_cov).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                sites.iter().map(|s| s.ctl_conv).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn read_parquet(path: &str) -> RecordBatch {
    let file = std::fs::File::open(path).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
    let schema = batches[0].schema();
    arrow::compute::concat_batches(&schema, &batches).unwrap()
}

fn strings(batch: &RecordBatch, col: &str) -> Vec<String> {
    let a = batch
        .column_by_name(col)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    (0..a.len()).map(|i| a.value(i).to_string()).collect()
}

/// One batch, six cells. Cells C1..C4 carry counts; C5 is empty in the count
/// matrix; C6 has a single gene. Sites: S1 strong and seen in 4 cells, S2
/// strong but seen in 1 cell, S3 weak (p = 0.6), S4 thin coverage.
fn build_fixture(dir: &str) {
    let cells = ["C1", "C2", "C3", "C4", "C5", "C6"];
    let genes = [
        "G1/count/spliced",
        "G1/count/unspliced",
        "G2/count/spliced",
        "G3/count/spliced",
    ];
    let mut entries = Vec::new();
    for c in 0..4 {
        for r in 0..4 {
            entries.push((r, c, 5.0 + c as f32));
        }
    }
    entries.push((0, 5, 1.0)); // C6: one gene only
    write_matrix(dir, "b1_count", &genes, &cells, &entries);

    // Site matrix: {gene}/m6a/{chr}:{pos}/{channel}; pos = conversion_pos.
    let site_rows = [
        "G1/m6a/chr1:100/methylated",
        "G1/m6a/chr1:100/unmethylated",
        "G1/m6a/chr1:200/methylated",
        "G1/m6a/chr1:200/unmethylated",
        "G2/m6a/chr1:300/methylated",
        "G2/m6a/chr1:300/unmethylated",
        "G3/m6a/chr1:400/methylated",
        "G3/m6a/chr1:400/unmethylated",
    ];
    let mut site_entries = Vec::new();
    for c in 0..4 {
        site_entries.push((0, c, 2.0)); // S1 methylated in C1..C4
        site_entries.push((1, c, 3.0));
        site_entries.push((5, c, 4.0)); // S3 unmethylated everywhere
    }
    site_entries.push((2, 0, 3.0)); // S2 methylated in C1 only
    site_entries.push((3, 0, 1.0));
    site_entries.push((4, 1, 1.0)); // S3 methylated in C2
    site_entries.push((6, 4, 1.0)); // S4 methylated in the dropped cell C5 only
    site_entries.push((7, 4, 1.0));
    write_matrix(dir, "b1_m6a_site", &site_rows, &cells, &site_entries);

    // Producer's gene-level pool (all sites), which qc must re-derive.
    write_matrix(
        dir,
        "b1_m6a",
        &[
            "G1/m6a/methylated",
            "G1/m6a/unmethylated",
            "G2/m6a/methylated",
            "G2/m6a/unmethylated",
        ],
        &cells,
        &[(0, 0, 99.0), (1, 0, 99.0), (2, 1, 99.0), (3, 1, 99.0)],
    );

    write_sites(
        &format!("{dir}/m6a_sites.parquet"),
        &[
            Site {
                chr: "chr1",
                gene: "G1",
                pos: 100,
                pv: 1e-4,
                log_odds: Some(3.0),
                cov: 20,
                conv: 8,
                ctl_cov: 20,
                ctl_conv: 0,
            },
            Site {
                chr: "chr1",
                gene: "G1",
                pos: 200,
                pv: 1e-3,
                log_odds: Some(2.0),
                cov: 10,
                conv: 3,
                ctl_cov: 10,
                ctl_conv: 0,
            },
            Site {
                chr: "chr1",
                gene: "G2",
                pos: 300,
                pv: 0.6,
                log_odds: Some(0.1),
                cov: 20,
                conv: 1,
                ctl_cov: 20,
                ctl_conv: 1,
            },
            Site {
                chr: "chr1",
                gene: "G3",
                pos: 400,
                pv: 1e-2,
                log_odds: Some(1.0),
                cov: 2,
                conv: 1,
                ctl_cov: 0,
                ctl_conv: 0,
            },
        ],
    );
    std::fs::write(format!("{dir}/pipeline_summary.json"), "{}").unwrap();
}

#[test]
fn qc_filters_cells_sites_and_repools_gene_level() {
    let tmp = tempfile::tempdir().unwrap();
    let input = format!("{}/in", tmp.path().display());
    let output = format!("{}/out", tmp.path().display());
    std::fs::create_dir_all(&input).unwrap();
    build_fixture(&input);

    let status = faba()
        .args([
            "qc",
            &input,
            "-o",
            &output,
            "--no-cell-qc",
            "--column-nnz-cutoff",
            "2",
            "--site-max-pv",
            "0.05",
            "--site-min-coverage",
            "3",
            "--site-min-cells",
            "2",
            "--no-zip",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "faba qc failed");

    // Cells: C1..C4 (nnz 4 >= 2); C5 empty and C6 (nnz 1) drop.
    let count =
        open_sparse_matrix(&format!("{output}/b1_count.zarr"), &SparseIoBackend::Zarr).unwrap();
    let cols = count.column_names().unwrap();
    assert_eq!(
        cols.iter().map(|c| c.as_ref()).collect::<Vec<_>>(),
        ["C1", "C2", "C3", "C4"]
    );

    // Sites: S1 kept; S2 dropped (1 cell < 2); S3 dropped (equal rates in both
    // arms: raw log odds exactly 0, checked before its p of 0.6); S4 dropped
    // (coverage 2 < 3, checked before cells).
    let kept = read_parquet(&format!("{output}/m6a_sites.parquet"));
    assert_eq!(kept.num_rows(), 1);
    assert_eq!(strings(&kept, "gene"), ["G1"]);
    let dropped = read_parquet(&format!("{output}/m6a_sites_dropped.parquet"));
    assert_eq!(dropped.num_rows(), 3);
    let reasons = strings(&dropped, "reason");
    assert_eq!(reasons, ["cells", "log_odds", "coverage"]);

    // Site matrix: both channels of S1 only, over the kept cells.
    let site = open_sparse_matrix(
        &format!("{output}/b1_m6a_site.zarr"),
        &SparseIoBackend::Zarr,
    )
    .unwrap();
    let rows = site.row_names().unwrap();
    assert_eq!(
        rows.iter().map(|r| r.as_ref()).collect::<Vec<_>>(),
        ["G1/m6a/chr1:100/methylated", "G1/m6a/chr1:100/unmethylated"]
    );
    assert_eq!(site.num_columns(), Some(4));

    // Gene level re-pooled from the kept site rows, not the producer's 99s:
    // G1 methylated = 2 per cell, unmethylated = 3 per cell; G2 is gone.
    let gene =
        open_sparse_matrix(&format!("{output}/b1_m6a.zarr"), &SparseIoBackend::Zarr).unwrap();
    let rows = gene.row_names().unwrap();
    assert_eq!(
        rows.iter().map(|r| r.as_ref()).collect::<Vec<_>>(),
        ["G1/m6a/methylated", "G1/m6a/unmethylated"]
    );
    let (_, _, triplets) = gene.read_triplets_by_columns((0..4).collect()).unwrap();
    let mut sums = [0.0f32; 2];
    for (r, _, x) in triplets {
        sums[r as usize] += x;
    }
    assert_eq!(sums, [8.0, 12.0]);

    assert!(std::path::Path::new(&format!("{output}/pipeline_summary.json")).exists());
    assert!(std::path::Path::new(&format!("{output}/b1_cells.tsv.gz")).exists());
    assert!(std::path::Path::new(&format!("{output}/qc_summary.tsv")).exists());
}

#[test]
fn qc_report_sweeps_every_criterion() {
    let tmp = tempfile::tempdir().unwrap();
    let input = format!("{}/in", tmp.path().display());
    std::fs::create_dir_all(&input).unwrap();
    build_fixture(&input);
    let prefix = format!("{}/rep", tmp.path().display());

    let status = faba()
        .args(["qc-report", &input, "-o", &prefix, "--quiet"])
        .status()
        .unwrap();
    assert!(status.success(), "faba qc-report failed");

    let report = read_parquet(&format!("{prefix}.qc_report.parquet"));
    let modality = strings(&report, "modality");
    let criterion = strings(&report, "criterion");
    let n_kept = report
        .column_by_name("n_kept")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let threshold = report
        .column_by_name("threshold")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();

    let mut seen = std::collections::BTreeSet::new();
    for i in 0..report.num_rows() {
        seen.insert((modality[i].clone(), criterion[i].clone()));
        // A permissive threshold keeps everything: p <= 1 keeps all 4 sites.
        if modality[i] == "m6a" && criterion[i] == "max_pv" && threshold.value(i) == 1.0 {
            assert_eq!(n_kept.value(i), 4);
        }
        // min_cells 2 keeps S1 and S3 only (S3's methylated row is in C2 and,
        // via the producer matrix, that is one cell; S1 in four) -> 1 site.
        if modality[i] == "m6a" && criterion[i] == "min_cells" && threshold.value(i) == 2.0 {
            assert_eq!(n_kept.value(i), 1);
        }
    }
    for (m, c) in [
        ("m6a", "neglog10_pv_hist"),
        ("m6a", "max_pv"),
        ("m6a", "min_log_odds"),
        ("m6a", "min_fold"),
        ("m6a", "min_coverage"),
        ("m6a", "min_converted"),
        ("m6a", "min_edit_ratio"),
        ("m6a", "min_cells"),
        ("count", "min_cells"),
        ("count", "min_counts"),
        ("count", "min_genes_per_cell"),
    ] {
        assert!(
            seen.contains(&(m.to_string(), c.to_string())),
            "missing panel {m}/{c}"
        );
    }
}
