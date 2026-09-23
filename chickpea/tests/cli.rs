//! `chickpea peak-to-gene` as a user runs it.

mod common;

use common::{write_fixture, write_gene_coords};
use std::process::Command;

fn chickpea() -> Command {
    Command::new(env!("CARGO_BIN_EXE_chickpea"))
}

const SMALL: [&str; 16] = [
    "--embedding-dim",
    "8",
    "--epochs",
    "60",
    "--num-levels",
    "2",
    "--sort-dim",
    "3",
    "--proj-dim",
    "8",
    "--feature-modules",
    "4",
    "--attention-rank",
    "4",
    "--attention-epochs",
    "50",
];

#[test]
fn peak_to_gene_runs_end_to_end_and_writes_the_link_tables() {
    let dir = tempfile::tempdir().unwrap();
    let (rna, atac) = write_fixture(dir.path());
    let genes = write_gene_coords(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let st = chickpea()
        .args([
            "peak-to-gene",
            "--rna",
            &rna,
            "--atac",
            &atac,
            "--gene-coords",
            &genes,
        ])
        .args(["--cis-window", "10000", "-o", &out])
        .args(SMALL)
        .output()
        .unwrap();
    assert!(
        st.status.success(),
        "{}",
        String::from_utf8_lossy(&st.stderr)
    );
    for stem in [
        "links",
        "links_by_cluster",
        "cell_clusters",
        "peaks",
        "gene_embedding",
    ] {
        let f = format!("{out}.{stem}.parquet");
        assert!(std::path::Path::new(&f).exists(), "missing {f}");
    }
}

#[test]
fn gene_positions_are_required() {
    let dir = tempfile::tempdir().unwrap();
    let (rna, atac) = write_fixture(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let st = chickpea()
        .args(["peak-to-gene", "--rna", &rna, "--atac", &atac, "-o", &out])
        .output()
        .unwrap();
    assert!(!st.status.success());
    let err = String::from_utf8_lossy(&st.stderr);
    assert!(err.contains("gene-coords") && err.contains("gff"), "{err}");
}
