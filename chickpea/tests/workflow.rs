//! The whole link workflow on the planted fixture.

mod common;

use chickpea::p2g::two_track::{TwoTrackConfig, TwoTrackInput};
use chickpea::p2g::workflow::{run_links, LinkConfig};
use common::{gene_positions, kernel, write_fixture, N_CELLS, N_GENES, PEAKS_PER_GENE};
use legume_numeric::matrix::parquet::read_table_columns;
use std::collections::HashMap;

fn small() -> LinkConfig {
    LinkConfig {
        embed: TwoTrackConfig {
            embedding_dim: 8,
            epochs: 60,
            num_levels: 2,
            sort_dim: 3,
            proj_dim: 8,
            feature_modules: 4,
            peak_modules: 4,
            phase1_cells_per_pb: 4,
            module_only_min_rows: 30,
            ..TwoTrackConfig::default()
        },
        ..LinkConfig::default()
    }
}

#[test]
fn the_workflow_writes_consistent_tables() {
    let dir = tempfile::tempdir().unwrap();
    let (rna, atac) = write_fixture(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let genes = gene_positions();
    let input = TwoTrackInput {
        rna_file: &rna,
        atac_file: &atac,
        batch_file: None,
        gene_positions: &genes,
        kernel: &kernel(),
        work_prefix: &out,
    };
    let summary = run_links(&input, &small(), None).unwrap();
    // Pairs to near-empty or scattered peaks are dropped, never added.
    let n_pairs = summary.n_pairs;
    assert!(
        n_pairs > 0 && n_pairs <= N_GENES * PEAKS_PER_GENE,
        "{n_pairs} pairs"
    );
    assert!(summary.n_clusters >= 2, "{} clusters", summary.n_clusters);
    assert!(summary.theta0.is_finite(), "θ0={}", summary.theta0);
    assert!(summary.align_gap.is_finite(), "gap {}", summary.align_gap);
    // The gates train: the shared scalars leave their start (θ₀ = 1, θ₁ = ½).
    let moved = (summary.theta0 - 1.0).abs() + (summary.theta1 - 0.5).abs();
    assert!(moved > 1e-3, "the gates never moved: {summary:?}");

    for stem in [
        "gene_embedding",
        "cell_embedding",
        "cell_clusters",
        "peak_embedding",
        "peaks",
        "links",
        "links_by_cluster",
    ] {
        let f = format!("{out}.{stem}.parquet");
        assert!(std::path::Path::new(&f).exists(), "missing {f}");
    }

    let links = format!("{out}.links.parquet");
    let (s, n) = read_table_columns(
        &links,
        &["gene", "peak"],
        &["distance", "abc", "gate", "corr", "score"],
    )
    .unwrap();
    assert_eq!(s[0].len(), n_pairs);
    let mut per_gene: HashMap<&str, f64> = HashMap::new();
    let mut gate_per_gene: HashMap<&str, f64> = HashMap::new();
    for (i, g) in s[0].iter().enumerate() {
        *per_gene.entry(g).or_default() += n[1][i];
        *gate_per_gene.entry(g).or_default() += n[2][i];
        assert!(n[2][i] >= 0.0, "gate weight negative");
        // The link score is the distance prior times the data's evidence.
        assert!(n[3][i].abs() <= 1.0 + 1e-5, "corr {}", n[3][i]);
        assert!(
            (n[4][i] - n[2][i] * n[3][i]).abs() < 1e-5,
            "score {} ≠ gate {} × corr {}",
            n[4][i],
            n[2][i],
            n[3][i]
        );
    }
    assert!(!per_gene.is_empty() && per_gene.len() <= N_GENES);
    for (g, abc) in per_gene {
        assert!((abc - 1.0).abs() < 1e-5, "{g}: ABC sums to {abc}");
    }
    // The gate is the gene's share of each pair in its ATAC half.
    for (g, gate) in gate_per_gene {
        assert!((gate - 1.0).abs() < 1e-4, "{g}: gate shares sum to {gate}");
    }

    // Per cluster: ABC (and renormalised gate) shares sum to 1 per gene.
    let by = format!("{out}.links_by_cluster.parquet");
    let (_, n) = read_table_columns(&by, &[], &["gene_idx", "cluster", "gate", "abc"]).unwrap();
    let mut per: HashMap<(i32, i32), (f64, f64)> = HashMap::new();
    for (((&g, &k), &gate), &abc) in n[0].iter().zip(&n[1]).zip(&n[2]).zip(&n[3]) {
        let e = per.entry((g as i32, k as i32)).or_default();
        e.0 += gate;
        e.1 += abc;
    }
    for ((g, k), (gate, abc)) in per {
        assert!(
            (gate - 1.0).abs() < 1e-4,
            "gene {g} in cluster {k}: gate shares sum to {gate}"
        );
        assert!(
            (abc - 1.0).abs() < 1e-4,
            "gene {g} in cluster {k}: ABC sums to {abc}"
        );
    }

    let (cells, _) = read_table_columns(
        &format!("{out}.cell_clusters.parquet"),
        &["cell", "cluster"],
        &[],
    )
    .unwrap();
    assert_eq!(cells[0].len(), N_CELLS);
}

#[test]
fn qc_failed_cells_are_left_out_of_every_cell_output() {
    let dir = tempfile::tempdir().unwrap();
    let (rna, atac) = write_fixture(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let genes = gene_positions();
    let input = TwoTrackInput {
        rna_file: &rna,
        atac_file: &atac,
        batch_file: None,
        gene_positions: &genes,
        kernel: &kernel(),
        work_prefix: &out,
    };
    let dropped = ["CELL1", "CELL2", "CELL7"];
    let keep: rustc_hash::FxHashSet<Box<str>> = common::names("CELL", N_CELLS)
        .into_iter()
        .filter(|b| !dropped.contains(&b.as_ref()))
        .collect();
    run_links(&input, &small(), Some(&keep)).unwrap();

    let (cells, _) =
        read_table_columns(&format!("{out}.cell_clusters.parquet"), &["cell"], &[]).unwrap();
    assert_eq!(cells[0].len(), N_CELLS - dropped.len());
    assert!(cells[0].iter().all(|c| !dropped.contains(&c.as_ref())));
    let (emb, _) =
        read_table_columns(&format!("{out}.cell_embedding.parquet"), &["cell"], &[]).unwrap();
    assert_eq!(emb[0].len(), N_CELLS - dropped.len());
}
