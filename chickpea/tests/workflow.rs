//! The whole link workflow on the planted fixture: embed, fold peaks in,
//! train the attention, and write every table.

mod common;

use chickpea::p2g::attention::AttentionConfig;
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
            phase1_cells_per_pb: 4,
            ..TwoTrackConfig::default()
        },
        attention: AttentionConfig {
            rank: 4,
            epochs: 50,
            ..AttentionConfig::default()
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
    let summary = run_links(&input, &small()).unwrap();
    let n_pairs = N_GENES * PEAKS_PER_GENE;
    assert_eq!(summary.n_pairs, n_pairs);
    assert!(summary.n_clusters >= 2, "{} clusters", summary.n_clusters);
    assert!(
        summary.attention_loss.last() < summary.attention_loss.first(),
        "attention loss did not fall: {:?}",
        summary.attention_loss
    );

    for stem in [
        "gene_embedding",
        "gene_atac_embedding",
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

    // One row per pair; attention and ABC shares each sum to 1 per gene.
    let links = format!("{out}.links.parquet");
    let (s, n) =
        read_table_columns(&links, &["gene", "peak"], &["distance", "abc", "attention"]).unwrap();
    assert_eq!(s[0].len(), n_pairs);
    let mut per_gene: HashMap<&str, (f64, f64)> = HashMap::new();
    for (i, g) in s[0].iter().enumerate() {
        let e = per_gene.entry(g).or_default();
        e.0 += n[1][i];
        e.1 += n[2][i];
    }
    assert_eq!(per_gene.len(), N_GENES);
    for (g, (abc, att)) in per_gene {
        assert!((abc - 1.0).abs() < 1e-5, "{g}: ABC sums to {abc}");
        assert!((att - 1.0).abs() < 1e-5, "{g}: attention sums to {att}");
    }

    // Per cluster: shares sum to 1 per gene within each cluster.
    let by = format!("{out}.links_by_cluster.parquet");
    let (s, n) = read_table_columns(&by, &["gene", "cluster"], &["attention", "abc"]).unwrap();
    let mut per: HashMap<(&str, &str), (f64, f64)> = HashMap::new();
    for i in 0..s[0].len() {
        let e = per.entry((&s[0][i], &s[1][i])).or_default();
        e.0 += n[0][i];
        e.1 += n[1][i];
    }
    for ((g, k), (att, abc)) in per {
        assert!(
            (att - 1.0).abs() < 1e-4,
            "{g} in {k}: attention sums to {att}"
        );
        assert!((abc - 1.0).abs() < 1e-4, "{g} in {k}: ABC sums to {abc}");
    }

    let (cells, _) = read_table_columns(
        &format!("{out}.cell_clusters.parquet"),
        &["cell", "cluster"],
        &[],
    )
    .unwrap();
    assert_eq!(cells[0].len(), N_CELLS);
}
