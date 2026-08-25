//! Round-trip tests for the run manifest.
//!
//! The manifest is a published contract: `pinto plot`, `pinto prop` and
//! `pinto lra` all locate their inputs through it, so a field that fails to
//! serialize is a broken pipeline rather than a cosmetic defect.

use crate::util::metadata::*;

#[test]
fn metadata_roundtrip_lc() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into(), "b.h5".into()];
    let coord_cols: Vec<Box<str>> = vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
    let meta = create_lc_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: Some("a.tsv,b.tsv"),
            coord_columns: &coord_cols,
            n_cells: 1234,
            n_genes: 18000,
            n_edges: 55555,
            k: 12,
        },
        Some(DictMergeSummary {
            min_nnz: 1,
            genes_scored: 10,
        }),
        // A channelized `lc` run reports the structural fact of its feature
        // axis. The three `delta_*` fields stay `None` because `lc` runs no
        // splice sampler: "no contrast was fit" and "the contrast came out
        // empty" are different findings and must stay distinguishable.
        Some(SpliceTrackInfo {
            n_rows: 36000,
            n_delta_identified: 13000,
            nascent_count_fraction: 0.21,
            delta_base: DELTA_BASE_SPLICED.to_string(),
            delta_from_refresh: None,
            delta_median_counts: None,
            delta_counts_per_pseudobulk: None,
        }),
        &[0, 1, 2],
    );
    let path = dir.path().join("run.pinto.json");
    meta.write(&path).unwrap();
    let back = PintoMetadata::read(&path).unwrap();
    assert_eq!(back.command, "lc");
    assert_eq!(back.n_cells, 1234);
    assert_eq!(back.n_communities, Some(12));
    let levels = back.levels.expect("levels");
    // 3 cascade levels + final = 4 (final carries the merged consensus)
    assert_eq!(levels.len(), 4);
    assert_eq!(levels[0].tag, "L0");
    assert_eq!(levels[3].tag, "final");
    assert_eq!(levels[3].entropy_present, Some(true));
    assert!(back.outputs.dict_merge.is_some());
    assert!(back.outputs.lr_activity.is_none());
    assert_eq!(
        back.outputs.coord_columns.as_deref(),
        Some(
            &[
                "pxl_row_in_fullres".to_string(),
                "pxl_col_in_fullres".to_string()
            ][..]
        )
    );

    // The splice block is what tells a consumer the axis was channelized, so a
    // round trip that drops it would be silent.
    let splice = back.splice.as_ref().expect("splice block must survive");
    assert_eq!(splice.n_rows, 36000);
    assert_eq!(splice.n_delta_identified, 13000);
    assert_eq!(splice.delta_base, DELTA_BASE_SPLICED);
    assert!(
        splice.delta_from_refresh.is_none(),
        "lc runs no splice sampler"
    );
}

#[test]
fn metadata_roundtrip_cage() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into()];
    let coord_cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    let meta = create_cage_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: Some("a.csv"),
            coord_columns: &coord_cols,
            n_cells: 1000,
            n_genes: 20000,
            n_edges: 5000,
            k: 16, // edge clusters
        },
        true,
        Some(SpliceTrackInfo {
            n_rows: 40000,
            n_delta_identified: 15000,
            nascent_count_fraction: 0.23,
            delta_base: DELTA_BASE_SPLICED.to_string(),
            delta_from_refresh: Some(true),
            delta_median_counts: Some(19.0),
            delta_counts_per_pseudobulk: Some(0.14),
        }),
    );
    let path = dir.path().join("run.pinto.json");
    meta.write(&path).unwrap();
    let back = PintoMetadata::read(&path).unwrap();
    assert_eq!(back.command, "cage");
    assert_eq!(back.n_cells, 1000);
    assert_eq!(back.n_communities, Some(16));
    assert!(back.outputs.cell_embedding.is_some());
    // The trained unit is the PB: pb tables + the cell->pb map ship,
    // and there is no per-cell bias to report.
    assert!(back.outputs.cell_bias.is_none());
    assert!(back.outputs.pb_embedding.is_some());
    assert!(back.outputs.pb_bias.is_some());
    assert!(back.outputs.cell_pb.is_some());
    assert_eq!(
        back.outputs.feature_posterior_mean,
        Some(format!("{prefix}.feature_posterior_mean.parquet"))
    );
    assert!(back.outputs.feature_embedding.is_some());
    assert!(back.outputs.gene_bias.is_some());
    assert!(back.outputs.scores.is_some());
    // A channelized run reports GENES on `n_genes` and keeps the matrix's
    // own row count in the splice block — reading `n_genes` as a row count
    // is exactly the confusion the two-field split exists to prevent.
    let splice = back.splice.expect("splice block round-trips");
    assert_eq!(back.n_genes, 20000);
    assert_eq!(splice.n_rows, 40000);
    assert_eq!(splice.n_delta_identified, 15000);
    assert_eq!(splice.delta_base, "spliced");
    assert_eq!(splice.delta_from_refresh, Some(true));
    // Structural identifiability and usable evidence are different findings
    // and both have to survive the round-trip.
    assert_eq!(splice.delta_counts_per_pseudobulk, Some(0.14));
    assert!(back.outputs.batch_effects.is_some());
    assert!(back.outputs.clusters.is_none());
    let levels = back.levels.expect("levels");
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].tag, "final");
    // The point of the pair projection: cage's level is the SAME shape lc and
    // dsvd publish — a real propensity (with entropy), a per-edge community
    // table, and a gene x community dictionary.
    assert!(levels[0].propensity.ends_with(".propensity.parquet"));
    assert!(levels[0]
        .link_community
        .as_deref()
        .unwrap()
        .ends_with(".link_community.parquet"));
    assert!(levels[0]
        .gene_community
        .as_deref()
        .unwrap()
        .ends_with(".gene_community.parquet"));
    assert_eq!(levels[0].entropy_present, Some(true));
}

#[test]
fn metadata_roundtrip_cage_no_batch() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into()];
    let meta = create_cage_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: None,
            coord_columns: &[],
            n_cells: 100,
            n_genes: 200,
            n_edges: 300,
            k: 8,
        },
        false,
        None,
    );
    let json = serde_json::to_string(&meta).unwrap();
    let back: PintoMetadata = serde_json::from_str(&json).unwrap();
    assert!(back.outputs.batch_effects.is_none());
    // Absent, not zeroed: "this input had no channels" and "this input had
    // channels that identified nothing" are different findings.
    assert!(back.splice.is_none());
    assert!(!json.contains("splice"));
}

#[test]
fn metadata_roundtrip_lc_merge_no_collapse() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let data_files: Vec<Box<str>> = vec!["a.h5".into()];
    let meta = create_lc_metadata(
        &RunInputs {
            prefix: &prefix,
            data_files: &data_files,
            coord_file: None,
            coord_columns: &[],
            n_cells: 100,
            n_genes: 200,
            n_edges: 300,
            k: 8,
        },
        None,
        None,
        &[],
    );
    let path = dir.path().join("run.pinto.json");
    meta.write(&path).unwrap();
    let back = PintoMetadata::read(&path).unwrap();
    let levels = back.levels.expect("levels");
    // 0 cascade levels + final = 1
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].tag, "final");
    assert!(back.outputs.dict_merge.is_none());
}
