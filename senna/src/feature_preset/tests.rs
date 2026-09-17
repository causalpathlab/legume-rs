use super::*;
use auxiliary_data::feature_types::write_feature_types;
use graph_embedding_util::PresetMode;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;

/// A `senna fne`-shaped run: genes, one term and one word share the embedding
/// table, and `feature_types.parquet` says which is which.
fn write_fne_like(dir: &std::path::Path, with_types: bool) -> String {
    let prefix = dir.join("run").to_string_lossy().into_owned();
    let names: Vec<Box<str>> = ["TP53", "GATA1", "GO:0006915", "apoptosis"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    let mut e = DMatrix::<f32>::zeros(4, 3);
    for i in 0..4 {
        for k in 0..3 {
            e[(i, k)] = (i * 3 + k) as f32 * 0.5 - 1.0;
        }
    }
    e.to_parquet_with_names(
        &format!("{prefix}.feature_embedding.parquet"),
        (Some(&names), Some("feature")),
        None,
    )
    .unwrap();
    if with_types {
        let types: Vec<Box<str>> = ["gene", "gene", "term", "word"]
            .iter()
            .map(|s| Box::from(*s))
            .collect();
        write_feature_types(&prefix, &names, &types).unwrap();
    }
    prefix
}

#[test]
fn gene_rows_match_the_axis_by_canonical_name_and_other_types_are_left_out() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), true);
    // The word `apoptosis` also names a feature on this axis; only the GENE row
    // of the source may pin anything.
    let axis: Vec<Box<str>> = ["ENSG1_GATA1", "ENSG2_MYC", "apoptosis", "ENSG3_TP53"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    let f = load_preset_genes(
        &prefix,
        PresetMode::Freeze,
        &axis,
        &ge::FeatureNameKind::Gene { delim: '_' },
    )
    .unwrap();
    assert_eq!(f.gene, vec![0, 3]);
    // GATA1 is source row 1, TP53 source row 0.
    assert_eq!(f.rows, vec![0.5, 1.0, 1.5, -1.0, -0.5, 0.0]);
}

#[test]
fn without_a_types_table_every_row_is_a_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), false);
    let axis: Vec<Box<str>> = ["apoptosis", "ENSG3_TP53"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    let f = load_preset_genes(
        &prefix,
        PresetMode::Freeze,
        &axis,
        &ge::FeatureNameKind::Gene { delim: '_' },
    )
    .unwrap();
    assert_eq!(f.gene, vec![0, 1]);
}

#[test]
fn no_matching_gene_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), true);
    let axis: Vec<Box<str>> = vec![Box::from("ENSG2_MYC")];
    assert!(load_preset_genes(
        &prefix,
        PresetMode::Freeze,
        &axis,
        &ge::FeatureNameKind::Gene { delim: '_' }
    )
    .is_err());
}

#[test]
fn the_mode_is_carried_and_a_rank_the_table_cannot_hold_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = write_fne_like(dir.path(), true);
    let axis: Vec<Box<str>> = vec![Box::from("ENSG3_TP53")];
    let kind = ge::FeatureNameKind::Gene { delim: '_' };
    for mode in [
        PresetMode::Freeze,
        PresetMode::Init,
        PresetMode::Lora {
            rank: 2,
            lr_ratio: 16.0,
        },
    ] {
        assert_eq!(
            load_preset_genes(&prefix, mode, &axis, &kind).unwrap().mode,
            mode
        );
    }
    // The table is H = 3 wide: rank 3 is no residual.
    assert!(load_preset_genes(
        &prefix,
        PresetMode::Lora {
            rank: 3,
            lr_ratio: 1.0
        },
        &axis,
        &kind
    )
    .is_err());
}
