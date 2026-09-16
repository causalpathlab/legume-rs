//! The typed feature table that rides beside a mixed-type embedding:
//! `{prefix}.feature_types.parquet`, string columns `feature` and `type`, one
//! row per embedding row in the same order. `senna fne` and `gene-text` write
//! it; a consumer that wants only one type of row (`senna bge` pinning gene
//! rows, say) reads it. The type names are the shared vocabulary.

use matrix_util::parquet::{read_parquet_string_columns_by_name, write_named_table, Column};
use std::path::Path;

/// Nodes whose names are canonicalised as gene symbols.
pub const GENE_TYPE: &str = "gene";
/// Ontology terms and gene sets.
pub const TERM_TYPE: &str = "term";
/// Fixed genomic windows.
pub const REGION_TYPE: &str = "region";
/// Vocabulary words of a text relation.
pub const WORD_TYPE: &str = "word";

pub fn feature_types_path(prefix: &str) -> String {
    format!("{prefix}.feature_types.parquet")
}

/// Write the table for `names[i]` of type `types[i]`.
pub fn write_feature_types(
    prefix: &str,
    names: &[Box<str>],
    types: &[Box<str>],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        names.len() == types.len(),
        "feature types: {} names for {} types",
        names.len(),
        types.len()
    );
    write_named_table(
        &feature_types_path(prefix),
        "feature",
        names,
        &[(Box::from("type"), Column::Str(types))],
    )
}

/// One row of the table: the feature's name and its type.
pub type FeatureType = (Box<str>, Box<str>);

/// The run's rows; `None` when the run wrote no table, which a caller reads as
/// "every row is of the one type it expects".
pub fn read_feature_types(prefix: &str) -> anyhow::Result<Option<Vec<FeatureType>>> {
    let path = feature_types_path(prefix);
    if !Path::new(&path).exists() {
        return Ok(None);
    }
    let mut cols = read_parquet_string_columns_by_name(&path, &["feature", "type"])?;
    let types = cols.pop().expect("two columns requested");
    let names = cols.pop().expect("two columns requested");
    Ok(Some(names.into_iter().zip(types).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_absence() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("run").to_string_lossy().into_owned();
        assert!(read_feature_types(&prefix).unwrap().is_none());
        let names: Vec<Box<str>> = vec!["TP53".into(), "GO:1".into()];
        let types: Vec<Box<str>> = vec![GENE_TYPE.into(), TERM_TYPE.into()];
        write_feature_types(&prefix, &names, &types).unwrap();
        let rows = read_feature_types(&prefix).unwrap().unwrap();
        assert_eq!(
            rows,
            vec![
                ("TP53".into(), GENE_TYPE.into()),
                ("GO:1".into(), TERM_TYPE.into())
            ]
        );
        assert!(write_feature_types(&prefix, &names, &types[..1]).is_err());
    }
}
