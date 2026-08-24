//! Reader tolerance for the two coord_pairs batch schemas: the current
//! writer emits string labels, but tables from older runs carry the numeric
//! batch pseudo-coordinate under the same column names. Those files do not
//! vanish, so the reader must load either, stringifying numerics.

use crate::lr_activity::io::{attach_batch_from_coord_pairs, EdgeRecord};
use matrix_util::parquet::{write_named_table, Column};

fn records(n: usize) -> Vec<EdgeRecord> {
    (0..n)
        .map(|i| EdgeRecord {
            left_cell: format!("cell{i}").into(),
            right_cell: format!("cell{}", i + 1).into(),
            community: 0,
            batch: None,
            is_spatial: true,
        })
        .collect()
}

/// A legacy table's `left_batch` / `right_batch` are FLOAT columns holding
/// the batch pseudo-coordinate. They still read: integral floats become
/// their integer string, and a straddling edge still resolves to None.
#[test]
fn a_legacy_numeric_batch_table_still_reads() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("legacy.coord_pairs.parquet");
    let path = path.to_str().unwrap();

    let left_cells: Vec<Box<str>> = vec!["cell0".into(), "cell1".into(), "cell2".into()];
    let right_cells: Vec<Box<str>> = vec!["cell1".into(), "cell2".into(), "cell3".into()];
    let left_b = [0.0f32, 0.0, 1000.0];
    let right_b = [0.0f32, 1000.0, 1000.0];
    let row_names: Vec<Box<str>> = (0..3).map(|i| i.to_string().into()).collect();
    write_named_table(
        path,
        "cell_pair",
        &row_names,
        &[
            ("left_cell".into(), Column::Str(&left_cells)),
            ("right_cell".into(), Column::Str(&right_cells)),
            ("left_batch".into(), Column::F32(&left_b)),
            ("right_batch".into(), Column::F32(&right_b)),
        ],
    )?;

    let mut edges = records(3);
    attach_batch_from_coord_pairs(&mut edges, path)?;
    assert_eq!(edges[0].batch.as_deref(), Some("0"));
    assert_eq!(edges[1].batch, None, "straddling edge belongs to no batch");
    assert_eq!(edges[2].batch.as_deref(), Some("1000"));
    Ok(())
}

/// The current schema's string labels pass through verbatim.
#[test]
fn string_batch_labels_pass_through_verbatim() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("current.coord_pairs.parquet");
    let path = path.to_str().unwrap();

    let left_cells: Vec<Box<str>> = vec!["cell0".into(), "cell1".into()];
    let right_cells: Vec<Box<str>> = vec!["cell1".into(), "cell2".into()];
    let left_b: Vec<Box<str>> = vec!["coreA".into(), "coreA".into()];
    let right_b: Vec<Box<str>> = vec!["coreA".into(), "coreB".into()];
    let row_names: Vec<Box<str>> = (0..2).map(|i| i.to_string().into()).collect();
    write_named_table(
        path,
        "cell_pair",
        &row_names,
        &[
            ("left_cell".into(), Column::Str(&left_cells)),
            ("right_cell".into(), Column::Str(&right_cells)),
            ("left_batch".into(), Column::Str(&left_b)),
            ("right_batch".into(), Column::Str(&right_b)),
        ],
    )?;

    let mut edges = records(2);
    attach_batch_from_coord_pairs(&mut edges, path)?;
    assert_eq!(edges[0].batch.as_deref(), Some("coreA"));
    assert_eq!(edges[1].batch, None);
    Ok(())
}
