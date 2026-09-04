//! `plot --colour-by topic` argmaxes a topic table (columns carry their IDs)
//! or an embedding (axes numbered by position), and refuses anything else.

use super::argmax_topics;
use crate::embed_common::Mat;
use matrix_util::traits::IoOps;

fn write(dir: &std::path::Path, name: &str, cols: &[&str]) -> String {
    let m = Mat::from_row_slice(3, 3, &[0.1, 0.9, 0.0, 0.7, 0.2, 0.1, 0.0, 0.0, 1.0]);
    let rows: Vec<Box<str>> = ["a", "b", "c"].iter().map(|s| Box::from(*s)).collect();
    let cols: Vec<Box<str>> = cols.iter().map(|s| Box::from(*s)).collect();
    let path = dir.join(name).to_string_lossy().into_owned();
    m.to_parquet_with_names(&path, (Some(&rows), Some("cell")), Some(&cols))
        .unwrap();
    path
}

#[test]
fn topic_columns_yield_their_ids_not_their_positions() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(dir.path(), "t.parquet", &["T5", "T2", "T9"]);
    assert_eq!(argmax_topics(&p, 3).unwrap(), vec![2, 5, 9]);
}

/// An embedding's `h{c}` axes carry no IDs of their own.
#[test]
fn embedding_axes_are_numbered_by_position() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(dir.path(), "h.parquet", &["h0", "h1", "h2"]);
    assert_eq!(argmax_topics(&p, 3).unwrap(), vec![1, 0, 2]);
}

/// Anything else is not a composition and must not be argmaxed as one.
#[test]
fn other_column_names_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(dir.path(), "x.parquet", &["alpha", "beta", "gamma"]);
    assert!(argmax_topics(&p, 3).is_err());
}

#[test]
fn a_row_count_mismatch_is_still_refused() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(dir.path(), "t.parquet", &["T0", "T1", "T2"]);
    assert!(argmax_topics(&p, 4).is_err());
}
