//! Cluster-parquet alignment for [`super::load_cluster_labels`].

use super::load_cluster_labels;
use crate::mat_io::Mat;
use legume_numeric::matrix::traits::IoOps;

fn write_clusters(dir: &std::path::Path, name: &str, cells: &[&str], labels: &[f32]) -> String {
    let mut m = Mat::zeros(cells.len(), 1);
    for (i, &lab) in labels.iter().enumerate() {
        m[(i, 0)] = lab;
    }
    let rows: Vec<Box<str>> = cells.iter().map(|s| Box::from(*s)).collect();
    let cols: Vec<Box<str>> = vec!["cluster".into()];
    let path = dir.join(name).to_string_lossy().into_owned();
    m.to_parquet_with_names(&path, (Some(&rows), Some("cell")), Some(&cols))
        .unwrap();
    path
}

#[test]
fn aligns_cluster_parquet_to_data_cell_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_clusters(dir.path(), "c.parquet", &["b", "a", "c"], &[1.0, 0.0, 2.0]);
    let cell_names: Vec<Box<str>> = ["a", "b", "c"].iter().map(|s| Box::from(*s)).collect();
    let (labels, n_clusters) = load_cluster_labels(&path, &cell_names).unwrap();
    assert_eq!(labels, vec![0, 1, 2]);
    assert_eq!(n_clusters, 3);
}

#[test]
fn rejects_parquet_with_no_overlap_to_data_cells() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_clusters(dir.path(), "c.parquet", &["x", "y"], &[0.0, 1.0]);
    let cell_names: Vec<Box<str>> = vec!["a".into(), "b".into()];
    assert!(load_cluster_labels(&path, &cell_names).is_err());
}
