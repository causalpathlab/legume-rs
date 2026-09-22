use super::*;
use crate::multiome_layout::RunMultiome;
use crate::run_manifest::RunKind;
use data_beans::sparse_io_vector::ColumnAlignment;

fn manifest(inputs: &[&str]) -> RunManifest {
    let mut m = RunManifest::new(RunKind::Bge, "/runs/r");
    m.data.input = inputs.iter().map(|s| (*s).to_string()).collect();
    m
}

/// A single-modality run re-opens its counts as a plain load: files stacked
/// as cells, rows as given.
#[test]
fn a_plain_run_reopens_its_counts_as_a_plain_load() {
    let m = manifest(&["a.zarr", "/abs/b.zarr"]);
    let args = raw_counts_load(&m, Path::new("/runs"), false).unwrap();
    assert_eq!(
        args.data_files,
        vec![Box::from("/runs/a.zarr"), Box::from("/abs/b.zarr")]
    );
    assert!(args.per_file_feature_suffix.is_none());
    assert_eq!(args.column_alignment, ColumnAlignment::Disjoint);
}

/// A multiome run's files are modalities of ONE cell set: they must be glued
/// by barcode and namespaced exactly as training did, or every cell appears
/// once per modality and none matches the latent.
#[test]
fn a_multiome_run_reopens_its_counts_under_the_recorded_layout() {
    let mut m = manifest(&["rna.zarr", "atac.zarr"]);
    m.data.multiome = Some(RunMultiome {
        modality: vec!["m0".into(), "m1".into()],
        group: vec!["g0".into(), "g0".into()],
        barcode_tagged: false,
    });
    let args = raw_counts_load(&m, Path::new("/runs"), false).unwrap();
    assert_eq!(args.column_alignment, ColumnAlignment::Union);
    assert_eq!(
        args.per_file_feature_suffix,
        Some(vec![Box::from("m0"), Box::from("m1")])
    );
    assert!(args.per_file_barcode_suffix.is_none());
}
