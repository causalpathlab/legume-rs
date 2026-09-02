use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-6, "entry {i}: got {g}, want {w}");
    }
}

/// The count backend numbers batches by sorted name, the unified data by first
/// appearance. The fold rows must follow the UNIFIED ids, matched by name.
#[test]
fn maps_collapse_batches_by_name_not_by_column() {
    // rows = genes, columns = collapse batches ["early", "late"] (sorted order)
    let delta = DMatrix::from_row_slice(3, 2, &[1.0, 0.5, 2.0, 1.0, 4.0, 8.0]);
    let off = batch_gene_fold(FoldSource {
        delta: &delta,
        collapse_batch_names: &names(&["early", "late"]),
        unified_batch_names: &names(&["late", "early"]),
        n_features: 3,
        feature_to_backend: &[0, 1, 2],
    })
    .unwrap()
    .expect("two batches ⇒ a fold");
    assert_eq!(off.n_batches(), 2);
    assert_eq!(off.batch_names, names(&["late", "early"]));
    assert_close(off.row(0), &[0.5, 1.0, 8.0]);
    assert_close(off.row(1), &[1.0, 2.0, 4.0]);
}

/// Backend rows are gathered onto the unified feature axis through
/// `feature_to_backend`, like every other collapse table.
#[test]
fn gathers_rows_onto_the_unified_feature_axis() {
    let delta = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 4.0, 5.0]);
    let off = batch_gene_fold(FoldSource {
        delta: &delta,
        collapse_batch_names: &names(&["a", "b"]),
        unified_batch_names: &names(&["a", "b"]),
        n_features: 2,
        feature_to_backend: &[3, 1],
    })
    .unwrap()
    .unwrap();
    assert_eq!(off.n_features, 2);
    assert_close(off.row(0), &[4.0, 2.0]);
    assert_close(off.row(1), &[5.0, 3.0]);
}

#[test]
fn one_batch_yields_no_fold() {
    let delta = DMatrix::from_row_slice(2, 1, &[3.0, 0.2]);
    let off = batch_gene_fold(FoldSource {
        delta: &delta,
        collapse_batch_names: &names(&["only"]),
        unified_batch_names: &names(&["only"]),
        n_features: 2,
        feature_to_backend: &[0, 1],
    })
    .unwrap();
    assert!(off.is_none());
}

#[test]
fn unified_batch_missing_from_the_collapse_is_an_error() {
    let delta = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    let err = batch_gene_fold(FoldSource {
        delta: &delta,
        collapse_batch_names: &names(&["a", "b"]),
        unified_batch_names: &names(&["a", "c"]),
        n_features: 2,
        feature_to_backend: &[0, 1],
    })
    .expect_err("unknown batch must fail");
    assert!(err.to_string().contains('c'), "{err}");
}

#[test]
fn nonpositive_delta_is_floored() {
    let delta = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 2.0]);
    let off = batch_gene_fold(FoldSource {
        delta: &delta,
        collapse_batch_names: &names(&["a", "b"]),
        unified_batch_names: &names(&["a", "b"]),
        n_features: 2,
        feature_to_backend: &[0, 1],
    })
    .unwrap()
    .unwrap();
    assert_close(off.row(0), &[DELTA_FLOOR, DELTA_FLOOR]);
    assert_close(off.row(1), &[1.0, 2.0]);
}
