use super::gem_hvg_row_weights;
use crate::gem::tracks::assign_tracks;
use data_beans::alg::hvg::HvgCliArgs;
use data_beans::sparse_io::{create_sparse_from_dmatrix, SparseIoBackend};
use graph_embedding_util as ge;
use nalgebra::DMatrix;

fn boxes(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&s| s.into()).collect()
}

/// A single-file synthetic axis: `rows` x `cells`, one row per `values`
/// entry (must match `rows.len()` x `cells.len()`, row-major).
fn fixture(
    dir: &std::path::Path,
    rows: &[&str],
    cells: &[&str],
    values: &[f32],
) -> ge::UnifiedData {
    let path = dir.join("fixture.zarr");
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_row_slice(rows.len(), cells.len(), values);
    let mut b = create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr))
        .expect("create synthetic backend");
    b.register_row_names_vec(&boxes(rows));
    b.register_column_names_vec(&boxes(cells));
    drop(b);

    ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: vec![path],
        feature_kind: Some(ge::FeatureNameKind::Exact),
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::Disjoint,
        ..Default::default()
    })
    .expect("load_unified_data")
}

/// GENE1 carries three rows (base, an unspliced offset, and an m6a
/// methylated channel); GENE2 and GENE3 carry only their base row. GENE1's
/// own base + unspliced rows are FLAT (zero variance); all its variance
/// sits on the m6a row. GENE2/GENE3 are flat too, so only pooling can ever
/// select GENE1.
const ROWS: [&str; 5] = [
    "GENE1/count/spliced",
    "GENE1/count/unspliced",
    "GENE1/m6a/methylated",
    "GENE2/count/spliced",
    "GENE3/count/spliced",
];
const CELLS: [&str; 6] = ["C1", "C2", "C3", "C4", "C5", "C6"];
#[rustfmt::skip]
const VALUES: [f32; 30] = [
    // GENE1/count/spliced: flat.
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    // GENE1/count/unspliced: flat (a different constant, still zero variance).
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    // GENE1/m6a/methylated: alternating -- all of GENE1's dispersion.
    1.0, 9.0, 1.0, 9.0, 1.0, 9.0,
    // GENE2/count/spliced: flat.
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    // GENE3/count/spliced: flat.
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
];

fn mixed_axis(dir: &std::path::Path) -> ge::UnifiedData {
    fixture(dir, &ROWS, &CELLS, &VALUES)
}

fn selection_on() -> HvgCliArgs {
    HvgCliArgs {
        n_hvg: 1,
        feature_list_file: None,
        must_train_features: None,
    }
}

fn selection_off() -> HvgCliArgs {
    HvgCliArgs {
        n_hvg: 0,
        feature_list_file: None,
        must_train_features: None,
    }
}

#[test]
fn a_gene_whose_variance_sits_only_in_its_m6a_row_is_still_selected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let unified = mixed_axis(dir.path());
    let plan = assign_tracks(&unified.feature_names).expect("assign_tracks");

    let w = gem_hvg_row_weights(&unified, &plan, &selection_on(), None)
        .expect("gem_hvg_row_weights")
        .expect("selection is on");

    // Row 0 = GENE1/count/spliced (GENE1's base row) — selected.
    assert_eq!(
        w[0], 1.0,
        "GENE1 must be selected on its pooled (incl. m6a) variance"
    );
    // GENE2's and GENE3's base rows (3, 4) are NOT selected.
    assert_eq!(w[3], 0.0);
    assert_eq!(w[4], 0.0);
}

#[test]
fn weights_are_zero_on_every_non_base_row_of_a_selected_gene() {
    let dir = tempfile::tempdir().expect("tempdir");
    let unified = mixed_axis(dir.path());
    let plan = assign_tracks(&unified.feature_names).expect("assign_tracks");

    let w = gem_hvg_row_weights(&unified, &plan, &selection_on(), None)
        .expect("gem_hvg_row_weights")
        .expect("selection is on");

    assert_eq!(
        w[0], 1.0,
        "the base row of the selected gene carries the weight"
    );
    assert_eq!(w[1], 0.0, "GENE1/count/unspliced is not a base row");
    assert_eq!(w[2], 0.0, "GENE1/m6a/methylated is not a base row");
}

#[test]
fn selection_off_with_modality_rows_returns_the_base_mask() {
    let dir = tempfile::tempdir().expect("tempdir");
    let unified = mixed_axis(dir.path());
    let plan = assign_tracks(&unified.feature_names).expect("assign_tracks");

    let w = gem_hvg_row_weights(&unified, &plan, &selection_off(), None)
        .expect("gem_hvg_row_weights")
        .expect("a non-base track exists, so the base mask is returned, not None");

    let expected: Vec<f32> = plan
        .base_rows
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();
    assert_eq!(w, expected);
    // Concretely: rows 0, 3, 4 are base (count/spliced); 1, 2 are not.
    assert_eq!(w, vec![1.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn spliced_only_and_selection_off_returns_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let rows = ["GENE1/count/spliced", "GENE2/count/spliced"];
    let cells = ["C1", "C2"];
    let values = [5.0, 5.0, 3.0, 3.0];
    let unified = fixture(dir.path(), &rows, &cells, &values);
    let plan = assign_tracks(&unified.feature_names).expect("assign_tracks");

    let w =
        gem_hvg_row_weights(&unified, &plan, &selection_off(), None).expect("gem_hvg_row_weights");
    assert_eq!(w, None, "a plain spliced-only axis has nothing to mask");
}
