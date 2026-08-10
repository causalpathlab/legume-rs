//! The reference extends across rounds; it is never re-summarized.
//!
//! Under `--use-pb-reference` the collapse keeps each carried column in its
//! own singleton finest group, so emission is append-only: the parent's
//! columns pass through from their **observed** evidence (their stored,
//! already-adjusted frame — exactly `y·w / w`), while groups holding new mass
//! emit from the **adjusted** posterior, corrected toward that frame. Routing
//! a carried column through `mu_adjusted` instead would re-apply the near-1
//! anchor δ every round — a frame that quietly drifts, compounding with each
//! generation, which is the failure append-only exists to prevent.

use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::TwoStatParam;
use nalgebra::DMatrix;
use senna::pb_reference::{self, PbReferenceMeta, REFERENCE_BATCH};

const D: usize = 3;

/// A finest collapse over 2 groups: group 0 holds two new cells, group 1 is
/// one carried column (weight 10). Observed and adjusted evidence are given
/// deliberately different values so the emitted backend betrays which plane
/// each group was read from.
fn finest() -> data_beans_alg::collapse_data::CollapsedOut {
    let observed_rate = DMatrix::from_row_slice(D, 2, &[
        4.0, 7.0, //
        5.0, 8.0, //
        0.0, 9.0, //
    ]);
    let adjusted_rate = DMatrix::from_row_slice(D, 2, &[
        2.0, 6.0, //
        2.5, 6.5, //
        0.0, 7.0, //
    ]);
    // Evidence = sum/denom with the prior excluded; sizes: group 0 = 2 cells,
    // group 1 = 10 carried cells.
    let denom = DMatrix::from_row_slice(D, 2, &[2.0, 10.0, 2.0, 10.0, 2.0, 10.0]);
    let mut mu_observed = GammaMatrix::new((D, 2), 1.0, 1.0);
    mu_observed.update_stat(&observed_rate.component_mul(&denom), &denom);
    let mut mu_adjusted = GammaMatrix::new((D, 2), 1.0, 1.0);
    mu_adjusted.update_stat(&adjusted_rate.component_mul(&denom), &denom);
    data_beans_alg::collapse_data::CollapsedOut {
        mu_observed,
        mu_adjusted: Some(mu_adjusted),
        mu_residual: None,
        gamma: None,
        delta: None,
    }
}

fn read_backend_dense(prefix: &str) -> DMatrix<f32> {
    let backend = data_beans::sparse_io::open_sparse_matrix(
        &pb_reference::backend_path(prefix),
        &data_beans::sparse_io::SparseIoBackend::Zarr,
    )
    .expect("open emitted reference");
    let (rows, cols) = (backend.num_rows().unwrap(), backend.num_columns().unwrap());
    let csc = backend
        .read_columns_csc((0..cols).collect())
        .expect("read emitted columns");
    let mut dense = DMatrix::<f32>::zeros(rows, cols);
    for (j, col) in csc.col_iter().enumerate() {
        for (&g, &v) in col.row_indices().iter().zip(col.values().iter()) {
            dense[(g, j)] = v;
        }
    }
    dense
}

#[test]
fn carried_columns_pass_through_and_new_groups_are_adjusted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let prefix = dir.path().join("child").to_string_lossy().into_owned();

    // Loaded cohort: 2 new cells (cols 0, 1 → group 0) + 1 carried column
    // (col 2 → its own group 1) weighing 10 cells.
    let cell_to_pb = [0usize, 0, 1];
    let weight = [1.0f32, 1.0, 10.0];
    let genes: Vec<Box<str>> = (0..D).map(|g| format!("g{g}").into_boxed_str()).collect();
    let parent = PbReferenceMeta {
        senna_version: "0.0.0".into(),
        cell_counts: vec![10.0],
        batch_label: REFERENCE_BATCH.into(),
        batch_adjusted: true,
        generation: 3,
        column_generation: vec![2],
    };

    pb_reference::write(
        &prefix,
        &finest(),
        &cell_to_pb,
        Some(&weight),
        &genes,
        1,
        Some(&parent),
    )
    .expect("write");

    let dense = read_backend_dense(&prefix);
    assert_eq!(dense.shape(), (D, 2));
    // Group 0 holds new mass → adjusted evidence.
    assert_eq!(dense.column(0).as_slice(), &[2.0, 2.5, 0.0]);
    // Group 1 is the carried column → observed evidence, i.e. the stored
    // frame back out, NOT the re-adjusted 6.0/6.5/7.0.
    assert_eq!(dense.column(1).as_slice(), &[7.0, 8.0, 9.0]);

    let meta = pb_reference::read_meta(&prefix)
        .expect("read sidecar")
        .expect("present");
    assert_eq!(meta.cell_counts, vec![2.0, 10.0], "mass conserved per group");
    assert_eq!(meta.generation, 4, "parent gen 3 + this round");
    assert_eq!(
        meta.column_generation,
        vec![4, 2],
        "new group takes this round; the carried column keeps its origin round"
    );
}

/// A fresh fit (no carried tail) is the old behavior exactly: everything from
/// the adjusted posterior, every column stamped with generation 1.
#[test]
fn a_fresh_fit_emits_adjusted_evidence_everywhere() {
    let dir = tempfile::tempdir().expect("tempdir");
    let prefix = dir.path().join("fresh").to_string_lossy().into_owned();
    let genes: Vec<Box<str>> = (0..D).map(|g| format!("g{g}").into_boxed_str()).collect();

    pb_reference::write(&prefix, &finest(), &[0, 0, 1], None, &genes, 0, None).expect("write");

    let dense = read_backend_dense(&prefix);
    assert_eq!(dense.column(0).as_slice(), &[2.0, 2.5, 0.0]);
    assert_eq!(dense.column(1).as_slice(), &[6.0, 6.5, 7.0]);

    let meta = pb_reference::read_meta(&prefix)
        .expect("read sidecar")
        .expect("present");
    assert_eq!(meta.generation, 1);
    assert_eq!(meta.column_generation, vec![1, 1]);
    assert_eq!(meta.cell_counts, vec![2.0, 1.0]);
}
