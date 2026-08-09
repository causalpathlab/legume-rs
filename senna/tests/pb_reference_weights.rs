//! Carried pseudobulks are weighted by *position*, so position has to be right.
//!
//! `weights_for` puts each stored cell-count on the column it belongs to by
//! assuming the loader appended the reference last. If that ever silently
//! fails — a reordering, a different `ColumnAlignment`, another backend pushed
//! after it — the weights land on the wrong columns: plausible numbers in the
//! wrong places, with no shape mismatch and no NaN to notice.

use senna::pb_reference::{ReferenceInput, COLUMN_PREFIX};

fn names(new: usize, refs: usize) -> Vec<Box<str>> {
    let mut v: Vec<Box<str>> = (0..new)
        .map(|i| format!("cell{i}").into_boxed_str())
        .collect();
    v.extend((0..refs).map(|i| format!("{COLUMN_PREFIX}{i}@parent.pb_reference").into_boxed_str()));
    v
}

fn input(counts: &[f32]) -> ReferenceInput {
    ReferenceInput {
        backend: "parent.pb_reference.zarr".into(),
        batch_file: "parent.pb_reference_batch.txt".into(),
        cell_counts: counts.to_vec(),
    }
}

#[test]
fn new_cells_weigh_one_and_carried_columns_carry_their_counts() {
    let r = input(&[3.0, 40.0, 7.0]);
    let w = r.weights_for(&names(5, 3)).expect("well-formed layout");

    assert_eq!(w.len(), 8);
    assert_eq!(&w[..5], &[1.0; 5], "real cells weigh one apiece");
    assert_eq!(&w[5..], &[3.0, 40.0, 7.0], "carried counts, in order");
    assert_eq!(r.cells_represented(), 50.0);
}

/// Every way the layout can be wrong. The stray-column case keeps a well-formed
/// tail on purpose, so it exercises the contiguity check rather than tripping
/// the tail check first.
#[test]
fn a_misplaced_reference_column_is_refused() {
    let r = input(&[3.0, 40.0]);

    let mut stray: Vec<Box<str>> = vec![
        "cell0".into(),
        format!("{COLUMN_PREFIX}99@parent.pb_reference").into_boxed_str(),
        "cell1".into(),
    ];
    stray.extend(names(0, 2));
    let err = r.weights_for(&stray).expect_err("stray reference column");
    assert!(err.to_string().contains("among the new cells"), "{err}");

    // A later backend pushed after the reference, so the tail is not ours.
    let mut stolen = names(3, 2);
    stolen.push("cell_from_a_later_backend".into());
    let err = r
        .weights_for(&stolen)
        .expect_err("tail is not the reference");
    assert!(
        err.to_string().contains("append the reference last"),
        "{err}"
    );

    // Nothing new to absorb is not an update.
    let err = r.weights_for(&names(0, 2)).expect_err("no new cells");
    assert!(err.to_string().contains("no new cells"), "{err}");
}

/// `update` reads the sidecar back to build the weights, and treats an absent
/// one as "this run carries none" rather than as a failure.
#[test]
fn sidecar_round_trips_and_absence_is_not_an_error() {
    use senna::pb_reference::{read_meta, sidecar_path, PbReferenceMeta, REFERENCE_BATCH};

    let dir = tempfile::tempdir().expect("tempdir");
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    assert!(read_meta(&prefix)
        .expect("absent is not an error")
        .is_none());

    let meta = PbReferenceMeta {
        senna_version: "9.9.9".into(),
        cell_counts: vec![1.0, 2.5, 300.0],
        batch_label: REFERENCE_BATCH.into(),
        batch_adjusted: true,
        generation: 4,
    };
    std::fs::write(
        sidecar_path(&prefix),
        serde_json::to_string_pretty(&meta).expect("serialize"),
    )
    .expect("write");

    let back = read_meta(&prefix).expect("read").expect("present");
    assert_eq!(back.cell_counts, meta.cell_counts);
    assert_eq!(back.generation, 4);
    assert!(back.batch_adjusted);
    assert_eq!(back.batch_label.as_ref(), REFERENCE_BATCH);
}
