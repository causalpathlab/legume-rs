//! Carried pseudobulks are weighted by *position*, so position has to be right.
//!
//! `weights_for` puts each stored cell-count on the column it belongs to by
//! assuming the loader appended the reference last. If that ever silently
//! fails — a reordering, a different `ColumnAlignment`, another backend pushed
//! after it — the weights land on the wrong columns: plausible numbers in the
//! wrong places, with no shape mismatch and no NaN to notice.

use senna::pb_reference::{weights_for, ReferenceInput, COLUMN_PREFIX};

fn names(new: usize, refs: usize) -> Vec<Box<str>> {
    let mut v: Vec<Box<str>> = (0..new)
        .map(|i| format!("cell{i}").into_boxed_str())
        .collect();
    v.extend((0..refs).map(|i| format!("{COLUMN_PREFIX}{i}@parent.pb_reference").into_boxed_str()));
    v
}

fn input(counts: &[f32]) -> ReferenceInput {
    ReferenceInput {
        parent: "parent".into(),
        backend: "parent.pb_reference.zarr".into(),
        batch_file: "parent.pb_reference_batch.txt".into(),
        cell_counts: counts.to_vec(),
    }
}

/// Through `weight_fn`, which is what the loader is actually handed.
#[test]
fn new_cells_weigh_one_and_carried_columns_carry_their_counts() {
    let r = input(&[3.0, 40.0, 7.0]);
    let w = r.weight_fn()(&names(5, 3)).expect("well-formed layout");

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
    let counts = [3.0, 40.0];

    let mut stray: Vec<Box<str>> = vec![
        "cell0".into(),
        format!("{COLUMN_PREFIX}99@parent.pb_reference").into_boxed_str(),
        "cell1".into(),
    ];
    stray.extend(names(0, 2));
    let err = weights_for(&counts, &stray).expect_err("stray reference column");
    assert!(err.to_string().contains("among the new cells"), "{err}");

    // A later backend pushed after the reference, so the tail is not ours.
    let mut stolen = names(3, 2);
    stolen.push("cell_from_a_later_backend".into());
    let err = weights_for(&counts, &stolen).expect_err("tail is not the reference");
    assert!(
        err.to_string().contains("append the reference last"),
        "{err}"
    );

    // Nothing new to absorb is not an update.
    let err = weights_for(&counts, &names(0, 2)).expect_err("no new cells");
    assert!(err.to_string().contains("no new cells"), "{err}");
}

/// Carried pseudobulks train the model but must not reach the per-cell
/// outputs, or every downstream artifact gains rows whose "barcode" is
/// `PBREF_37`. Composes with whatever QC already dropped.
#[test]
fn carried_columns_are_held_out_of_the_per_cell_outputs() {
    let r = input(&[3.0, 40.0, 7.0]);

    // 5 real cells + 3 carried.
    assert_eq!(r.keep_new_cells_only(8, None), vec![0, 1, 2, 3, 4]);

    // QC already dropped cells 1 and 3; the carried columns go too.
    assert_eq!(
        r.keep_new_cells_only(8, Some(vec![0, 2, 4, 5, 6, 7])),
        vec![0, 2, 4],
    );

    // No reference at all leaves the mask exactly as QC left it.
    assert_eq!(
        senna::pb_reference::exclude_carried(None, 8, Some(vec![0, 2])),
        Some(vec![0, 2]),
    );
    assert_eq!(senna::pb_reference::exclude_carried(None, 8, None), None);
}

/// Cell mass survives a round of accumulation.
///
/// A carried column stands for many cells, so re-emitting it has to add its
/// count, not 1. Counting columns instead cost the old cohort most of its
/// weight every round — 900 cells + 400 new came out as 735 — so a long-lived
/// model would slowly forget everything it had already absorbed, which is the
/// exact failure carrying pseudobulks forward exists to prevent.
#[test]
fn re_emitting_conserves_the_cells_behind_each_column() {
    use senna::pb_reference::cell_counts_from;

    // 4 new cells and 3 carried columns standing for 900 cells between them,
    // landing in 2 pseudobulks.
    let cell_to_pb = [0, 0, 1, 1, 0, 1, 1];
    let weight = [1.0, 1.0, 1.0, 1.0, 500.0, 300.0, 100.0];

    let counts = cell_counts_from(&cell_to_pb, 2, &weight);
    assert_eq!(counts, vec![502.0, 402.0]);
    assert_eq!(counts.iter().sum::<f32>(), weight.iter().sum::<f32>());

    // No multiplicity registered at all — every column is one cell.
    assert_eq!(cell_counts_from(&cell_to_pb, 2, &[]), vec![3.0, 4.0]);
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
