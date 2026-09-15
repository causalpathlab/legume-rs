use super::*;
use crate::fit::config::{TrackInfo, TrackSpec};
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;

/// `rows × 4` planted counts; row `r` of the returned backend holds
/// `triplets` entries whose row index is `rows_kept[r]` in the full layout, so
/// the two constructions below hold the SAME numbers in the same row order.
fn backend(rows: usize, rows_kept: &[usize]) -> SparseIoVec {
    let value = |r: usize, c: usize| (1 + (r * 7 + c * 3) % 11) as f32;
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for (new_r, &old_r) in rows_kept.iter().enumerate() {
        for c in 0..4usize {
            triplets.push((new_r as u64, c as u64, value(old_r, c)));
        }
    }
    let shape = (rows, 4usize, triplets.len());
    let mut b = create_sparse_from_triplets(&triplets, shape, None, Some(&SparseIoBackend::Zarr))
        .expect("backend");
    b.register_row_names_vec(
        &(0..rows)
            .map(|r| format!("r{r}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..4)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    v
}

/// Six feature rows over three genes: rows 0..3 on the base track, rows 3..6 on
/// a second track.
fn two_track_spec() -> TrackSpec {
    TrackSpec {
        track_of_row: vec![0, 0, 0, 1, 1, 1],
        gene_of_row: vec![0, 1, 2, 0, 1, 2],
        tracks: vec![
            TrackInfo {
                name: "t0".into(),
                is_count: true,
            },
            TrackInfo {
                name: "t1".into(),
                is_count: true,
            },
        ],
    }
}

#[test]
fn a_one_track_axis_masks_nothing() {
    assert!(base_track_row_mask(&TrackSpec::base(6), &(0..6).collect::<Vec<_>>(), 6).is_none());
}

#[test]
fn the_row_mask_keeps_the_base_tracks_backend_rows() {
    // A feature axis narrowed by an earlier pass: unified feature `i` lives on
    // backend row `2 * i`.
    let f2b: Vec<usize> = (0..6).map(|i| i * 2).collect();
    let keep = base_track_row_mask(&two_track_spec(), &f2b, 12).expect("two tracks ⇒ a mask");
    let kept: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter(|&(_, &k)| k)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(kept, vec![0, 2, 4]);
}

#[test]
fn weights_follow_the_rows_they_belong_to() {
    let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let keep = vec![false, true, true, false, true];
    assert_eq!(subset_kept(&w, &keep), vec![2.0, 3.0, 5.0]);
}

/// The projection that seeds the collapse must see the base track alone, so a
/// row-masked clone of a two-track backend has to sketch exactly like a backend
/// that only ever held those rows.
#[test]
fn the_masked_sketch_equals_a_base_only_backends_sketch() {
    let full = backend(6, &[0, 1, 2, 3, 4, 5]);
    let base_only = backend(3, &[0, 1, 2]);
    let keep = base_track_row_mask(&two_track_spec(), &(0..6).collect::<Vec<_>>(), 6)
        .expect("two tracks ⇒ a mask");
    let mut view = full.clone_for_collapse();
    view.mask_rows(&keep).expect("mask");
    assert_eq!(view.num_rows(), 3);

    let none: Option<&[Box<str>]> = None;
    let masked = project_backend(&view, 3, None, none, None, 1234).expect("masked sketch");
    let direct = project_backend(&base_only, 3, None, none, None, 1234).expect("base sketch");
    assert_eq!(masked.proj.shape(), direct.proj.shape());
    for (a, b) in masked.proj.iter().zip(direct.proj.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    // The same, weighted: the weights are subset to the kept rows in order.
    let w_full = vec![1.0f32, 0.0, 2.0, 9.0, 9.0, 9.0];
    let w_base = subset_kept(&w_full, &keep);
    assert_eq!(w_base, vec![1.0, 0.0, 2.0]);
    let masked_w = project_backend(&view, 3, None, none, Some(&w_base), 1234).expect("masked");
    let direct_w = project_backend(&base_only, 3, None, none, Some(&w_base), 1234).expect("base");
    for (a, b) in masked_w.proj.iter().zip(direct_w.proj.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
    // …and the weights actually bite: dropping a row changes the sketch.
    assert!(masked_w
        .proj
        .iter()
        .zip(masked.proj.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6));
}
