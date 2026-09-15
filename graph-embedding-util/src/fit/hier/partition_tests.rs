use super::*;
use crate::data::Triplet;
use crate::fit::config::{TrackInfo, TrackSpec};
use crate::fit::hier::units::UnitTable;

fn t(cell: u32, feature: u32, count: f32) -> Triplet {
    Triplet {
        cell,
        feature,
        count,
    }
}

#[test]
fn from_labels_groups_members_sorted_and_keeps_empty_modules() {
    let p = Partition::from_labels(&[1, 0, 1, 1], 3);
    assert_eq!(p.n_modules(), 3);
    assert_eq!(p.members[0], vec![1]);
    assert_eq!(p.members[1], vec![0, 2, 3]);
    assert!(p.members[2].is_empty());
    assert_eq!(p.slot_of(), vec![0, 0, 1, 2]);
}

#[test]
fn composition_is_count_share_per_module_and_by_module_uses_slots() {
    // genes 0,2 → module 1; gene 1 → module 0
    let p = Partition::from_labels(&[1, 0, 1], 2);
    let l0 = vec![t(0, 0, 1.0), t(0, 1, 3.0), t(0, 2, 1.0)];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[1], &[], None, 3);
    let um = UnitModules::new(&u, &p);
    assert_eq!(um.n_um, vec![3.0, 2.0]);
    assert!((um.q[0] - 0.6).abs() < 1e-6 && (um.q[1] - 0.4).abs() < 1e-6);
    assert_eq!(um.by_module[0].len(), 2);
    assert_eq!(um.by_module[0][0].0, (0, 0));
    assert_eq!(um.by_module[0][0].1, vec![(0, 3.0)]);
    assert_eq!(um.by_module[0][1].0, (0, 1));
    assert_eq!(um.by_module[0][1].1, vec![(0, 1.0), (1, 1.0)]); // gene 0 → slot 0, gene 2 → slot 1
}

#[test]
fn labels_from_membership_takes_the_row_argmax_with_low_index_ties() {
    let pi = nalgebra::DMatrix::<f32>::from_row_slice(3, 2, &[0.2, 0.8, 0.9, 0.1, 0.0, 0.0]);
    assert_eq!(labels_from_membership(&pi), vec![1, 0, 0]);
}

#[test]
fn an_empty_unit_has_a_zero_composition_row() {
    let p = Partition::from_labels(&[0, 0], 1);
    let l0 = vec![t(1, 0, 2.0)];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[2], &[], None, 2);
    let um = UnitModules::new(&u, &p);
    assert_eq!(um.q[0], 0.0);
    assert!(um.by_module[0].is_empty());
    assert_eq!(um.q[1], 1.0);
}

/// Row layout `[g0/t0, g1/t0, g0/t1, g1/t1]` — two tracks over two genes.
fn two_track_spec() -> TrackSpec {
    let s = TrackSpec {
        track_of_row: vec![0, 0, 1, 1],
        gene_of_row: vec![0, 1, 0, 1],
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
    };
    s.validate(4).unwrap();
    s
}

#[test]
fn composition_and_buckets_are_per_track_over_one_shared_gene_partition() {
    // gene 0 → module 0, gene 1 → module 1
    let p = Partition::from_labels(&[0, 1], 2);
    // unit 0 counts on all four rows; unit 1 only on track 0
    let l0 = vec![
        t(0, 0, 3.0),
        t(0, 1, 1.0),
        t(0, 2, 2.0),
        t(0, 3, 6.0),
        t(1, 0, 5.0),
    ];
    let u =
        UnitTable::from_pseudobulks_and_cells_tracked(&[&l0], &[2], &[], None, 4, two_track_spec());
    let um = UnitModules::new(&u, &p);
    assert_eq!(um.n_tracks, 2);
    assert_eq!(um.n_modules, 2);
    assert_eq!(um.idx(1, 1, 0), 6); // (u*T + t)*M + m = (1*2 + 1)*2 + 0
                                    // track 0 of unit 0: 3 of 4 in module 0; track 1: 2 of 8 in module 0
    assert_eq!(um.n_um[um.idx(0, 0, 0)], 3.0);
    assert_eq!(um.n_um[um.idx(0, 1, 1)], 6.0);
    assert!((um.q[um.idx(0, 0, 0)] - 0.75).abs() < 1e-6);
    assert!((um.q[um.idx(0, 0, 1)] - 0.25).abs() < 1e-6);
    assert!((um.q[um.idx(0, 1, 0)] - 0.25).abs() < 1e-6);
    assert!((um.q[um.idx(0, 1, 1)] - 0.75).abs() < 1e-6);
    // buckets keyed by (track, module), sorted
    let keys: Vec<(u32, u32)> = um.by_module[0].iter().map(|&(k, _)| k).collect();
    assert_eq!(keys, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    assert_eq!(um.by_module[0][2].1, vec![(0, 2.0)]); // (track 1, module 0): gene 0 at slot 0
}

#[test]
fn a_track_a_unit_has_no_counts_on_gets_no_bucket_and_a_zero_composition() {
    let p = Partition::from_labels(&[0, 1], 2);
    let l0 = vec![t(0, 0, 3.0), t(0, 1, 1.0), t(1, 2, 2.0)];
    let u =
        UnitTable::from_pseudobulks_and_cells_tracked(&[&l0], &[2], &[], None, 4, two_track_spec());
    let um = UnitModules::new(&u, &p);
    // unit 0 has nothing on track 1
    assert!(um.by_module[0].iter().all(|&((t, _), _)| t == 0));
    assert_eq!(um.q[um.idx(0, 1, 0)], 0.0);
    assert_eq!(um.q[um.idx(0, 1, 1)], 0.0);
    // unit 1 has nothing on track 0
    assert!(um.by_module[1].iter().all(|&((t, _), _)| t == 1));
    assert_eq!(um.q[um.idx(1, 0, 0)], 0.0);
}
