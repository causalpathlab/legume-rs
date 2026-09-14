use super::*;
use crate::data::Triplet;
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
    assert_eq!(um.by_module[0][0].0, 0);
    assert_eq!(um.by_module[0][0].1, vec![(0, 3.0)]);
    assert_eq!(um.by_module[0][1].0, 1);
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
