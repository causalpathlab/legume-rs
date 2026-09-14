use super::*;
use crate::data::Triplet;

fn t(cell: u32, feature: u32, count: f32) -> Triplet {
    Triplet {
        cell,
        feature,
        count,
    }
}

#[test]
fn pseudobulk_levels_then_cells_in_order_with_sorted_unique_features() {
    let l0 = vec![t(0, 2, 3.0), t(0, 0, 1.0), t(1, 1, 5.0)];
    let l1 = vec![t(0, 3, 2.0)];
    let cf = [1u32, 3];
    let cc = [4.0f32, 6.0];
    let cells = vec![(7u32, &cf[..], &cc[..])];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0, &l1], &cells, None, 4);
    assert_eq!(u.n_units(), 4);
    assert_eq!(u.level, vec![0, 0, 1, 2]);
    assert_eq!(u.source_index, vec![0, 1, 0, 7]);
    assert_eq!(u.feats[0], vec![0, 2]);
    assert_eq!(u.counts[0], vec![1.0, 3.0]);
    assert_eq!(u.total, vec![4.0, 5.0, 2.0, 10.0]);
    assert_eq!(u.feats[3], vec![1, 3]);
}

#[test]
fn weights_scale_with_sqrt_total_and_average_to_one() {
    let l0 = vec![t(0, 0, 1.0), t(1, 0, 9.0)];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[], None, 1);
    // totals 1 and 9 → sqrt 1 and 3 → normalized to mean 1: 0.5 and 1.5
    assert!((u.weight[0] - 0.5).abs() < 1e-6);
    assert!((u.weight[1] - 1.5).abs() < 1e-6);
}

#[test]
fn a_pseudobulk_with_no_edges_is_kept_as_an_empty_unit() {
    // pb 1 has no triplets but pb 2 does: the table must still hold three rows.
    let l0 = vec![t(0, 0, 1.0), t(2, 1, 2.0)];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[], None, 2);
    assert_eq!(u.n_units(), 3);
    assert!(u.feats[1].is_empty());
    assert_eq!(u.total[1], 0.0);
    assert_eq!(u.weight[1], 0.0);
}

#[test]
fn zero_and_negative_counts_are_dropped() {
    let l0 = vec![t(0, 0, 0.0), t(0, 1, 2.0), t(0, 2, -1.0)];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[], None, 3);
    assert_eq!(u.feats[0], vec![1]);
}
