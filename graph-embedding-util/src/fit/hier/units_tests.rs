use super::*;
use crate::data::Triplet;
use crate::fit::batch_fold::BatchGeneFold;

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
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0, &l1], &[2, 1], &cells, None, 4);
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
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[2], &[], None, 1);
    // totals 1 and 9 → sqrt 1 and 3 → normalized to mean 1: 0.5 and 1.5
    assert!((u.weight[0] - 0.5).abs() < 1e-6);
    assert!((u.weight[1] - 1.5).abs() < 1e-6);
}

#[test]
fn a_pseudobulk_with_no_edges_is_kept_as_an_empty_unit() {
    // pb 1 has no triplets but pb 2 does: the table must still hold three rows.
    let l0 = vec![t(0, 0, 1.0), t(2, 1, 2.0)];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[3], &[], None, 2);
    assert_eq!(u.n_units(), 3);
    assert!(u.feats[1].is_empty());
    assert_eq!(u.total[1], 0.0);
    assert_eq!(u.weight[1], 0.0);
}

#[test]
fn zero_and_negative_counts_are_dropped() {
    let l0 = vec![t(0, 0, 0.0), t(0, 1, 2.0), t(0, 2, -1.0)];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[1], &[], None, 3);
    assert_eq!(u.feats[0], vec![1]);
}

#[test]
fn a_trailing_pseudobulk_with_no_edges_is_still_a_row() {
    // one level, n_pb = 3, triplets only for pb 0
    let l0 = vec![t(0, 0, 1.0)];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[3], &[], None, 2);
    assert_eq!(u.n_units(), 3);
    assert_eq!(u.feats[0], vec![0]);
    assert!(u.feats[1].is_empty());
    assert!(u.feats[2].is_empty());
    assert_eq!(u.level, vec![0, 0, 0]);
    assert_eq!(u.source_index, vec![0, 1, 2]);
}

#[test]
fn cell_counts_are_divided_by_the_batch_fold() {
    // Build a BatchGeneFold with one batch, two features, fold [2.0, 4.0]
    let fold_table = BatchGeneFold {
        delta: vec![2.0, 4.0],
        n_features: 2,
        batch_names: vec!["batch0".into()],
    };
    let cell_to_batch = [0u32]; // cell 0 → batch 0
    let cf = CellBatchFold {
        fold: &fold_table,
        cell_to_batch: &cell_to_batch,
    };

    // A cell with counts [8.0, 8.0] on features [0, 1] should be divided by fold
    let cells = vec![(0u32, &[0u32, 1u32][..], &[8.0f32, 8.0f32][..])];
    let u = UnitTable::from_pseudobulks_and_cells(&[], &[], &cells, Some(cf), 2);

    assert_eq!(u.n_units(), 1);
    // 8.0 / 2.0 = 4.0, 8.0 / 4.0 = 2.0
    assert_eq!(u.counts[0], vec![4.0, 2.0]);
    assert_eq!(u.total[0], 6.0);
}

#[test]
fn all_zero_totals_give_all_zero_weights() {
    // one level, n_pb = 2, no triplets → all totals 0 → all weights 0
    let l0: Vec<Triplet> = vec![];
    let u = UnitTable::from_pseudobulks_and_cells(&[&l0], &[2], &[], None, 1);
    assert_eq!(u.weight, vec![0.0, 0.0]);
    assert!(!u.weight[0].is_nan());
    assert!(!u.weight[1].is_nan());
}
