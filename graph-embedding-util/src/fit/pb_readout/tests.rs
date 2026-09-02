use super::*;

#[test]
fn majority_batch_follows_the_member_cells() {
    // pb 0: cells 0,1,2 in batches 0,0,1 → 0; pb 1: cells 3,4 in batch 1 → 1;
    // pb 2: empty → u32::MAX.
    let cell_to_pb = vec![0usize, 0, 0, 1, 1];
    let batch_of_cell = vec![0u32, 0, 1, 1, 1];
    assert_eq!(
        majority_batch_per_pb(&cell_to_pb, &batch_of_cell, 3),
        vec![0, 1, u32::MAX]
    );
}

#[test]
fn ties_resolve_to_the_lowest_batch() {
    let cell_to_pb = vec![0usize, 0];
    let batch_of_cell = vec![2u32, 1];
    assert_eq!(
        majority_batch_per_pb(&cell_to_pb, &batch_of_cell, 1),
        vec![1]
    );
}
