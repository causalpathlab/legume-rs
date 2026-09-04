use super::*;

/// One round is the whole budget; several split it evenly with the remainder
/// on the first rounds; the rounds always sum to the budget.
#[test]
fn epochs_split_evenly_with_the_remainder_first() {
    assert_eq!(
        (0..3)
            .map(|r| epochs_for_round(10, 3, r))
            .collect::<Vec<_>>(),
        [4, 3, 3]
    );
    assert_eq!(epochs_for_round(10, 1, 0), 10);
    assert_eq!(
        (0..3)
            .map(|r| epochs_for_round(0, 3, r))
            .collect::<Vec<_>>(),
        [0, 0, 0]
    );
    for (total, rounds) in [(1usize, 4usize), (7, 2), (100, 7), (5, 5), (3, 8)] {
        let sum: usize = (0..rounds)
            .map(|r| epochs_for_round(total, rounds, r))
            .sum();
        assert_eq!(sum, total, "total {total} over {rounds} rounds");
    }
}
