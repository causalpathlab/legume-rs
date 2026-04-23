//! PB-membership permutation helper for the sample-permutation null.
//!
//! The Efron–Tibshirani row-randomization moments and the actual ES walks
//! over permuted profiles are inlined in `orchestrate.rs` (one tight loop
//! per permutation that handles matmul → specificity → ranking → ES per
//! (k, c)). This module owns just the index-permutation primitive.

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Draw a permutation of `[0, p)`. When `batch_labels` is provided, shuffle
/// within each batch block (positions sharing a batch label are permuted
/// among themselves).
pub fn permute_indices(p: usize, batch_labels: Option<&[u32]>, rng: &mut SmallRng) -> Vec<usize> {
    let mut pi: Vec<usize> = (0..p).collect();
    match batch_labels {
        None => {
            pi.shuffle(rng);
        }
        Some(labels) => {
            assert_eq!(labels.len(), p);
            let mut by_batch: rustc_hash::FxHashMap<u32, Vec<usize>> =
                rustc_hash::FxHashMap::default();
            for (i, &b) in labels.iter().enumerate() {
                by_batch.entry(b).or_default().push(i);
            }
            for (_, idxs) in by_batch.iter_mut() {
                let mut shuffled = idxs.clone();
                shuffled.shuffle(rng);
                for (orig, new) in idxs.iter().zip(shuffled.iter()) {
                    pi[*orig] = *new;
                }
            }
        }
    }
    pi
}

/// Convenience: seed a `SmallRng`.
pub fn seeded_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permute_indices_is_a_permutation() {
        let mut rng = seeded_rng(42);
        let pi = permute_indices(10, None, &mut rng);
        let mut sorted = pi.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn permute_indices_within_batch_preserves_blocks() {
        let mut rng = seeded_rng(7);
        let labels = vec![0u32, 0, 0, 1, 1, 1];
        let pi = permute_indices(6, Some(&labels), &mut rng);
        for (i, &new) in pi.iter().enumerate() {
            assert_eq!(labels[i], labels[new]);
        }
    }
}
