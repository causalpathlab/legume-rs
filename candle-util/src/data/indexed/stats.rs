//! Per-sample / per-feature aggregate statistics on indexed samples.

use super::types::IndexedSample;
use rayon::prelude::*;

/// Compute log selection frequency for each feature from indexed samples.
///
/// `q_d = (# samples containing feature d) / (total samples)`, clamped to
/// `[1/n, 1]`. Returns `log(q_d)` for each of `n_features` features.
/// Used for importance-weighted conditional softmax (Jean et al., 2015,
/// "On Using Very Large Target Vocabulary for Neural Machine Translation").
pub fn compute_log_selection_freq(samples: &[IndexedSample], n_features: usize) -> Vec<f32> {
    let n = samples.len().max(1) as f32;
    let mut counts = vec![0u32; n_features];
    for sample in samples {
        for &idx in &sample.indices {
            counts[idx as usize] += 1;
        }
    }
    counts
        .par_iter()
        .map(|&c| ((c.max(1) as f32) / n).ln())
        .collect()
}

/// Sum of every value across all samples — the per-epoch decoder count
/// total. Invariant to minibatch shuffling, so it can be cached once.
pub fn sum_sample_values(samples: &[IndexedSample]) -> f32 {
    samples
        .par_iter()
        .map(|s| s.values.iter().sum::<f32>())
        .sum()
}
