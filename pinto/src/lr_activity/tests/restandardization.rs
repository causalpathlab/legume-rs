//! The restandardization calibration guard.
//!
//! `z_re` asks whether a pair stands out against the OTHER pairs in its
//! stratum. That question only has an answer when the cross-pair spread is
//! commensurate with a single pair's own permutation noise; when the pairs
//! all sit in one tight clump, the clump's width is a noise floor, and
//! dividing by it manufactures astronomical significance for trivial
//! values. A real case: a pair at permutation z = 0.2 (p = 0.42) came out
//! at z_re = 12.3, p_re = 1e-34, and sorting by p_re built a plausible
//! signalling story the permutation null flatly rejected.

use crate::lr_activity::fit::restandardization_scale;

/// The field-report failure shape: a tight clump of near-identical
/// statistics whose spread sits far below the pairs' own permutation
/// noise. No scale may be returned, whatever an absolute floor says.
#[test]
fn a_clump_far_below_permutation_noise_is_uncalibrated() {
    // Cross-pair spread ~1e-3, per-pair permutation sd ~0.05: ratio ~0.02.
    let stats: Vec<f32> = (0..200)
        .map(|i| 0.009 + 1e-3 * ((i % 7) as f32 / 7.0))
        .collect();
    let null_sds = vec![0.054f32; 200];
    assert!(
        restandardization_scale(&stats, &null_sds).is_none(),
        "a clump below the noise floor cannot calibrate anything"
    );
}

/// A healthy stratum: cross-pair spread on the same order as the pairs'
/// permutation noise. The scale comes back, and it is the robust MAD scale.
#[test]
fn commensurate_spread_calibrates() {
    let stats: Vec<f32> = (0..200)
        .map(|i| 0.05 * ((i as f32) - 100.0) / 100.0)
        .collect();
    let null_sds = vec![0.05f32; 200];
    let sigma = restandardization_scale(&stats, &null_sds)
        .expect("a spread matching the noise scale must calibrate");
    assert!(
        sigma > 0.0 && sigma < 0.1,
        "robust scale in range, got {sigma}"
    );
}

/// The guard is scale-RELATIVE: shrinking both the statistics and the
/// permutation noise by the same factor must not change the verdict. An
/// absolute floor fails exactly this.
#[test]
fn the_guard_is_scale_relative() {
    let stats: Vec<f32> = (0..200)
        .map(|i| 5e-5 * ((i as f32) - 100.0) / 100.0)
        .collect();
    let null_sds = vec![5e-5f32; 200];
    assert!(
        restandardization_scale(&stats, &null_sds).is_some(),
        "tiny but commensurate scales are calibrated; only the RATIO matters"
    );
}

/// Empty input calibrates nothing.
#[test]
fn empty_input_is_uncalibrated() {
    assert!(restandardization_scale(&[], &[]).is_none());
}
