//! [`ContrastiveIndex::calibrate_anchor_bias`] — the anchor intercept must make
//! the NULL model reproduce each anchor's observed total count exactly, because
//! that null is the baseline every spike-and-slab inclusion decision is measured
//! against.

use super::*;
use crate::posterior::lnpdf::poisson_ll;

/// A 3-anchor toy index over `n_other` frozen rows with spread-out biases, so
/// the partition sum is not a constant that would hide a scaling error.
fn toy(n_other: usize, h: usize, partition_scale: f64) -> ContrastiveIndex {
    let other_b: Vec<f32> = (0..n_other).map(|o| (o as f32) * 0.25 - 1.0).collect();
    ContrastiveIndex {
        other_e: vec![0.0; n_other * h],
        other_b,
        h,
        // totals: 6, 1, 0 — including an anchor with no observed counts.
        pos: vec![vec![(0, 4.0), (2, 2.0)], vec![(1, 1.0)], vec![]],
        anchor_b: vec![7.0, -3.0, 0.5], // deliberately far from any log-rate
        partition: (0..n_other as u32).collect(),
        partition_scale,
        anchor_offset: None,
    }
}

/// The defining property: at `θ = 0`, the modelled rate sum equals the observed
/// total, for every anchor with counts.
#[test]
fn calibrated_null_reproduces_the_observed_total() {
    let (h, n) = (4usize, 8usize);
    let mut idx = toy(n, h, 1.0);
    idx.calibrate_anchor_bias();

    let side = idx.frozen_side();
    let zeros = vec![0.0f32; h];
    for (a, nodes) in idx.node_terms().iter().enumerate() {
        let total: f64 = idx.pos[a].iter().map(|&(_, c)| f64::from(c)).sum();
        if total == 0.0 {
            continue;
        }
        // ll(0) = Σ n·s − scale·Σ exp(s); at the calibrated bias the rate term
        // must equal the total, so reconstruct it from the two pieces.
        let b = f64::from(idx.anchor_b[a]);
        let data_term: f64 = idx.pos[a]
            .iter()
            .map(|&(o, c)| f64::from(c) * (b + f64::from(idx.other_b[o as usize])))
            .sum();
        let ll = f64::from(poisson_ll(&zeros, b, nodes, &side));
        let rate = data_term - ll; // = scale · Σ exp(s)
        assert!(
            (rate - total).abs() < 1e-3 * total.max(1.0),
            "anchor {a}: modelled rate {rate} != observed total {total}"
        );
    }
}

/// `partition_scale` folds a sampled slate up to the full sum, so it must enter
/// the calibration — otherwise a subsampled partition would leave every anchor's
/// rate short by exactly that factor.
#[test]
fn calibration_accounts_for_the_partition_scale() {
    let (h, n) = (4usize, 8usize);
    let mut plain = toy(n, h, 1.0);
    let mut scaled = toy(n, h, 4.0);
    plain.calibrate_anchor_bias();
    scaled.calibrate_anchor_bias();
    // A 4× larger scale means each anchor needs a 4× smaller rate per row: ln 4.
    for a in 0..2 {
        let d = plain.anchor_b[a] - scaled.anchor_b[a];
        assert!(
            (d - 4.0f32.ln()).abs() < 1e-4,
            "anchor {a}: shift {d} != ln 4"
        );
    }
}

/// An anchor with no counts has no rate to match; it must be parked at a ~zero
/// rate rather than at `ln(0) = -inf`, which would poison every downstream sum.
#[test]
fn empty_anchor_is_parked_not_infinite() {
    let mut idx = toy(8, 4, 1.0);
    let cal = idx.calibrate_anchor_bias();
    assert_eq!(cal.n_empty, 1);
    assert!(idx.anchor_b[2].is_finite(), "empty anchor must stay finite");
    assert!(
        idx.anchor_b[2] < -20.0,
        "empty anchor must predict ~no rate"
    );
}

/// With a frozen offset the null is "offset only", not "nothing": the calibrated
/// rate must still equal the observed total once `⟨offset, e_o⟩` is in the score.
/// Getting this wrong would silently mis-baseline every velocity-gate inclusion.
#[test]
fn calibration_absorbs_the_frozen_offset() {
    let (h, n) = (3usize, 6usize);
    let mut idx = toy(n, h, 1.0);
    // Non-zero frozen side, so the offset actually changes the partition sum.
    idx.other_e = (0..n * h).map(|k| (k % 5) as f32 * 0.2 - 0.4).collect();
    // Anchor 0 carries a real offset; anchor 1's is zero, so it must land on the
    // same bias the no-offset path gives — the two cases have to agree there.
    let mut off = vec![0.0f32; 3 * h];
    off[..h].copy_from_slice(&[0.7, -0.3, 0.5]);
    idx.anchor_offset = Some(off);
    idx.calibrate_anchor_bias();

    let side = idx.frozen_side();
    let zeros = vec![0.0f32; h];
    let nodes = idx.node_terms();
    for a in 0..2 {
        let total: f64 = idx.pos[a].iter().map(|&(_, c)| f64::from(c)).sum();
        let b = f64::from(idx.anchor_b[a]);
        let offset = idx.anchor_offset.as_ref().unwrap();
        let off_a = &offset[a * h..(a + 1) * h];
        let data_term: f64 = idx.pos[a]
            .iter()
            .map(|&(o, c)| {
                let e_o = &idx.other_e[o as usize * h..(o as usize + 1) * h];
                let dot: f64 = off_a
                    .iter()
                    .zip(e_o)
                    .map(|(f, e)| f64::from(*f) * f64::from(*e))
                    .sum();
                f64::from(c) * (dot + b + f64::from(idx.other_b[o as usize]))
            })
            .sum();
        let rate = data_term - f64::from(poisson_ll(&zeros, b, &nodes[a], &side));
        assert!(
            (rate - total).abs() < 1e-3 * total.max(1.0),
            "anchor {a}: offset-aware rate {rate} != total {total}"
        );
    }
}

/// A zero offset must be indistinguishable from no offset — otherwise the
/// per-anchor path and the shared fast path have quietly diverged.
#[test]
fn zero_offset_matches_the_no_offset_path() {
    let (h, n) = (3usize, 6usize);
    let mut plain = toy(n, h, 2.0);
    let mut zeroed = toy(n, h, 2.0);
    plain.other_e = (0..n * h).map(|k| (k % 5) as f32 * 0.2 - 0.4).collect();
    zeroed.other_e = plain.other_e.clone();
    zeroed.anchor_offset = Some(vec![0.0; 3 * h]);
    plain.calibrate_anchor_bias();
    zeroed.calibrate_anchor_bias();
    for a in 0..3 {
        assert!(
            (plain.anchor_b[a] - zeroed.anchor_b[a]).abs() < 1e-4,
            "anchor {a}: {} vs {}",
            plain.anchor_b[a],
            zeroed.anchor_b[a]
        );
    }
}

/// The reported shift is what tells a caller how far the frozen model was from
/// being Poisson-calibrated, so it must reflect the actual move.
#[test]
fn reported_shift_matches_the_actual_move() {
    let mut idx = toy(8, 4, 1.0);
    let before = idx.anchor_b.clone();
    let cal = idx.calibrate_anchor_bias();
    // Anchor 2 is empty and is excluded from the reported statistics.
    let mut moved: Vec<f32> = before
        .iter()
        .zip(&idx.anchor_b)
        .take(2)
        .map(|(b0, b1)| (b1 - b0).abs())
        .collect();
    moved.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((cal.median_abs_shift - moved[moved.len() / 2]).abs() < 1e-5);
    assert!((cal.max_abs_shift - moved[moved.len() - 1]).abs() < 1e-5);
}
