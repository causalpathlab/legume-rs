//! Tests for the cell-level delta `Δθ = θ^u − θ^s` and the common-mode gauge.
//!
//! The arithmetic is small; what can silently go wrong is the **convention**.
//! A sign flip produces a perfectly well-formed velocity field pointing into
//! the past and nothing downstream would notice — `faba lineage` would happily
//! lay out a trajectory running backwards. Hence the sign test.
//!
//! The estimator changed (a dictionary operator `P·θ` before, a difference of
//! two post-hoc per-track fits now) but the convention did not, which is why
//! this file survives the change.

use super::*;

const K: usize = 3;

/// Pack rows of log θ, the layout `Inferred` uses.
fn log_theta(rows: &[[f32; K]]) -> Vec<f32> {
    rows.iter().flat_map(|r| r.iter().map(|p| p.ln())).collect()
}

fn inferred(nascent: &[[f32; K]], mature: &[[f32; K]]) -> Inferred {
    Inferred {
        latent: log_theta(mature),
        latent_mature: log_theta(mature),
        latent_nascent: log_theta(nascent),
    }
}

/// Identical tracks ⇒ no movement. If a cell's nascent and mature compositions
/// agree, it is at steady state and the velocity must be exactly zero, not
/// merely small.
#[test]
fn identical_tracks_give_zero_velocity() {
    let same = [[0.5f32, 0.3, 0.2], [0.1, 0.1, 0.8]];
    let inf = inferred(&same, &same);
    let v: Vec<f32> = inf
        .latent_nascent
        .iter()
        .zip(inf.latent_mature.iter())
        .map(|(u, s)| u.exp() - s.exp())
        .collect();
    for (i, x) in v.iter().enumerate() {
        assert!(x.abs() < 1e-6, "steady-state cell moved: component {i} = {x}");
    }
}

/// THE SIGN CONVENTION. A factor the cell is moving TOWARD — more of it in the
/// nascent (newly transcribed) pool than in the mature one — must get a
/// POSITIVE velocity component.
///
/// Nascent is what was just transcribed, so an excess there is the future, not
/// the past. Reverse this and every downstream trajectory runs backwards while
/// looking entirely well-formed.
#[test]
fn a_factor_enriched_in_nascent_gets_positive_velocity() {
    // Cell moving toward topic 2: nascent has more of it than mature.
    let nascent = [[0.2f32, 0.2, 0.6]];
    let mature = [[0.4f32, 0.4, 0.2]];
    let inf = inferred(&nascent, &mature);
    let v: Vec<f32> = inf
        .latent_nascent
        .iter()
        .zip(inf.latent_mature.iter())
        .map(|(u, s)| u.exp() - s.exp())
        .collect();
    assert!(v[2] > 0.0, "topic 2 is nascent-enriched, so it is the FUTURE: {v:?}");
    assert!(v[0] < 0.0 && v[1] < 0.0, "topics it is leaving must be negative: {v:?}");
    let total: f32 = v.iter().sum();
    assert!(
        total.abs() < 1e-5,
        "a difference of two simplex points must sum to zero, got {total}"
    );
}

/// `center_velocity` removes the population mean per axis and hands back the
/// offset, so the uncentred field is recoverable.
#[test]
fn centering_removes_the_population_mean_and_returns_it() {
    let mut v = vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0];
    let offset = center_velocity(&mut v, 2, 3, "test");
    assert_eq!(offset.len(), 3);
    for (a, b) in offset.iter().zip([2.0f32, 3.0, 4.0].iter()) {
        assert!((a - b).abs() < 1e-6, "offset {offset:?}");
    }
    for axis in 0..3 {
        let m: f32 = (0..2).map(|c| v[c * 3 + axis]).sum::<f32>() / 2.0;
        assert!(m.abs() < 1e-6, "axis {axis} not centred: mean {m}");
    }
}
