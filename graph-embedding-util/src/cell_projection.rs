//! Shared constants for the analytical Poisson-MAP projections.
//!
//! Both `senna bge` and `senna gem` train in two phases: phase 1 fits the
//! shared feature side, phase 2 re-estimates the cell side.

/// Clamp on the linear predictor before `exp`.
///
/// Shared crate-wide: every Poisson fit here exponentiates the same linear
/// predictor in f32 (which overflows at 88), so the bound must move as one.
pub const SCORE_CLAMP: f64 = 30.0;
