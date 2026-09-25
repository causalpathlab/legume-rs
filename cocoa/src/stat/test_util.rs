//! Helpers shared by the stat tests.

/// The logistic function.
pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
