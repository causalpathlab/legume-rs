//! Primitive neural-network building blocks shared across encoders,
//! decoders, and topic models.
//!
//! - [`linear`]: linear variants (incl. non-negative weights, indexed softmax)
//! - [`layers`]: activations, IAF flows, sparsemax, misc layer helpers
//! - [`module`]: rank-embedding and small `Module` impls
//! - [`gcn`]: sparse residual γ-gated GCN block over packed top-K reps
//! - [`batch_norm`]: VarMap-aware BatchNorm (device-transfer safe)

pub mod batch_norm;
pub mod gcn;
pub mod layers;
pub mod linear;
pub mod module;
