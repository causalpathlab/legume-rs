// Re-export the upstream candle crates so downstream code can refer to
// `candle_util::candle_core` / `candle_util::candle_nn` without taking a
// direct dep.
pub use candle_core;
pub use candle_nn;

// Primitive layers, model traits, loaders, encoders, decoders, and
// training drivers — grouped by role.
pub mod cli;
pub mod data;
pub mod decoder;
pub mod encoder;
pub mod frozen_features;
pub mod loss;
pub mod mcmc;
pub mod nn;
pub mod sgvb;
pub mod topic_refinement;
pub mod traits;
pub mod value_transform;
pub mod vae;
