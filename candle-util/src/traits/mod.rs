//! Encoder / decoder trait abstractions shared across topic models.
//!
//! - [`model`]: dense `EncoderModuleT` / `DecoderModuleT` interfaces
//! - [`indexed`]: sparse/indexed variants (`IndexedEncoderT`, `IndexedDecoderT`,
//!   `CellEncoderT`) used by top-K embedding pipelines

pub mod indexed;
pub mod model;
