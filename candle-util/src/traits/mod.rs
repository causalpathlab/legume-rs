//! Encoder / decoder trait abstractions shared across topic models.
//!
//! - [`model`]: dense `EncoderModuleT` / `DecoderModuleT` interfaces
//! - [`indexed`]: sparse/indexed variants (`IndexedEncoderT`,
//!   `IndexedDecoderT`) used by top-K embedding pipelines

pub mod indexed;
pub mod model;

pub use indexed::{IndexedDecoderT, IndexedEncoderT};
pub use model::{
    joint_multinomial_llik, DecoderModuleT, EncoderModuleT, EssLlikFn, JointDecoderModuleT,
    JointEncoderModuleT, MatchedDecoderData, MatchedDecoderModuleT, MatchedDecoderRecon,
    MatchedEncoderData, MatchedEncoderLatent, MatchedEncoderModuleT, NewDecoder,
};
