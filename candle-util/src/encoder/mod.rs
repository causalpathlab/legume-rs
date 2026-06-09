//! Encoder modules for VAE-style topic / link-community models.
//!
//! - [`softmax`]: dense log-softmax encoder (baseline)
//! - [`gaussian`]: dense Gaussian (scVI-style) encoder — raw continuous latent
//! - [`softmax_iaf`]: log-softmax encoder with IAF flow head
//! - [`joint_softmax`]: paired/multi-view variant
//! - [`indexed`]: sparse top-K `IndexedEmbeddingEncoder` (with optional GCN)

pub mod gaussian;
pub mod indexed;
pub mod joint_softmax;
pub mod softmax;
pub mod softmax_iaf;

pub use gaussian::{GaussianEncoder, GaussianEncoderArgs};
pub use indexed::{IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs};
pub use joint_softmax::{LogSoftmaxJointEncoder, LogSoftmaxJointEncoderArgs};
pub use softmax::{LogSoftmaxEncoder, LogSoftmaxEncoderArgs};
pub use softmax_iaf::{LogSoftmaxIAFEncoder, LogSoftmaxIAFEncoderArgs};
