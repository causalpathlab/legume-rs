//! Encoder modules for VAE-style topic / link-community models.
//!
//! - [`softmax`]: dense log-softmax encoder (baseline)
//! - [`softmax_iaf`]: log-softmax encoder with IAF flow head
//! - [`joint_softmax`]: paired/multi-view variant
//! - [`indexed`]: sparse top-K `IndexedEmbeddingEncoder` (with optional GCN)
//! - [`cell_embedded`]: per-cell embedding-table encoder

pub mod cell_embedded;
pub mod indexed;
pub mod joint_softmax;
pub mod softmax;
pub mod softmax_iaf;

pub use cell_embedded::{CellEmbeddedEncoder, CellEmbeddedEncoderArgs};
pub use indexed::{IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs};
pub use joint_softmax::{LogSoftmaxJointEncoder, LogSoftmaxJointEncoderArgs};
pub use softmax::{LogSoftmaxEncoder, LogSoftmaxEncoderArgs};
pub use softmax_iaf::{LogSoftmaxIAFEncoder, LogSoftmaxIAFEncoderArgs};
