//! Encoder modules for VAE-style topic / link-community models.
//!
//! - [`softmax`]: dense log-softmax encoder (baseline)
//! - [`gaussian`]: dense Gaussian (scVI-style) encoder — raw continuous latent
//! - [`softmax_iaf`]: log-softmax encoder with IAF flow head
//! - [`joint_softmax`]: paired/multi-view variant
//! - [`indexed`]: sparse top-K `IndexedEmbeddingEncoder`

pub mod coarse_pool;
pub mod dense_pool;
pub mod gaussian;
pub mod indexed;
pub mod joint_softmax;
pub mod pair_head;
pub mod pooled;
pub mod scatter_pool;
pub mod softmax;
pub mod softmax_iaf;

pub use gaussian::{GaussianEncoder, GaussianEncoderArgs};
pub use indexed::{IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs};
pub use joint_softmax::{LogSoftmaxJointEncoder, LogSoftmaxJointEncoderArgs};
pub use pair_head::{SymmetricPairHead, SymmetricPairHeadArgs};
pub use pooled::{PooledGeneEncoder, PooledGeneEncoderArgs};
pub use softmax::{LogSoftmaxEncoder, LogSoftmaxEncoderArgs};
pub use softmax_iaf::{LogSoftmaxIAFEncoder, LogSoftmaxIAFEncoderArgs};
