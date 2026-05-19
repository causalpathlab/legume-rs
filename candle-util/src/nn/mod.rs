//! Primitive neural-network building blocks shared across encoders,
//! decoders, and topic models.
//!
//! - [`linear`]: linear variants (incl. non-negative weights, indexed softmax)
//! - [`layers`]: activations, IAF flows, sparsemax, misc layer helpers
//! - [`gcn`]: sparse residual γ-gated GCN block over packed top-K reps
//! - [`batch_norm`]: VarMap-aware BatchNorm (device-transfer safe)

pub mod batch_norm;
pub mod gcn;
pub mod layers;
pub mod linear;

pub use batch_norm::{batch_norm, BatchNorm, BatchNormConfig};
pub use gcn::GcnBlock;
pub use layers::{iaf_stack_linear, sparsemax, stack_relu_linear, IAFLayers, StackLayers};
pub use linear::{
    aggregate_linear, aggregate_linear_hard, log_softmax_linear, log_softmax_linear_nobias,
    logsumexp_forward, non_neg_linear, sparsemax_linear, AggregateLinear, NonNegLinear,
    SoftmaxLinear, SparsemaxLinear,
};
