//! Data loaders for VAE training pipelines.
//!
//! - [`loader`]: generic dense `DataLoader` trait + in-memory impl
//! - [`loader_util`]: shared minibatch helpers
//! - [`joint`]: paired/multi-view loader (joint encoders)
//! - [`indexed`]: sparse/top-K indexed loader (used by `IndexedEmbeddingEncoder`)

pub mod indexed;
pub mod joint;
pub mod loader;
pub mod loader_util;

pub use indexed::{
    build_indexed_samples, build_sparse_edges_from_tensor, build_union_and_scatter_pos,
    compute_log_selection_freq, csc_columns_to_indexed_samples, gather_per_feature_at_indices,
    labeled_bar, pack_indices_values, slice_log_q_at_union, sum_sample_values,
    top_k_indices_weighted, GraphCsr, IndexedInMemoryArgs, IndexedInMemoryData,
    IndexedMinibatchData, IndexedSample, SparseEdgeBatch,
};
pub use joint::{JointDataLoader, JointInMemoryArgs, JointInMemoryData, JointMinibatchData};
pub use loader::{DataLoader, InMemoryArgs, InMemoryData, MinibatchData};
pub use loader_util::{copy_shuffled, take_lb_ub, take_shuffled, Minibatches};
