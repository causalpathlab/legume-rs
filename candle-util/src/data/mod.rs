//! Data loaders for VAE training pipelines.
//!
//! - [`loader`]: generic dense `DataLoader` trait + in-memory impl
//! - [`loader_util`]: shared minibatch helpers
//! - [`joint`]: paired/multi-view loader (joint encoders)
//! - [`indexed`]: sparse/top-K packing helpers and the two-track gem loader
//! - [`masked_dense`]: window-free masked loader — dense rows, mask over every gene

pub mod indexed;
pub mod joint;
pub mod loader;
pub mod loader_util;
pub mod masked_dense;

pub use indexed::{
    build_indexed_samples, csc_columns_to_indexed_samples, gather_per_feature_at_indices,
    labeled_bar, pack_indices_values, top_k_indices_weighted, IndexedMinibatchData, IndexedSample,
};
pub use joint::{JointDataLoader, JointInMemoryArgs, JointInMemoryData, JointMinibatchData};
pub use loader::{DataLoader, InMemoryArgs, InMemoryData, MinibatchData};
pub use loader_util::{copy_shuffled, take_lb_ub, take_shuffled, Minibatches};
