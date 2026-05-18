//! Data loaders for VAE training pipelines.
//!
//! - [`loader`]: generic dense `DataLoader` trait + in-memory impl
//! - [`loader_util`]: shared minibatch helpers
//! - [`joint`]: paired/multi-view loader (joint encoders)
//! - [`cell_grouped`]: per-cell grouped minibatching for pseudobulk-aware training
//! - [`indexed`]: sparse/top-K indexed loader (used by `IndexedEmbeddingEncoder`)

pub mod cell_grouped;
pub mod indexed;
pub mod joint;
pub mod loader;
pub mod loader_util;
