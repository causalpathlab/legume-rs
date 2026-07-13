//! Back-compatibility re-export.
//!
//! The k-NN implementation moved to the [`crate::knn`] module directory when the
//! backend was swapped from `hnsw_rs` to a seeded instant-distance index. This
//! path is preserved so existing `matrix_util::knn_match::…` imports keep working
//! without churn across the workspace.
pub use crate::knn::*;
