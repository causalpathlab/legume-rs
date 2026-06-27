//! Logging / progress-bar helpers.
//!
//! The implementation now lives in [`auxiliary_data::logging`] so the whole
//! workspace shares one progress-bar style and one `MULTI_PROGRESS`. This module
//! is kept as `crate::logging` so existing senna call sites need no change.

pub use auxiliary_data::logging::{init_logger, new_progress_bar};
