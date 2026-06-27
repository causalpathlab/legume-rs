//! Senna-style progress bar for rayon-parallel builds.
//!
//! The implementation now lives in [`auxiliary_data::logging`] so the whole
//! workspace shares one style and one `MULTI_PROGRESS` (previously this file
//! held a byte-for-byte copy that was *not* registered with senna's bars). This
//! module is kept as `crate::progress` so existing call sites need no change.

pub use auxiliary_data::logging::new_progress_bar;
