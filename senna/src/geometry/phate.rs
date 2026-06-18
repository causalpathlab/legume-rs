//! PHATE diffusion embedding — the implementation now lives in
//! `matrix_util::layout` so it can be shared with `faba gem-annotate`.
//! Re-exported here to keep the `crate::geometry::phate::{..}` paths (and
//! the `From<&PhateCliArgs>` conversion in `postprocess::fit_layout_common`)
//! stable.

pub use matrix_util::layout::{phate_layout_2d, PhateArgs};
