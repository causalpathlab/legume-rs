//! PHATE diffusion embedding — the implementation now lives in
//! `legume_numeric::matrix::layout` so it can be shared with `senna annotate-by-projection`.
//! Re-exported here to keep the `crate::geometry::phate::{..}` paths (and
//! the `From<&PhateCliArgs>` conversion in `postprocess::fit_layout_common`)
//! stable.

pub use legume_numeric::matrix::layout::{phate_layout_2d, PhateArgs};
