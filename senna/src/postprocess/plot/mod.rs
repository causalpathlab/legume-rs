//! `senna plot` — publication-quality rasterized scatter with vector
//! labels over transparent background.
//!
//! The rasterizer / palette / hull / SVG-emit / render primitives live
//! in the shared workspace crate `plot-utils`; senna (and pinto) both
//! import from there so the output path stays identical across tools.

pub use plot_utils::{hull, palette, rasterize, svg_emit};
