//! `senna plot` — publication-quality rasterized scatter with vector
//! labels over transparent background.
//!
//! Pipeline: load cell coordinates → assign group per cell (topic,
//! cluster, or pb_id) → rasterize one transparent PNG layer per group
//! via `tiny-skia` → wrap in a vector SVG frame with convex-hull
//! polygons and text labels at per-group medians. Optional flattened PNG
//! and true-vector PDF fall out via `resvg` and `svg2pdf`.

pub mod hull;
pub mod palette;
pub mod rasterize;
pub mod svg_emit;
