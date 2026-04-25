//! Reusable 2D scatter-plotting building blocks for legume-rs tools.
//!
//! The pipeline mirrors the senna-era workflow: data points in data
//! coordinates → pixel mapping → per-layer transparent PNG rasters →
//! vector SVG frame with hulls + labels → optional flattened PNG and
//! true-vector PDF.
//!
//! Consumers compose the pieces in this crate (senna plot and pinto plot
//! share the same primitives; each owns its own CLI wrapper).

pub mod hull;
pub mod palette;
pub mod rasterize;
pub mod render;
pub mod svg_emit;

pub use hull::{convex_hull, hull_centroid, median_xy, trim_outliers_by_median, Pt};
pub use palette::{color as palette_color, resolve as palette_resolve, Palette, Rgb};
pub use rasterize::{
    rasterize_arrow_layer_png, rasterize_group_png, rasterize_segment_layer_png, DataBounds,
    Extent, PointShape, RadiusSpec, Segment,
};
pub use render::{render_pdf, render_png};
pub use svg_emit::{emit_svg, SvgOpts, TopicLayer};
