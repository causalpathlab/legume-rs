//! Per-group scatter rasterization via `tiny-skia`.
//!
//! A single call produces one transparent-background PNG per group,
//! ready to embed as an `<image>` layer in the vector SVG frame. Points
//! are antialiased filled circles; group assignment is opaque — callers
//! index their own `(x, y, group_id, color)` list.

use super::palette::Rgb;
use clap::ValueEnum;
use tiny_skia::{FillRule, Paint, Path, PathBuilder, Pixmap, Transform};

/// Marker shape for scatter points. All shapes share the same bounding
/// half-extent `r` (so `--point-size` is comparable across shapes even
/// though geometric area differs slightly).
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum PointShape {
    Circle,
    Square,
    Triangle,
    Diamond,
}

impl PointShape {
    const CYCLE: &'static [PointShape] = &[
        PointShape::Circle,
        PointShape::Triangle,
        PointShape::Square,
        PointShape::Diamond,
    ];

    /// The `i`-th cycle shape (wraps).
    #[must_use]
    pub fn cycle_nth(i: usize) -> PointShape {
        Self::CYCLE[i % Self::CYCLE.len()]
    }

    /// Build a tiny-skia path for this shape centered at origin with
    /// bounding half-extent `r` (pixels).
    #[must_use]
    pub fn build_path(self, r: f32) -> Option<Path> {
        let mut pb = PathBuilder::new();
        match self {
            PointShape::Circle => {
                pb.push_circle(0.0, 0.0, r);
            }
            PointShape::Square => {
                pb.move_to(-r, -r);
                pb.line_to(r, -r);
                pb.line_to(r, r);
                pb.line_to(-r, r);
                pb.close();
            }
            PointShape::Triangle => {
                // Equilateral pointing up, inscribed in circle of radius r.
                let h = r * 3f32.sqrt() / 2.0;
                pb.move_to(0.0, -r);
                pb.line_to(h, r / 2.0);
                pb.line_to(-h, r / 2.0);
                pb.close();
            }
            PointShape::Diamond => {
                pb.move_to(0.0, -r);
                pb.line_to(r, 0.0);
                pb.line_to(0.0, r);
                pb.line_to(-r, 0.0);
                pb.close();
            }
        }
        pb.finish()
    }
}

/// Pixel extents of the plotting surface (width × height, both in pixels).
#[derive(Clone, Copy)]
pub struct Extent {
    pub w: u32,
    pub h: u32,
}

/// Data-space axis-aligned bounding box.
#[derive(Clone, Copy)]
pub struct DataBounds {
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
}

impl DataBounds {
    /// Inflate by 2% so extreme points aren't clipped by sub-pixel AA.
    #[must_use]
    pub fn from_minmax(xmin: f32, xmax: f32, ymin: f32, ymax: f32) -> Self {
        let (dx, dy) = ((xmax - xmin).max(1e-6), (ymax - ymin).max(1e-6));
        let pad = 0.02;
        Self {
            xmin: xmin - dx * pad,
            xmax: xmax + dx * pad,
            ymin: ymin - dy * pad,
            ymax: ymax + dy * pad,
        }
    }
}

/// Rasterize one group's points into a transparent `Pixmap`. Returns a
/// PNG-encoded byte buffer.
///
/// - `pts_px`: points already mapped to pixel coordinates.
/// - `radius_px`: bounding half-extent in pixels (interpretation depends
///   on `shape`; see [`PointShape`]).
/// - `alpha`: 0..=1 opacity.
pub fn rasterize_group_png(
    pts_px: &[(f32, f32)],
    ext: Extent,
    radius_px: f32,
    color: Rgb,
    alpha: f32,
    shape: PointShape,
) -> anyhow::Result<Vec<u8>> {
    let mut pixmap = Pixmap::new(ext.w, ext.h)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc failed ({}x{})", ext.w, ext.h))?;

    let mut paint = Paint::default();
    let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
    paint.set_color_rgba8(color.0, color.1, color.2, a);
    paint.anti_alias = true;

    // Build the marker path once at origin, re-translate per point.
    let marker = shape
        .build_path(radius_px)
        .ok_or_else(|| anyhow::anyhow!("invalid marker path for {shape:?}"))?;

    for &(x, y) in pts_px {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let t = Transform::from_translate(x, y);
        pixmap.fill_path(&marker, &paint, FillRule::Winding, t, None);
    }

    pixmap
        .encode_png()
        .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"))
}
