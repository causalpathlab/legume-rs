//! Per-group scatter and per-group segment rasterization via `tiny-skia`.
//!
//! Each call produces one transparent-background PNG, ready to embed as
//! an `<image>` layer in the vector SVG frame.
//!
//! Two rasterizers share layout conventions:
//! - [`rasterize_group_png`]: filled shape per point. Radius is either a
//!   single scalar or a per-point slice ([`RadiusSpec`]); the latter
//!   quantizes to 0.25-pixel bins so the marker path is reused within a
//!   bin.
//! - [`rasterize_segment_layer_png`]: packs all segments for one group
//!   into a single antialiased stroked path (used for cell-cell mesh
//!   plots colored by link-community).
//!
//! Both keep color and shape uniform per call (one layer = one hue);
//! heatmap-style per-point color is handled upstream by bucketing into
//! multiple layers (see e.g. pinto plot's viridis quantization).
//!
//! Data→pixel mapping lives in [`DataBounds::to_pixel`] so both senna
//! plot and pinto plot project points consistently and hull vertices /
//! label anchors line up with the raster layers.

use super::hull::Pt;
use super::palette::Rgb;
use clap::ValueEnum;
use tiny_skia::{FillRule, LineCap, Paint, Path, PathBuilder, Pixmap, Stroke, Transform};

/// One segment in pixel space (both endpoints share the layer's color).
pub type Segment = (Pt, Pt);

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
    Hexagon,
}

impl PointShape {
    const CYCLE: &'static [PointShape] = &[
        PointShape::Hexagon,
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
            PointShape::Hexagon => {
                // Flat-top regular hexagon. Tilted 30° from pointy-top so
                // neighboring hexagons share full vertical edges and pack
                // tightly without the star-shaped interstitial gaps that
                // appear with pointy-top tilings on a square layout.
                // Apothem = r (matches a Circle of radius r in extent);
                // circumradius = r · 2/√3.
                let rr = r * 2.0 / 3f32.sqrt();
                let a = r;
                pb.move_to(-rr, 0.0);
                pb.line_to(-rr / 2.0, -a);
                pb.line_to(rr / 2.0, -a);
                pb.line_to(rr, 0.0);
                pb.line_to(rr / 2.0, a);
                pb.line_to(-rr / 2.0, a);
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
    /// Build from raw extrema, inflating by 2% so edge points aren't
    /// clipped by sub-pixel antialiasing.
    #[must_use]
    pub fn from_minmax(xmin: f32, xmax: f32, ymin: f32, ymax: f32) -> Self {
        Self::from_minmax_padded(xmin, xmax, ymin, ymax, 0.02)
    }

    /// Build from raw extrema with custom pad fraction.
    #[must_use]
    pub fn from_minmax_padded(xmin: f32, xmax: f32, ymin: f32, ymax: f32, pad: f32) -> Self {
        let (dx, dy) = ((xmax - xmin).max(1e-6), (ymax - ymin).max(1e-6));
        Self {
            xmin: xmin - dx * pad,
            xmax: xmax + dx * pad,
            ymin: ymin - dy * pad,
            ymax: ymax + dy * pad,
        }
    }

    /// Data → pixel mapping (no y-flip: larger data-y → larger pixel-y).
    /// Shared by every callsite that needs hull vertices or label anchors
    /// aligned with the raster layers.
    #[must_use]
    pub fn to_pixel(&self, p: (f32, f32), ext: Extent) -> (f32, f32) {
        let (w, h) = (ext.w as f32, ext.h as f32);
        let tx = (p.0 - self.xmin) / (self.xmax - self.xmin) * w;
        let ty = (p.1 - self.ymin) / (self.ymax - self.ymin) * h;
        (tx, ty)
    }
}

/// Per-point radius spec. Scalar is the fast path (path allocated once);
/// Per quantizes to 0.25-pixel bins so the path is still reused within a
/// bin.
#[derive(Clone, Copy)]
pub enum RadiusSpec<'a> {
    Scalar(f32),
    Per(&'a [f32]),
}

/// Radius quantization step used by [`RadiusSpec::Per`]. Small enough to
/// preserve visual gradation, large enough that typical propensity/expr
/// mapping yields only a handful of distinct path objects per layer.
const RADIUS_QUANT_PX: f32 = 0.25;

/// Rasterize one group's points into a transparent `Pixmap`. Returns a
/// PNG-encoded byte buffer.
///
/// - `pts_px`: points already mapped to pixel coordinates.
/// - `radius`: scalar (uniform) or per-point half-extent in pixels.
/// - `alpha`: 0..=1 opacity.
pub fn rasterize_group_png(
    pts_px: &[(f32, f32)],
    ext: Extent,
    radius: RadiusSpec<'_>,
    color: Rgb,
    alpha: f32,
    shape: PointShape,
) -> anyhow::Result<Vec<u8>> {
    let mut pixmap = Pixmap::new(ext.w, ext.h)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc failed ({}x{})", ext.w, ext.h))?;

    let paint = fill_paint(color, alpha);

    match radius {
        RadiusSpec::Scalar(r) => {
            let marker = shape
                .build_path(r.max(0.1))
                .ok_or_else(|| anyhow::anyhow!("invalid marker path for {shape:?}"))?;
            for &(x, y) in pts_px {
                if !x.is_finite() || !y.is_finite() {
                    continue;
                }
                let t = Transform::from_translate(x, y);
                pixmap.fill_path(&marker, &paint, FillRule::Winding, t, None);
            }
        }
        RadiusSpec::Per(radii) => {
            if radii.len() != pts_px.len() {
                anyhow::bail!(
                    "per-point radius len {} != pts len {}",
                    radii.len(),
                    pts_px.len()
                );
            }
            // Group points by quantized radius so the marker path is
            // allocated once per bin rather than per point.
            let mut by_bin: std::collections::BTreeMap<i32, Vec<(f32, f32)>> = Default::default();
            for (&(x, y), &r) in pts_px.iter().zip(radii) {
                if !x.is_finite() || !y.is_finite() || !r.is_finite() || r <= 0.0 {
                    continue;
                }
                let bin = (r / RADIUS_QUANT_PX).round() as i32;
                by_bin.entry(bin.max(1)).or_default().push((x, y));
            }
            for (bin, pts) in by_bin {
                let r = (bin as f32) * RADIUS_QUANT_PX;
                let marker = match shape.build_path(r) {
                    Some(m) => m,
                    None => continue,
                };
                for (x, y) in pts {
                    let t = Transform::from_translate(x, y);
                    pixmap.fill_path(&marker, &paint, FillRule::Winding, t, None);
                }
            }
        }
    }

    pixmap
        .encode_png()
        .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"))
}

/// Rasterize one group's cell-cell segments as an antialiased stroked
/// polyline into a transparent `Pixmap`.
///
/// - `segs_px`: segment endpoints already mapped to pixel coordinates.
/// - `stroke_px`: line width in pixels.
pub fn rasterize_segment_layer_png(
    segs_px: &[Segment],
    ext: Extent,
    stroke_px: f32,
    color: Rgb,
    alpha: f32,
) -> anyhow::Result<Vec<u8>> {
    let mut pixmap = Pixmap::new(ext.w, ext.h)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc failed ({}x{})", ext.w, ext.h))?;

    let paint = fill_paint(color, alpha);
    let stroke = Stroke {
        width: stroke_px.max(0.1),
        line_cap: LineCap::Round,
        ..Stroke::default()
    };

    let mut pb = PathBuilder::new();
    let mut any = false;
    for &((x0, y0), (x1, y1)) in segs_px {
        if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
            continue;
        }
        pb.move_to(x0, y0);
        pb.line_to(x1, y1);
        any = true;
    }
    if any {
        if let Some(path) = pb.finish() {
            pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }
    pixmap
        .encode_png()
        .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"))
}

/// Rasterize a vector field of directed arrows. Each `(start, end)`
/// segment is drawn as a short shaft + filled triangular arrowhead at
/// `end`, oriented along the segment direction. Intended for quiver
/// plots: callers pass *short* segments (e.g. centred on edge midpoints)
/// rather than full cell-to-cell spans, so the arrows act as direction
/// glyphs rather than tracks.
pub fn rasterize_arrow_layer_png(
    segs_px: &[Segment],
    ext: Extent,
    stroke_px: f32,
    head_len_px: f32,
    color: Rgb,
    alpha: f32,
) -> anyhow::Result<Vec<u8>> {
    let mut pixmap = Pixmap::new(ext.w, ext.h)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc failed ({}x{})", ext.w, ext.h))?;

    let paint = fill_paint(color, alpha);
    let stroke = Stroke {
        width: stroke_px.max(0.1),
        line_cap: LineCap::Round,
        ..Stroke::default()
    };

    let head_len = head_len_px.max(1.0);
    // Slim arrowhead: half-width 35% of length → ~38° tip angle, more
    // arrow-like than the previous 57°-tip stub.
    let head_half_w = (head_len * 0.35).max(0.4);

    let mut shafts = PathBuilder::new();
    let mut heads = PathBuilder::new();
    let mut any_shaft = false;
    let mut any_head = false;

    for &((x0, y0), (x1, y1)) in segs_px {
        if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
            continue;
        }
        let (dx, dy) = (x1 - x0, y1 - y0);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-3 {
            continue;
        }
        let (ux, uy) = (dx / len, dy / len);
        let (px, py) = (-uy, ux);

        // Shaft stops at the arrowhead base so the head sits cleanly at end.
        let shaft_len = (len - head_len).max(len * 0.1);
        let (sx, sy) = (x0 + ux * shaft_len, y0 + uy * shaft_len);
        shafts.move_to(x0, y0);
        shafts.line_to(sx, sy);
        any_shaft = true;

        let (bx, by) = (x1 - ux * head_len, y1 - uy * head_len);
        let (lx, ly) = (bx + px * head_half_w, by + py * head_half_w);
        let (rx, ry) = (bx - px * head_half_w, by - py * head_half_w);
        heads.move_to(x1, y1);
        heads.line_to(lx, ly);
        heads.line_to(rx, ry);
        heads.close();
        any_head = true;
    }

    if any_shaft {
        if let Some(path) = shafts.finish() {
            pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }
    if any_head {
        if let Some(path) = heads.finish() {
            pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }
    pixmap
        .encode_png()
        .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"))
}

fn fill_paint(color: Rgb, alpha: f32) -> Paint<'static> {
    let mut paint = Paint::default();
    let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
    paint.set_color_rgba8(color.0, color.1, color.2, a);
    paint.anti_alias = true;
    paint
}
