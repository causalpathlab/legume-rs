//! Emit a transparent-background SVG wrapping pre-rasterized per-group
//! PNG layers, convex-hull polygons, and vector text labels.
//!
//! Output structure (named `<g>` groups — easy to toggle in
//! Illustrator / Inkscape):
//!
//! ```text
//! <svg viewBox="0 0 W H">
//!   <g id="raster-layers">
//!     <image id="topic-0" .../>
//!     ...
//!   </g>
//!   <g id="hulls">           (if enabled)
//!     <polygon id="hull-0" .../>
//!     ...
//!   </g>
//!   <g id="labels">
//!     <text id="label-0" .../>
//!     ...
//!   </g>
//! </svg>
//! ```
//!
//! The whole SVG lives in pixel space (matching the raster layers), so
//! label (x, y) must be pre-mapped to pixels by the caller.

use super::hull::Pt;
use super::palette::Rgb;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use std::fmt::Write;

/// White halo stroke width as a fraction of font size. Matches typical
/// SVG label-on-dense-raster conventions — wide enough to read over any
/// background without visibly eating into glyph strokes.
const LABEL_HALO_WIDTH_FACTOR: f32 = 0.35;
/// Minimum halo width for a main group label (pixels). Large enough that
/// even at small --label-font-size the halo still separates text from
/// raster.
const MAIN_LABEL_HALO_MIN_PX: f32 = 1.5;

/// Per-topic inputs to the SVG assembler. All coordinates are in pixels,
/// in the same pixel-space as the rasterized layers.
#[derive(Clone)]
pub struct TopicLayer {
    /// Display label, e.g. `"Neuron"` or `"T3"`.
    pub label: String,
    /// Pre-rasterized transparent PNG of this topic's points.
    pub png: Vec<u8>,
    /// Convex hull polygon in pixel space (empty → no hull drawn).
    pub hull_px: Vec<Pt>,
    /// Label anchor in pixel space.
    pub label_xy_px: Pt,
    /// Topic color (for hull stroke + label fill).
    pub color: Rgb,
}

/// Default canvas when a caller uses `..SvgOpts::default()` without sizing it.
/// A derived `Default` would hand out a 0×0 canvas, which every rasterizer
/// downstream rejects; 8 inches at 96 dpi is a plausible figure instead.
const DEFAULT_CANVAS_PX: u32 = 768;

/// SVG assembly options.
///
/// [`Default`] gives a transparent [`DEFAULT_CANVAS_PX`]-square canvas with no
/// hulls, labels or frame — so a caller names only the fields it cares about
/// (`SvgOpts { width_px, height_px, ..Default::default() }`) and adding a field
/// here stops breaking every call site.
pub struct SvgOpts {
    pub width_px: u32,
    pub height_px: u32,
    pub draw_hulls: bool,
    pub draw_labels: bool,
    pub label_font_size_px: f32,
    /// Hull stroke width in pixels.
    pub hull_stroke_px: f32,
    /// Hull fill opacity (0..=1). 0 = no fill.
    pub hull_fill_alpha: f32,
    /// Outline box stroke width in pixels around the full canvas
    /// (0 = no frame).
    pub frame_stroke_px: f32,
    /// Canvas background. `None` (the default) leaves the SVG transparent, which
    /// the PNG rasterizer then flattens to **black** — fine when the caller wants
    /// that, surprising otherwise. `Some(rgb)` paints a full-canvas rect first,
    /// e.g. R's `gray90` = `(229, 229, 229)`.
    pub background: Option<Rgb>,
}

impl Default for SvgOpts {
    fn default() -> Self {
        Self {
            width_px: DEFAULT_CANVAS_PX,
            height_px: DEFAULT_CANVAS_PX,
            draw_hulls: false,
            draw_labels: false,
            label_font_size_px: 12.0,
            hull_stroke_px: 1.0,
            hull_fill_alpha: 0.0,
            frame_stroke_px: 0.0,
            background: None,
        }
    }
}

/// Assemble a full SVG document as a `String`.
#[must_use]
pub fn emit_svg(layers: &[TopicLayer], opts: &SvgOpts) -> String {
    let mut s = String::with_capacity(layers.iter().map(|l| l.png.len() * 2).sum::<usize>() + 4096);
    let _ = write!(
        s,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <svg xmlns=\"http://www.w3.org/2000/svg\" \
             xmlns:xlink=\"http://www.w3.org/1999/xlink\" \
             viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\">\n",
        w = opts.width_px,
        h = opts.height_px,
    );

    // Background, before every raster layer so it never occludes them.
    if let Some((r, g, b)) = opts.background {
        let _ = writeln!(
            &mut s,
            "  <rect id=\"background\" x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" \
             fill=\"rgb({r},{g},{b})\"/>",
            w = opts.width_px,
            h = opts.height_px,
        );
    }

    // Raster layers.
    let _ = writeln!(&mut s, "  <g id=\"raster-layers\">");
    for (i, layer) in layers.iter().enumerate() {
        let b64 = BASE64.encode(&layer.png);
        let _ = writeln!(
            &mut s,
            "    <image id=\"topic-{i}\" x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" \
             preserveAspectRatio=\"none\" href=\"data:image/png;base64,{b64}\"/>",
            w = opts.width_px,
            h = opts.height_px,
        );
    }
    let _ = writeln!(&mut s, "  </g>");

    // Hulls.
    if opts.draw_hulls {
        let _ = writeln!(&mut s, "  <g id=\"hulls\">");
        for (i, layer) in layers.iter().enumerate() {
            if layer.hull_px.len() < 3 {
                continue;
            }
            let (r, g, b) = layer.color;
            let pts: String = layer
                .hull_px
                .iter()
                .map(|(x, y)| format!("{x:.2},{y:.2}"))
                .collect::<Vec<_>>()
                .join(" ");
            let fill_alpha = opts.hull_fill_alpha.clamp(0.0, 1.0);
            let fill_attr = if fill_alpha > 0.0 {
                format!("rgb({r},{g},{b})")
            } else {
                "none".to_string()
            };
            let _ = writeln!(
                &mut s,
                "    <polygon id=\"hull-{i}\" points=\"{pts}\" \
                 fill=\"{fill_attr}\" fill-opacity=\"{fa:.3}\" \
                 stroke=\"rgb({r},{g},{b})\" stroke-width=\"{sw}\" \
                 stroke-linejoin=\"round\"/>",
                fa = fill_alpha,
                sw = opts.hull_stroke_px,
            );
        }
        let _ = writeln!(&mut s, "  </g>");
    }

    if opts.frame_stroke_px > 0.0 {
        let sw = opts.frame_stroke_px;
        let inset = sw * 0.5;
        let fw = (opts.width_px as f32 - sw).max(0.0);
        let fh = (opts.height_px as f32 - sw).max(0.0);
        let _ = writeln!(&mut s, "  <g id=\"frame\">");
        let _ = writeln!(
            &mut s,
            "    <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" \
             fill=\"none\" stroke=\"black\" stroke-width=\"{sw:.2}\"/>",
            x = inset,
            y = inset,
            w = fw,
            h = fh,
        );
        let _ = writeln!(&mut s, "  </g>");
    }

    if opts.draw_labels {
        let _ = writeln!(&mut s, "  <g id=\"labels\">");
        for (i, layer) in layers.iter().enumerate() {
            emit_halo_text(
                &mut s,
                &format!("label-{i}"),
                layer.label_xy_px,
                opts.label_font_size_px,
                MAIN_LABEL_HALO_MIN_PX,
                layer.color,
                &layer.label,
            );
        }
        let _ = writeln!(&mut s, "  </g>");
    }

    let _ = writeln!(&mut s, "</svg>");
    s
}

/// Emit a single `<text>` element with a white halo stroke + colored
/// fill, centered on `xy`. Skips emission silently for non-finite
/// positions (a common no-op when a group has no points).
fn emit_halo_text(
    s: &mut String,
    id: &str,
    xy: Pt,
    font_size_px: f32,
    halo_min_px: f32,
    color: Rgb,
    text: &str,
) {
    let (x, y) = xy;
    if !x.is_finite() || !y.is_finite() || text.is_empty() {
        return;
    }
    let (r, g, b) = color;
    let hw = (font_size_px * LABEL_HALO_WIDTH_FACTOR).max(halo_min_px);
    let esc = escape_xml(text);
    let _ = writeln!(
        s,
        "    <text id=\"{id}\" x=\"{x:.2}\" y=\"{y:.2}\" \
         font-family=\"Helvetica, Arial, sans-serif\" \
         font-size=\"{font_size_px:.2}\" text-anchor=\"middle\" \
         dominant-baseline=\"central\" \
         paint-order=\"stroke\" \
         stroke=\"white\" stroke-width=\"{hw:.2}\" stroke-linejoin=\"round\" \
         fill=\"rgb({r},{g},{b})\">{esc}</text>",
    );
}

/// Composite all per-layer raster PNGs into a single PNG and return a new
/// layer set with one composite raster + the original hulls/labels
/// preserved (raster-empty). Used by PDF/PNG emit so the output carries
/// one image stream instead of K stacked full-canvas rasters — drops PDF
/// size roughly K× on multi-community plots and lets viewers open them
/// instantly. Multi-layer SVG is still produced when `--svg` is on so
/// Illustrator/Inkscape can toggle per-topic groups.
pub fn flatten_raster_layers(
    layers: &[TopicLayer],
    width_px: u32,
    height_px: u32,
) -> anyhow::Result<Vec<TopicLayer>> {
    use tiny_skia::{Pixmap, PixmapPaint, Transform};

    let mut canvas = Pixmap::new(width_px, height_px)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc failed ({width_px}x{height_px})"))?;
    let paint = PixmapPaint::default();
    let mut any_raster = false;
    for layer in layers {
        if layer.png.is_empty() {
            continue;
        }
        let pm = Pixmap::decode_png(&layer.png)
            .map_err(|e| anyhow::anyhow!("layer PNG decode failed: {e}"))?;
        canvas.draw_pixmap(0, 0, pm.as_ref(), &paint, Transform::identity(), None);
        any_raster = true;
    }

    let mut out = Vec::with_capacity(layers.len() + 1);
    if any_raster {
        let composite = canvas
            .encode_png()
            .map_err(|e| anyhow::anyhow!("composite PNG encode failed: {e}"))?;
        out.push(TopicLayer {
            label: String::new(),
            png: composite,
            hull_px: Vec::new(),
            label_xy_px: (f32::NAN, f32::NAN),
            color: (0, 0, 0),
        });
    }
    for layer in layers {
        if !layer.hull_px.is_empty() || !layer.label.is_empty() {
            out.push(TopicLayer {
                label: layer.label.clone(),
                png: Vec::new(),
                hull_px: layer.hull_px.clone(),
                label_xy_px: layer.label_xy_px,
                color: layer.color,
            });
        }
    }
    Ok(out)
}

/// Escape the XML-significant characters (`& < > " '`). Use anywhere SVG
/// element text or attribute strings are built from arbitrary input.
#[must_use]
pub fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}
