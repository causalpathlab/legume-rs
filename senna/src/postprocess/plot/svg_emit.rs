//! Emit a transparent-background SVG wrapping pre-rasterized per-group
//! PNG layers, convex-hull polygons, and vector text labels.
//!
//! Output structure (three named `<g>` groups — easy to toggle in
//! Illustrator / Inkscape):
//!
//! ```text
//! <svg viewBox="0 0 W H">
//!   <g id="raster-layers">
//!     <image id="topic-0" .../>
//!     ...
//!   </g>
//!   <g id="hulls">          (if enabled)
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

/// Per-topic inputs to the SVG assembler. All coordinates are in pixels,
/// in the same pixel-space as the rasterized layers.
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

/// SVG assembly options.
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

    // Labels — white halo stroke + colored fill for legibility over
    // dense raster regions. Still a single editable `<text>` element.
    if opts.draw_labels {
        let _ = writeln!(&mut s, "  <g id=\"labels\">");
        for (i, layer) in layers.iter().enumerate() {
            let (lx, ly) = layer.label_xy_px;
            if !lx.is_finite() || !ly.is_finite() {
                continue;
            }
            let (r, g, b) = layer.color;
            let esc = escape_xml(&layer.label);
            let _ = writeln!(
                &mut s,
                "    <text id=\"label-{i}\" x=\"{lx:.2}\" y=\"{ly:.2}\" \
                 font-family=\"Helvetica, Arial, sans-serif\" \
                 font-size=\"{fs}\" text-anchor=\"middle\" \
                 dominant-baseline=\"central\" \
                 paint-order=\"stroke\" \
                 stroke=\"white\" stroke-width=\"{hw}\" stroke-linejoin=\"round\" \
                 fill=\"rgb({r},{g},{b})\">{esc}</text>",
                fs = opts.label_font_size_px,
                hw = (opts.label_font_size_px * 0.35).max(1.5),
            );
        }
        let _ = writeln!(&mut s, "  </g>");
    }

    let _ = writeln!(&mut s, "</svg>");
    s
}

fn escape_xml(s: &str) -> String {
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
