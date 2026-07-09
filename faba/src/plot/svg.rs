//! Vector overlay for `faba plot` — the text, star, legend and colourbar spliced
//! on top of the rasterized layers. Geometry only; colours and sizes come from
//! [`super::style`].

use plot_utils::palette::{self, Rgb};
use plot_utils::svg_emit::escape_xml;
use std::fmt::Write as _;

use super::style::ROOT_RED;

/// One legend row: display name + swatch colour, in draw order.
pub(super) struct LegendEntry {
    pub(super) label: String,
    pub(super) color: Rgb,
}

/// A white-haloed, centred `<text>` label (matches `plot-utils` label style).
pub(super) fn emit_halo_text(s: &mut String, x: f32, y: f32, font_px: f32, color: Rgb, text: &str) {
    if !x.is_finite() || !y.is_finite() || text.is_empty() {
        return;
    }
    let (r, g, b) = color;
    let hw = (font_px * 0.35).max(1.2);
    let esc = escape_xml(text);
    let _ = writeln!(
        s,
        "    <text x=\"{x:.2}\" y=\"{y:.2}\" \
         font-family=\"Helvetica, Arial, sans-serif\" font-size=\"{font_px:.2}\" \
         text-anchor=\"middle\" dominant-baseline=\"central\" paint-order=\"stroke\" \
         stroke=\"white\" stroke-width=\"{hw:.2}\" stroke-linejoin=\"round\" \
         fill=\"rgb({r},{g},{b})\">{esc}</text>"
    );
}

/// A red 5-point star centred at `(cx, cy)` with outer radius `r_out`.
pub(super) fn emit_star(s: &mut String, cx: f32, cy: f32, r_out: f32) {
    if !cx.is_finite() || !cy.is_finite() {
        return;
    }
    let r_in = r_out * 0.4;
    let mut pts = String::new();
    for k in 0..10 {
        let ang = -std::f32::consts::FRAC_PI_2 + k as f32 * std::f32::consts::PI / 5.0;
        let rr = if k % 2 == 0 { r_out } else { r_in };
        let _ = write!(
            pts,
            "{:.2},{:.2} ",
            cx + rr * ang.cos(),
            cy + rr * ang.sin()
        );
    }
    let (r, g, b) = ROOT_RED;
    let _ = writeln!(
        s,
        "    <polygon points=\"{pts}\" fill=\"rgb({r},{g},{b})\" \
         stroke=\"white\" stroke-width=\"{sw:.2}\" stroke-linejoin=\"round\"/>",
        sw = (r_out * 0.25).max(1.0)
    );
}

/// A top-left legend box: one swatch + type name per row.
pub(super) fn emit_legend(entries: &[LegendEntry], font_px: f32) -> String {
    if entries.is_empty() {
        return String::new();
    }
    let pad = font_px * 0.6;
    let sw = font_px; // swatch side
    let row_h = font_px * 1.5;
    let x0 = pad;
    let y0 = pad;
    let max_chars = entries
        .iter()
        .map(|e| e.label.chars().count())
        .max()
        .unwrap_or(1);
    let box_w = sw + pad + max_chars as f32 * font_px * 0.62 + pad;
    let box_h = entries.len() as f32 * row_h + pad;

    let mut s = String::from("  <g id=\"legend\">\n");
    let _ = writeln!(
        s,
        "    <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" rx=\"{rx:.2}\" \
         fill=\"white\" fill-opacity=\"0.72\" stroke=\"rgb(120,120,120)\" stroke-width=\"1\"/>",
        x = x0 - pad * 0.5,
        y = y0 - pad * 0.5,
        w = box_w,
        h = box_h,
        rx = font_px * 0.3,
    );
    for (i, e) in entries.iter().enumerate() {
        let ry = y0 + i as f32 * row_h;
        let (r, g, b) = e.color;
        let _ = writeln!(
            s,
            "    <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{sw:.2}\" height=\"{sw:.2}\" \
             fill=\"rgb({r},{g},{b})\" stroke=\"rgb(80,80,80)\" stroke-width=\"0.5\"/>",
            x = x0,
            y = ry,
        );
        let esc = escape_xml(&e.label);
        let _ = writeln!(
            s,
            "    <text x=\"{tx:.2}\" y=\"{ty:.2}\" \
             font-family=\"Helvetica, Arial, sans-serif\" font-size=\"{font_px:.2}\" \
             text-anchor=\"start\" dominant-baseline=\"central\" fill=\"rgb(20,20,20)\">{esc}</text>",
            tx = x0 + sw + pad * 0.5,
            ty = ry + sw * 0.5,
        );
    }
    s.push_str("  </g>\n");
    s
}

/// A vertical blue→red colourbar (sampled in stacked strips) on the right,
/// annotated with the pseudotime min/max.
pub(super) fn emit_colourbar(
    width_px: u32,
    height_px: u32,
    font_px: f32,
    lo: f32,
    hi: f32,
) -> String {
    let w = width_px as f32;
    let h = height_px as f32;
    let bar_w = font_px * 1.2;
    let bar_h = (h * 0.32).max(font_px * 6.0);
    let x0 = w - bar_w - font_px * 4.5;
    let y0 = font_px * 1.5;
    let n_strip = 48usize;
    let strip_h = bar_h / n_strip as f32;

    let mut s = String::from("  <g id=\"colourbar\">\n");
    for k in 0..n_strip {
        // Top of the bar = high pseudotime (red), bottom = low (blue).
        let t = 1.0 - k as f32 / (n_strip - 1).max(1) as f32;
        let (r, g, b) = palette::sample_blue_red(t);
        let _ = writeln!(
            s,
            "    <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{bw:.2}\" height=\"{sh:.3}\" \
             fill=\"rgb({r},{g},{b})\"/>",
            x = x0,
            y = y0 + k as f32 * strip_h,
            bw = bar_w,
            sh = strip_h + 0.5,
        );
    }
    let _ = writeln!(
        s,
        "    <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{bw:.2}\" height=\"{bh:.2}\" \
         fill=\"none\" stroke=\"rgb(80,80,80)\" stroke-width=\"1\"/>",
        x = x0,
        y = y0,
        bw = bar_w,
        bh = bar_h,
    );
    let tx = x0 + bar_w + font_px * 0.4;
    let _ = writeln!(
        s,
        "    <text x=\"{tx:.2}\" y=\"{y:.2}\" font-family=\"Helvetica, Arial, sans-serif\" \
         font-size=\"{font_px:.2}\" text-anchor=\"start\" dominant-baseline=\"central\" \
         fill=\"rgb(20,20,20)\">{hi:.2}</text>",
        y = y0,
    );
    let _ = writeln!(
        s,
        "    <text x=\"{tx:.2}\" y=\"{y:.2}\" font-family=\"Helvetica, Arial, sans-serif\" \
         font-size=\"{font_px:.2}\" text-anchor=\"start\" dominant-baseline=\"central\" \
         fill=\"rgb(20,20,20)\">{lo:.2}</text>",
        y = y0 + bar_h,
    );
    let _ = writeln!(
        s,
        "    <text x=\"{tx:.2}\" y=\"{y:.2}\" font-family=\"Helvetica, Arial, sans-serif\" \
         font-size=\"{fs:.2}\" text-anchor=\"middle\" dominant-baseline=\"central\" \
         transform=\"rotate(90 {tx:.2} {y:.2})\" fill=\"rgb(20,20,20)\">pseudotime</text>",
        tx = tx + font_px * 1.6,
        y = y0 + bar_h * 0.5,
        fs = font_px,
    );
    s.push_str("  </g>\n");
    s
}

/////////////////////
// Parquet helpers //
/////////////////////
