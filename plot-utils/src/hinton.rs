//! Hinton-diagram primitive: rows × cols grid of squares whose area
//! encodes a magnitude.
//!
//! Used for "marker gene × community" summaries (pinto plot) and
//! similar (rows × cols) overviews where readers want a square-area
//! visual instead of a heatmap. Reuses [`crate::order::diagonalize_order`]
//! for the canonical block-diagonal layout.
//!
//! The renderer is layout-only (no I/O): it returns an SVG string so
//! callers can pipe to [`crate::render_pdf`] / [`crate::render_png`] or
//! write SVG directly. Width/height scale with `(ncols, nrows)` so each
//! cell stays square at any aspect.

use crate::palette::Rgb;
use crate::svg_emit::escape_xml;
use std::fmt::Write;

/// How to map raw magnitudes to box side-length (as a fraction of the
/// cell tile, in `[0, 1]`).
#[derive(Copy, Clone, Debug, Default)]
pub enum HintonScale {
    /// `frac = sqrt(v / max)`. Default — gentle compression, no offset.
    #[default]
    Sqrt,
    /// `frac = log1p(v) / log1p(max)`. Heavier compression for
    /// long-tailed distributions.
    Log1p,
    /// `frac = v / max`. Linear; box area is then ∝ `v²`.
    Linear,
}

impl HintonScale {
    fn frac(self, v: f32, max: f32) -> f32 {
        if !v.is_finite() || v <= 0.0 || max <= 0.0 {
            return 0.0;
        }
        let f = match self {
            HintonScale::Sqrt => (v / max).sqrt(),
            HintonScale::Log1p => v.ln_1p() / max.ln_1p().max(f32::EPSILON),
            HintonScale::Linear => v / max,
        };
        f.clamp(0.0, 1.0)
    }

    fn legend_label(self) -> &'static str {
        match self {
            HintonScale::Sqrt => "size ∝ √(mean)",
            HintonScale::Log1p => "size ∝ log1p(mean)",
            HintonScale::Linear => "size ∝ mean",
        }
    }
}

/// Render-time options. Labels and colors are passed in *original*
/// order; pass `row_order` / `col_order` to re-arrange without copying
/// the matrix.
pub struct HintonOpts<'a> {
    /// Row labels (e.g. gene names), length = `nrows`. `None` skips the row legend gutter.
    pub row_labels: Option<&'a [Box<str>]>,
    /// Column labels (e.g. community ids), length = `ncols`. `None` skips the column header.
    pub col_labels: Option<&'a [Box<str>]>,
    /// Display-order permutation of rows. `None` ⇒ identity.
    pub row_order: Option<&'a [usize]>,
    /// Display-order permutation of columns. `None` ⇒ identity.
    pub col_order: Option<&'a [usize]>,
    /// Optional per-column fill color, in original (pre-permutation) order.
    /// `None` ⇒ all boxes drawn in mid-gray.
    pub col_colors: Option<&'a [Rgb]>,
    /// Optional per-cell fill color (length = `nrows * ncols`, row-major,
    /// pre-permutation order). When `Some`, takes precedence over
    /// `col_colors` for each cell. Use this for "color = which group
    /// dominates this (row, column)" heatmaps.
    pub cell_colors: Option<&'a [Rgb]>,
    /// Magnitude → side-length mapping.
    pub scale: HintonScale,
    /// Cell tile size (px). Box side = `cell_px * frac`. Default 18.
    pub cell_px: f32,
    /// Vector text size (px). Default 11.
    pub font_px: f32,
    /// Optional title rendered above the grid.
    pub title: Option<&'a str>,
    /// Stroke width (px) for the cell-boundary grid. `0.0` disables the
    /// grid (default). Small positive values (≈ 0.3–0.5) give a faint
    /// guideline that helps the eye line up rows and columns.
    pub grid_stroke_px: f32,
    /// Grid stroke color, in (r, g, b) bytes. Only used when
    /// `grid_stroke_px > 0`.
    pub grid_color: Rgb,
    /// Optional categorical color legend: list of `(label, color)`
    /// entries rendered below the magnitude legend, one swatch per
    /// entry. Use this when `cell_colors` encodes a categorical
    /// variable (e.g. "color of each box = major community"). `None`
    /// (default) skips the color legend.
    pub color_legend: Option<&'a [(Box<str>, Rgb)]>,
}

impl<'a> Default for HintonOpts<'a> {
    fn default() -> Self {
        HintonOpts {
            row_labels: None,
            col_labels: None,
            row_order: None,
            col_order: None,
            col_colors: None,
            cell_colors: None,
            scale: HintonScale::default(),
            cell_px: 18.0,
            font_px: 11.0,
            title: None,
            grid_stroke_px: 0.0,
            grid_color: (220, 220, 220),
            color_legend: None,
        }
    }
}

/// Total SVG size in pixels.
pub struct HintonSize {
    pub width_px: u32,
    pub height_px: u32,
}

/// Compute the SVG canvas size for a Hinton diagram. Useful so callers
/// can size the rasterizer (PNG) without parsing the SVG.
#[must_use]
pub fn hinton_size(nrows: usize, ncols: usize, opts: &HintonOpts<'_>) -> HintonSize {
    let pad_left = label_left_px(opts);
    let pad_top = label_top_px(opts);
    let grid_w = ncols as f32 * opts.cell_px;
    let grid_h = nrows as f32 * opts.cell_px;
    let legend_w = LEGEND_WIDTH_FACTOR * opts.cell_px;
    let legend_h = legend_total_height(opts);
    let content_h = grid_h.max(legend_h);
    HintonSize {
        width_px: (pad_left + grid_w + legend_w).ceil() as u32,
        height_px: (pad_top + content_h + 8.0).ceil() as u32,
    }
}

fn legend_total_height(opts: &HintonOpts<'_>) -> f32 {
    let cell = opts.cell_px;
    // Magnitude block: header + 4 swatches.
    let mag_h = opts.font_px * 1.6 + 4.0 * (cell + 3.0);
    let color_h = match opts.color_legend {
        Some(entries) if !entries.is_empty() => {
            // Header gap + entries (swatch ~ 0.7 * cell, line spacing).
            opts.font_px * 1.6 + entries.len() as f32 * (cell * 0.7 + 4.0)
        }
        _ => 0.0,
    };
    mag_h + color_h + 8.0
}

/// Render a Hinton-style square-grid summary as a standalone SVG document.
///
/// `mat` is row-major, length = `nrows * ncols`.
#[must_use]
pub fn render_hinton(mat: &[f32], nrows: usize, ncols: usize, opts: &HintonOpts<'_>) -> String {
    debug_assert_eq!(mat.len(), nrows * ncols);
    let size = hinton_size(nrows, ncols, opts);
    let pad_left = label_left_px(opts);
    let pad_top = label_top_px(opts);
    let cell = opts.cell_px;
    let inset = (cell * CELL_INSET_FRAC).max(0.5);

    let row_idx =
        |display_pos: usize| -> usize { opts.row_order.map_or(display_pos, |o| o[display_pos]) };
    let col_idx =
        |display_pos: usize| -> usize { opts.col_order.map_or(display_pos, |o| o[display_pos]) };

    let max_v = mat
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max);

    let mut s = String::with_capacity(8192 + nrows * ncols * 96);
    let _ = write!(
        s,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" \
              width=\"{w}\" height=\"{h}\">\n",
        w = size.width_px,
        h = size.height_px,
    );

    if let Some(title) = opts.title {
        let _ = writeln!(
            &mut s,
            "  <text x=\"{x:.1}\" y=\"{y:.1}\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"{fs:.1}\" font-weight=\"bold\">{t}</text>",
            x = pad_left,
            y = opts.font_px * 1.2,
            fs = opts.font_px * 1.1,
            t = escape_xml(title),
        );
    }

    if opts.grid_stroke_px > 0.0 {
        let (gr, gg, gb) = opts.grid_color;
        let grid_w = ncols as f32 * cell;
        let grid_h = nrows as f32 * cell;
        let _ = writeln!(
            &mut s,
            "  <g id=\"grid\" stroke=\"rgb({gr},{gg},{gb})\" stroke-width=\"{sw:.2}\" \
             shape-rendering=\"crispEdges\">",
            sw = opts.grid_stroke_px,
        );
        for cc in 0..=ncols {
            let x = pad_left + cc as f32 * cell;
            let _ = writeln!(
                &mut s,
                "    <line x1=\"{x:.2}\" y1=\"{y0:.2}\" x2=\"{x:.2}\" y2=\"{y1:.2}\"/>",
                y0 = pad_top,
                y1 = pad_top + grid_h,
            );
        }
        for rr in 0..=nrows {
            let y = pad_top + rr as f32 * cell;
            let _ = writeln!(
                &mut s,
                "    <line x1=\"{x0:.2}\" y1=\"{y:.2}\" x2=\"{x1:.2}\" y2=\"{y:.2}\"/>",
                x0 = pad_left,
                x1 = pad_left + grid_w,
            );
        }
        let _ = writeln!(&mut s, "  </g>");
    }

    let _ = writeln!(&mut s, "  <g id=\"cells\">");
    for rr in 0..nrows {
        let r = row_idx(rr);
        for cc in 0..ncols {
            let c = col_idx(cc);
            let v = mat[r * ncols + c];
            let frac = opts.scale.frac(v, max_v);
            let side = (cell - 2.0 * inset) * frac;
            if side <= 0.0 {
                continue;
            }
            let cx = pad_left + (cc as f32 + 0.5) * cell;
            let cy = pad_top + (rr as f32 + 0.5) * cell;
            let (rgb_r, rgb_g, rgb_b) = opts
                .cell_colors
                .and_then(|cs| cs.get(r * ncols + c).copied())
                .or_else(|| opts.col_colors.and_then(|cs| cs.get(c).copied()))
                .unwrap_or((90, 90, 90));
            let _ = writeln!(
                &mut s,
                "    <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{side:.2}\" height=\"{side:.2}\" \
                 fill=\"rgb({rgb_r},{rgb_g},{rgb_b})\"/>",
                x = cx - side * 0.5,
                y = cy - side * 0.5,
            );
        }
    }
    let _ = writeln!(&mut s, "  </g>");

    if let Some(labels) = opts.row_labels {
        let _ = writeln!(
            &mut s,
            "  <g id=\"row-labels\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"{fs:.1}\" text-anchor=\"end\">",
            fs = opts.font_px,
        );
        for rr in 0..nrows {
            let r = row_idx(rr);
            let label = labels.get(r).map_or("", AsRef::as_ref);
            let y = pad_top + (rr as f32 + 0.5) * cell + opts.font_px * 0.35;
            let _ = writeln!(
                &mut s,
                "    <text x=\"{x:.1}\" y=\"{y:.1}\">{t}</text>",
                x = pad_left - 4.0,
                t = escape_xml(label),
            );
        }
        let _ = writeln!(&mut s, "  </g>");
    }

    if let Some(labels) = opts.col_labels {
        let _ = writeln!(
            &mut s,
            "  <g id=\"col-labels\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"{fs:.1}\">",
            fs = opts.font_px,
        );
        for cc in 0..ncols {
            let c = col_idx(cc);
            let label = labels.get(c).map_or("", AsRef::as_ref);
            let x = pad_left + (cc as f32 + 0.5) * cell;
            let y = pad_top - 4.0;
            let _ = writeln!(
                &mut s,
                "    <text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"start\" \
                 transform=\"rotate(-45 {x:.1} {y:.1})\">{t}</text>",
                t = escape_xml(label),
            );
        }
        let _ = writeln!(&mut s, "  </g>");
    }

    write_legend(&mut s, &size, pad_left, pad_top, ncols, opts);

    let _ = writeln!(&mut s, "</svg>");
    s
}

const CELL_INSET_FRAC: f32 = 0.06;
const LEGEND_WIDTH_FACTOR: f32 = 6.0;

fn label_left_px(opts: &HintonOpts<'_>) -> f32 {
    match opts.row_labels {
        Some(_) => opts.font_px * 9.0,
        None => 4.0,
    }
}

fn label_top_px(opts: &HintonOpts<'_>) -> f32 {
    let title = if opts.title.is_some() {
        opts.font_px * 1.6
    } else {
        0.0
    };
    let cols = match opts.col_labels {
        Some(_) => opts.font_px * 3.5,
        None => 4.0,
    };
    title + cols
}

fn write_legend(
    s: &mut String,
    size: &HintonSize,
    pad_left: f32,
    pad_top: f32,
    ncols: usize,
    opts: &HintonOpts<'_>,
) {
    let cell = opts.cell_px;
    let inset = (cell * CELL_INSET_FRAC).max(0.5);
    let lx = pad_left + ncols as f32 * cell + 14.0;
    if lx + cell > size.width_px as f32 {
        return;
    }
    let _ = writeln!(
        s,
        "  <g id=\"legend\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"{fs:.1}\">",
        fs = opts.font_px,
    );
    let _ = writeln!(
        s,
        "    <text x=\"{x:.1}\" y=\"{y:.1}\" font-weight=\"bold\">{t}</text>",
        x = lx,
        y = pad_top + opts.font_px,
        t = opts.scale.legend_label(),
    );
    let mut ly = pad_top + opts.font_px * 1.6;
    for &frac_label in &[0.25_f32, 0.5, 0.75, 1.0] {
        // Legend boxes are sized as if the *raw* value already encoded
        // the fraction (so a "50%" swatch is half-side regardless of scale).
        let side = (cell - 2.0 * inset) * frac_label;
        let cx = lx + cell * 0.5;
        let cy = ly + cell * 0.5;
        let _ = writeln!(
            s,
            "    <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{side:.2}\" height=\"{side:.2}\" \
             fill=\"rgb(90,90,90)\"/>",
            x = cx - side * 0.5,
            y = cy - side * 0.5,
        );
        let _ = writeln!(
            s,
            "    <text x=\"{x:.1}\" y=\"{y:.1}\">{p:.0}%</text>",
            x = lx + cell + 4.0,
            y = ly + cell * 0.5 + opts.font_px * 0.35,
            p = frac_label * 100.0,
        );
        ly += cell + 3.0;
    }

    if let Some(entries) = opts.color_legend {
        if !entries.is_empty() {
            ly += opts.font_px * 0.6;
            let _ = writeln!(
                s,
                "    <text x=\"{x:.1}\" y=\"{y:.1}\" font-weight=\"bold\">color</text>",
                x = lx,
                y = ly + opts.font_px,
            );
            ly += opts.font_px * 1.6;
            let swatch = cell * 0.7;
            for (label, (rgb_r, rgb_g, rgb_b)) in entries {
                let _ = writeln!(
                    s,
                    "    <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{sw:.2}\" height=\"{sw:.2}\" \
                     fill=\"rgb({rgb_r},{rgb_g},{rgb_b})\"/>",
                    x = lx,
                    y = ly,
                    sw = swatch,
                );
                let _ = writeln!(
                    s,
                    "    <text x=\"{x:.1}\" y=\"{y:.1}\">{t}</text>",
                    x = lx + swatch + 4.0,
                    y = ly + swatch * 0.5 + opts.font_px * 0.35,
                    t = escape_xml(label),
                );
                ly += swatch + 4.0;
            }
        }
    }

    let _ = writeln!(s, "  </g>");
}
