//! Faceted Miami-plot SVG assembly + PNG/PDF rendering.
//!
//! One stacked panel per cell type, all sharing the genomic x-axis. Each
//! panel is a mirrored pair around a central gene-model band: epi sites as
//! lollipops rising up, read depth as a filled step-area descending down.
//! We hand-build the SVG string (mirroring `senna .../strand/render.rs`)
//! and render PNG/PDF through the shared `plot_utils` resvg/svg2pdf path.
//!
//! Vector everywhere except the per-site dot layer, which rasterizes once
//! a panel exceeds `raster_threshold` sites (a base64 PNG `<image>` sized
//! to the panel's pixel rect, so it lines up with the vector axes).

use super::bin::{robust_max, BinEdges};
use super::genemodel::{gene_model_svg, GeneModel};
use crate::site_analysis::pileup::fmt_thousands;
use plot_utils::palette::{self, Palette, Rgb};
use plot_utils::rasterize::{rasterize_group_png, Extent, PointShape, RadiusSpec};
use std::fmt::Write as _;
use std::path::Path;

/// Per-cell-type data for one panel.
pub struct PanelData {
    pub celltype: Box<str>,
    /// Raw epi sites `(genomic_pos, signal)` for the top lollipop track.
    pub epi_sites: Vec<(i64, f64)>,
    /// Pre-binned read depth (on the shared [`BinEdges`] grid) for the
    /// bottom track.
    pub depth_bins: Vec<f64>,
}

/// Rendering options resolved from the CLI.
pub struct FigOpts {
    pub out_prefix: Box<str>,
    pub width_in: f32,
    pub dpi: u32,
    pub palette: Palette,
    pub want_svg: bool,
    pub want_png: bool,
    pub want_pdf: bool,
    pub raster_threshold: usize,
    /// Title line (e.g. `BRCA2  chr13:32,000,000-32,100,000`).
    pub title: Box<str>,
    /// Top-track signal label (e.g. `m6A`).
    pub top_label: Box<str>,
}

/// Build + render the faceted Miami figure. Returns the number of files
/// written.
pub fn render_miami(
    panels: &[PanelData],
    models: &[GeneModel],
    edges: &BinEdges,
    opts: &FigOpts,
) -> anyhow::Result<usize> {
    let dpi = opts.dpi as f32;
    let w = (opts.width_in * dpi).round();
    let left = (1.1 * dpi).round();
    let right = (0.3 * dpi).round();
    let top = (0.7 * dpi).round();
    let bottom = (0.5 * dpi).round();
    let plot_w = (w - left - right).max(1.0);

    let top_h = (0.85 * dpi).round();
    let model_h = (0.30 * dpi).round();
    let bottom_h = (0.85 * dpi).round();
    let panel_h = top_h + model_h + bottom_h;
    let panel_gap = (0.35 * dpi).round();
    let n = panels.len().max(1) as f32;
    let h = top + bottom + n * panel_h + (n - 1.0).max(0.0) * panel_gap;

    let title_fs = 0.16 * dpi;
    let label_fs = 0.10 * dpi;
    let tick_fs = 0.075 * dpi;
    let band_h = (model_h * 0.45).max(4.0);
    let binpx = plot_w / edges.num_bins as f32;

    // Global, spike-robust scales so panels are comparable and one outlier
    // doesn't flatten the rest. Separate for the two mirrored tracks.
    let site_max = robust_max(
        panels
            .iter()
            .flat_map(|p| p.epi_sites.iter().map(|&(_, v)| v)),
    );
    let depth_max = robust_max(panels.iter().flat_map(|p| p.depth_bins.iter().copied()));

    // Per-cell-type hue.
    let pal = palette::resolve(&opts.palette, panels.len());

    let mut svg = String::with_capacity(64 * 1024);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" \
         width=\"{w:.0}\" height=\"{h:.0}\" viewBox=\"0 0 {w:.0} {h:.0}\">"
    );
    let _ = write!(
        svg,
        "<rect x=\"0\" y=\"0\" width=\"{w:.0}\" height=\"{h:.0}\" fill=\"white\"/>"
    );

    // Title.
    let _ = write!(
        svg,
        "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"{:.1}\" \
         font-weight=\"bold\" fill=\"#222\">{} \u{2014} {} \u{2191} / depth \u{2193}</text>",
        left,
        top * 0.55,
        title_fs,
        escape(&opts.title),
        escape(&opts.top_label)
    );

    for (pi, panel) in panels.iter().enumerate() {
        let panel_top = top + pi as f32 * (panel_h + panel_gap);
        let epi_baseline = panel_top + top_h;
        let model_mid = epi_baseline + model_h / 2.0;
        let depth_baseline = epi_baseline + model_h;
        let color = palette::color(&pal, pi);
        let hex = rgb_hex(color);

        // Cell-type label + color swatch (left margin, at the model band).
        let label = if panel.celltype.is_empty() {
            "all cells".to_string()
        } else {
            panel.celltype.to_string()
        };
        let _ = write!(
            svg,
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{hex}\"/>",
            left - 0.95 * dpi,
            model_mid - label_fs * 0.5,
            label_fs * 0.7,
            label_fs * 0.7
        );
        let _ = write!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"start\" font-family=\"sans-serif\" \
             font-size=\"{:.1}\" fill=\"#333\">{}</text>",
            left - 0.82 * dpi,
            model_mid + label_fs * 0.35,
            label_fs,
            escape(&label)
        );

        // Baselines.
        for &y in &[epi_baseline, depth_baseline] {
            let _ = write!(
                svg,
                "<line x1=\"{left:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" \
                 stroke=\"#bbb\" stroke-width=\"0.6\"/>",
                left + plot_w
            );
        }

        // ---- Top: epi lollipops (rasterize the dot layer when dense) ----
        if panel.epi_sites.len() > opts.raster_threshold {
            svg.push_str(&rasterized_dots(
                &panel.epi_sites,
                edges,
                left,
                panel_top,
                plot_w,
                top_h,
                site_max,
                color,
            )?);
        } else {
            svg.push_str(&lollipops(
                &panel.epi_sites,
                edges,
                left,
                plot_w,
                epi_baseline,
                top_h,
                site_max,
                &hex,
            ));
        }

        // ---- Middle: gene model band ----
        for g in models {
            svg.push_str(&gene_model_svg(g, edges, left, plot_w, model_mid, band_h));
        }

        // ---- Bottom: read-depth filled step-area (mirrored down) ----
        svg.push_str(&area_path(
            &panel.depth_bins,
            left,
            binpx,
            depth_baseline,
            bottom_h,
            depth_max,
            &hex,
        ));
    }

    // Shared x-axis ticks at the figure bottom.
    svg.push_str(&x_axis(
        edges,
        left,
        plot_w,
        h - bottom + 0.12 * dpi,
        tick_fs,
        dpi,
    ));

    svg.push_str("</svg>");

    // ---- Render ----
    let base = opts.out_prefix.as_ref();
    let mut written = 0usize;
    if opts.want_svg {
        let pth = format!("{base}.miami.svg");
        std::fs::write(&pth, svg.as_bytes())?;
        log::info!("wrote {pth}");
        written += 1;
    }
    let png_task = opts.want_png.then(|| format!("{base}.miami.png"));
    let pdf_task = opts.want_pdf.then(|| format!("{base}.miami.pdf"));
    let (png_res, pdf_res) = rayon::join(
        || match &png_task {
            Some(p) => {
                plot_utils::render_png(&svg, w as u32, h as u32, Path::new(p)).map(|()| Some(p))
            }
            None => Ok(None),
        },
        || match &pdf_task {
            Some(p) => plot_utils::render_pdf(&svg, Path::new(p)).map(|()| Some(p)),
            None => Ok(None),
        },
    );
    if let Some(p) = png_res? {
        log::info!("wrote {p}");
        written += 1;
    }
    if let Some(p) = pdf_res? {
        log::info!("wrote {p}");
        written += 1;
    }
    Ok(written)
}

/// Filled step-area `<path>` mirrored below `baseline` (depth track).
fn area_path(
    bins: &[f64],
    x_left: f32,
    binpx: f32,
    baseline: f32,
    max_h: f32,
    denom: f64,
    color: &str,
) -> String {
    if denom <= 0.0 || bins.iter().all(|&v| v <= 0.0) {
        return String::new();
    }
    let mut d = String::with_capacity(bins.len() * 24);
    let _ = write!(d, "M {:.2} {:.2} ", x_left, baseline);
    for (b, &v) in bins.iter().enumerate() {
        let frac = (v / denom).clamp(0.0, 1.0) as f32;
        let hpx = frac * max_h;
        let x0 = x_left + b as f32 * binpx;
        let x1 = x0 + binpx;
        let y = baseline + hpx;
        let _ = write!(d, "L {:.2} {:.2} L {:.2} {:.2} ", x0, y, x1, y);
    }
    let xr = x_left + bins.len() as f32 * binpx;
    let _ = write!(d, "L {:.2} {:.2} Z", xr, baseline);
    format!("<path d=\"{d}\" fill=\"{color}\" fill-opacity=\"0.85\" stroke=\"none\"/>")
}

/// Vector lollipops (stem + dot) at site positions, rising above
/// `baseline`. Dot height = `scale(val)/scale(denom) * top_h`.
#[allow(clippy::too_many_arguments)]
fn lollipops(
    sites: &[(i64, f64)],
    edges: &BinEdges,
    x_left: f32,
    plot_w: f32,
    baseline: f32,
    top_h: f32,
    denom: f64,
    color: &str,
) -> String {
    if denom <= 0.0 {
        return String::new();
    }
    let mut s = String::with_capacity(sites.len() * 48);
    for &(pos, val) in sites {
        if val <= 0.0 {
            continue;
        }
        let x = edges.x_px(pos, x_left, plot_w);
        let frac = (val / denom).clamp(0.0, 1.0) as f32;
        let y = baseline - frac * top_h;
        let _ = write!(
            s,
            "<line x1=\"{x:.1}\" y1=\"{baseline:.1}\" x2=\"{x:.1}\" y2=\"{y:.1}\" \
             stroke=\"{color}\" stroke-width=\"0.8\" stroke-opacity=\"0.6\"/>\
             <circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"2.0\" fill=\"{color}\"/>"
        );
    }
    s
}

/// Rasterized dot layer for dense panels: one transparent PNG sized to the
/// top track's pixel rect, embedded as a base64 `<image>`.
#[allow(clippy::too_many_arguments)]
fn rasterized_dots(
    sites: &[(i64, f64)],
    edges: &BinEdges,
    x_left: f32,
    panel_top: f32,
    plot_w: f32,
    top_h: f32,
    denom: f64,
    color: Rgb,
) -> anyhow::Result<String> {
    let iw = plot_w.round().max(1.0) as u32;
    let ih = top_h.round().max(1.0) as u32;
    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(sites.len());
    if denom > 0.0 {
        for &(pos, val) in sites {
            if val <= 0.0 {
                continue;
            }
            // Image-local pixels: x across [0, plot_w], y=0 at top so the
            // dot rises (smaller y = taller).
            let x = edges.x_px(pos, 0.0, plot_w);
            let frac = (val / denom).clamp(0.0, 1.0) as f32;
            let y = top_h - frac * top_h;
            pts.push((x, y));
        }
    }
    let png = rasterize_group_png(
        &pts,
        Extent { w: iw, h: ih },
        RadiusSpec::Scalar(1.6),
        color,
        0.85,
        PointShape::Circle,
    )?;
    Ok(format!(
        "<image x=\"{x_left:.1}\" y=\"{panel_top:.1}\" width=\"{plot_w:.1}\" height=\"{top_h:.1}\" \
         xlink:href=\"data:image/png;base64,{}\"/>",
        b64(&png)
    ))
}

/// Shared bottom x-axis with ~6 ticks labeled in genomic bp.
fn x_axis(edges: &BinEdges, x_left: f32, plot_w: f32, y: f32, fs: f32, dpi: f32) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "<line x1=\"{x_left:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" \
         stroke=\"#444\" stroke-width=\"0.8\"/>",
        x_left + plot_w
    );
    let n_ticks = 6usize;
    for i in 0..=n_ticks {
        let frac = i as f32 / n_ticks as f32;
        let x = x_left + frac * plot_w;
        let pos = edges.min_pos + ((edges.max_pos - edges.min_pos) as f64 * frac as f64) as i64;
        let _ = write!(
            s,
            "<line x1=\"{x:.1}\" y1=\"{y:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" \
             stroke=\"#444\" stroke-width=\"0.8\"/>",
            y + 0.05 * dpi
        );
        let _ = write!(
            s,
            "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" \
             font-size=\"{fs:.1}\" fill=\"#444\">{}</text>",
            y + 0.05 * dpi + fs,
            escape(&fmt_thousands(pos))
        );
    }
    s
}

fn rgb_hex(c: Rgb) -> String {
    format!("#{:02x}{:02x}{:02x}", c.0, c.1, c.2)
}

fn escape(s: &str) -> String {
    plot_utils::svg_emit::escape_xml(s)
}

/// Minimal standard base64 (no padding omitted) for embedding PNG bytes.
fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[(n >> 18 & 63) as usize] as char);
        out.push(T[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(b64(b""), "");
        assert_eq!(b64(b"f"), "Zg==");
        assert_eq!(b64(b"fo"), "Zm8=");
        assert_eq!(b64(b"foo"), "Zm9v");
        assert_eq!(b64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn area_path_empty_when_flat() {
        assert!(area_path(&[0.0, 0.0], 0.0, 1.0, 10.0, 5.0, 1.0, "#abc").is_empty());
        let p = area_path(&[0.0, 1.0], 0.0, 1.0, 10.0, 5.0, 1.0, "#abc");
        assert!(p.contains("<path"));
    }

    #[test]
    fn rgb_hex_formats() {
        assert_eq!(rgb_hex((255, 0, 16)), "#ff0010");
    }
}
