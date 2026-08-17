//! SVG → PNG / PDF rendering helpers (usvg + resvg + svg2pdf).
//!
//! Lifted from `senna/src/postprocess/fit_plot.rs` so both senna plot
//! and pinto plot share the exact same parse/render path (including
//! system-font loading, without which resvg silently skips vector text
//! labels).

use std::fs;
use std::path::Path;

/// Render the SVG to a flattened PNG via usvg + resvg. Loads system
/// fonts so vector `<text>` labels get rasterized (resvg's default
/// options ship an empty font database).
pub fn render_png(svg: &str, w: u32, h: u32, out: &Path) -> anyhow::Result<()> {
    let mut options = usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = usvg::Tree::from_str(svg, &options)
        .map_err(|e| anyhow::anyhow!("usvg parse failed: {e}"))?;
    let mut pixmap = tiny_skia::Pixmap::new(w, h)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc failed ({w}x{h})"))?;
    resvg::render(
        &tree,
        tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );
    pixmap
        .save_png(out)
        .map_err(|e| anyhow::anyhow!("PNG save failed: {e}"))?;
    Ok(())
}

/// Render the SVG to a true-vector PDF via svg2pdf.
pub fn render_pdf(svg: &str, out: &Path) -> anyhow::Result<()> {
    let mut options = svg2pdf::usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = svg2pdf::usvg::Tree::from_str(svg, &options)
        .map_err(|e| anyhow::anyhow!("svg2pdf/usvg parse failed: {e}"))?;
    let pdf = svg2pdf::to_pdf(
        &tree,
        svg2pdf::ConversionOptions::default(),
        svg2pdf::PageOptions::default(),
    )
    .map_err(|e| anyhow::anyhow!("svg2pdf render failed: {e}"))?;
    fs::write(out, &pdf)?;
    Ok(())
}

/// Which files a figure should be written out as.
#[derive(Copy, Clone, Debug)]
pub struct FigureFormats {
    pub svg: bool,
    pub png: bool,
    pub pdf: bool,
}

/// Write one SVG string out as `{base}.svg` / `.png` / `.pdf`, logging each.
///
/// Every plot command reaches the same place — one finished SVG and a choice of
/// containers — and each had grown its own copy of this: the same `rayon::join`,
/// the same PDF-by-default-PNG-opt-in policy, the same comment explaining it.
/// Four copies of a decision that is not per-figure.
///
/// PNG and PDF share the SVG and are independent, so they render concurrently:
/// both pay a usvg parse of the same string, and that parse is the bulk of the
/// cost on a large scatter.
///
/// Returns how many files were written.
pub fn write_figure(
    svg: &str,
    width_px: u32,
    height_px: u32,
    base: &str,
    want: FigureFormats,
) -> anyhow::Result<usize> {
    let mut written = 0usize;
    if want.svg {
        let path = format!("{base}.svg");
        fs::write(&path, svg.as_bytes()).map_err(|e| anyhow::anyhow!("writing {path}: {e}"))?;
        log::info!("Wrote {path}");
        written += 1;
    }

    let png_task = want.png.then(|| format!("{base}.png"));
    let pdf_task = want.pdf.then(|| format!("{base}.pdf"));
    let (png_res, pdf_res) = rayon::join(
        || match &png_task {
            Some(p) => render_png(svg, width_px, height_px, Path::new(p)).map(|()| Some(p.clone())),
            None => Ok(None),
        },
        || match &pdf_task {
            Some(p) => render_pdf(svg, Path::new(p)).map(|()| Some(p.clone())),
            None => Ok(None),
        },
    );
    if let Some(p) = png_res? {
        log::info!("Wrote {p} ({width_px}x{height_px})");
        written += 1;
    }
    if let Some(p) = pdf_res? {
        log::info!("Wrote {p}");
        written += 1;
    }
    Ok(written)
}
