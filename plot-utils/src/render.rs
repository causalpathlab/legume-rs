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
