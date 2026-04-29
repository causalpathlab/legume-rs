//! Per-cell stacked-bar ("structure plot") rasterizer.
//!
//! Rasterizes a single panel: cells along x, K topic proportions stacked
//! along y. One filled rect per (cell, topic) cell. Returns a transparent
//! PNG, mirroring the contract of [`crate::rasterize::rasterize_group_png`]
//! so callers can embed multiple panels as `<image>` layers in an SVG.
//!
//! `props` is row-major `[N × K]` and rows must already be in display
//! order (callers sort by argmax topic + dominant prob, by 1-D embedding,
//! etc., before calling this). Each row is renormalized to sum to 1
//! defensively — input log-prob → exp may not sum exactly to 1 because
//! of log-sum-exp rounding.

use crate::palette::Rgb;
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Transform};

/// Rasterize a structure-plot panel.
///
/// - `props`: row-major `[n_cells × n_topics]` non-negative weights;
///   each row is renormalized to sum to 1 (rows that sum to 0 are skipped).
/// - `width_px`, `height_px`: panel pixel size.
/// - `topic_colors`: indexed by *column position* `0..n_topics` in the
///   `props` matrix (not by topic id) — pass colors permuted to match
///   any column reorder you've already applied to `props`.
///
/// Returns transparent PNG bytes.
pub fn structure_bar_png(
    props: &[f32],
    n_cells: usize,
    n_topics: usize,
    width_px: u32,
    height_px: u32,
    topic_colors: &[Rgb],
) -> anyhow::Result<Vec<u8>> {
    if props.len() != n_cells * n_topics {
        anyhow::bail!(
            "structure_bar: props len {} != n_cells {} * n_topics {}",
            props.len(),
            n_cells,
            n_topics,
        );
    }
    if topic_colors.len() < n_topics {
        anyhow::bail!(
            "structure_bar: topic_colors len {} < n_topics {}",
            topic_colors.len(),
            n_topics,
        );
    }
    let mut pixmap = Pixmap::new(width_px, height_px)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc failed ({width_px}x{height_px})"))?;
    if n_cells == 0 || n_topics == 0 {
        return pixmap
            .encode_png()
            .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"));
    }

    let w = width_px as f32;
    let h = height_px as f32;
    let n = n_cells as f32;

    // Per-topic Paint, allocated once.
    let mut paints: Vec<Paint> = Vec::with_capacity(n_topics);
    for &(r, g, b) in &topic_colors[..n_topics] {
        let mut p = Paint::default();
        p.set_color_rgba8(r, g, b, 255);
        p.anti_alias = false;
        paints.push(p);
    }

    // Accumulate all rectangles per topic into a single PathBuilder, then
    // fill once per topic. With N=20k cells × K=10 topics this collapses
    // ~200k tiny path allocations into K. Stacked rects within one
    // topic share the same fill, so a single Path covers them all.
    let mut paths: Vec<PathBuilder> = (0..n_topics).map(|_| PathBuilder::new()).collect();

    for i in 0..n_cells {
        // Snap column edges to integer pixels so adjacent cells share an
        // edge with no antialias seam.
        let x_left = ((i as f32) / n * w).round();
        let x_right = (((i + 1) as f32) / n * w).round();
        let cell_w = (x_right - x_left).max(0.0);
        if cell_w <= 0.0 {
            continue;
        }

        let row_off = i * n_topics;
        let row = &props[row_off..row_off + n_topics];
        let mut sum = 0.0f32;
        for &v in row {
            if v.is_finite() && v > 0.0 {
                sum += v;
            }
        }
        if sum <= 0.0 {
            continue;
        }

        let mut y_top = 0.0f32;
        for (k, &v) in row.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                continue;
            }
            let frac = v / sum;
            let y_bot = (y_top + frac * h).min(h);
            let cell_h = (y_bot - y_top).max(0.0);
            if cell_h > 0.0 {
                if let Some(rect) = tiny_skia::Rect::from_xywh(x_left, y_top, cell_w, cell_h) {
                    paths[k].push_rect(rect);
                }
            }
            y_top = y_bot;
        }
    }

    let identity = Transform::identity();
    for (k, pb) in paths.into_iter().enumerate() {
        if let Some(path) = pb.finish() {
            pixmap.fill_path(&path, &paints[k], FillRule::Winding, identity, None);
        }
    }

    pixmap
        .encode_png()
        .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"))
}
