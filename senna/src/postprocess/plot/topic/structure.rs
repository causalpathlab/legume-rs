//! Structure-bar (admixture) plot rendering: one panel per group, with
//! a shared topic legend on the right and a horizontal gap between
//! adjacent panels.

use super::labels::cells_by_batch;
use super::order::{global_topic_order, order_by_argmax, order_by_coord};
use super::output::{emit_outputs, sanitize};
use super::{CellOrder, PlotTopicArgs, ResolvedInputs};
use crate::embed_common::*;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use plot_utils::palette::Rgb;
use rayon::prelude::*;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// Layout knobs for a structure-plot SVG (single-panel or combined).
/// Pixel-level dimensions are derived from `args.{width,height,dpi}` once
/// per call so the per-cell pixel density is identical between the
/// standalone per-group panels and the combined `all.*` plot.
struct StructDims {
    bar_h: u32,
    label_band: u32,
    legend_band: u32,
    /// Horizontal gap between adjacent panels in the combined view.
    panel_gap: u32,
}

struct PanelOut {
    batch: Box<str>,
    n_cells: usize,
    width_px: u32,
    png: Vec<u8>,
}

const PANEL_LABEL_FONT_FRAC: f32 = 0.55;

#[allow(clippy::too_many_arguments)]
pub(super) fn render_structure_plots(
    probs: &[f32],
    n_topics: usize,
    topic_ids: &[i64],
    topic_colors: &[Rgb],
    group_labels: &[Box<str>],
    args: &PlotTopicArgs,
    resolved: &ResolvedInputs,
    plot_root: &str,
) -> anyhow::Result<()> {
    let groups = cells_by_batch(group_labels);
    if groups.is_empty() {
        info!("No cells to plot for structure plot");
        return Ok(());
    }

    let struct_dir = format!("{plot_root}/struct");
    let by_group_dir = format!("{struct_dir}/by_{}", args.group_by.subdir_suffix());
    // Clear per-group outputs so renaming (e.g., index labels → basenames,
    // or batch reruns with a different cohort) doesn't leave stale files
    // alongside the new ones. Only this dir is owned exclusively by
    // plot-topic's per-group writer; sibling dirs (e.g. `by_celltype/`
    // when this run uses `by_batch/`) are preserved.
    if Path::new(&by_group_dir).exists() {
        fs::remove_dir_all(&by_group_dir)?;
    }
    fs::create_dir_all(&by_group_dir)?;

    // Global topic display order: descending total prevalence across all
    // cells. Cells dominated by the same topic land at the same x-band
    // in every panel, so structure-bar blocks read consistently across
    // batches. `topic_rank[j]` = position of topic-column `j` in the
    // display order; `topic_display_order[i]` = topic-column at slot `i`.
    let topic_display_order = global_topic_order(probs, n_topics);
    let mut topic_rank = vec![0usize; n_topics];
    for (pos, &j) in topic_display_order.iter().enumerate() {
        topic_rank[j] = pos;
    }

    let ordered: Vec<(Box<str>, Vec<usize>)> = match args.order {
        CellOrder::Argmax => groups
            .iter()
            .map(|(b, cs)| (b.clone(), order_by_argmax(cs, probs, n_topics, &topic_rank)))
            .collect(),
        CellOrder::Coord => groups
            .iter()
            .map(|(b, cs)| {
                let o = order_by_coord(
                    cs,
                    resolved.cell_coords.as_deref(),
                    probs,
                    n_topics,
                    &topic_rank,
                )?;
                Ok::<_, anyhow::Error>((b.clone(), o))
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
    };

    let total_cells: usize = ordered.iter().map(|(_, v)| v.len()).sum();
    if total_cells == 0 {
        info!("No cells in any group, skipping structure plot");
        return Ok(());
    }

    let dims = StructDims {
        bar_h: (args.height * args.dpi as f32).round().max(1.0) as u32,
        label_band: (args.dpi as f32 * 0.35).round().max(20.0) as u32,
        legend_band: (args.dpi as f32 * 0.6).round().max(40.0) as u32,
        // ~0.06 in @ 300 DPI ≈ 18 px; small but visible separator.
        panel_gap: (args.dpi as f32 * 0.06).round().max(8.0) as u32,
    };
    // Reserve gap room when budgeting per-panel widths so the combined
    // SVG width still respects `args.width`. With a single panel there
    // are no gaps.
    let n_panels = ordered.len();
    let total_gap_px = if n_panels > 1 {
        dims.panel_gap * (n_panels as u32 - 1)
    } else {
        0
    };
    let usable_width_px = (args.width * args.dpi as f32).round().max(1.0) as u32;
    let total_width_px = usable_width_px.saturating_sub(total_gap_px).max(1);

    // Per-group raster panels (rayon-parallel; each panel is an
    // independent tiny-skia render).
    let panels: Vec<PanelOut> = ordered
        .par_iter()
        .map(|(batch, order)| -> anyhow::Result<PanelOut> {
            let n = order.len();
            let panel_w = ((n as f64 / total_cells as f64) * total_width_px as f64)
                .round()
                .max(1.0) as u32;
            // Reorder probs into a contiguous [n × K] row-major slice so
            // structure_bar_png can iterate linearly.
            let mut buf = Vec::with_capacity(n * n_topics);
            for &cell in order {
                buf.extend_from_slice(&probs[cell * n_topics..(cell + 1) * n_topics]);
            }
            let png = plot_utils::structure_bar_png(
                &buf,
                n,
                n_topics,
                panel_w,
                dims.bar_h,
                topic_colors,
            )?;
            Ok(PanelOut {
                batch: batch.clone(),
                n_cells: n,
                width_px: panel_w,
                png,
            })
        })
        .collect::<anyhow::Result<_>>()?;

    panels.par_iter().try_for_each(|p| -> anyhow::Result<()> {
        let svg = emit_struct_svg(
            std::slice::from_ref(p),
            &dims,
            topic_ids,
            topic_colors,
            &topic_display_order,
        );
        let base = format!("{by_group_dir}/{}", sanitize(&p.batch));
        emit_outputs(
            &svg,
            p.width_px + dims.legend_band,
            dims.label_band + dims.bar_h,
            &base,
            args,
        )
    })?;

    let combined_bars_w: u32 = panels.iter().map(|p| p.width_px).sum();
    let svg = emit_struct_svg(&panels, &dims, topic_ids, topic_colors, &topic_display_order);
    let combined_gap_w: u32 = if panels.len() > 1 {
        dims.panel_gap * (panels.len() as u32 - 1)
    } else {
        0
    };
    emit_outputs(
        &svg,
        combined_bars_w + combined_gap_w + dims.legend_band,
        dims.label_band + dims.bar_h,
        &format!("{struct_dir}/all"),
        args,
    )?;

    Ok(())
}

/// Emit a structure-plot SVG laying out `panels` left-to-right with a
/// single shared topic legend on the right. A 1-element slice produces
/// the standalone per-group view; a multi-element slice produces the
/// combined `all.*` view — the layout is identical.
fn emit_struct_svg(
    panels: &[PanelOut],
    d: &StructDims,
    topic_ids: &[i64],
    topic_colors: &[Rgb],
    topic_display_order: &[usize],
) -> String {
    let bars_w: u32 = panels.iter().map(|p| p.width_px).sum();
    let n_gaps = panels.len().saturating_sub(1) as u32;
    let total_gaps_w = d.panel_gap * n_gaps;
    let total_w = bars_w + total_gaps_w + d.legend_band;
    let total_h = d.label_band + d.bar_h;
    let label_fs = (d.label_band as f32 * PANEL_LABEL_FONT_FRAC).round();

    let mut s = String::with_capacity(panels.iter().map(|p| p.png.len()).sum::<usize>() * 2 + 4096);
    let _ = write!(
        s,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <svg xmlns=\"http://www.w3.org/2000/svg\" \
             xmlns:xlink=\"http://www.w3.org/1999/xlink\" \
             viewBox=\"0 0 {total_w} {total_h}\" width=\"{total_w}\" height=\"{total_h}\">\n",
    );
    // White background — PDF backends don't have a defined fill behind
    // the raster <image>, so missing this paints garbage on some viewers.
    let _ = writeln!(
        s,
        "  <rect x=\"0\" y=\"0\" width=\"{total_w}\" height=\"{total_h}\" fill=\"white\"/>"
    );

    let mut x_offset: u32 = 0;
    for (panel_idx, p) in panels.iter().enumerate() {
        let label = format!("{} (n={})", plot_utils::escape_xml(&p.batch), p.n_cells);
        let _ = writeln!(
            s,
            "  <text x=\"{x}\" y=\"{y}\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"{label_fs}\" text-anchor=\"middle\" dominant-baseline=\"central\" \
             fill=\"black\">{label}</text>",
            x = x_offset + p.width_px / 2,
            y = d.label_band / 2,
        );
        let b64 = BASE64.encode(&p.png);
        let _ = writeln!(
            s,
            "  <image x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" \
             preserveAspectRatio=\"none\" href=\"data:image/png;base64,{b64}\"/>",
            x = x_offset,
            y = d.label_band,
            w = p.width_px,
            h = d.bar_h,
        );
        let _ = writeln!(
            s,
            "  <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" \
             fill=\"none\" stroke=\"black\" stroke-width=\"1\"/>",
            x = x_offset as f32 + 0.5,
            y = d.label_band as f32 + 0.5,
            w = p.width_px as f32 - 1.0,
            h = d.bar_h as f32 - 1.0,
        );
        x_offset += p.width_px;
        if panel_idx + 1 < panels.len() {
            x_offset += d.panel_gap;
        }
    }

    // Legend lists topics in display order so the top swatch matches the
    // dominant block at the leftmost x in every panel.
    emit_topic_legend(
        &mut s,
        bars_w + total_gaps_w,
        d,
        topic_ids,
        topic_colors,
        topic_display_order,
    );
    let _ = writeln!(s, "</svg>");
    s
}

fn emit_topic_legend(
    s: &mut String,
    bar_x_end: u32,
    d: &StructDims,
    topic_ids: &[i64],
    topic_colors: &[Rgb],
    topic_display_order: &[usize],
) {
    let n = topic_ids.len();
    if n == 0 || d.legend_band < 8 {
        return;
    }
    let pad_left = 8.0;
    let swatch = (d.legend_band as f32 * 0.18).clamp(8.0, 18.0);
    let line_h = swatch + 4.0;
    let total_legend_h = line_h * n as f32;
    let start_y = d.label_band as f32 + ((d.bar_h as f32 - total_legend_h) * 0.5).max(0.0);

    let _ = writeln!(s, "  <g id=\"legend\">");
    for (i, &j) in topic_display_order.iter().enumerate() {
        let tid = topic_ids[j];
        let (r, g, b) = topic_colors[j];
        let y = start_y + i as f32 * line_h;
        let x = bar_x_end as f32 + pad_left;
        let _ = writeln!(
            s,
            "    <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{sw:.1}\" height=\"{sw:.1}\" \
             fill=\"rgb({r},{g},{b})\" stroke=\"black\" stroke-width=\"0.5\"/>",
            sw = swatch,
        );
        let _ = writeln!(
            s,
            "    <text x=\"{tx:.1}\" y=\"{ty:.1}\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"{fs:.1}\" dominant-baseline=\"central\" fill=\"black\">T{tid}</text>",
            tx = x + swatch + 4.0,
            ty = y + swatch * 0.5,
            fs = swatch * 0.85,
        );
    }
    let _ = writeln!(s, "  </g>");
}
