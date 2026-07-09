//! SVG assembly for `senna plot-strand`.
//!
//! One figure per cell type: chromosomes stacked vertically, each a
//! mirrored Watson(up)/Crick(down) filled step-area around a shared
//! axis. We hand-build the SVG string and render it to PDF/PNG through
//! the shared `plot_utils` resvg/svg2pdf path.

use super::place::{bin_height, BinGrid, ChrGeom, Placement, Placements};
use super::{HeightScale, PlotStrandArgs, Strands};
use crate::postprocess::plot::sanitize_filename as sanitize;
use plot_utils::svg_emit::escape_xml;
use std::fmt::Write as _;
use std::path::Path;

/// Build + render one cell-type (or consensus) figure. Returns the
/// number of files written.
#[allow(clippy::too_many_arguments)]
pub(super) fn render_one(
    name: &str,
    grid: &BinGrid,
    strands: Strands,
    robust_max: f32,
    p: &Placements,
    out_dir: &str,
    args: &PlotStrandArgs,
) -> anyhow::Result<usize> {
    let dpi = args.dpi as f32;
    let w = (args.width * dpi).round();
    let track_h = (args.track_height * dpi).round();
    let gap = (track_h * 0.45).round();
    let left = (1.0 * dpi).round();
    let right = (0.25 * dpi).round();
    let top = (0.55 * dpi).round();
    let bottom = (0.3 * dpi).round();
    let plot_w = (w - left - right).max(1.0);
    let n_chr = p.chromosomes.len() as f32;
    let h = top + bottom + n_chr * (track_h + gap);

    let title_fs = 0.17 * dpi;
    let chr_fs = 0.10 * dpi;
    let gene_fs = 0.075 * dpi;
    let half = (track_h / 2.0 - 1.0).max(1.0);

    let mut svg = String::with_capacity(64 * 1024);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.0}\" height=\"{h:.0}\" \
         viewBox=\"0 0 {w:.0} {h:.0}\">",
    ));
    svg.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{w:.0}\" height=\"{h:.0}\" fill=\"white\"/>"
    ));

    // Title.
    let title = if name == "_consensus" {
        "Consensus  (Watson \u{2191} / Crick \u{2193})".to_string()
    } else {
        format!("{name}  (Watson \u{2191} / Crick \u{2193})")
    };
    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"{:.1}\" \
         font-weight=\"bold\" fill=\"#222\">{}</text>",
        left,
        top * 0.6,
        title_fs,
        escape_xml(&title)
    ));

    for (ci, chr) in p.chromosomes.iter().enumerate() {
        let y_mid = top + ci as f32 * (track_h + gap) + track_h / 2.0;
        let chr_w = plot_w * (chr.span as f32 / p.max_span as f32);

        // Chromosome label.
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-family=\"sans-serif\" \
             font-size=\"{:.1}\" fill=\"#333\">chr{}</text>",
            left - 0.08 * dpi,
            y_mid + chr_fs * 0.35,
            chr_fs,
            escape_xml(&chr.name)
        ));

        // Watson (up) + Crick (down) filled step areas.
        let binpx = chr_w / chr.n_bins as f32;
        let geom = StrandPath {
            x_left: left,
            binpx,
            y_mid,
            half,
            robust_max,
            scale: args.scale,
        };
        svg.push_str(&strand_path(&geom, &grid.watson[ci], true, strands.up));
        svg.push_str(&strand_path(&geom, &grid.crick[ci], false, strands.down));

        // Midline (chromosome axis).
        svg.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#888\" \
             stroke-width=\"{:.2}\"/>",
            left,
            y_mid,
            left + chr_w,
            y_mid,
            (0.006 * dpi).max(0.5)
        ));

        // Optional top-gene labels.
        if args.top_genes > 0 {
            svg.push_str(&top_gene_labels(
                ci, chr, left, binpx, y_mid, half, gene_fs, dpi, grid, p, args,
            ));
        }
    }

    svg.push_str("</svg>");

    ////////////
    // Render //
    ////////////
    let base = format!("{out_dir}/{}", sanitize(name));
    let mut written = 0usize;
    if args.svg {
        let pth = format!("{base}.svg");
        std::fs::write(&pth, svg.as_bytes())?;
        written += 1;
    }
    let png_task = args.png.then(|| format!("{base}.png"));
    let pdf_task = (!args.no_pdf).then(|| format!("{base}.pdf"));
    let (png_res, pdf_res) = rayon::join(
        || match &png_task {
            Some(pth) => {
                plot_utils::render_png(&svg, w as u32, h as u32, Path::new(pth)).map(|()| 1usize)
            }
            None => Ok(0),
        },
        || match &pdf_task {
            Some(pth) => plot_utils::render_pdf(&svg, Path::new(pth)).map(|()| 1usize),
            None => Ok(0),
        },
    );
    written += png_res? + pdf_res?;
    Ok(written)
}

/// Shared geometry + scaling for one chromosome's mirrored strand pair.
struct StrandPath {
    x_left: f32,
    binpx: f32,
    y_mid: f32,
    half: f32,
    robust_max: f32,
    scale: HeightScale,
}

/// Build a filled step-area `<path>` for one strand of one chromosome.
/// `up = true` draws above the midline (Watson), else mirrored below.
/// Heights are remapped through `g.scale` (and the same scale is applied
/// to `robust_max`) so the normalized fraction stays in `[0, 1]`.
fn strand_path(g: &StrandPath, heights: &[f32], up: bool, color: &str) -> String {
    if heights.iter().all(|&v| v <= 0.0) {
        return String::new();
    }
    let denom = g.scale.apply(g.robust_max).max(f32::MIN_POSITIVE);
    let mut d = String::with_capacity(heights.len() * 24);
    let _ = write!(d, "M {:.2} {:.2} ", g.x_left, g.y_mid);
    for (b, &v) in heights.iter().enumerate() {
        let frac = (g.scale.apply(v) / denom).clamp(0.0, 1.0);
        let hpx = frac * g.half;
        let x0 = g.x_left + b as f32 * g.binpx;
        let x1 = x0 + g.binpx;
        let y = if up { g.y_mid - hpx } else { g.y_mid + hpx };
        let _ = write!(d, "L {x0:.2} {y:.2} L {x1:.2} {y:.2} ");
    }
    let x_right = g.x_left + heights.len() as f32 * g.binpx;
    let _ = write!(d, "L {:.2} {:.2} Z", x_right, g.y_mid);
    format!("<path d=\"{d}\" fill=\"{color}\" fill-opacity=\"0.9\" stroke=\"none\"/>")
}

/// Tick + name labels for the top-N genes (by activity) on one
/// chromosome of one group. Forward genes label above, reverse below.
/// Labels are de-cluttered: once N are placed we stop, and a candidate
/// too close (in x) to an already-placed label on the same strand is
/// skipped so names stay legible.
#[allow(clippy::too_many_arguments)]
fn top_gene_labels(
    ci: usize,
    chr: &ChrGeom,
    x_left: f32,
    binpx: f32,
    y_mid: f32,
    half: f32,
    gene_fs: f32,
    dpi: f32,
    grid: &BinGrid,
    p: &Placements,
    args: &PlotStrandArgs,
) -> String {
    // Rank this chromosome's placed genes by the binned height of their
    // own strand (peak proximity), then label the top-N.
    let mut on_chr: Vec<&Placement> = p.placed.iter().filter(|pl| pl.chr_idx == ci).collect();
    on_chr.sort_by(|a, b| {
        bin_height(grid, b)
            .partial_cmp(&bin_height(grid, a))
            .unwrap()
    });
    let chr_w = binpx * chr.n_bins as f32;
    let min_gap = gene_fs * 4.0;
    let mut placed_up: Vec<f32> = Vec::new();
    let mut placed_down: Vec<f32> = Vec::new();
    let mut out = String::new();
    let mut n = 0usize;
    for pl in on_chr {
        if n >= args.top_genes || bin_height(grid, pl) <= 0.0 {
            break;
        }
        let x = x_left + ((pl.tss - chr.min_pos) as f32 / chr.span as f32) * chr_w;
        let placed = if pl.forward {
            &mut placed_up
        } else {
            &mut placed_down
        };
        if placed.iter().any(|&px| (px - x).abs() < min_gap) {
            continue; // would collide with an already-placed label
        }
        placed.push(x);
        n += 1;

        let (y_tick, y_text) = if pl.forward {
            (y_mid - half, y_mid - half - 0.02 * dpi)
        } else {
            (y_mid + half, y_mid + half + gene_fs)
        };
        out.push_str(&format!(
            "<line x1=\"{x:.1}\" y1=\"{y_mid:.1}\" x2=\"{x:.1}\" y2=\"{y_tick:.1}\" \
             stroke=\"#444\" stroke-width=\"{:.2}\"/>",
            (0.004 * dpi).max(0.4)
        ));
        out.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{y_text:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{gene_fs:.1}\" fill=\"#444\">{}</text>",
            escape_xml(&pl.symbol)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> StrandPath {
        StrandPath {
            x_left: 0.0,
            binpx: 1.0,
            y_mid: 10.0,
            half: 5.0,
            robust_max: 1.0,
            scale: HeightScale::Linear,
        }
    }

    #[test]
    fn strand_path_empty_when_all_zero() {
        let zero = strand_path(&geom(), &[0.0, 0.0], true, "#abc");
        assert!(zero.is_empty());
        let nonzero = strand_path(&geom(), &[0.0, 0.5], true, "#abc");
        assert!(nonzero.contains("<path"));
        assert!(nonzero.contains("#abc"));
    }
}
