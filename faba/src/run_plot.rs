//! Entry point for `faba plot` — render the outputs of `faba lineage` into a
//! publication-style figure (PDF by default; opt-in PNG/SVG): an annotated
//! trajectory laid over the 2D (PHATE) embedding.
//!
//! Reads the `{from}.*_2d.parquet` layout tables written by
//! `faba lineage --markers` (with the default `--layout phate`) plus the
//! per-cell annotation and the labeled trajectory:
//!
//! - `{from}.cells_2d.parquet`            — cell × [x, y] (the PHATE coords)
//! - `{from}.lineage_annot.annot.parquet` — cell × coarse_label (per-cell type)
//! - `{from}.curves_2d.parquet`           — Slingshot principal curves (polylines)
//! - `{from}.nodes_2d.parquet`            — MST node positions
//! - `{from}.trajectory_annotation.parquet` — node → role → cell_type → confidence
//! - `{from}.pseudotime.parquet`          — cell × pseudotime (for `--color-by pseudotime`)
//!
//! The render follows the shared `plot-utils` pipeline (also used by `senna
//! plot`): each colour group is rasterized to a transparent PNG layer
//! ([`plot_utils::rasterize::rasterize_group_png`]); the curves + nodes are
//! rasterized as dark overlay layers; [`plot_utils::svg_emit::emit_svg`] stacks
//! the layers as base64 `<image>` elements; then this module splices vector
//! node labels, the root star, and a legend / colourbar on top. The stacked SVG
//! is then written as `{out}.plot.pdf` (vector, via `svg2pdf`; the default) and,
//! opt-in, `{out}.plot.png` (resvg) / `{out}.plot.svg`. The point cloud stays a
//! raster layer, so the PDF is a hybrid: vector text over a raster scatter at `--dpi`.

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use log::{info, warn};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;

use plot_utils::palette::{self, Palette, Rgb};
use plot_utils::rasterize::{
    rasterize_arrow_layer_png, rasterize_group_png, rasterize_per_point_png,
    rasterize_segment_layer_png, DataBounds, Extent, PointShape,
};
use plot_utils::render::{render_pdf, render_png};
use plot_utils::svg_emit::{emit_svg, escape_xml, SvgOpts, TopicLayer};
use plot_utils::RadiusSpec;

/// Points → pixels conversion base (72 pt per inch), shared with `senna plot`.
const PT_PER_INCH: f32 = 72.0;

/// Dark ink used for the trajectory nodes.
const INK: Rgb = (35, 35, 40);
/// Light grey for the principal curves + direction arrows — high contrast on the
/// dark plot background (dark INK would vanish).
const CURVE: Rgb = (225, 225, 235);
/// Red used for the root marker (a star).
const ROOT_RED: Rgb = (214, 39, 40);
/// Node cell type that is never labeled (the un-differentiated progenitor pool).
const CYCLING: &str = "Cycling_Progenitor";

/// What the cells are coloured by.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
#[clap(rename_all = "kebab-case")]
pub enum ColorBy {
    /// Per-cell coarse cell-type label (`coarse_label`), one colour per type
    /// from a qualitative palette, plus a legend. (default)
    #[default]
    Celltype,
    /// Continuous pseudotime on a blue→red ramp, plus a colourbar.
    Pseudotime,
}

#[derive(Args, Debug)]
pub struct PlotArgs {
    #[arg(
        long,
        short = 'f',
        help = "lineage output prefix (reads {from}.cells_2d.parquet, .curves_2d, \
                .nodes_2d, .lineage_annot.annot, .trajectory_annotation, .pseudotime)"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (default: the --from prefix); writes {out}.plot.pdf (+ .png / .svg with --png / --svg)"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long = "color-by",
        alias = "colour-by",
        value_enum,
        default_value_t = ColorBy::Celltype,
        help = "Colour cells by coarse cell type (default) or by pseudotime"
    )]
    pub color_by: ColorBy,

    #[arg(long, default_value_t = 9.0, help = "Plot width (inches)")]
    pub width: f32,

    #[arg(long, default_value_t = 8.0, help = "Plot height (inches)")]
    pub height: f32,

    #[arg(long, default_value_t = 150, help = "Output DPI")]
    pub dpi: u32,

    #[arg(long, default_value_t = 3.0, help = "Cell point size (pt)")]
    pub point_size: f32,

    #[arg(long, default_value_t = 0.7, help = "Cell point alpha (0..=1)")]
    pub alpha: f32,

    #[arg(
        long,
        value_enum,
        default_value_t = Palette::Auto,
        help = "Qualitative palette for --color-by celltype"
    )]
    pub palette: Palette,

    #[arg(
        long,
        default_value_t = 11.0,
        help = "Node / legend label font size (pt)"
    )]
    pub label_font_size: f32,

    #[arg(
        long,
        default_value_t = false,
        help = "Also emit SVG (default: PDF only)"
    )]
    pub svg: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Also emit flattened PNG (default: PDF only). The scatter/curves/nodes \
                are raster layers, so the PDF is a hybrid — vector text/legend/star over \
                a raster point cloud rendered at --dpi. Raise --dpi (300-600) for print."
    )]
    pub png: bool,

    #[arg(long, default_value_t = false, help = "Skip PDF output")]
    pub no_pdf: bool,
}

pub fn run_plot(args: &PlotArgs) -> Result<()> {
    let prefix = args.from.as_ref();
    let out = args.out.as_deref().unwrap_or(prefix).to_string();
    mkdir_parent(&out)?;

    // ---- cell coordinates (PHATE 2D) ----
    let cells_path = format!("{prefix}.cells_2d.parquet");
    let cells = DMatrix::<f32>::from_parquet(&cells_path)
        .with_context(|| format!("reading cell coordinates {cells_path}"))?;
    let cell_names = cells.rows;
    let xi = col_index(&cells.cols, "x", &cells_path)?;
    let yi = col_index(&cells.cols, "y", &cells_path)?;
    let n = cells.mat.nrows();
    anyhow::ensure!(n >= 1, "no cells in {cells_path}");
    let cx: Vec<f32> = (0..n).map(|i| cells.mat[(i, xi)]).collect();
    let cy: Vec<f32> = (0..n).map(|i| cells.mat[(i, yi)]).collect();
    info!("loaded {n} cells from {cells_path}");

    // ---- extents / bounds / pixel scale ----
    let (mut xmin, mut xmax) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..n {
        if cx[i].is_finite() && cy[i].is_finite() {
            xmin = xmin.min(cx[i]);
            xmax = xmax.max(cx[i]);
            ymin = ymin.min(cy[i]);
            ymax = ymax.max(cy[i]);
        }
    }
    anyhow::ensure!(
        xmin.is_finite() && xmax.is_finite(),
        "no finite cell coordinates in {cells_path}"
    );
    let width_px = (args.width * args.dpi as f32).round().max(1.0) as u32;
    let height_px = (args.height * args.dpi as f32).round().max(1.0) as u32;
    let ext = Extent {
        w: width_px,
        h: height_px,
    };
    let bounds = DataBounds::from_minmax(xmin, xmax, ymin, ymax);
    let radius_px = (args.point_size * args.dpi as f32 / PT_PER_INCH / 2.0).max(0.3);
    let font_px = args.label_font_size * args.dpi as f32 / PT_PER_INCH;

    // ---- cell colour layers (celltype or pseudotime) + the on-top overlay ----
    let mut layers: Vec<TopicLayer> = Vec::new();
    // Extra vector SVG spliced in just before </svg> (legend or colourbar).
    let side_overlay = match args.color_by {
        ColorBy::Celltype => {
            let legend = build_celltype_layers(
                prefix,
                &cell_names,
                &cx,
                &cy,
                &bounds,
                ext,
                radius_px,
                args,
                &mut layers,
            )?;
            emit_legend(&legend, font_px)
        }
        ColorBy::Pseudotime => {
            let (lo, hi) = build_pseudotime_layer(
                prefix,
                &cell_names,
                &cx,
                &cy,
                &bounds,
                ext,
                radius_px,
                args,
                &mut layers,
            )?;
            emit_colourbar(width_px, height_px, font_px, lo, hi)
        }
    };

    // ---- principal-curve overlay (thin polylines + periodic direction arrows) ----
    match build_curve_layer(prefix, &bounds, ext, radius_px) {
        Ok(curve_layers) if !curve_layers.is_empty() => layers.extend(curve_layers),
        Ok(_) => warn!("no principal-curve segments to draw"),
        Err(e) => warn!("curves overlay skipped: {e}"),
    }

    // ---- trajectory nodes: positions (+ dark point layer) and labels/root ----
    let node_overlay = match build_nodes(prefix, &bounds, ext, radius_px, font_px) {
        Ok((layer, overlay)) => {
            layers.push(layer);
            overlay
        }
        Err(e) => {
            warn!("nodes overlay skipped: {e}");
            String::new()
        }
    };

    anyhow::ensure!(!layers.is_empty(), "nothing to plot");

    // ---- assemble SVG: base raster stack, then splice vector overlays ----
    let svg = emit_svg(
        &layers,
        &SvgOpts {
            width_px,
            height_px,
            draw_hulls: false,
            draw_labels: false,
            label_font_size_px: font_px,
            hull_stroke_px: 1.0,
            hull_fill_alpha: 0.0,
            frame_stroke_px: 0.0,
        },
    );
    let mut overlay = String::new();
    overlay.push_str(&node_overlay);
    overlay.push_str(&side_overlay);
    let svg = if overlay.is_empty() {
        svg
    } else {
        svg.replacen("</svg>", &format!("{overlay}</svg>"), 1)
    };

    // SVG is opt-in (the intermediate source); PDF is the default; PNG is opt-in.
    if args.svg {
        let svg_path = format!("{out}.plot.svg");
        std::fs::write(&svg_path, svg.as_bytes()).with_context(|| format!("writing {svg_path}"))?;
        info!("Wrote {svg_path}");
    }

    // PNG + PDF share the same SVG string and are independent; render concurrently to
    // hide resvg/svg2pdf parse latency. Default is PDF-only; PNG is opt-in.
    let png_task = args.png.then(|| format!("{out}.plot.png"));
    let pdf_task = (!args.no_pdf).then(|| format!("{out}.plot.pdf"));
    let (png_res, pdf_res) = rayon::join(
        || match &png_task {
            Some(p) => {
                render_png(&svg, width_px, height_px, Path::new(p)).map(|()| Some(p.clone()))
            }
            None => Ok(None),
        },
        || match &pdf_task {
            Some(p) => render_pdf(&svg, Path::new(p)).map(|()| Some(p.clone())),
            None => Ok(None),
        },
    );
    if let Some(p) = png_res? {
        info!("Wrote {p} ({width_px}x{height_px})");
    }
    if let Some(p) = pdf_res? {
        info!("Wrote {p}");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////
// Cell colour layers
////////////////////////////////////////////////////////////////////////

/// One legend row: display name + swatch colour, in draw order.
struct LegendEntry {
    label: String,
    color: Rgb,
}

/// Build one `rasterize_group_png` layer per coarse cell type and return the
/// legend (type → colour). Cells missing an annotation → `unassigned`.
#[allow(clippy::too_many_arguments)]
fn build_celltype_layers(
    prefix: &str,
    cell_names: &[Box<str>],
    cx: &[f32],
    cy: &[f32],
    bounds: &DataBounds,
    ext: Extent,
    radius_px: f32,
    args: &PlotArgs,
    layers: &mut Vec<TopicLayer>,
) -> Result<Vec<LegendEntry>> {
    let annot_path = format!("{prefix}.lineage_annot.annot.parquet");
    let label_by_cell: HashMap<Box<str>, Box<str>> =
        match read_str_columns(&annot_path, &["cell", "coarse_label"]) {
            Ok(mut cols) => {
                let labels = cols.pop().unwrap();
                let cells = cols.pop().unwrap();
                cells.into_iter().zip(labels).collect()
            }
            Err(e) => {
                warn!("no per-cell annotation ({e}); colouring every cell as 'unassigned'");
                HashMap::new()
            }
        };

    // Per-cell type name (join by barcode).
    let unassigned: Box<str> = Box::from("unassigned");
    let per_cell: Vec<Box<str>> = cell_names
        .iter()
        .map(|c| {
            label_by_cell
                .get(c)
                .cloned()
                .unwrap_or_else(|| unassigned.clone())
        })
        .collect();

    // Stable colour assignment: sort unique types, push `unassigned` last so it
    // never steals the leading palette slot.
    let mut types: Vec<Box<str>> = per_cell.clone();
    types.sort_unstable();
    types.dedup();
    if let Some(pos) = types.iter().position(|t| t.as_ref() == "unassigned") {
        let last = types.remove(pos);
        types.push(last);
    }
    let pal = palette::resolve(&args.palette, types.len());
    let type_color: HashMap<Box<str>, Rgb> = types
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), palette::color(&pal, i)))
        .collect();

    // Bucket cells (in pixel space) per type.
    let mut pts_by_type: HashMap<Box<str>, Vec<(f32, f32)>> = HashMap::new();
    for i in 0..cell_names.len() {
        if !cx[i].is_finite() || !cy[i].is_finite() {
            continue;
        }
        pts_by_type
            .entry(per_cell[i].clone())
            .or_default()
            .push(bounds.to_pixel((cx[i], cy[i]), ext));
    }

    // One raster layer per type, in the stable legend order.
    let mut legend = Vec::with_capacity(types.len());
    for t in &types {
        let color = type_color[t];
        if let Some(pts) = pts_by_type.get(t) {
            let png = rasterize_group_png(
                pts,
                ext,
                RadiusSpec::Scalar(radius_px),
                color,
                args.alpha,
                PointShape::Circle,
            )?;
            layers.push(TopicLayer {
                label: String::new(),
                png,
                hull_px: Vec::new(),
                label_xy_px: (f32::NAN, f32::NAN),
                color,
            });
        }
        legend.push(LegendEntry {
            label: t.to_string(),
            color,
        });
    }
    info!("coloured {} cell types", legend.len());
    Ok(legend)
}

/// Build a single continuous pseudotime layer (blue→red ramp) and return the
/// `(min, max)` pseudotime for the colourbar labels.
#[allow(clippy::too_many_arguments)]
fn build_pseudotime_layer(
    prefix: &str,
    cell_names: &[Box<str>],
    cx: &[f32],
    cy: &[f32],
    bounds: &DataBounds,
    ext: Extent,
    radius_px: f32,
    args: &PlotArgs,
    layers: &mut Vec<TopicLayer>,
) -> Result<(f32, f32)> {
    let pt_path = format!("{prefix}.pseudotime.parquet");
    let pt = DMatrix::<f32>::from_parquet(&pt_path)
        .with_context(|| format!("reading pseudotime {pt_path}"))?;
    // Prefer the `pseudotime` column, else the first numeric column.
    let j = pt
        .cols
        .iter()
        .position(|c| c.as_ref() == "pseudotime")
        .unwrap_or(0);
    anyhow::ensure!(pt.mat.ncols() > j, "pseudotime parquet has no data column");
    let value_by_cell: HashMap<Box<str>, f32> = pt
        .rows
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), pt.mat[(i, j)]))
        .collect();

    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for c in cell_names {
        if let Some(&v) = value_by_cell.get(c) {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    anyhow::ensure!(
        lo.is_finite() && hi > lo,
        "pseudotime has no finite range (all NaN/constant?)"
    );
    let span = hi - lo;

    let mut pts_px: Vec<(f32, f32)> = Vec::with_capacity(cell_names.len());
    let mut colors: Vec<Rgb> = Vec::with_capacity(cell_names.len());
    for (i, c) in cell_names.iter().enumerate() {
        let v = value_by_cell.get(c).copied().unwrap_or(f32::NAN);
        if !v.is_finite() || !cx[i].is_finite() || !cy[i].is_finite() {
            continue;
        }
        pts_px.push(bounds.to_pixel((cx[i], cy[i]), ext));
        colors.push(palette::sample_blue_red((v - lo) / span));
    }
    let png = rasterize_per_point_png(
        &pts_px,
        &colors,
        ext,
        radius_px,
        args.alpha,
        PointShape::Circle,
    )?;
    layers.push(TopicLayer {
        label: String::new(),
        png,
        hull_px: Vec::new(),
        label_xy_px: (f32::NAN, f32::NAN),
        color: (0, 0, 0),
    });
    info!(
        "coloured {} cells by pseudotime [{lo:.3}, {hi:.3}]",
        pts_px.len()
    );
    Ok((lo, hi))
}

////////////////////////////////////////////////////////////////////////
// Trajectory overlays (curves, nodes, root, labels)
////////////////////////////////////////////////////////////////////////

/// Rasterize the principal curves (`curves_2d`) as one dark segment layer:
/// group points by lineage, order by `grid`, connect consecutive points.
/// Returns `None` when there are no finite segments.
fn build_curve_layer(
    prefix: &str,
    bounds: &DataBounds,
    ext: Extent,
    radius_px: f32,
) -> Result<Vec<TopicLayer>> {
    let path = format!("{prefix}.curves_2d.parquet");
    let c =
        DMatrix::<f32>::from_parquet(&path).with_context(|| format!("reading curves {path}"))?;
    let li = col_index(&c.cols, "lineage", &path)?;
    let gi = col_index(&c.cols, "grid", &path)?;
    let xi = col_index(&c.cols, "x", &path)?;
    let yi = col_index(&c.cols, "y", &path)?;

    // Gather (grid, x, y) per lineage.
    let mut by_lineage: HashMap<i64, Vec<(f32, f32, f32)>> = HashMap::new();
    for i in 0..c.mat.nrows() {
        let l = c.mat[(i, li)] as i64;
        by_lineage
            .entry(l)
            .or_default()
            .push((c.mat[(i, gi)], c.mat[(i, xi)], c.mat[(i, yi)]));
    }
    // Thin polyline segments (the smooth curve) + a few periodic direction arrows
    // per lineage. `grid` order runs root → terminal, so each arrow points downstream.
    const ARROWS_PER_CURVE: usize = 6;
    let mut line_segs: Vec<((f32, f32), (f32, f32))> = Vec::new();
    let mut arrow_segs: Vec<((f32, f32), (f32, f32))> = Vec::new();
    let px = |p: &(f32, f32, f32)| bounds.to_pixel((p.1, p.2), ext);
    for pts in by_lineage.values_mut() {
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for w in pts.windows(2) {
            if w[0].1.is_finite() && w[0].2.is_finite() && w[1].1.is_finite() && w[1].2.is_finite()
            {
                line_segs.push((px(&w[0]), px(&w[1])));
            }
        }
        // Space the arrows evenly along the curve; each spans one grid step so the
        // head reads as a small local direction marker, not a long shaft.
        let n = pts.len();
        if n >= 3 {
            let step = (n / (ARROWS_PER_CURVE + 1)).max(1);
            let mut i = step;
            while i < n {
                let (a, b) = (&pts[i - 1], &pts[i]);
                if a.1.is_finite() && a.2.is_finite() && b.1.is_finite() && b.2.is_finite() {
                    arrow_segs.push((px(a), px(b)));
                }
                i += step;
            }
        }
    }
    if line_segs.is_empty() {
        return Ok(Vec::new());
    }
    let line_stroke = (radius_px * 0.45).max(0.8); // thinner than the cell radius
    let arrow_stroke = (radius_px * 0.55).max(1.0);
    let head_len = (radius_px * 2.6).max(6.0);
    let mk = |png| TopicLayer {
        label: String::new(),
        png,
        hull_px: Vec::new(),
        label_xy_px: (f32::NAN, f32::NAN),
        color: CURVE,
    };
    let mut out = vec![mk(rasterize_segment_layer_png(
        &line_segs,
        ext,
        line_stroke,
        CURVE,
        0.55,
    )?)];
    if !arrow_segs.is_empty() {
        out.push(mk(rasterize_arrow_layer_png(
            &arrow_segs,
            ext,
            arrow_stroke,
            head_len,
            CURVE,
            0.95,
        )?));
    }
    Ok(out)
}

/// Build the trajectory-node point layer (dark dots) plus the vector overlay
/// carrying: a red star at the root node and a haloed `cell_type` label at each
/// non-`Cycling_Progenitor` node.
fn build_nodes(
    prefix: &str,
    bounds: &DataBounds,
    ext: Extent,
    radius_px: f32,
    font_px: f32,
) -> Result<(TopicLayer, String)> {
    let nodes_path = format!("{prefix}.nodes_2d.parquet");
    let nodes = DMatrix::<f32>::from_parquet(&nodes_path)
        .with_context(|| format!("reading nodes {nodes_path}"))?;
    let xi = col_index(&nodes.cols, "x", &nodes_path)?;
    let yi = col_index(&nodes.cols, "y", &nodes_path)?;
    let node_names = nodes.rows;

    // node name → pixel position
    let mut pos_by_node: HashMap<Box<str>, (f32, f32)> = HashMap::new();
    let mut pts_px: Vec<(f32, f32)> = Vec::with_capacity(node_names.len());
    for (i, name) in node_names.iter().enumerate() {
        let (x, y) = (nodes.mat[(i, xi)], nodes.mat[(i, yi)]);
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let p = bounds.to_pixel((x, y), ext);
        pos_by_node.insert(name.clone(), p);
        pts_px.push(p);
    }

    let node_r = (radius_px * 1.6).max(2.5);
    let png = rasterize_group_png(
        &pts_px,
        ext,
        RadiusSpec::Scalar(node_r),
        INK,
        1.0,
        PointShape::Circle,
    )?;
    let layer = TopicLayer {
        label: String::new(),
        png,
        hull_px: Vec::new(),
        label_xy_px: (f32::NAN, f32::NAN),
        color: INK,
    };

    // Labels + root star from the trajectory annotation.
    let traj_path = format!("{prefix}.trajectory_annotation.parquet");
    let mut overlay = String::new();
    match read_str_columns(&traj_path, &["node", "role", "cell_type"]) {
        Ok(cols) => {
            let node_col = &cols[0];
            let role_col = &cols[1];
            let type_col = &cols[2];
            overlay.push_str("  <g id=\"trajectory-labels\">\n");
            let mut n_labeled = 0usize;
            for i in 0..node_col.len() {
                let Some(&(px, py)) = pos_by_node.get(&node_col[i]) else {
                    continue;
                };
                if role_col[i].as_ref() == "root" {
                    emit_star(&mut overlay, px, py, node_r * 2.4);
                }
                let ct = type_col[i].as_ref();
                if !ct.is_empty() && !ct.eq_ignore_ascii_case(CYCLING) {
                    // Nudge the label just above the node marker.
                    emit_halo_text(
                        &mut overlay,
                        px,
                        py - node_r - font_px * 0.6,
                        font_px,
                        INK,
                        ct,
                    );
                    n_labeled += 1;
                }
            }
            overlay.push_str("  </g>\n");
            info!("labeled {n_labeled} trajectory node(s)");
        }
        Err(e) => warn!("no trajectory annotation ({e}); nodes drawn without labels"),
    }
    Ok((layer, overlay))
}

////////////////////////////////////////////////////////////////////////
// Vector SVG helpers (spliced on top of the raster stack)
////////////////////////////////////////////////////////////////////////

/// A white-haloed, centred `<text>` label (matches `plot-utils` label style).
fn emit_halo_text(s: &mut String, x: f32, y: f32, font_px: f32, color: Rgb, text: &str) {
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
fn emit_star(s: &mut String, cx: f32, cy: f32, r_out: f32) {
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
fn emit_legend(entries: &[LegendEntry], font_px: f32) -> String {
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
fn emit_colourbar(width_px: u32, height_px: u32, font_px: f32, lo: f32, hi: f32) -> String {
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

////////////////////////////////////////////////////////////////////////
// Parquet helpers
////////////////////////////////////////////////////////////////////////

/// Position of `name` in a column-name list, with a file-qualified error.
fn col_index(cols: &[Box<str>], name: &str, path: &str) -> Result<usize> {
    cols.iter()
        .position(|c| c.as_ref() == name)
        .ok_or_else(|| anyhow::anyhow!("column '{name}' not found in {path}"))
}

/// Read the named (string / `BYTE_ARRAY`) columns from a parquet file, one
/// `Vec<Box<str>>` per requested name, in request order. Non-string cells fall
/// back to the empty string.
fn read_str_columns(path: &str, wanted: &[&str]) -> Result<Vec<Vec<Box<str>>>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
    let reader = SerializedFileReader::new(file)?;
    let fields = reader
        .metadata()
        .file_metadata()
        .schema()
        .get_fields()
        .to_vec();
    let idx: Vec<usize> = wanted
        .iter()
        .map(|w| {
            fields
                .iter()
                .position(|f| f.name() == *w)
                .ok_or_else(|| anyhow::anyhow!("column '{w}' not found in {path}"))
        })
        .collect::<Result<_>>()?;

    let mut out: Vec<Vec<Box<str>>> = vec![Vec::new(); wanted.len()];
    for record in reader.get_row_iter(None)? {
        let row = record?;
        for (k, &j) in idx.iter().enumerate() {
            let v = row
                .get_string(j)
                .map(|s| s.clone().into_boxed_str())
                .unwrap_or_else(|_| Box::from(""));
            out[k].push(v);
        }
    }
    Ok(out)
}
