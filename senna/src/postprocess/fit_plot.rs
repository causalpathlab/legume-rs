//! `senna plot` — publication-quality rasterized scatter with vector
//! labels over transparent background.
//!
//! See `plot::` module docs for the overall pipeline. This file is the
//! clap entry point and glue: reads coord + group-source parquets,
//! dispatches rasterization, emits SVG / PNG / PDF.

use crate::embed_common::*;
use crate::postprocess::plot::hull::{
    convex_hull, hull_centroid, median_xy, trim_outliers_by_median, Pt,
};
use crate::postprocess::plot::palette::{self, Palette};
use crate::postprocess::plot::rasterize::{
    rasterize_group_png, DataBounds, Extent, PointShape,
};
use crate::postprocess::plot::svg_emit::{emit_svg, SvgOpts, TopicLayer};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fs;
use std::path::Path;

const PT_PER_INCH: f32 = 72.0;

/// Source of the per-cell group ID used for coloring.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum ColorBy {
    /// Use the `cluster` column in `cell_coords.parquet`.
    Cluster,
    /// Use the `pb_id` column in `cell_coords.parquet`.
    PbId,
    /// Argmax of a separate topic-proportions parquet (`--topics`).
    Topic,
}

/// Label placement strategy per group.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum LabelPosition {
    /// Coordinate-wise median of the group's points (robust default).
    Median,
    /// Area-weighted centroid of the group's convex hull.
    HullCentroid,
}

#[derive(Args, Debug)]
pub struct PlotArgs {
    #[arg(
        long,
        short = 'c',
        required = true,
        help = "Cell coordinates parquet (from `senna viz`)"
    )]
    pub cell_coords: Box<str>,

    #[arg(
        long,
        short = 'o',
        required = true,
        help = "Output prefix",
        long_help = "Writes {out}.plot.svg, {out}.plot.png, {out}.plot.pdf."
    )]
    pub out: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value_t = ColorBy::Cluster,
        help = "Color source"
    )]
    pub color_by: ColorBy,

    #[arg(
        long,
        help = "Topic proportions parquet (cells × K); required with --color-by topic"
    )]
    pub topics: Option<Box<str>>,

    #[arg(
        long,
        help = "TSV mapping group_id<TAB>display_name (one per line). Missing IDs fall back to T{id}."
    )]
    pub labels: Option<Box<str>>,

    #[arg(long, default_value_t = 6.0, help = "Plot width (inches)")]
    pub width: f32,

    #[arg(long, default_value_t = 6.0, help = "Plot height (inches)")]
    pub height: f32,

    #[arg(long, default_value_t = 300, help = "Output DPI (raster layers)")]
    pub dpi: u32,

    #[arg(long, default_value_t = 0.5, help = "Point size (pt)")]
    pub point_size: f32,

    #[arg(long, default_value_t = 0.6, help = "Point alpha (0..=1)")]
    pub alpha: f32,

    #[arg(
        long,
        value_enum,
        default_value_t = PointShape::Circle,
        help = "Marker shape"
    )]
    pub point_shape: PointShape,

    #[arg(
        long,
        default_value_t = false,
        help = "Cycle marker shape per group (circle→triangle→square→diamond)"
    )]
    pub point_shape_cycle: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = Palette::Auto,
        help = "Qualitative palette"
    )]
    pub palette: Palette,

    #[arg(
        long,
        value_enum,
        default_value_t = LabelPosition::Median,
        help = "Label placement strategy"
    )]
    pub label_position: LabelPosition,

    #[arg(long, default_value_t = 10.0, help = "Label font size (pt)")]
    pub label_font_size: f32,

    #[arg(long, default_value_t = false, help = "Suppress vector text labels")]
    pub no_labels: bool,

    #[arg(long, default_value_t = false, help = "Suppress convex hull polygons")]
    pub no_hull: bool,

    #[arg(
        long,
        default_value_t = 0.95,
        help = "Fraction of closest-to-median points used for each hull (1.0 = all)",
        long_help = "For each group, keep only the {coverage} fraction of points\n\
                     nearest the coordinate-wise median (Euclidean) before computing\n\
                     the convex hull. Strips a few fringe cells so one outlier can't\n\
                     drag the polygon across the plot. Set to 1.0 to use every point."
    )]
    pub hull_coverage: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Hull fill opacity (0..=1; 0 = outline only)"
    )]
    pub hull_fill_alpha: f32,

    #[arg(long, default_value_t = false, help = "Skip SVG output")]
    pub no_svg: bool,

    #[arg(long, default_value_t = false, help = "Skip flattened PNG output")]
    pub no_png: bool,

    #[arg(long, default_value_t = false, help = "Skip PDF output")]
    pub no_pdf: bool,
}

pub fn fit_plot(args: &PlotArgs) -> anyhow::Result<()> {
    let coords_by_name = read_cell_coords(&args.cell_coords)?;
    let x = coords_by_name
        .get("x")
        .ok_or_else(|| anyhow::anyhow!("cell_coords parquet missing 'x' column"))?;
    let y = coords_by_name
        .get("y")
        .ok_or_else(|| anyhow::anyhow!("cell_coords parquet missing 'y' column"))?;
    let n_cells = x.len();
    info!("Loaded {n_cells} cells from {}", args.cell_coords);

    let group_ids = match args.color_by {
        ColorBy::Cluster => coords_by_name
            .get("cluster")
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "--color-by cluster but cell_coords has no 'cluster' column; \
                     re-run `senna viz` with --clusters or pick a different --color-by"
                )
            })?
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<_>>(),
        ColorBy::PbId => coords_by_name
            .get("pb_id")
            .ok_or_else(|| anyhow::anyhow!("cell_coords missing 'pb_id' column"))?
            .iter()
            .map(|&v| if v.is_nan() { -1 } else { v as i64 })
            .collect::<Vec<_>>(),
        ColorBy::Topic => {
            let topics_path = args
                .topics
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("--color-by topic requires --topics PATH"))?;
            argmax_topics(topics_path, n_cells)?
        }
    };

    // Single pass: per-group bucketing + global bounds.
    let mut pts_by_group: FxHashMap<i64, Vec<Pt>> = FxHashMap::default();
    let (mut xmin, mut xmax) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..n_cells {
        let g = group_ids[i];
        if g < 0 || !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        pts_by_group.entry(g).or_default().push((x[i], y[i]));
        if x[i] < xmin {
            xmin = x[i];
        }
        if x[i] > xmax {
            xmax = x[i];
        }
        if y[i] < ymin {
            ymin = y[i];
        }
        if y[i] > ymax {
            ymax = y[i];
        }
    }
    if pts_by_group.is_empty() {
        anyhow::bail!("no valid group assignments found");
    }

    let mut unique: Vec<i64> = pts_by_group.keys().copied().collect();
    unique.sort_unstable();
    info!("Plotting {} groups", unique.len());

    let width_px = (args.width * args.dpi as f32).round() as u32;
    let height_px = (args.height * args.dpi as f32).round() as u32;
    let ext = Extent {
        w: width_px,
        h: height_px,
    };
    let bounds = DataBounds::from_minmax(xmin, xmax, ymin, ymax);

    let radius_px = args.point_size * args.dpi as f32 / PT_PER_INCH / 2.0;
    let label_font_px = args.label_font_size * args.dpi as f32 / PT_PER_INCH;
    let palette = palette::resolve(&args.palette, unique.len());

    let label_map = match &args.labels {
        Some(p) => read_labels_tsv(p)?,
        None => FxHashMap::default(),
    };

    // Per-group rasterization + hull + label placement in parallel.
    // Each group is independent and encodes its own PNG — outermost loop
    // is the right place to fan out rayon (per repo convention).
    let layers: Vec<TopicLayer> = unique
        .par_iter()
        .enumerate()
        .map(|(i, g)| -> anyhow::Result<TopicLayer> {
            let pts = pts_by_group.get(g).expect("group present");
            let pts_px: Vec<(f32, f32)> =
                pts.iter().map(|&p| to_pixel(p, &bounds, ext)).collect();

            let color = palette::color(&palette, i);
            let shape = if args.point_shape_cycle {
                PointShape::cycle_nth(i)
            } else {
                args.point_shape
            };
            let png = rasterize_group_png(&pts_px, ext, radius_px, color, args.alpha, shape)?;

            let hull_pts = trim_outliers_by_median(pts, args.hull_coverage);
            let hull = convex_hull(&hull_pts);
            let hull_px: Vec<Pt> = hull.iter().map(|&p| to_pixel(p, &bounds, ext)).collect();

            let label_xy_data = match args.label_position {
                LabelPosition::Median => median_xy(pts),
                LabelPosition::HullCentroid => hull_centroid(&hull),
            };
            let label_xy_px = to_pixel(label_xy_data, &bounds, ext);

            let label = label_map
                .get(g)
                .cloned()
                .unwrap_or_else(|| format!("T{g}"));

            Ok(TopicLayer {
                label,
                png,
                hull_px,
                label_xy_px,
                color,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let svg = emit_svg(
        &layers,
        &SvgOpts {
            width_px,
            height_px,
            draw_hulls: !args.no_hull,
            draw_labels: !args.no_labels,
            label_font_size_px: label_font_px,
            hull_stroke_px: (radius_px * 0.8).max(1.0),
            hull_fill_alpha: args.hull_fill_alpha,
        },
    );

    let base = args.out.to_string();
    if !args.no_svg {
        let svg_path = format!("{base}.plot.svg");
        fs::write(&svg_path, svg.as_bytes())?;
        info!("Wrote {svg_path}");
    }

    // PNG + PDF share the same SVG string and are independent; render
    // concurrently to hide resvg/svg2pdf parse latency.
    let png_task = (!args.no_png).then(|| format!("{base}.plot.png"));
    let pdf_task = (!args.no_pdf).then(|| format!("{base}.plot.pdf"));
    let (png_res, pdf_res) = rayon::join(
        || match &png_task {
            Some(p) => render_png(&svg, width_px, height_px, p).map(|()| Some(p.clone())),
            None => Ok(None),
        },
        || match &pdf_task {
            Some(p) => render_pdf(&svg, p).map(|()| Some(p.clone())),
            None => Ok(None),
        },
    );
    if let Some(p) = png_res? {
        info!("Wrote {p}");
    }
    if let Some(p) = pdf_res? {
        info!("Wrote {p}");
    }

    Ok(())
}

/// Data→pixel mapping with y-axis flipped (larger data-y → higher on
/// screen). Used for both hull vertices and label anchors so they align
/// exactly with the rasterized PNG layers.
#[must_use]
fn to_pixel(p: Pt, bounds: &DataBounds, ext: Extent) -> (f32, f32) {
    let (w, h) = (ext.w as f32, ext.h as f32);
    let tx = (p.0 - bounds.xmin) / (bounds.xmax - bounds.xmin) * w;
    let ty = h - (p.1 - bounds.ymin) / (bounds.ymax - bounds.ymin) * h;
    (tx, ty)
}

fn read_cell_coords(path: &str) -> anyhow::Result<FxHashMap<String, Vec<f32>>> {
    let MatWithNames { cols, mat, .. } = Mat::from_parquet(path)?;
    let mut by_name: FxHashMap<String, Vec<f32>> = FxHashMap::default();
    for (j, name) in cols.iter().enumerate() {
        let col: Vec<f32> = (0..mat.nrows()).map(|i| mat[(i, j)]).collect();
        by_name.insert(name.to_string(), col);
    }
    Ok(by_name)
}

/// Argmax over cells × K topic proportions.
fn argmax_topics(path: &str, n_cells_expected: usize) -> anyhow::Result<Vec<i64>> {
    let MatWithNames { mat, .. } = Mat::from_parquet(path)?;
    if mat.nrows() != n_cells_expected {
        anyhow::bail!(
            "topics parquet has {} rows but cell_coords has {}",
            mat.nrows(),
            n_cells_expected
        );
    }
    let mut out = Vec::with_capacity(mat.nrows());
    for i in 0..mat.nrows() {
        let mut best_j = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for j in 0..mat.ncols() {
            let v = mat[(i, j)];
            if v > best_v {
                best_v = v;
                best_j = j;
            }
        }
        out.push(best_j as i64);
    }
    Ok(out)
}

/// Parse a two-column TSV of `group_id<TAB>display_name`. Blank lines
/// and lines starting with `#` are skipped.
fn read_labels_tsv(path: &str) -> anyhow::Result<FxHashMap<i64, String>> {
    let content = fs::read_to_string(Path::new(path))?;
    let mut map: FxHashMap<i64, String> = FxHashMap::default();
    for (line_no, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let id_str = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("labels TSV line {}: missing id", line_no + 1))?;
        let name = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("labels TSV line {}: missing name", line_no + 1))?;
        let id: i64 = id_str
            .parse()
            .map_err(|e| anyhow::anyhow!("labels TSV line {}: bad id: {e}", line_no + 1))?;
        map.insert(id, name.to_string());
    }
    Ok(map)
}

/// Render the SVG to a flattened PNG via usvg + resvg. Loads system
/// fonts so vector `<text>` labels get rasterized (resvg's default
/// options ship an empty font database).
fn render_png(svg: &str, w: u32, h: u32, out: &str) -> anyhow::Result<()> {
    let mut options = usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = usvg::Tree::from_str(svg, &options)
        .map_err(|e| anyhow::anyhow!("usvg parse failed: {e}"))?;
    let mut pixmap = tiny_skia::Pixmap::new(w, h)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc failed ({w}x{h})"))?;
    resvg::render(&tree, tiny_skia::Transform::identity(), &mut pixmap.as_mut());
    pixmap
        .save_png(out)
        .map_err(|e| anyhow::anyhow!("PNG save failed: {e}"))?;
    Ok(())
}

/// Render the SVG to a true-vector PDF via svg2pdf.
fn render_pdf(svg: &str, out: &str) -> anyhow::Result<()> {
    let mut options = svg2pdf::usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = svg2pdf::usvg::Tree::from_str(svg, &options)
        .map_err(|e| anyhow::anyhow!("svg2pdf/usvg parse failed: {e}"))?;
    let pdf = svg2pdf::to_pdf(&tree, svg2pdf::ConversionOptions::default(), svg2pdf::PageOptions::default())
        .map_err(|e| anyhow::anyhow!("svg2pdf render failed: {e}"))?;
    fs::write(out, &pdf)?;
    Ok(())
}
