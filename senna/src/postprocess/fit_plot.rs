//! `senna plot` — publication-quality rasterized scatter with vector
//! labels over transparent background.
//!
//! Preferred invocation is `senna plot --from {prefix}.senna.json`
//! (manifest produced by `senna topic` + enriched by `senna viz`); it
//! fills in cell-coords / topics / labels / colour-by / palette from
//! the manifest's `viz{}`, `outputs{}`, and `defaults{}` sections.
//! Explicit CLI flags still override whatever the manifest provides.
//!
//! See `plot::` module docs for the overall SVG→PNG/PDF pipeline. This
//! file is the clap entry point and glue: resolves manifest + overrides
//! into a `ResolvedInputs`, buckets cells by group, dispatches per-group
//! rasterization via rayon, emits SVG, then renders PNG + PDF.

use crate::embed_common::*;
use crate::postprocess::plot::hull::{
    convex_hull, hull_centroid, median_xy, trim_outliers_by_median, Pt,
};
use crate::postprocess::plot::palette::{self, Palette};
use crate::postprocess::plot::rasterize::{rasterize_group_png, DataBounds, Extent, PointShape};
use crate::postprocess::plot::svg_emit::{emit_svg, SvgOpts, TopicLayer};
use crate::run_manifest::{self, RunManifest};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fs;
use std::path::{Path, PathBuf};

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
        short = 'f',
        help = "Run manifest JSON from `senna topic`/`itopic`/`joint-topic` (+ updated by `senna viz`)",
        long_help = "If set, fills in --cell-coords, --topics, --labels, --colour-by, \
                     and --palette from the manifest's viz/outputs/defaults sections. \
                     Any explicit CLI flag still overrides the manifest value. Paths \
                     inside the manifest are resolved relative to the manifest's own \
                     directory so you can move a run directory around freely."
    )]
    pub from: Option<Box<str>>,

    #[arg(
        long,
        short = 'c',
        help = "Cell coordinates parquet (from `senna viz`); required unless --from provides it"
    )]
    pub cell_coords: Option<Box<str>>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (defaults to the manifest's `prefix` when --from is used)",
        long_help = "Writes {out}.plot.svg, {out}.plot.png, {out}.plot.pdf."
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long = "colour-by",
        alias = "color-by",
        value_enum,
        help = "Colour source (default: manifest's `defaults.colour_by`, else `cluster`)"
    )]
    pub colour_by: Option<ColorBy>,

    #[arg(
        long,
        help = "Topic proportions parquet (cells × K); required with --colour-by topic"
    )]
    pub topics: Option<Box<str>>,

    #[arg(
        long,
        help = "TSV mapping group_id<TAB>display_name (one per line). Missing IDs fall back to T{id}."
    )]
    pub labels: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Drop groups with fewer than N cells (0 = keep all)"
    )]
    pub min_topic_cells: usize,

    #[arg(long, default_value_t = 6.0, help = "Plot width (inches)")]
    pub width: f32,

    #[arg(long, default_value_t = 6.0, help = "Plot height (inches)")]
    pub height: f32,

    #[arg(long, default_value_t = 300, help = "Output DPI (raster layers)")]
    pub dpi: u32,

    #[arg(long, default_value_t = 2.0, help = "Point size (pt)")]
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
        help = "Qualitative palette (default: manifest's `defaults.palette`, else `auto`)"
    )]
    pub palette: Option<Palette>,

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

    #[arg(
        long,
        default_value_t = false,
        help = "Draw convex hull polygons around each group (off by default: scRNA groups are rarely separable in 2D, and hulls overstate that)"
    )]
    pub hull: bool,

    #[arg(
        long,
        default_value_t = 0.95,
        help = "Fraction of closest-to-median points used for each hull (1.0 = all)",
        long_help = "Only applies when --hull is enabled. For each group, keep only\n\
                     the {coverage} fraction of points nearest the coordinate-wise\n\
                     median (Euclidean) before computing the convex hull. Strips a\n\
                     few fringe cells so one outlier can't drag the polygon across\n\
                     the plot. Set to 1.0 to use every point."
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
    let resolved = resolve_inputs(args)?;
    let coords_by_name = read_cell_coords(&resolved.cell_coords)?;
    let x = coords_by_name
        .get("x")
        .ok_or_else(|| anyhow::anyhow!("cell_coords parquet missing 'x' column"))?;
    let y = coords_by_name
        .get("y")
        .ok_or_else(|| anyhow::anyhow!("cell_coords parquet missing 'y' column"))?;
    let n_cells = x.len();
    info!("Loaded {n_cells} cells from {}", resolved.cell_coords);

    let group_ids = match resolved.colour_by {
        ColorBy::Cluster => coords_by_name
            .get("cluster")
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "--colour-by cluster but cell_coords has no 'cluster' column; \
                     re-run `senna viz` with --clusters or pick a different --colour-by"
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
            let topics_path = resolved.topics.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--colour-by topic requires --topics PATH (or manifest outputs.topics)"
                )
            })?;
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

    if args.min_topic_cells > 0 {
        let before = pts_by_group.len();
        pts_by_group.retain(|_g, pts| pts.len() >= args.min_topic_cells);
        let dropped = before - pts_by_group.len();
        if dropped > 0 {
            info!(
                "Dropped {dropped} groups with fewer than {} cells",
                args.min_topic_cells
            );
        }
        if pts_by_group.is_empty() {
            anyhow::bail!(
                "--min-topic-cells {} dropped every group",
                args.min_topic_cells
            );
        }
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
    let palette = palette::resolve(&resolved.palette, unique.len());

    let label_map = match &resolved.labels {
        Some(p) => read_labels_tsv(p)?,
        None => FxHashMap::default(),
    };

    // Skip hull/trim computation when nothing needs it — hull polygons
    // are opt-in now and median is the default label position.
    let need_hull_geometry = args.hull || args.label_position == LabelPosition::HullCentroid;

    // Per-group rasterization + hull + label placement in parallel.
    // Each group is independent and encodes its own PNG — outermost loop
    // is the right place to fan out rayon (per repo convention).
    let layers: Vec<TopicLayer> = unique
        .par_iter()
        .enumerate()
        .map(|(i, g)| -> anyhow::Result<TopicLayer> {
            let pts = pts_by_group.get(g).expect("group present");
            let pts_px: Vec<(f32, f32)> = pts.iter().map(|&p| to_pixel(p, &bounds, ext)).collect();

            let color = palette::color(&palette, i);
            let shape = if args.point_shape_cycle {
                PointShape::cycle_nth(i)
            } else {
                args.point_shape
            };
            let png = rasterize_group_png(&pts_px, ext, radius_px, color, args.alpha, shape)?;

            let (hull_px, hull_data) = if need_hull_geometry {
                let trimmed = trim_outliers_by_median(pts, args.hull_coverage);
                let h = convex_hull(&trimmed);
                let px: Vec<Pt> = h.iter().map(|&p| to_pixel(p, &bounds, ext)).collect();
                (px, h)
            } else {
                (Vec::new(), Vec::new())
            };

            let label_xy_data = match args.label_position {
                LabelPosition::Median => median_xy(pts),
                LabelPosition::HullCentroid => hull_centroid(&hull_data),
            };
            let label_xy_px = to_pixel(label_xy_data, &bounds, ext);

            let label = label_map.get(g).cloned().unwrap_or_else(|| format!("T{g}"));

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
            draw_hulls: args.hull,
            draw_labels: !args.no_labels,
            label_font_size_px: label_font_px,
            hull_stroke_px: (radius_px * 0.8).max(1.0),
            hull_fill_alpha: args.hull_fill_alpha,
        },
    );

    let base = resolved.out.clone();
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

/// Resolved inputs after merging `--from` manifest defaults with
/// explicit CLI flags. CLI wins; missing-and-not-in-manifest yields a
/// clear error. All paths here are absolute (or at least resolved
/// relative to the manifest's directory) so downstream readers don't
/// have to guess what they're relative to.
struct ResolvedInputs {
    cell_coords: String,
    topics: Option<String>,
    labels: Option<String>,
    out: String,
    colour_by: ColorBy,
    palette: Palette,
}

fn resolve_inputs(args: &PlotArgs) -> anyhow::Result<ResolvedInputs> {
    let (manifest, manifest_dir): (Option<RunManifest>, PathBuf) = match &args.from {
        Some(p) => {
            let (m, dir) = RunManifest::load(Path::new(p.as_ref()))?;
            info!("Loaded run manifest {} (kind: {})", p, m.kind);
            (Some(m), dir)
        }
        None => (None, PathBuf::from(".")),
    };

    let resolve_opt = |s: &str| {
        run_manifest::resolve(&manifest_dir, s)
            .to_string_lossy()
            .into_owned()
    };

    let cell_coords = args
        .cell_coords
        .as_deref()
        .map(String::from)
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|m| m.layout.cell_coords.as_deref())
                .map(resolve_opt)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no --cell-coords given and manifest {} has no layout.cell_coords \
                 (did you run `senna layout` against this manifest?)",
                args.from.as_deref().unwrap_or("(none)")
            )
        })?;

    let topics = args.topics.as_deref().map(String::from).or_else(|| {
        manifest
            .as_ref()
            .and_then(|m| m.outputs.latent.as_deref())
            .map(resolve_opt)
    });

    let labels = args.labels.as_deref().map(String::from).or_else(|| {
        manifest
            .as_ref()
            .and_then(|m| m.outputs.anchor_labels.as_deref())
            .map(resolve_opt)
    });

    let out = args
        .out
        .as_deref()
        .map(String::from)
        .or_else(|| manifest.as_ref().map(|m| m.prefix.clone()))
        .ok_or_else(|| anyhow::anyhow!("no --out given and no manifest prefix available"))?;

    let colour_by = args
        .colour_by
        .clone()
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|m| m.defaults.colour_by.as_deref())
                .and_then(parse_colour_by)
        })
        .unwrap_or(ColorBy::Cluster);

    let palette = args
        .palette
        .clone()
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|m| m.defaults.palette.as_deref())
                .and_then(parse_palette)
        })
        .unwrap_or(Palette::Auto);

    Ok(ResolvedInputs {
        cell_coords,
        topics,
        labels,
        out,
        colour_by,
        palette,
    })
}

fn parse_colour_by(s: &str) -> Option<ColorBy> {
    match s {
        "topic" => Some(ColorBy::Topic),
        "cluster" => Some(ColorBy::Cluster),
        "pb-id" | "pb_id" => Some(ColorBy::PbId),
        _ => None,
    }
}

fn parse_palette(s: &str) -> Option<Palette> {
    use clap::ValueEnum;
    Palette::from_str(s, true).ok()
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
fn render_pdf(svg: &str, out: &str) -> anyhow::Result<()> {
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
