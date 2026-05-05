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
    /// Per-cell celltype label from `senna annotate`'s argmax TSV
    /// (`--annotation`, defaults to `manifest.annotate.argmax`).
    Annotation,
    /// Continuous scalar from `senna pseudotime`'s `.pseudotime.parquet`
    /// (defaults to `manifest.pseudotime.pseudotime`). Cells are coloured
    /// on a sequential blue→red ramp (ColorBrewer RdYlBu reversed). When
    /// the manifest also has `pseudotime.nodes_2d` + `pseudotime.edges`,
    /// the principal-graph tree is drawn as a black overlay on top.
    Pseudotime,
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
        help = "Annotation argmax TSV from `senna annotate` (cell\\tcell_type\\tprobability). Defaults to manifest's annotate.argmax."
    )]
    pub annotation: Option<Box<str>>,

    #[arg(
        long,
        help = "Pseudotime parquet from `senna pseudotime` (cells × 1). Defaults to manifest's pseudotime.pseudotime."
    )]
    pub pseudotime: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload data when auto-running `senna layout` for a manifest missing layout.cell_coords (no-op if cell_coords already exists)"
    )]
    pub preload_data: bool,

    #[arg(
        long,
        help = "TSV mapping group_id<TAB>display_name (one per line). Missing IDs fall back to T{id}."
    )]
    pub labels: Option<Box<str>>,

    #[arg(
        long,
        help = "Drop groups with fewer than N cells (0 = keep all)",
        long_help = "Filter out small/dead groups before rendering. When unset, \
                     defaults to max(50, n_cells / 200) for --colour-by topic \
                     (kills argmax ghosts on dead topics) and 0 otherwise. \
                     Pass --min-topic-cells 0 to opt out of the auto threshold."
    )]
    pub min_topic_cells: Option<usize>,

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

    #[arg(
        long,
        default_value_t = false,
        help = "Also emit SVG (default: PDF only)"
    )]
    pub svg: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Also emit flattened PNG (default: PDF only)"
    )]
    pub png: bool,

    #[arg(long, default_value_t = false, help = "Skip PDF output")]
    pub no_pdf: bool,
}

pub fn fit_plot(args: &PlotArgs) -> anyhow::Result<()> {
    let mut resolved = resolve_inputs(args)?;
    mkdir_parent(&resolved.out)?;
    let (cell_names, coords_by_name) = read_cell_coords(&resolved.cell_coords)?;
    let x = coords_by_name
        .get("x")
        .ok_or_else(|| anyhow::anyhow!("cell_coords parquet missing 'x' column"))?;
    let y = coords_by_name
        .get("y")
        .ok_or_else(|| anyhow::anyhow!("cell_coords parquet missing 'y' column"))?;
    let n_cells = x.len();
    info!("Loaded {n_cells} cells from {}", resolved.cell_coords);

    // Auto-built id → display name map for `ColorBy::Annotation` (replaces
    // the empty default below). The `--labels` TSV, when present, layers
    // on top so users can still rename specific celltypes.
    let mut auto_label_map: FxHashMap<i64, String> = FxHashMap::default();

    // For continuous coloring (pseudotime) we keep a per-cell normalized
    // value in [0, 1] (NaN for invalid) and feed it to `sample_blue_red`
    // at rasterization time — no binning, no quantization stairs.
    // `group_ids` is still used to drive the bucketing pass that gives
    // bounds + skip-mask, but for pseudotime mode every valid cell shares
    // group 0 (single-bucket) and the float vector below is what colors
    // them.
    let mut pseudotime_norm: Option<Vec<f32>> = None;

    let group_ids = match resolved.colour_by {
        ColorBy::Cluster => {
            let ids = match coords_by_name.get("cluster") {
                Some(col) => col
                    .iter()
                    .map(|&v| if v.is_nan() || v < 0.0 { -1 } else { v as i64 })
                    .collect::<Vec<_>>(),
                None => resolve_cluster_ids_for_plot(&mut resolved, &cell_names)?,
            };
            // Label cluster groups as `C{g}` so a downstream `--labels` TSV
            // can still rename them by id, but the default isn't the
            // topic-style `T{g}`.
            for &g in &ids {
                if g >= 0 {
                    auto_label_map.entry(g).or_insert_with(|| format!("C{g}"));
                }
            }
            ids
        }
        ColorBy::PbId => coords_by_name
            .get("pb_id")
            .ok_or_else(|| anyhow::anyhow!("cell_coords missing 'pb_id' column"))?
            .iter()
            .map(|&v| if v.is_nan() { -1 } else { v as i64 })
            .collect::<Vec<_>>(),
        ColorBy::Annotation => {
            let path = resolved.annotation.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--colour-by annotation requires --annotation PATH \
                     (or `senna annotate` must have populated manifest.annotate.argmax)"
                )
            })?;
            let (ids, label_map) = argmax_annotation(path, &cell_names)?;
            auto_label_map = label_map;
            ids
        }
        ColorBy::Topic => {
            let topics_path = resolved.topics.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--colour-by topic requires --topics PATH (or manifest outputs.topics)"
                )
            })?;
            argmax_topics(topics_path, n_cells)?
        }
        ColorBy::Pseudotime => {
            let path = resolved.pseudotime.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--colour-by pseudotime requires --pseudotime PATH \
                     (or `senna pseudotime --from manifest` to populate \
                     manifest.pseudotime.pseudotime)"
                )
            })?;
            let norm = read_pseudotime_normalized(path, n_cells)?;
            let g: Vec<i64> = norm
                .iter()
                .map(|&t| if t.is_finite() { 0 } else { -1 })
                .collect();
            pseudotime_norm = Some(norm);
            g
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

    // Auto-threshold for `--colour-by topic` kills argmax ghosts on dead
    // topics. Floor of 50 protects small-N runs.
    let min_topic_cells: usize = args.min_topic_cells.unwrap_or_else(|| {
        if matches!(resolved.colour_by, ColorBy::Topic) {
            (n_cells / 200).max(50)
        } else {
            0
        }
    });
    if min_topic_cells > 0 {
        let before = pts_by_group.len();
        pts_by_group.retain(|_g, pts| pts.len() >= min_topic_cells);
        let dropped = before - pts_by_group.len();
        if dropped > 0 {
            info!("Dropped {dropped} groups with fewer than {min_topic_cells} cells");
        }
        if pts_by_group.is_empty() {
            anyhow::bail!("--min-topic-cells {min_topic_cells} dropped every group");
        }
    }

    let mut unique: Vec<i64> = pts_by_group.keys().copied().collect();
    unique.sort_unstable();
    if pseudotime_norm.is_some() {
        let n_total: usize = pts_by_group.values().map(|v| v.len()).sum();
        info!("Plotting {n_total} cells (continuous pseudotime, single-PNG layer)");
    } else {
        info!("Plotting {} groups", unique.len());
    }

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

    // Start from the auto-built map (populated for ColorBy::Annotation,
    // empty otherwise), then layer the user-supplied --labels TSV on
    // top so per-id renames win over the auto names.
    let mut label_map = auto_label_map;
    if let Some(p) = resolved.labels.as_deref() {
        for (id, name) in read_labels_tsv(p)? {
            label_map.insert(id, name);
        }
    }

    // Skip hull/trim computation when nothing needs it — hull polygons
    // are opt-in now and median is the default label position.
    let need_hull_geometry = args.hull || args.label_position == LabelPosition::HullCentroid;

    // Continuous coloring (pseudotime): emit a single PNG with per-cell
    // colors so the SVG embeds one raster layer instead of one per bin.
    // The per-cell normalized float is sampled directly through the
    // blue→red ramp — no quantization, smooth gradient.
    let mut layers: Vec<TopicLayer> = if let Some(norm) = pseudotime_norm.as_deref() {
        let style = PointStyle {
            radius_px,
            alpha: args.alpha,
            shape: args.point_shape,
        };
        vec![rasterize_continuous_layer(
            x, y, norm, &bounds, ext, &style,
        )?]
    } else {
        unique
            .par_iter()
            .map(|g| -> anyhow::Result<TopicLayer> {
                let pts = pts_by_group.get(g).expect("group present");
                let pts_px: Vec<(f32, f32)> =
                    pts.iter().map(|&p| to_pixel(p, &bounds, ext)).collect();

                // Key palette by group id, not enumerate index, so colors
                // stay aligned across every view that maps id → color.
                let color_idx = (*g).max(0) as usize;
                let color = palette::color(&palette, color_idx);
                let shape = if args.point_shape_cycle {
                    PointShape::cycle_nth(color_idx)
                } else {
                    args.point_shape
                };
                let png = rasterize_group_png(
                    &pts_px,
                    ext,
                    plot_utils::RadiusSpec::Scalar(radius_px),
                    color,
                    args.alpha,
                    shape,
                )?;

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
            .collect::<anyhow::Result<Vec<_>>>()?
    };

    // Principal-graph overlay: drawn unconditionally when both
    // pseudotime.nodes_2d and pseudotime.edges are available in the
    // manifest. Drawn last so the tree sits on top of the rasterized
    // cells.
    if let (Some(nodes_path), Some(edges_path)) = (
        resolved.principal_graph_nodes_2d.as_deref(),
        resolved.principal_graph_edges.as_deref(),
    ) {
        match build_principal_graph_layer(nodes_path, edges_path, &bounds, ext, radius_px) {
            Ok(Some(layer)) => layers.push(layer),
            Ok(None) => log::warn!(
                "principal-graph overlay skipped: nodes_2d {nodes_path} produced no finite segments"
            ),
            Err(e) => log::warn!("principal-graph overlay skipped: {e}"),
        }
    }

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
            frame_stroke_px: 0.0,
        },
    );

    let base = resolved.out.clone();
    if args.svg {
        let svg_path = format!("{base}.plot.svg");
        fs::write(&svg_path, svg.as_bytes())?;
        info!("Wrote {svg_path}");
    }

    // PNG + PDF share the same SVG string and are independent; render
    // concurrently to hide resvg/svg2pdf parse latency. Default is
    // PDF-only; SVG/PNG are opt-in.
    let png_task = args.png.then(|| format!("{base}.plot.png"));
    let pdf_task = (!args.no_pdf).then(|| format!("{base}.plot.pdf"));
    let (png_res, pdf_res) = rayon::join(
        || match &png_task {
            Some(p) => plot_utils::render_png(&svg, width_px, height_px, std::path::Path::new(p))
                .map(|()| Some(p.clone())),
            None => Ok(None),
        },
        || match &pdf_task {
            Some(p) => {
                plot_utils::render_pdf(&svg, std::path::Path::new(p)).map(|()| Some(p.clone()))
            }
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
    annotation: Option<String>,
    /// Cluster assignments parquet path (cells × 1, "cluster" column).
    /// Used by the `ColorBy::Cluster` fallback when `cell_coords` lacks
    /// a `cluster` column. Populated from `manifest.cluster.clusters`
    /// when `--from` is in effect.
    clusters: Option<String>,
    labels: Option<String>,
    pseudotime: Option<String>,
    /// Principal-graph node 2D coordinates parquet (K × 2) — drawn as a
    /// tree overlay when ColorBy::Pseudotime is in effect.
    principal_graph_nodes_2d: Option<String>,
    /// Principal-graph edges parquet (E × 3: from, to, weight).
    principal_graph_edges: Option<String>,
    out: String,
    colour_by: ColorBy,
    palette: Palette,
    /// Loaded manifest (when `--from` is in effect). The cluster fallback
    /// may write back to it (`cluster.clusters` field) and persist to
    /// `manifest_path`.
    manifest: Option<RunManifest>,
    manifest_path: Option<String>,
    manifest_dir: PathBuf,
}

fn resolve_inputs(args: &PlotArgs) -> anyhow::Result<ResolvedInputs> {
    let (mut manifest, mut manifest_dir): (Option<RunManifest>, PathBuf) = match &args.from {
        Some(p) => {
            let (m, dir) = RunManifest::load(Path::new(p.as_ref()))?;
            info!("Loaded run manifest {} (kind: {})", p, m.kind);
            (Some(m), dir)
        }
        None => (None, PathBuf::from(".")),
    };

    // Auto-layout: when the user passed --from but the manifest has no
    // layout yet, drive `senna layout phate` against it (defaults) so
    // the natural workflow `train → annotate → plot` works without an
    // explicit layout step. CLI --cell-coords still wins; only kick in
    // when both manifest and user are silent on cell_coords.
    if let Some(from_path) = args.from.as_deref() {
        let manifest_has_coords = manifest
            .as_ref()
            .and_then(|m| m.layout.cell_coords.as_deref())
            .is_some();
        if args.cell_coords.is_none() && !manifest_has_coords {
            info!(
                "manifest {} has no layout.cell_coords; running `senna layout phate` first",
                from_path
            );
            crate::postprocess::run_default_phate_layout(from_path, args.preload_data)?;
            // layout rewrote the manifest in place — reload to pick up
            // the freshly-populated layout.cell_coords / pb_coords.
            let (m, dir) = RunManifest::load(Path::new(from_path))?;
            manifest = Some(m);
            manifest_dir = dir;
        }
    }

    let resolve_opt = |s: &str| {
        run_manifest::resolve(&manifest_dir, s)
            .to_string_lossy()
            .into_owned()
    };

    // When the user asked for `--colour-by pseudotime` and the manifest
    // has the Reingold-Tilford tree layout populated by `senna pseudotime`,
    // prefer that over the generic PHATE/UMAP layout. The tree layout
    // gives a Monocle-2-style trajectory plot — guaranteed tree-shaped
    // because cells are placed directly on the principal-graph edges.
    // Explicit --cell-coords still wins over either default.
    let want_tree_layout = matches!(args.colour_by, Some(ColorBy::Pseudotime))
        || manifest
            .as_ref()
            .and_then(|m| m.defaults.colour_by.as_deref())
            .and_then(parse_colour_by)
            == Some(ColorBy::Pseudotime);
    let manifest_cell_coords = manifest.as_ref().and_then(|m| {
        if want_tree_layout {
            m.pseudotime
                .tree_cell_coords
                .as_deref()
                .or(m.layout.cell_coords.as_deref())
        } else {
            m.layout.cell_coords.as_deref()
        }
    });
    let cell_coords = args
        .cell_coords
        .as_deref()
        .map(String::from)
        .or_else(|| manifest_cell_coords.map(resolve_opt))
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

    let annotation = args.annotation.as_deref().map(String::from).or_else(|| {
        manifest
            .as_ref()
            .and_then(|m| m.annotate.argmax.as_deref())
            .map(resolve_opt)
    });

    let clusters = manifest
        .as_ref()
        .and_then(|m| m.cluster.clusters.as_deref())
        .map(resolve_opt);

    let pseudotime = args.pseudotime.as_deref().map(String::from).or_else(|| {
        manifest
            .as_ref()
            .and_then(|m| m.pseudotime.pseudotime.as_deref())
            .map(resolve_opt)
    });
    // Edges are always the same. Nodes_2d depends on whether the plot
    // is using the tree layout (pseudotime mode) or the PHATE/UMAP
    // layout — pick the matching nodes so the overlay aligns.
    let principal_graph_nodes_2d = manifest
        .as_ref()
        .and_then(|m| {
            if want_tree_layout {
                m.pseudotime
                    .tree_nodes_2d
                    .as_deref()
                    .or(m.pseudotime.nodes_2d.as_deref())
            } else {
                m.pseudotime.nodes_2d.as_deref()
            }
        })
        .map(resolve_opt);
    let principal_graph_edges = manifest
        .as_ref()
        .and_then(|m| m.pseudotime.edges.as_deref())
        .map(resolve_opt);

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

    // Colour-by precedence:
    //   1. explicit CLI flag
    //   2. manifest `defaults.colour_by`
    //   3. Annotation, if `manifest.annotate.argmax` is populated
    //   4. Cluster (existing fallback)
    // Step 3 makes annotation the natural default for the
    // train → annotate → plot workflow without forcing users to set it.
    let colour_by = args
        .colour_by
        .clone()
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|m| m.defaults.colour_by.as_deref())
                .and_then(parse_colour_by)
        })
        .or_else(|| annotation.as_ref().map(|_| ColorBy::Annotation))
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
        annotation,
        clusters,
        labels,
        pseudotime,
        principal_graph_nodes_2d,
        principal_graph_edges,
        out,
        colour_by,
        palette,
        manifest,
        manifest_path: args.from.as_deref().map(String::from),
        manifest_dir,
    })
}

fn parse_colour_by(s: &str) -> Option<ColorBy> {
    match s {
        "topic" => Some(ColorBy::Topic),
        "cluster" => Some(ColorBy::Cluster),
        "pb-id" | "pb_id" => Some(ColorBy::PbId),
        "annotation" => Some(ColorBy::Annotation),
        "pseudotime" => Some(ColorBy::Pseudotime),
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

/// Read pseudotime parquet (cells × 1) and rescale to `[0, 1]` using
/// the (min, max) of finite values. Cells with non-finite pseudotime
/// get NaN so the rasterizer skips them.
fn read_pseudotime_normalized(path: &str, n_cells_expected: usize) -> anyhow::Result<Vec<f32>> {
    let MatWithNames { mat, .. } = Mat::from_parquet(path)?;
    if mat.nrows() != n_cells_expected {
        anyhow::bail!(
            "pseudotime parquet has {} rows but cell_coords has {}",
            mat.nrows(),
            n_cells_expected
        );
    }
    if mat.ncols() < 1 {
        anyhow::bail!("pseudotime parquet has no columns");
    }
    let vals: Vec<f32> = (0..mat.nrows()).map(|i| mat[(i, 0)]).collect();
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &vals {
        if v.is_finite() {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        anyhow::bail!("pseudotime has no finite range (all NaN/constant?)");
    }
    let span = hi - lo;
    Ok(vals
        .into_iter()
        .map(|v| {
            if v.is_finite() {
                (v - lo) / span
            } else {
                f32::NAN
            }
        })
        .collect())
}

/// Per-point style shared by the rasterizers in this file.
struct PointStyle {
    radius_px: f32,
    alpha: f32,
    shape: PointShape,
}

/// Pack every cell into one PNG with per-cell colors sampled from the
/// blue→red ramp. One raster layer instead of one-per-bin keeps the
/// SVG/PDF small even at 10k+ cells.
fn rasterize_continuous_layer(
    x: &[f32],
    y: &[f32],
    norm: &[f32],
    bounds: &DataBounds,
    ext: Extent,
    style: &PointStyle,
) -> anyhow::Result<TopicLayer> {
    let n_cells = x.len();
    let mut pts_px: Vec<(f32, f32)> = Vec::with_capacity(n_cells);
    let mut colors: Vec<plot_utils::palette::Rgb> = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let t = norm[i];
        if !t.is_finite() || !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        pts_px.push(to_pixel((x[i], y[i]), bounds, ext));
        colors.push(palette::sample_blue_red(t));
    }
    let png = plot_utils::rasterize::rasterize_per_point_png(
        &pts_px,
        &colors,
        ext,
        style.radius_px,
        style.alpha,
        style.shape,
    )?;
    Ok(TopicLayer {
        label: String::new(),
        png,
        hull_px: Vec::new(),
        label_xy_px: (f32::NAN, f32::NAN),
        color: (0, 0, 0),
    })
}

/// Build a single overlay layer carrying the principal graph drawn as
/// black line segments on a transparent canvas. Returns `None` when the
/// graph has no usable (finite) edges — caller logs and skips.
fn build_principal_graph_layer(
    nodes_2d_path: &str,
    edges_path: &str,
    bounds: &DataBounds,
    ext: Extent,
    radius_px: f32,
) -> anyhow::Result<Option<TopicLayer>> {
    let MatWithNames { mat: nodes, .. } = Mat::from_parquet(nodes_2d_path)?;
    if nodes.ncols() < 2 {
        anyhow::bail!("principal-graph nodes_2d at {nodes_2d_path} has < 2 columns");
    }
    let MatWithNames {
        cols: edge_cols,
        mat: edges,
        ..
    } = Mat::from_parquet(edges_path)?;
    let from_col = edge_cols
        .iter()
        .position(|c| c.as_ref() == "from")
        .unwrap_or(0);
    let to_col = edge_cols
        .iter()
        .position(|c| c.as_ref() == "to")
        .unwrap_or(1);

    let n_nodes = nodes.nrows();
    let mut segs_px: Vec<((f32, f32), (f32, f32))> = Vec::with_capacity(edges.nrows());
    for i in 0..edges.nrows() {
        let a = edges[(i, from_col)] as i64;
        let b = edges[(i, to_col)] as i64;
        if a < 0 || b < 0 || (a as usize) >= n_nodes || (b as usize) >= n_nodes {
            continue;
        }
        let (ax, ay) = (nodes[(a as usize, 0)], nodes[(a as usize, 1)]);
        let (bx, by) = (nodes[(b as usize, 0)], nodes[(b as usize, 1)]);
        if !ax.is_finite() || !ay.is_finite() || !bx.is_finite() || !by.is_finite() {
            continue;
        }
        let p0 = to_pixel((ax, ay), bounds, ext);
        let p1 = to_pixel((bx, by), bounds, ext);
        segs_px.push((p0, p1));
    }
    if segs_px.is_empty() {
        return Ok(None);
    }
    let stroke_px = (radius_px * 1.6).max(1.0);
    let png = plot_utils::rasterize::rasterize_segment_layer_png(
        &segs_px,
        ext,
        stroke_px,
        (0, 0, 0),
        1.0,
    )?;
    Ok(Some(TopicLayer {
        label: String::new(),
        png,
        hull_px: Vec::new(),
        label_xy_px: (f32::NAN, f32::NAN),
        color: (0, 0, 0),
    }))
}

type CellCoords = (Vec<Box<str>>, FxHashMap<String, Vec<f32>>);

/// Returns `(cell_names, columns_by_name)`. Cell names are the parquet
/// row labels (in data column order), needed when matching against an
/// annotation TSV by cell name.
fn read_cell_coords(path: &str) -> anyhow::Result<CellCoords> {
    let MatWithNames { rows, cols, mat } = Mat::from_parquet(path)?;
    let mut by_name: FxHashMap<String, Vec<f32>> = FxHashMap::default();
    for (j, name) in cols.iter().enumerate() {
        let col: Vec<f32> = (0..mat.nrows()).map(|i| mat[(i, j)]).collect();
        by_name.insert(name.to_string(), col);
    }
    Ok((rows, by_name))
}

/// Argmax over cells × K topic proportions, returning the **topic ID**
/// per row (parsed from the column name), not the column position. So a
/// downstream `T5` legend swatch always means topic 5, even if the
/// columns aren't in 0..K-1 order on disk.
fn argmax_topics(path: &str, n_cells_expected: usize) -> anyhow::Result<Vec<i64>> {
    let MatWithNames { cols, mat, .. } = Mat::from_parquet(path)?;
    if mat.nrows() != n_cells_expected {
        anyhow::bail!(
            "topics parquet has {} rows but cell_coords has {}",
            mat.nrows(),
            n_cells_expected
        );
    }
    let topic_ids = crate::embed_common::try_parse_axis_ids(&cols, "T").ok_or_else(|| {
        anyhow::anyhow!(
            "topics parquet at {path} has columns that aren't topic IDs \
             (expected \"T{{c}}\" or bare integer)"
        )
    })?;
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
        out.push(topic_ids[best_j]);
    }
    Ok(out)
}

/// Cluster fallback for `ColorBy::Cluster` when the `cell_coords`
/// parquet has no `cluster` column. Mirrors `senna annotate`'s strategy:
///   1. If `manifest.cluster.clusters` is set, load that parquet.
///   2. Otherwise run Leiden against `manifest.outputs.latent`, write
///      `{out}.clusters.parquet`, and update the manifest in place.
///
/// Returns per-cell ids aligned to `cell_names` (cell_coords row order),
/// with `-1` for unassigned/missing cells.
fn resolve_cluster_ids_for_plot(
    resolved: &mut ResolvedInputs,
    cell_names: &[Box<str>],
) -> anyhow::Result<Vec<i64>> {
    let Some(manifest) = resolved.manifest.as_mut() else {
        anyhow::bail!(
            "--colour-by cluster but cell_coords has no 'cluster' column \
             and no manifest is loaded; re-run `senna layout` with --clusters \
             or pass --from <manifest>"
        );
    };

    // Path 1: manifest already has a cluster parquet.
    if let Some(path) = resolved.clusters.as_deref() {
        info!("Loading clusters from {path}");
        let (labels_usize, n_clusters) =
            crate::annotate::inputs::load_cluster_labels(path, cell_names)?;
        info!("Loaded {n_clusters} clusters from manifest.cluster.clusters");
        return Ok(usize_to_signed(&labels_usize));
    }

    // Path 2: leiden on the manifest's latent. Defaults mirror annotate.
    let leiden_args = crate::annotate::inputs::LeidenArgs {
        knn: 15,
        resolution: 1.0,
        num_clusters: None,
        min_cluster_size: 2,
        seed: Some(42),
    };
    let manifest_dir = resolved.manifest_dir.clone();
    let resolve = |rel: &str| -> String {
        run_manifest::resolve(&manifest_dir, rel)
            .to_string_lossy()
            .into_owned()
    };
    info!(
        "No 'cluster' column in cell_coords and manifest.cluster.clusters is unset; \
         running internal Leiden on the manifest latent"
    );
    let (labels_usize, n_clusters) = crate::annotate::inputs::compute_clusters_from_latent(
        manifest,
        &resolve,
        cell_names,
        &leiden_args,
    )?;
    info!("Internal Leiden produced {n_clusters} clusters");

    // Persist `{out}.clusters.parquet` next to the plot output and patch
    // the manifest so future runs (annotate/plot) reuse it.
    let parquet_path = format!("{}.clusters.parquet", resolved.out);
    write_cluster_assignments_parquet(&parquet_path, cell_names, &labels_usize)?;
    info!("Wrote {parquet_path}");

    if let Some(manifest_path) = resolved.manifest_path.as_deref() {
        let rel = run_manifest::rel_to_manifest(&resolved.manifest_dir, &parquet_path);
        manifest.cluster.clusters = Some(rel);
        manifest.save(Path::new(manifest_path))?;
    }
    resolved.clusters = Some(parquet_path);

    Ok(usize_to_signed(&labels_usize))
}

fn usize_to_signed(labels: &[usize]) -> Vec<i64> {
    labels
        .iter()
        .map(|&v| if v == usize::MAX { -1 } else { v as i64 })
        .collect()
}

fn write_cluster_assignments_parquet(
    path: &str,
    cell_names: &[Box<str>],
    labels: &[usize],
) -> anyhow::Result<()> {
    use matrix_util::traits::IoOps;
    let mut data = Mat::zeros(cell_names.len(), 1);
    for (i, &c) in labels.iter().enumerate() {
        data[(i, 0)] = if c == usize::MAX { f32::NAN } else { c as f32 };
    }
    let cols: Vec<Box<str>> = vec!["cluster".into()];
    data.to_parquet_with_names(path, (Some(cell_names), Some("cell")), Some(&cols))?;
    Ok(())
}

/// Read `senna annotate`'s argmax TSV and produce per-cell integer
/// group ids + a stable id → celltype-name label map.
///
/// Format: `cell\tcell_type\tprobability` with optional header. Cells
/// absent from the TSV map to `-1` (filtered downstream by the same
/// `g < 0` skip used for unassigned clusters / NaN pb_ids). Celltype
/// strings are sorted alphabetically before id assignment so the same
/// celltype gets the same colour across reruns and across sibling
/// commands (e.g. `plot-topic --group-by annotation`, which sorts
/// celltype panels alphabetically too — see plot/topic/mod.rs:533).
fn argmax_annotation(
    path: &str,
    cell_names: &[Box<str>],
) -> anyhow::Result<(Vec<i64>, FxHashMap<i64, String>)> {
    let content = fs::read_to_string(Path::new(path))?;
    let mut by_cell: FxHashMap<Box<str>, Box<str>> = FxHashMap::default();
    for (line_no, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split('\t');
        let cell = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("annotation TSV line {}: missing cell", line_no + 1))?;
        let label = parts.next().ok_or_else(|| {
            anyhow::anyhow!("annotation TSV line {}: missing cell_type", line_no + 1)
        })?;
        if cell == "cell" && label == "cell_type" {
            continue;
        }
        by_cell.insert(cell.into(), label.into());
    }

    // Stable id assignment: sort unique celltype strings, assign 0..N.
    let mut unique: Vec<Box<str>> = {
        let mut set: Vec<Box<str>> = by_cell.values().cloned().collect();
        set.sort_unstable();
        set.dedup();
        set
    };
    // Push "unassigned" to the very end of the colour cycle when present,
    // so the argmax-thresholded cells don't steal the leading palette
    // slot.
    if let Some(pos) = unique.iter().position(|s| s.as_ref() == "unassigned") {
        let last = unique.remove(pos);
        unique.push(last);
    }
    let name_to_id: FxHashMap<Box<str>, i64> = unique
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i as i64))
        .collect();

    let mut group_ids = Vec::with_capacity(cell_names.len());
    let mut n_missing = 0usize;
    for c in cell_names {
        match by_cell.get(c) {
            Some(label) => group_ids.push(*name_to_id.get(label).unwrap_or(&-1)),
            None => {
                group_ids.push(-1);
                n_missing += 1;
            }
        }
    }
    if n_missing > 0 {
        info!(
            "annotation: {n_missing}/{} cells absent from {path} → dropped from plot",
            cell_names.len()
        );
    }

    let label_map: FxHashMap<i64, String> = name_to_id
        .into_iter()
        .map(|(name, id)| (id, String::from(name)))
        .collect();
    Ok((group_ids, label_map))
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
