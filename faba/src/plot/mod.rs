//! `faba plot` — render the outputs of `faba lineage` into a
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
//! selective node labels (see `--label-nodes`), the root star, and a legend / colourbar on top. The stacked SVG
//! is then written as `{out}.plot.pdf` (vector, via `svg2pdf`; the default) and,
//! opt-in, `{out}.plot.png` (resvg) / `{out}.plot.svg`. The point cloud stays a
//! raster layer, so the PDF is a hybrid: vector text over a raster scatter at `--dpi`.
//!
//! Every colour, stroke, alpha and size lives in [`style`]; this module decides only
//! *what* to draw. Adding a graphic constant here rather than there is a smell.

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use log::{info, warn};
use std::path::Path;

use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use plot_utils::palette::Palette;
use plot_utils::rasterize::{DataBounds, Extent};
use plot_utils::render::{render_pdf, render_png};
use plot_utils::svg_emit::{emit_svg, flatten_raster_layers, SvgOpts, TopicLayer};

mod io;
mod layers;
pub mod style;
mod svg;

use io::{
    col_index, load_curves, load_node_positions, load_trajectory_edges, load_velocity_grid, Curves,
};
use layers::{
    build_celltype_layers, build_curve_layer, build_grid_velocity_arrows, build_nodes,
    build_pseudotime_layer, build_tree_layer, build_velocity_arrows, CellScatter,
};
use style::{CurveWidth, BACKGROUND, PT_PER_INCH};
use svg::{emit_colourbar, emit_legend};

/// Which trajectory nodes carry a `cell_type` label.
///
/// There are as many MST nodes as `faba lineage --n-centroids` asked for (200 by
/// default), and labeling each one prints the same handful of type names over and
/// over — on a cord-blood run, `Late_Erythroid` 48 times and `EoBasoMast_Precursor`
/// 38, stacked into an unreadable mat. The label's job is to name each *type* once,
/// not to restate every node's call.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
#[clap(rename_all = "kebab-case")]
pub enum NodeLabels {
    /// One label per called cell type, on its most-differentiated node
    /// (terminal > internal). The root is always labeled. (default)
    #[default]
    PerType,
    /// One label on every terminal node, plus the root.
    Terminal,
    /// The root node only.
    Root,
    /// No node labels; the cell legend still names the types.
    None,
}

/// Above this many lineages the per-lineage principal curves overplot into an
/// opaque mat, so `--trajectory auto` falls back to the tree. At 16 lineages the
/// curves read cleanly; at 97 they do not. See [`Trajectory`] for why the count
/// grows the way it does.
const AUTO_CURVE_MAX_LINEAGES: usize = 24;

/// How the trajectory backbone is drawn.
///
/// There is only ever ONE backbone: the MST over the `--n-centroids` node
/// centroids. Slingshot fits one smooth curve per *lineage*, and a lineage is a
/// root→leaf path, so the curve count is `leaves − 1` (98 leaves → 97 curves on a
/// K=200 cord-blood run) and every one of them redraws the shared trunk.
/// Slingshot's own figures show 2–4 curves because it runs over ~5–20 cell
/// *clusters*; `faba lineage` defaults to `K = min(cells/10, 200)`, a fine mesh.
/// Lowering K to get Slingshot-like curves costs cell-type resolution — at K=12 the
/// HSC_MPP nodes vanish and the root falls on EoBasoMast — so above
/// [`AUTO_CURVE_MAX_LINEAGES`] we draw the one backbone, which is what Monocle's
/// principal graph shows.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
#[clap(rename_all = "kebab-case")]
pub enum Trajectory {
    /// Slingshot curves when there are few enough lineages to read (≤ 24),
    /// otherwise the single MST backbone. (default)
    #[default]
    Auto,
    /// The Slingshot principal curves, one per lineage.
    Curves,
    /// The MST drawn once, stroke width ∝ how many root→leaf paths cross each
    /// edge. No overplotting — use when the curve count gets large.
    Tree,
    /// No backbone at all — cells + the velocity field only. Drops the MST edges
    /// AND the node dots, root star and labels (the nodes are a `--n-centroids`
    /// artefact that wobbles run-to-run).
    None,
}

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

/// Non-linear remap of the pseudotime → colour ramp, so a few late-time outliers do not
/// compress the whole progenitor bulk into one end of the spectrum.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
#[clap(rename_all = "kebab-case")]
pub enum PseudotimeScale {
    /// Linear in pseudotime.
    Linear,
    /// `√` — mild spread of the crowded low end. (default)
    #[default]
    Sqrt,
    /// `log10(1 + ·)` — stronger spread of the low end.
    Log10,
}

impl PseudotimeScale {
    /// Map a pseudotime `v ∈ [lo, hi]` to a ramp position in `[0, 1]` under the scale.
    pub fn frac(self, v: f32, lo: f32, span: f32) -> f32 {
        let x = ((v - lo) / span).clamp(0.0, 1.0); // linear position first
        match self {
            Self::Linear => x,
            Self::Sqrt => x.sqrt(),
            // span-normalised log so the endpoints stay 0 and 1 regardless of magnitude.
            Self::Log10 => (x * 9.0 + 1.0).log10(), // log10(1)=0 … log10(10)=1
        }
    }
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

    #[arg(
        long = "pseudotime-scale",
        value_enum,
        default_value_t = PseudotimeScale::Sqrt,
        help = "Non-linear remap of the --color-by pseudotime ramp (default sqrt) so late-time outliers don't dominate the spectrum"
    )]
    pub pseudotime_scale: PseudotimeScale,

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
        long = "trajectory",
        value_enum,
        default_value_t = Trajectory::Auto,
        help = "Backbone: auto (default; curves if few lineages, else the one MST tree)",
        long_help = "How the trajectory backbone is drawn.\n\
                     There is only ONE backbone (the MST);\n\
                     Slingshot draws one curve per root→leaf path through it,\n\
                     so the curve count is leaves-1.\n\
                     Direction is ALWAYS shown as velocity arrows (from `velocity_flux`),\n\
                     independent of this choice.\n\
                     \n\
                       auto (default) → the Slingshot curves\n\
                         when the run has ≤ 24 lineages, otherwise the tree.\n\
                         Lineage count grows with --n-centroids\n\
                         (K=40 → 16 lineages, K=200 → 97).\n\
                       tree → the MST drawn ONCE,\n\
                         stroke width proportional to how many root→leaf paths traverse each edge.\n\
                         `curves_2d` holds one principal curve per lineage\n\
                         and they all share the trunk,\n\
                         so on a 97-lineage run the trunk is drawn 97 times\n\
                         and saturates into an opaque mat;\n\
                         the tree carries the same information without it.\n\
                       curves → the smooth per-lineage principal curves\n\
                         (legible for a handful of lineages).\n\
                       none   → no backbone at all: cells + the velocity field only,\n\
                         no MST edges, node dots, root star or labels."
    )]
    pub trajectory: Trajectory,

    #[arg(
        long = "curve-width",
        value_enum,
        default_value_t = CurveWidth::Sqrt,
        help = "Scale principal-curve stroke by cell usage: sqrt (default) or log"
    )]
    pub curve_width: CurveWidth,

    #[arg(
        long = "label-nodes",
        value_enum,
        default_value_t = NodeLabels::PerType,
        help = "Which trajectory nodes get a cell-type label (default: one per type)",
        long_help = "Which trajectory nodes carry a cell_type label.\n\
                     `faba lineage` emits one MST node per --n-centroids (200 by default),\n\
                     so labeling every node repeats each type name dozens of times\n\
                     and the labels collide into an unreadable mat.\n\
                     \n\
                       per-type (default) → one label per called type,\n\
                         placed on its most-differentiated node (terminal preferred).\n\
                         Root always labeled.\n\
                       terminal → label every terminal node, plus the root.\n\
                       root     → the root only.\n\
                       none     → no node labels; the cell legend still names the types."
    )]
    pub label_nodes: NodeLabels,

    #[arg(
        long,
        default_value_t = false,
        help = "Colour every cell solid by its argmax `coarse_label`, ignoring the soft-set confidence fade",
        long_help = "By default cells are coloured by their LEADING fate (the argmax `coarse_label`) with the \
                     soft `label_set` setting opacity: a size-1 set (confident single call) draws SOLID, a \
                     size-≥2 set (an uncommitted cell between fates) draws FADED in the same leading-fate \
                     colour, and an empty set is `unassigned`. So the differentiation lean still shows in \
                     colour, but a transcriptionally-central progenitor is not given a false-confident solid \
                     call. Pass this flag to draw every cell solid by the argmax `coarse_label` (the old \
                     winner-take-all view, with no confidence fade)."
    )]
    pub label_argmax: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Draw `unassigned` cells too (default: dropped, so only called/mixed cells are shown)"
    )]
    pub show_unassigned: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Do not print a cell-type name at each type's centroid (default: label centroids, so the plot reads without decoding many leading/second colours)"
    )]
    pub no_celltype_labels: bool,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Scale factor on the velocity-field arrow length (default 0.5; 1.0 = the raw grid displacement)"
    )]
    pub velocity_scale: f32,

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

    /////////////////////////////////
    // cell coordinates (PHATE 2D) //
    /////////////////////////////////
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

    ////////////////////////////////////
    // extents / bounds / pixel scale //
    ////////////////////////////////////
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

    //////////////////////////////////////////////////////////////////////
    // cell colour layers (celltype or pseudotime) + the on-top overlay //
    //////////////////////////////////////////////////////////////////////
    let cells = CellScatter {
        names: &cell_names,
        x: &cx,
        y: &cy,
        bounds: &bounds,
        ext,
        radius_px,
    };
    let mut layers: Vec<TopicLayer> = Vec::new();
    // Haloed cell-type names at each type's centroid (celltype colouring only).
    let mut celltype_labels = String::new();
    // Extra vector SVG spliced in just before </svg> (legend or colourbar).
    let side_overlay = match args.color_by {
        ColorBy::Celltype => {
            let (legend, labels) =
                build_celltype_layers(prefix, &cells, args, font_px, &mut layers)?;
            celltype_labels = labels;
            emit_legend(&legend, font_px)
        }
        ColorBy::Pseudotime => {
            let (lo, hi) = build_pseudotime_layer(prefix, &cells, args, &mut layers)?;
            emit_colourbar(width_px, height_px, font_px, lo, hi)
        }
    };

    ///////////////////////////////////////////////////////////////
    // trajectory backbone: the MST once, or a curve per lineage //
    ///////////////////////////////////////////////////////////////
    // The curves are needed to *decide* `Auto` (their count is what overplots), so
    // load them before resolving rather than counting the file twice.
    let curves = match args.trajectory {
        Trajectory::Tree | Trajectory::None => None,
        _ => load_curves(prefix)
            .inspect_err(|e| warn!("principal curves unavailable ({e})"))
            .ok(),
    };
    let backbone = resolve_backbone(args.trajectory, curves.as_ref());

    if let (Trajectory::Curves, Some(curves)) = (backbone, &curves) {
        match build_curve_layer(curves, &bounds, ext, radius_px, args.curve_width) {
            Ok(curve_layers) if !curve_layers.is_empty() => layers.extend(curve_layers),
            Ok(_) => warn!("no principal-curve segments to draw"),
            Err(e) => warn!("curves overlay skipped: {e}"),
        }
    }

    // Node positions feed the tree, the arrows and the node markers alike; the edges
    // feed the first two. Read each once — a failure takes down everything that
    // needs it, which is every consumer of the same missing file.
    let nodes = load_node_positions(prefix, &bounds, ext)
        .inspect_err(|e| warn!("trajectory nodes unavailable ({e}); drawing cells only"))
        .ok();
    let graph = (nodes.is_some() && !matches!(backbone, Trajectory::None))
        .then(|| load_trajectory_edges(prefix))
        .and_then(|r| {
            r.inspect_err(|e| warn!("trajectory edges unavailable ({e})"))
                .ok()
        });

    if let (Some(nodes), Some(graph)) = (&nodes, &graph) {
        if matches!(backbone, Trajectory::Tree) {
            match build_tree_layer(nodes, graph, ext, radius_px) {
                Ok(tree_layers) if !tree_layers.is_empty() => layers.extend(tree_layers),
                Ok(_) => warn!("no trajectory-tree edges to draw"),
                Err(e) => warn!("tree overlay skipped: {e}"),
            }
        }
        // Direction is a property of the velocity field, not of the backbone we chose
        // to draw, so the arrows ride on top of whichever one it is.
        match build_velocity_arrows(nodes, graph, ext, radius_px) {
            Ok(Some(layer)) => layers.push(layer),
            Ok(None) => warn!("no edge had enough velocity flux to direct"),
            Err(e) => warn!("velocity arrows skipped: {e}"),
        }
    }

    // scVelo-style cell-velocity field (t-UMAP layout only): gridded local-flow arrows,
    // independent of the trajectory backbone. Absent file (e.g. PHATE) → no overlay.
    match load_velocity_grid(prefix, &bounds, ext, args.velocity_scale) {
        // `build_grid_velocity_arrows` already returns `Ok(None)` for an empty field.
        Ok(segs) => match build_grid_velocity_arrows(&segs, ext, radius_px) {
            Ok(Some(layer)) => layers.push(layer),
            Ok(None) => {}
            Err(e) => warn!("velocity-grid arrows skipped: {e}"),
        },
        Err(e) => warn!("velocity-grid arrows unavailable: {e}"),
    }

    //////////////////////////////////////////////////////////////
    // trajectory nodes: the dark point layer, plus labels/root //
    //////////////////////////////////////////////////////////////
    // `--trajectory none` drops the WHOLE backbone — the edges AND the node dots,
    // root star and labels — leaving cells + the velocity field. The MST nodes are a
    // `--n-centroids` artefact (their count and placement wobble run-to-run); once
    // the trajectory is off they are clutter, not signal.
    let node_overlay = match (&nodes, matches!(backbone, Trajectory::None)) {
        (Some(nodes), false) => {
            match build_nodes(prefix, nodes, ext, radius_px, font_px, args.label_nodes) {
                Ok((layer, overlay)) => {
                    layers.push(layer);
                    overlay
                }
                Err(e) => {
                    warn!("nodes overlay skipped: {e}");
                    String::new()
                }
            }
        }
        _ => String::new(),
    };

    anyhow::ensure!(!layers.is_empty(), "nothing to plot");

    //////////////////////////////////////////////////////////////////
    // assemble SVG: base raster stack, then splice vector overlays //
    //////////////////////////////////////////////////////////////////
    let opts = SvgOpts {
        background: Some(BACKGROUND),
        width_px,
        height_px,
        label_font_size_px: font_px,
        ..Default::default()
    };
    let overlay = format!("{node_overlay}{celltype_labels}{side_overlay}");
    let splice = |svg: String| match overlay.is_empty() {
        true => svg,
        false => svg.replacen("</svg>", &format!("{overlay}</svg>"), 1),
    };

    // One raster layer per cell type (15+ on a cord-blood run), each a full-canvas
    // RGBA PNG. Stacked, the PDF carries that many image streams and opens slowly;
    // composited, it carries one — 798 KB → 585 KB on a cord-blood figure. Blending
    // happens in the same order either way, but doing it once in tiny_skia rather
    // than K times in the PDF/PNG renderer rounds premultiplied alpha once: ~1% of
    // pixels shift by 1–2/255. Invisible, and worth the single image stream.
    let svg = splice(emit_svg(
        &flatten_raster_layers(&layers, width_px, height_px)?,
        &opts,
    ));

    // SVG is opt-in (the intermediate source); PDF is the default; PNG is opt-in.
    // It keeps the UNflattened stack so Illustrator / Inkscape can still toggle the
    // per-type `<g>` groups — the whole reason to ask for SVG.
    if args.svg {
        let svg_path = format!("{out}.plot.svg");
        let layered = splice(emit_svg(&layers, &opts));
        std::fs::write(&svg_path, layered.as_bytes())
            .with_context(|| format!("writing {svg_path}"))?;
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

/// Turn `--trajectory` into the backbone actually drawn: [`Trajectory::Auto`] picks
/// curves or the tree by lineage count, and an explicit curve request warns when it
/// is about to overplot. Unreadable curves (`None`) always mean the tree.
fn resolve_backbone(requested: Trajectory, curves: Option<&Curves>) -> Trajectory {
    let n = curves.map(Curves::n_lineages);
    match (requested, n) {
        (Trajectory::Auto, Some(n)) if n <= AUTO_CURVE_MAX_LINEAGES => {
            info!("trajectory auto: {n} lineage(s) ≤ {AUTO_CURVE_MAX_LINEAGES} → principal curves");
            Trajectory::Curves
        }
        (Trajectory::Auto, Some(n)) => {
            info!("trajectory auto: {n} lineage(s) would redraw the trunk {n}× → tree");
            Trajectory::Tree
        }
        (Trajectory::Auto, None) => Trajectory::Tree,
        (Trajectory::Curves, Some(n)) => {
            if n > AUTO_CURVE_MAX_LINEAGES {
                warn!(
                    "{n} lineages all start at the root, so the trunk is drawn {n}× and \
                     will saturate — use `--trajectory tree` (or lower `--n-centroids`)"
                );
            }
            Trajectory::Curves
        }
        (Trajectory::Curves, None) => Trajectory::Tree,
        (other, _) => other,
    }
}
