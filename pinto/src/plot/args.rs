//! CLI args for `pinto plot`.
//!
//! Auto-discovers pinto output files from a prefix (or its
//! `{prefix}.metadata.json` manifest, preferred), optionally reading
//! raw expression (`--data`) for marker-gene overlays AND for per-edge
//! L→R direction inference in the LR-activity overlay. Every plot
//! dimension (width, aspect, dot size, etc.) is user-overridable, with
//! batteries-included defaults.
//!
//! Splitting strategy: plots are emitted per-batch × per-data-file
//! using the `left_batch` / `right_batch` columns already present in
//! `{prefix}.coord_pairs.parquet` when pinto was fit with multiple
//! batches or data files. Single-batch runs get a single `all` core.
//!
//! Sub-modes:
//! - Default: community / propensity-argmax / per-community heatmap /
//!   mesh / marker-gene plots (per (level, core)).
//! - `--show-interfaces`: per-cell entropy as a grayscale + size
//!   signal, plus a TSV legend with neighborhood + top-gene info.
//! - LR-activity overlay (auto when an `lr_activity.json` sidecar is
//!   linked in the metadata): per significant LR pair, a quiver of
//!   directional arrows along edges incident to a boundary cell, color
//!   = diverging blue↔red on per-edge coexpression `sqrt(L·R)` minus
//!   the per-pair edge mean, plus thin per-community CC convex hulls.

use clap::Args;
use plot_utils::palette::Palette;
use plot_utils::rasterize::PointShape;

#[derive(Args, Debug)]
pub struct SrtPlotArgs {
    #[arg(
        long,
        short = 'f',
        required = true,
        help = "Input prefix or JSON metadata file. If path ends with .json or .metadata.json, reads metadata; otherwise discovers {prefix}.*.parquet files"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        help = "Expression data file (.h5/.zarr). Required only when --top-markers > 0. Multiple files comma-separated for multi-sample runs.",
        value_delimiter = ','
    )]
    pub data: Option<Vec<Box<str>>>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (defaults to --from). Writes {out}.plots/{kind}/{level}.*.pdf (per-kind subdirs) and {out}.plot.manifest.json."
    )]
    pub out: Option<Box<str>>,

    // ─── Figure size ──────────────────────────────────────────────────────
    #[arg(long, default_value_t = 5.0, help = "Plot width per panel (inches)")]
    pub width: f32,

    #[arg(
        long,
        default_value_t = 300,
        help = "Output DPI (raster layers; vector labels stay crisp at any DPI)"
    )]
    pub dpi: u32,

    #[arg(
        long,
        default_value_t = 3.0,
        help = "Max aspect ratio (h/w clamp). Wider-than-max or taller-than-max bounds are inflated symmetrically."
    )]
    pub max_aspect: f32,

    // ─── Scatter / point aesthetics ──────────────────────────────────────
    #[arg(long, default_value_t = 1.6, help = "Base point size (pt)")]
    pub point_size: f32,

    #[arg(long, default_value_t = 1.0, help = "Point alpha (0..=1)")]
    pub alpha: f32,

    #[arg(
        long,
        default_value_t = 3.0,
        help = "Max radius multiplier for propensity / expression size mapping (base_size * scale at p99)"
    )]
    pub size_scale: f32,

    #[arg(long, value_enum, default_value_t = PointShape::Hexagon, help = "Marker shape")]
    pub point_shape: PointShape,

    #[arg(long, value_enum, help = "Qualitative palette (default: auto by K)")]
    pub palette: Option<Palette>,

    // ─── Mesh plot ────────────────────────────────────────────────────────
    #[arg(long, default_value_t = 0.5, help = "Mesh edge stroke width (pt)")]
    pub mesh_stroke: f32,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "Mesh edge alpha (0..=1; dense graphs → keep low)"
    )]
    pub mesh_alpha: f32,

    #[arg(long, help = "Skip the mesh (cell-cell edge) plot")]
    pub no_mesh: bool,

    // ─── Marker genes ─────────────────────────────────────────────────────
    #[arg(
        long,
        default_value_t = 3,
        help = "Top-N marker genes per community (0 disables marker plots)"
    )]
    pub top_markers: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "Log-scale color bins for the marker-gene heatmap plot"
    )]
    pub heat_bins: usize,

    #[arg(
        long,
        default_value_t = 0.02,
        help = "Percentile clip for expression standardization (2 → p02/p98). Clamps outliers."
    )]
    pub expr_clip: f32,

    // ─── Partitioning ────────────────────────────────────────────────────
    #[arg(
        long,
        default_value_t = 100,
        help = "Skip cores (batches × data files) with fewer than N cells"
    )]
    pub min_core_cells: usize,

    #[arg(
        long,
        default_value_t = 0.005,
        help = "Percentile clip for coordinate bounds (0.005 → p0.5/p99.5). Prevents outlier cells from stretching the view; 0 = raw min/max."
    )]
    pub coord_clip: f32,

    #[arg(
        long,
        default_value = "all",
        help = "Which levels to plot: `all` | `final` | `draft` | comma-list like `final,L0,L2,draft`"
    )]
    pub levels: Box<str>,

    // ─── Output toggles ──────────────────────────────────────────────────
    // PDF is the default output; SVG/PNG are opt-in to avoid emitting
    // three copies of every figure on every run.
    #[arg(
        long,
        help = "Also emit SVG output (off by default; PDF is the default)"
    )]
    pub svg: bool,

    #[arg(
        long,
        help = "Also emit flattened PNG output (off by default; PDF is the default)"
    )]
    pub png: bool,

    #[arg(long, help = "Skip PDF output")]
    pub no_pdf: bool,

    // ─── Interface (high-entropy neighborhood) sub-mode ──────────────────
    #[arg(
        long,
        help = "Render high-entropy cells with their neighborhoods. Requires the propensity parquet to carry an `entropy` column (post-2026-04-25 runs)."
    )]
    pub show_interfaces: bool,

    #[arg(
        long,
        default_value_t = 0.95,
        help = "Quantile threshold (within each core) used to pick high-entropy focal cells. 0.95 → top 5%."
    )]
    pub entropy_quantile: f32,

    #[arg(
        long,
        default_value_t = 2,
        help = "Neighborhood depth from each focal cell. 1 = direct neighbors only; 2 = 2-hop (default)."
    )]
    pub neighborhood_hops: u8,

    #[arg(
        long,
        default_value_t = 5,
        help = "Top-N marker genes per neighbor community shown in interface panel legends."
    )]
    pub interface_top_genes: usize,

    #[arg(
        long,
        default_value_t = 200,
        help = "Cap on focal cells rendered per (level, core). When more qualify, top-N by entropy are kept."
    )]
    pub max_interface_cells: usize,

    // ─── LR-activity spatial overlay ────────────────────────────────────
    #[arg(
        long,
        help = "Path to a `pinto lr-activity` JSON sidecar.",
        long_help = "Path to a `pinto lr-activity` JSON sidecar. If omitted,\n\
                     looks up `outputs.lr_activity` from {prefix}.metadata.json\n\
                     (or skips silently if neither is present). One LR overlay\n\
                     PDF is written per (core × significant pair, capped by\n\
                     --lr-top-pairs)."
    )]
    pub lr_activity_json: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 10,
        help = "Global cap on significant LR pairs rendered (top-N by |z|, across all strata)."
    )]
    pub lr_top_pairs: usize,

    #[arg(long, help = "Skip rendering the LR-activity spatial overlays.")]
    pub no_lr_overlay: bool,

    #[arg(
        long,
        default_value_t = 0.9,
        help = "Propensity threshold above which a cell is `committed` (interior).",
        long_help = "Cells whose argmax community propensity is ≥ this threshold\n\
                     are treated as firmly committed (tissue interior) and dropped\n\
                     from the LR-overlay focal pool. Lower → wider boundary belt;\n\
                     higher → only the most uncommitted cells qualify. Default 0.9."
    )]
    pub lr_commit_threshold: f32,

    #[arg(
        long,
        default_value_t = 8,
        help = "Min cells per community connected-component to render its hull outline."
    )]
    pub lr_hull_min_cells: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Skip communities with fewer than this many edges (no hulls, markers, or LR overlays)"
    )]
    pub min_edges_per_community: usize,

    #[arg(
        long,
        help = "Skip the per-community convex-hull outlines on LR overlays."
    )]
    pub no_lr_hulls: bool,

    #[arg(
        long,
        default_value_t = 8,
        help = "Number of bins for the diverging blue↔red coexpression color ramp on LR arrows."
    )]
    pub lr_coexpr_bins: usize,
}
