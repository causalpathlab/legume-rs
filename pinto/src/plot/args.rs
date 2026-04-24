//! CLI args for `pinto plot`.
//!
//! Auto-discovers pinto output files from a prefix (`--from`),
//! optionally reading raw expression (`--data`) for marker gene
//! overlays. Every plot dimension (width, aspect, dot size, etc.) is
//! user-overridable, with batteries-included defaults.
//!
//! Splitting strategy: plots are emitted per-batch × per-data-file
//! using the `left_batch` / `right_batch` columns already present in
//! `{prefix}.coord_pairs.parquet` when pinto was fit with multiple
//! batches or data files. Single-batch runs get a single `all` core.

use clap::Args;
use plot_utils::palette::Palette;
use plot_utils::rasterize::PointShape;

#[derive(Args, Debug)]
pub struct SrtPlotArgs {
    #[arg(
        long,
        short = 'f',
        required = true,
        help = "Input prefix: reads {prefix}.coord_pairs.parquet, {prefix}.propensity.parquet, {prefix}.link_community.parquet, {prefix}.gene_topic.parquet, and any {prefix}.L{n}.* / {prefix}.bhc.* siblings"
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
        help = "Output prefix (defaults to --from). Writes {out}.plot.{level}.core{batch}.*.pdf and {out}.plot.manifest.json."
    )]
    pub out: Option<Box<str>>,

    // ─── Figure size ──────────────────────────────────────────────────────
    #[arg(long, default_value_t = 5.0, help = "Plot width per core (inches)")]
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
    #[arg(long, default_value_t = 0.3, help = "Base point size (pt)")]
    pub point_size: f32,

    #[arg(long, default_value_t = 0.6, help = "Point alpha (0..=1)")]
    pub alpha: f32,

    #[arg(
        long,
        default_value_t = 3.0,
        help = "Max radius multiplier for propensity / expression size mapping (base_size * scale at p99)"
    )]
    pub size_scale: f32,

    #[arg(long, value_enum, default_value_t = PointShape::Circle, help = "Marker shape")]
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
        default_value_t = 5,
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
        help = "Which levels to plot: `all` | `final` | `bhc` | comma-list like `final,L0,L2,bhc`"
    )]
    pub levels: Box<str>,

    // ─── Output toggles ──────────────────────────────────────────────────
    #[arg(long, help = "Skip SVG output")]
    pub no_svg: bool,

    #[arg(long, help = "Skip flattened PNG output")]
    pub no_png: bool,

    #[arg(long, help = "Skip PDF output")]
    pub no_pdf: bool,
}
