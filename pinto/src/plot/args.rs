//! CLI args for `pinto plot`.
//!
//! Auto-discovers pinto output files from a prefix (or its
//! `{prefix}.pinto.json` manifest, preferred), optionally reading
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
//!   = diverging blue↔red on per-edge co-detection minus the per-pair
//!   edge mean, plus thin per-community CC convex hulls.

use clap::{Args, ValueEnum};
use plot_utils::palette::Palette;
use plot_utils::rasterize::PointShape;

/// What the per-edge arrow color encodes in `pinto plot` LR overlays.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum LrColorMode {
    /// log((R+1)/(L+1)) on diverging red↔blue ramp (default).
    LogRatio,
    /// Outgoing/incoming/internal/external vs. community hull.
    Direction,
    /// Pair-centered co-detection deviation: is this contact one where
    /// both genes are detected, relative to how often that happens
    /// on this pair's contacts. The same co-detection notion as
    /// `lra --edge-scores-only`, computed per edge for display
    /// (plot does not read that table).
    Coexpr,
}

/// How the LR overlay renders its arrows.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum LrArrows {
    /// Every style: the field-lines figure under the bare pair name,
    /// plus `.edges` and `.grid` variants beside it (default).
    All,
    /// One arrow per contact.
    Edges,
    /// One arrow per grid bin: mean direction of the bin's cells,
    /// length scaled by their coherence; incoherent bins draw nothing.
    Grid,
    /// streamplot-style field lines through the smoothed per-cell
    /// gradient, over the propensity-coloured tissue; lines stop where
    /// the cells disagree.
    FieldLines,
}

#[derive(Args, Debug)]
pub struct SrtPlotArgs {
    #[arg(
        long,
        short = 'f',
        required = true,
        help = "Input prefix or JSON metadata file",
        long_help = "Input prefix, or a JSON metadata file.\n\
                     A path ending in .json or .pinto.json is read as metadata.\n\
                     Anything else is treated as a prefix,\n\
                     and the {prefix}.*.parquet files are discovered from it."
    )]
    pub from: Box<str>,

    #[arg(
        long,
        help = "Expression data file (.h5/.zarr); comma-separated for multi-sample runs",
        long_help = "Expression data file, .h5 or .zarr.\n\
                     It is required only when --top-markers > 0. Comma-separate several files for multi-sample runs.",
        value_delimiter = ','
    )]
    pub data: Option<Vec<Box<str>>>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (defaults to --from)",
        long_help = "Output prefix; it defaults to --from.\n\
                     Figures land in {out}.plots/{kind}/{level}.*.pdf,\n\
                     one subdirectory per kind. The run also writes {out}.plot.manifest.json."
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
        help = "Max aspect ratio (h/w clamp)",
        long_help = "Maximum aspect ratio, clamping height over width.\n\
                     Bounds outside the clamp are inflated symmetrically.",
        hide = true
    )]
    pub max_aspect: f32,

    // ─── Scatter / point aesthetics ──────────────────────────────────────
    #[arg(long, default_value_t = 1.6, help = "Base point size (pt)")]
    pub point_size: f32,

    #[arg(
        long,
        default_value_t = 3.0,
        help = "Max radius multiplier for propensity/expression size mapping",
        long_help = "Max radius multiplier for propensity / expression size mapping (base_size * scale at p99).",
        hide = true
    )]
    pub size_scale: f32,

    #[arg(long, value_enum, default_value_t = PointShape::Hexagon, help = "Marker shape", hide = true)]
    pub point_shape: PointShape,

    #[arg(long, value_enum, help = "Qualitative palette (default: auto by K)")]
    pub palette: Option<Palette>,

    // ─── Mesh plot ────────────────────────────────────────────────────────
    #[arg(
        long,
        default_value_t = 0.5,
        help = "Mesh edge stroke width (pt)",
        hide = true
    )]
    pub mesh_stroke: f32,

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
        help = "Log-scale color bins for the marker-gene heatmap plot",
        hide = true
    )]
    pub heat_bins: usize,

    #[arg(
        long,
        default_value_t = 0.02,
        help = "Percentile clip for expression standardization (2 → p02/p98).\n\
                Clamps outliers.",
        hide = true
    )]
    pub expr_clip: f32,

    #[arg(
        long,
        default_value_t = 0.02,
        help = "Min fraction of core cells with non-zero expression for a marker plot",
        long_help = "Detection floor for rendering a marker plot.\n\
                     It is the fraction of core cells with non-zero expression.\n\
                     The default of 0.02 means 2%.\n\
                     This skips sparse genes whose heatmap is mostly empty.",
        hide = true
    )]
    pub marker_min_frac: f32,

    // ─── Partitioning ────────────────────────────────────────────────────
    #[arg(
        long,
        default_value_t = 100,
        help = "Skip cores (batches × data files) with fewer than N cells",
        hide = true
    )]
    pub min_core_cells: usize,

    #[arg(
        long,
        default_value_t = 0.005,
        help = "Percentile clip for coordinate bounds (0.005 → p0.5/p99.5)",
        long_help = "Percentile clip for coordinate bounds.\n\
                     0.005 clips to p0.5 and p99.5.\n\
                     It stops outlier cells from stretching the view.\n\
                     Pass 0 to use the raw min and max.",
        hide = true
    )]
    pub coord_clip: f32,

    #[arg(
        long,
        default_value = "all",
        help = "Which levels to plot: `all` | `final` | `draft` | comma-list",
        long_help = "Which levels to plot. Accepts `all`, `final`, `draft`, or a comma-list.\n\
                     A comma-list looks like `final,L0,L2,draft`."
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
        help = "Render high-entropy cells with their neighborhoods",
        long_help = "Render high-entropy cells together with their neighborhoods.\n\
                     This needs an `entropy` column in the propensity parquet.\n\
                     Runs after 2026-04-25 carry one."
    )]
    pub show_interfaces: bool,

    #[arg(
        long,
        default_value_t = 0.95,
        help = "Quantile threshold for high-entropy focal cells (0.95 → top 5%)",
        long_help = "Quantile threshold for picking high-entropy focal cells.\n\
                     It applies within each core.\n\
                     0.95 keeps the top 5%.",
        hide = true
    )]
    pub entropy_quantile: f32,

    #[arg(
        long,
        default_value_t = 2,
        help = "Neighborhood depth from each focal cell (1 = direct; 2 = 2-hop)",
        long_help = "Neighborhood depth from each focal cell. 1 takes direct neighbours only.\n\
                     2 takes two hops, and is the default.",
        hide = true
    )]
    pub neighborhood_hops: u8,

    #[arg(
        long,
        default_value_t = 5,
        help = "Top-N marker genes per neighbor community in interface panel legends",
        hide = true
    )]
    pub interface_top_genes: usize,

    #[arg(
        long,
        default_value_t = 200,
        help = "Cap on focal cells rendered per (level, core); top-N by entropy kept",
        long_help = "Cap on focal cells rendered per (level, core). When more qualify,\n\
                     top-N by entropy are kept.",
        hide = true
    )]
    pub max_interface_cells: usize,

    // ─── LR-activity spatial overlay ────────────────────────────────────
    #[arg(
        long,
        help = "Path to a `pinto lr-activity` JSON sidecar.",
        long_help = "Path to a `pinto lr-activity` JSON sidecar. If omitted,\n\
                     {prefix}.pinto.json is consulted instead.\n\
                     Its `outputs.lr_activity` field supplies the path. With neither present,\n\
                     the overlays are skipped silently.\n\
                     \n\
                     Every LR figure draws only the sidecar's SIGNIFICANT rows:\n\
                     FWER below the lr-activity run's threshold AND a positive\n\
                     restandardized z, meaning the pair is active in that stratum\n\
                     specifically, not tissue-wide. A run with strong but ubiquitous\n\
                     pairs can therefore have FWER hits yet render nothing;\n\
                     re-run lr-activity with --json-fwer-threshold to widen the set.\n\
                     \n\
                     One overlay PDF is written per core and significant pair.\n\
                     --lr-top-pairs caps that by |z| within each stratum."
    )]
    pub lr_activity_json: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 10,
        help = "Per-stratum cap on LR pairs rendered (top-N by |z|)",
        long_help = "Per-stratum cap on the significant LR pairs rendered.\n\
                     Pairs are ranked by |z| within each (batch, community).\n\
                     Single-batch runs collapse that to per-community.",
        hide = true
    )]
    pub lr_top_pairs: usize,

    #[arg(
        long,
        help = "Keep homotypic LR pairs (L == R, e.g. CADM3-CADM3) in the overlay",
        long_help = "Keep homotypic LR pairs, where L == R, such as CADM3-CADM3.\n\
                     They are dropped by default.\n\
                     Homotypic adhesion pairs tend to dominate the top of the list.\n\
                     That crowds out heterotypic signalling.",
        hide = true
    )]
    pub lr_keep_homotypic: bool,

    #[arg(long, help = "Skip rendering the LR-activity spatial overlays.")]
    pub no_lr_overlay: bool,

    #[arg(
        long,
        default_value_t = 0.9,
        help = "Propensity threshold above which a cell is `committed` (interior).",
        long_help = "Commitment threshold on the argmax community propensity.\n\
                     Cells at or above it count as tissue interior.\n\
                     They are dropped from the LR-overlay focal pool.\n\
                     Lower values widen the boundary belt.\n\
                     Higher values keep only the most uncommitted cells.",
        hide = true
    )]
    pub lr_commit_threshold: f32,

    #[arg(
        long,
        default_value_t = 2,
        help = "Belt width (hops) around uncommitted cells for LR overlay focal set",
        long_help = "Belt width in graph hops around uncommitted cells.\n\
                     It sets the LR-overlay focal set. 1 takes direct neighbours only;\n\
                     2 takes two hops.",
        hide = true
    )]
    pub lr_belt_hops: u8,

    #[arg(
        long,
        default_value_t = 100,
        help = "Skip communities with fewer than this many edges (no markers or LR overlays)",
        hide = true
    )]
    pub min_edges_per_community: usize,

    #[arg(
        long,
        default_value_t = 25,
        help = "Skip communities with too few dominant cells per batch",
        long_help = "Skip a community whose dominant-cell count is below this.\n\
                     The count is taken within each batch, or core. Propensity,\n\
                     marker and LR plots are all skipped.\n\
                     This is independent of --min-edges-per-community,\n\
                     which instead applies across all batches.",
        hide = true
    )]
    pub min_cells_per_community: usize,

    #[arg(
        long,
        default_value_t = 30,
        help = "Min drawable arrows required to render an LR overlay",
        long_help = "Minimum drawable arrows for an LR overlay to be rendered.\n\
                     An arrow is an edge with non-zero L+R signal either way.\n\
                     This skips sparse pairs whose plot is a dust cloud.",
        hide = true
    )]
    pub lr_min_edges: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Cap on (ligand, receptor) pairs in the combined LR Hinton summary",
        long_help = "Cap on the pairs shown in the combined LR Hinton summary.\n\
                     That summary is `lr/summary.pdf`.\n\
                     Pairs rank by max |z| across communities.\n\
                     Rows and columns keep only those in the top-N.\n\
                     Per-community summaries are unaffected.",
        hide = true
    )]
    pub lr_summary_pairs: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "Bins for the diverging co-detection ramp (--lr-color-mode=coexpr only)",
        long_help = "Bins in the diverging blue↔red co-detection ramp on LR arrows.\n\
                     This is used only with --lr-color-mode=coexpr.",
        hide = true
    )]
    pub lr_coexpr_bins: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = LrArrows::All,
        help = "LR arrow rendering: all styles, or one of edges, grid, field-lines",
        long_help = "How the LR overlay renders its arrows.\n\
                     \n\
                     `all` (the default) writes every style per pair:\n\
                     the field-lines figure under the bare pair name,\n\
                     with `.edges` and `.grid` variants beside it.\n\
                     `edges` draws one arrow per contact.\n\
                     `grid` bins the cells' net contact directions\n\
                     and draws one arrow per bin,\n\
                     with length scaled by how much its cells agree.\n\
                     `field-lines` integrates streamplot-style lines\n\
                     through the smoothed field,\n\
                     drawn over the propensity-coloured tissue,\n\
                     with one arrowhead at each line's midpoint.\n\
                     \n\
                     The summaries are STATIC gradient views of the same\n\
                     per-contact directions, never a flow:\n\
                     the estimand is symmetric, and direction on an arrow\n\
                     is a display heuristic.\n\
                     Bins whose cells disagree draw nothing, by design.\n\
                     The direction colour mode stays per-edge;\n\
                     it has no scalar to average.",
        hide = true,
    )]
    pub lr_arrows: LrArrows,

    #[arg(
        long,
        value_enum,
        default_value_t = LrColorMode::LogRatio,
        help = "How LR-arrow colors are assigned (log-ratio | direction | coexpr)",
        long_help = "How LR-arrow colours are assigned.\n\
                     \n\
                     `log-ratio` is the default.\n\
                     It maps log((R_receiver+1)/(L_sender+1)) on a red↔blue ramp.\n\
                     Red means R≫L, so ligand-limited and activating. Blue means L≫R,\n\
                     so receptor-saturated and at plateau.\n\
                     \n\
                     `direction` colours by in, out and internal classes.\n\
                     `coexpr` colours by pair-centred co-detection deviation:\n\
                     red where both genes are detected across the contact,\n\
                     blue where at most one side is,\n\
                     centred on how often this pair co-occurs at all.\n\
                     It shares the co-detection notion\n\
                     of `lra --edge-scores-only`,\n\
                     computed per edge rather than per core.",
        hide = true,
    )]
    pub lr_color_mode: LrColorMode,
}
