//! Graphic parameters for `faba plot` — every colour, stroke, alpha and size
//! decision lives here, so `run_plot.rs` only decides *what* to draw.
//!
//! The rule this file exists to enforce: nothing below reaches for the data, and
//! nothing in `run_plot.rs` hard-codes a pixel or an RGB triple. Tuning the figure
//! means editing one file; changing what the figure *means* means editing the other.
//!
//! Sizes are expressed relative to something that moves with the figure — the
//! canvas's short side, or the cell-point radius (itself derived from
//! `--point-size` and `--dpi`) — never as bare pixels, so `--width` / `--height` /
//! `--dpi` scale the whole composition instead of just the raster layers.

use clap::ValueEnum;
use plot_utils::palette::Rgb;
use plot_utils::rasterize::Extent;

/// A pixel-space line segment, as the rasterizer wants it.
pub type Seg = ((f32, f32), (f32, f32));

/// Points → pixels conversion base (72 pt per inch), shared with `senna plot`.
pub const PT_PER_INCH: f32 = 72.0;

/// Which of `n_bins` equal-width stroke bins a `[0, 1]` intensity falls into.
///
/// One rasterized layer carries a single stroke width, so a continuous width would
/// mean one layer per line. Binning is how a continuous statistic — lineage usage,
/// edge traversal count — becomes a small stack of layers.
pub fn frac_to_bin(frac: f32, n_bins: usize) -> usize {
    let top = n_bins.saturating_sub(1);
    ((frac.clamp(0.0, 1.0) * n_bins as f32).floor() as usize).min(top)
}

/////////////
// Palette //
/////////////

/// Canvas background — R's `gray90`. The SVG itself is transparent, and the PNG
/// rasterizer flattens transparency to black; painting the canvas explicitly keeps
/// the figure light, which the white legend box and near-black label text were
/// already designed for.
pub const BACKGROUND: Rgb = (229, 229, 229);

/// Near-black ink for the trajectory node markers.
pub const INK: Rgb = (35, 35, 40);

/// Mid grey for the trajectory backbone: dark enough to read against
/// [`BACKGROUND`], light enough not to be confused with the [`INK`] nodes.
pub const CURVE: Rgb = (108, 108, 118);

/// Direction arrowheads — darker than [`CURVE`] so a velocity call reads as an
/// assertion, distinct from the backbone it sits on.
pub const ARROW: Rgb = (58, 58, 70);

/// Red star at the root node.
pub const ROOT_RED: Rgb = (214, 39, 40);

//////////////////////
// Backbone: curves //
//////////////////////

/// Stroke bins for the principal curves, keyed to [`CurveWidth`]-scaled usage.
pub const CURVE_BINS: usize = 4;

/// Base stroke for a principal curve, thinner than a cell marker.
pub fn curve_base(radius_px: f32) -> f32 {
    (radius_px * 0.42).max(0.8)
}

/// Stroke for curve bin `k`: `base × 1.0 … base × 3.4`, so a well-used lineage
/// reads as a trunk and a one-cell twig stays visible but recessive.
pub fn curve_stroke(base: f32, k: usize) -> f32 {
    base * (1.0 + 0.8 * k as f32)
}

/// Alpha for curve bin `k`, `0.45 … 0.80`.
pub fn curve_alpha(k: usize) -> f32 {
    0.45 + 0.117 * k as f32
}

/// How a principal curve's stroke width encodes its cell usage.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
#[clap(rename_all = "kebab-case")]
pub enum CurveWidth {
    /// `sqrt(usage)` — see [`CurveWidth::normalize`]. (default)
    #[default]
    Sqrt,
    /// `log1p(usage)` — flatter still; use when one lineage dwarfs the rest.
    Log,
}

impl CurveWidth {
    /// `usage` mapped to `[0, 1]` relative to the busiest lineage.
    ///
    /// Hard `argmax` ownership is the tempting input and is wrong here: it spans
    /// 1159× on a K=200 run, so minor curves would render at 3% of the trunk's
    /// width and vanish. Soft cell mass spans ~14×, which `sqrt` maps onto a usable
    /// `0.27..1.0`.
    pub fn normalize(self, u: f32, max_u: f32) -> f32 {
        match self {
            Self::Sqrt => (u.max(0.0).sqrt() / max_u.sqrt()).clamp(0.0, 1.0),
            Self::Log => (u.max(0.0).ln_1p() / max_u.ln_1p()).clamp(0.0, 1.0),
        }
    }
}

////////////////////////
// Backbone: MST tree //
////////////////////////

/// Stroke bins for the MST edges, keyed to log-scaled path-traversal count.
pub const TREE_BINS: usize = 3;

/// Base stroke for a tree edge.
pub fn tree_base(radius_px: f32) -> f32 {
    (radius_px * 0.40).max(0.7)
}

/// Stroke for tree bin `k`: the trunk reads heavy, the twigs stay legible.
pub fn tree_stroke(base: f32, k: usize) -> f32 {
    base * (1.0 + 1.7 * k as f32)
}

/// Alpha for tree bin `k`.
pub fn tree_alpha(k: usize) -> f32 {
    0.42 + 0.22 * k as f32
}

/////////////////////
// Velocity arrows //
/////////////////////

/// Arrowhead length as a fraction of the canvas's short side, so the heads keep
/// their visual weight across `--width` / `--height` / `--dpi`. Scaling off the
/// cell radius instead would shrink them to specks on a large canvas.
const ARROW_HEAD_FRAC: f32 = 0.011;

/// Arrowhead length in pixels for this canvas.
pub fn arrow_head_len(ext: Extent) -> f32 {
    (ext.w.min(ext.h) as f32 * ARROW_HEAD_FRAC).clamp(7.0, 30.0)
}

/// Shaft stroke for an arrow with the given head length.
pub fn arrow_stroke(head_len: f32, radius_px: f32) -> f32 {
    (head_len * 0.20).max(radius_px * 0.5)
}

/// Absolute floor on `|velocity_flux|` below which an edge's orientation is noise
/// and no arrowhead is drawn. The flux is a projected mean velocity, so this is a
/// magnitude in embedding units per node.
pub const MIN_VELOCITY_FLUX: f32 = 0.01;

/// Opacity of the arrow layer.
pub const ARROW_ALPHA: f32 = 0.95;

///////////
// Nodes //
///////////

/// Radius of a trajectory-node marker.
pub fn node_radius(radius_px: f32) -> f32 {
    (radius_px * 1.6).max(2.5)
}

/// Root star radius, relative to the node marker.
pub fn root_star_radius(node_r: f32) -> f32 {
    node_r * 2.4
}

/// Vertical offset that lifts a node's label clear of its marker.
pub fn node_label_dy(node_r: f32, font_px: f32) -> f32 {
    node_r + font_px * 0.6
}
