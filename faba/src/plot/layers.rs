//! Raster layer construction for `faba plot`: cells, the trajectory backbone
//! (principal curves or the MST), velocity arrows, and the trajectory nodes.
//!
//! Each builder answers "what is drawn"; every colour, stroke, alpha and size it
//! uses comes from [`super::style`].

use anyhow::{Context, Result};
use log::{info, warn};
use std::collections::HashMap;

use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use plot_utils::palette::{self, Rgb};
use plot_utils::rasterize::{
    rasterize_arrow_layer_png, rasterize_group_png, rasterize_per_point_png,
    rasterize_segment_layer_png, DataBounds, Extent, PointShape,
};
use plot_utils::svg_emit::TopicLayer;
use plot_utils::RadiusSpec;

use super::io::{read_str_columns, Curves, NodePositions, TrajectoryEdges, UNASSIGNED};
use super::style::{
    arrow_head_len, arrow_stroke, curve_alpha, curve_base, curve_stroke, frac_to_bin,
    node_label_dy, node_radius, root_star_radius, tree_alpha, tree_base, tree_stroke, CurveWidth,
    Seg, ARROW, ARROW_ALPHA, CURVE, CURVE_BINS, INK, MIN_VELOCITY_FLUX, TREE_BINS, VELOCITY_FIELD,
};
use super::svg::{emit_halo_text, emit_star, LegendEntry};
use super::{NodeLabels, PlotArgs};

/// The cell point cloud and the canvas it lands on — everything both cell-colour
/// builders need to turn a cell index into a pixel.
///
/// Bundled because the two builders otherwise take the same six parameters each,
/// and a six-parameter prefix is a struct wearing a disguise.
pub(super) struct CellScatter<'a> {
    /// Cell barcodes, in row order; the annotation tables join on these.
    pub names: &'a [Box<str>],
    pub x: &'a [f32],
    pub y: &'a [f32],
    pub bounds: &'a DataBounds,
    pub ext: Extent,
    pub radius_px: f32,
}

impl CellScatter<'_> {
    /// Every cell with a finite coordinate, as `(row index, pixel)`. Cells the
    /// layout could not place are silently absent — they have nowhere to be drawn.
    fn pixels(&self) -> impl Iterator<Item = (usize, (f32, f32))> + '_ {
        (0..self.names.len()).filter_map(|i| {
            let (x, y) = (self.x[i], self.y[i]);
            (x.is_finite() && y.is_finite()).then(|| (i, self.bounds.to_pixel((x, y), self.ext)))
        })
    }
}

/// A raster-only [`TopicLayer`]: no hull, no vector label. Every layer this module
/// builds is one of these — the labels are spliced in as vector SVG afterwards.
fn raster_layer(png: Vec<u8>, color: Rgb) -> TopicLayer {
    TopicLayer {
        label: String::new(),
        png,
        hull_px: Vec::new(),
        label_xy_px: (f32::NAN, f32::NAN),
        color,
    }
}

/// Rasterize one layer per non-empty stroke bin, widths and alphas rising with the
/// bin index. Shared by the principal curves and the MST tree, which differ only in
/// how they compute the bin and in their stroke/alpha ramps.
fn rasterize_binned_segments(
    bins: &[Vec<Seg>],
    ext: Extent,
    stroke: impl Fn(usize) -> f32,
    alpha: impl Fn(usize) -> f32,
) -> Result<Vec<TopicLayer>> {
    bins.iter()
        .enumerate()
        .filter(|(_, segs)| !segs.is_empty())
        .map(|(k, segs)| {
            let png = rasterize_segment_layer_png(segs, ext, stroke(k), CURVE, alpha(k))?;
            Ok(raster_layer(png, CURVE))
        })
        .collect()
}

/// Opacity of a MIXED (uncommitted, ≥2-label soft set) cell relative to a confident one:
/// it is drawn in its leading fate's colour but faded, so the differentiation lean shows
/// without manufacturing a false-confident solid call.
const MIXED_ALPHA_FACTOR: f32 = 0.3;

/// Per-cell `(display label, confident?)`. The display label is the LEADING fate, plus its
/// runner-up as `leading/second` when a second type is also significant — read from the
/// support-ordered `label_ranked` (largest share first), capped at the top 2. Confident = a
/// single significant label. `--label-argmax` forces the leading fate alone, every cell solid.
/// Falls back to the argmax `coarse_label` (no runner-up) when `label_ranked` is absent.
fn read_cell_labels(annot_path: &str, argmax: bool) -> HashMap<Box<str>, (Box<str>, bool)> {
    let mut out = HashMap::new();
    match read_str_columns(annot_path, &["cell", "label_ranked"])
        .or_else(|_| read_str_columns(annot_path, &["cell", "coarse_label"]))
    {
        Ok(cols) => {
            for (c, raw) in cols[0].iter().zip(cols[1].iter()) {
                let s = raw.trim();
                if s.is_empty() || s.eq_ignore_ascii_case(UNASSIGNED) {
                    out.insert(c.clone(), (Box::from(UNASSIGNED), true));
                    continue;
                }
                let mut it = s.split('/');
                let lead = it.next().unwrap_or(s);
                let second = it.next().filter(|_| !argmax);
                let label: Box<str> = match second {
                    Some(sec) => format!("{lead}/{sec}").into_boxed_str(),
                    None => Box::from(lead),
                };
                out.insert(c.clone(), (label, second.is_none()));
            }
        }
        Err(e) => warn!("no per-cell annotation ({e}); colouring every cell as '{UNASSIGNED}'"),
    }
    out
}

/// The colour-carrying LEADING fate of a display label (`leading/second` → `leading`).
fn leading_fate(display: &str) -> &str {
    display.split('/').next().unwrap_or(display)
}

/// Build one `rasterize_group_png` layer per coarse cell type and return the
/// legend (type → colour). Cells missing an annotation → `unassigned`.
pub(super) fn build_celltype_layers(
    prefix: &str,
    cells: &CellScatter,
    args: &PlotArgs,
    font_px: f32,
    layers: &mut Vec<TopicLayer>,
) -> Result<(Vec<LegendEntry>, String)> {
    let annot_path = format!("{prefix}.lineage_annot.annot.parquet");
    // Each cell gets its LEADING fate (argmax `coarse_label`) and whether the soft `label_set`
    // makes that call CONFIDENT (single label) or MIXED (≥2 labels — uncommitted, between
    // fates). Mixed cells are drawn in the leading fate's colour but faded, so the
    // differentiation lean shows without a false-confident solid call. `--label-argmax`
    // forces every cell confident (the old winner-take-all view).
    let label_by_cell = read_cell_labels(&annot_path, args.label_argmax);

    let unassigned: Box<str> = Box::from(UNASSIGNED);
    // One lookup per cell → (display label, confident?), then split for the passes below.
    let (display, confident): (Vec<Box<str>>, Vec<bool>) = cells
        .names
        .iter()
        .map(|c| {
            label_by_cell
                .get(c)
                .cloned()
                .unwrap_or_else(|| (unassigned.clone(), true))
        })
        .unzip();

    // Palette keyed by LEADING fate, so `HSPC` and `HSPC/Erythroid` share one hue. Sorted;
    // `unassigned` dropped unless --show-unassigned.
    let mut leads: Vec<Box<str>> = display.iter().map(|d| Box::from(leading_fate(d))).collect();
    leads.sort_unstable();
    leads.dedup();
    if let Some(p) = leads
        .iter()
        .position(|t| t.as_ref().eq_ignore_ascii_case(UNASSIGNED))
    {
        let u = leads.remove(p);
        if args.show_unassigned {
            leads.push(u);
        }
    }
    let pal = palette::resolve(&args.palette, leads.len());
    let type_color: HashMap<Box<str>, Rgb> = leads
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), palette::color(&pal, i)))
        .collect();
    let color_of = |d: &str| {
        type_color
            .get(leading_fate(d))
            .copied()
            .unwrap_or((150, 150, 150))
    };

    // Bucket pixels by DISPLAY label (`leading/second`), split confident vs mixed; also gather
    // pixels per LEADING fate for the centroid labels. `unassigned` dropped unless shown.
    let mut conf_pts: HashMap<Box<str>, Vec<(f32, f32)>> = HashMap::new();
    let mut mixed_pts: HashMap<Box<str>, Vec<(f32, f32)>> = HashMap::new();
    let mut fate_pts: HashMap<Box<str>, Vec<(f32, f32)>> = HashMap::new();
    for (i, px) in cells.pixels() {
        let d = &display[i];
        if !args.show_unassigned && d.as_ref().eq_ignore_ascii_case(UNASSIGNED) {
            continue;
        }
        let bucket = if confident[i] {
            &mut conf_pts
        } else {
            &mut mixed_pts
        };
        bucket.entry(d.clone()).or_default().push(px);
        fate_pts
            .entry(Box::from(leading_fate(d)))
            .or_default()
            .push(px);
    }

    // Legend order: display labels sorted (groups by leading-fate prefix, e.g. `HSPC` then
    // `HSPC/Erythroid`).
    let mut labels: Vec<Box<str>> = conf_pts.keys().chain(mixed_pts.keys()).cloned().collect();
    labels.sort_unstable();
    labels.dedup();

    // Pass 1: faded MIXED cells. Pass 2: solid CONFIDENT cells on top, + one legend entry each.
    let fade = args.alpha * MIXED_ALPHA_FACTOR;
    for d in &labels {
        if let Some(pts) = mixed_pts.get(d) {
            let png = rasterize_group_png(
                pts,
                cells.ext,
                RadiusSpec::Scalar(cells.radius_px),
                color_of(d),
                fade,
                PointShape::Circle,
            )?;
            layers.push(raster_layer(png, color_of(d)));
        }
    }
    let mut legend = Vec::with_capacity(labels.len());
    for d in &labels {
        let color = color_of(d);
        if let Some(pts) = conf_pts.get(d) {
            let png = rasterize_group_png(
                pts,
                cells.ext,
                RadiusSpec::Scalar(cells.radius_px),
                color,
                args.alpha,
                PointShape::Circle,
            )?;
            layers.push(raster_layer(png, color));
        }
        legend.push(LegendEntry {
            label: d.to_string(),
            color,
        });
    }
    let n_conf: usize = conf_pts.values().map(Vec::len).sum();
    let n_mixed: usize = mixed_pts.values().map(Vec::len).sum();
    info!(
        "coloured {} label(s) over {} lead fate(s): {n_conf} confident (solid) + {n_mixed} mixed (faded, leading/second)",
        labels.len(),
        leads.len()
    );

    // A haloed name at each LEADING fate's centroid (marginal median of its cells) — lets the
    // plot be read without decoding the many leading/second hues. `--no-celltype-labels` off.
    let mut overlay = String::new();
    if !args.no_celltype_labels {
        overlay.push_str("  <g id=\"celltype-labels\">\n");
        for fate in &leads {
            // Anchor on the CONFIDENT core (a tight cluster of single-label cells) so a scattered
            // fate's label lands on its body, not in the empty space between its lobes. Fall back
            // to all cells of the fate when it has no confident members.
            let pts = conf_pts
                .get(fate)
                .filter(|p| !p.is_empty())
                .or_else(|| fate_pts.get(fate));
            if let Some((mx, my)) = pts.and_then(|p| medoid_xy(p)) {
                emit_halo_text(&mut overlay, mx, my, font_px, INK, fate);
            }
        }
        overlay.push_str("  </g>\n");
        info!("labeled {} cell-type centroid(s)", leads.len());
    }
    Ok((legend, overlay))
}

/// Medoid of a pixel cloud — the member point minimising summed distance to the others, so
/// the label lands **on a real cell** in the type's densest lobe rather than in the empty space
/// a marginal median or centroid can fall into for a multimodal type. `None` when empty.
///
/// Exact medoid is `O(n²)`; the candidate set is capped (every `step`-th point, ≤ `CAP`) while
/// the cost is still summed over ALL points, so it stays `O(CAP·n)` on a large type.
fn medoid_xy(pts: &[(f32, f32)]) -> Option<(f32, f32)> {
    if pts.is_empty() {
        return None;
    }
    const CAP: usize = 512;
    let step = (pts.len() / CAP).max(1);
    pts.iter()
        .step_by(step)
        .map(|&p| {
            let cost: f64 = pts
                .iter()
                .map(|q| (f64::from(p.0 - q.0).powi(2) + f64::from(p.1 - q.1).powi(2)).sqrt())
                .sum();
            (p, cost)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(p, _)| p)
}

/// Build a single continuous pseudotime layer (blue→red ramp) and return the
/// `(min, max)` pseudotime for the colourbar labels.
pub(super) fn build_pseudotime_layer(
    prefix: &str,
    cells: &CellScatter,
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
    for c in cells.names {
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

    let mut pts_px: Vec<(f32, f32)> = Vec::with_capacity(cells.names.len());
    let mut colors: Vec<Rgb> = Vec::with_capacity(cells.names.len());
    for (i, px) in cells.pixels() {
        let v = value_by_cell
            .get(&cells.names[i])
            .copied()
            .unwrap_or(f32::NAN);
        if !v.is_finite() {
            continue;
        }
        pts_px.push(px);
        colors.push(palette::sample_blue_red(
            args.pseudotime_scale.frac(v, lo, span),
        ));
    }
    let png = rasterize_per_point_png(
        &pts_px,
        &colors,
        cells.ext,
        cells.radius_px,
        args.alpha,
        PointShape::Circle,
    )?;
    layers.push(raster_layer(png, (0, 0, 0)));
    info!(
        "coloured {} cells by pseudotime [{lo:.3}, {hi:.3}]",
        pts_px.len()
    );
    Ok((lo, hi))
}

///////////////////////////////////////////////////////
// Trajectory overlays (curves, nodes, root, labels) //
///////////////////////////////////////////////////////

/// Rasterize the principal curves as dark segment layers, one per stroke bin, with
/// each lineage's width scaled by the cell mass it carries. Empty when no lineage
/// has a finite segment.
pub(super) fn build_curve_layer(
    curves: &Curves,
    bounds: &DataBounds,
    ext: Extent,
    radius_px: f32,
    scale: CurveWidth,
) -> Result<Vec<TopicLayer>> {
    let mut bins: Vec<Vec<Seg>> = vec![Vec::new(); CURVE_BINS];
    let max_u = curves.usage.values().copied().fold(0.0f32, f32::max);
    for (lin, pts) in &curves.by_lineage {
        // `frac` ∈ [0,1]: this lineage's share of the busiest one, on the chosen
        // scale. With no weights every curve draws at the widest stroke.
        let frac = match (max_u > 0.0, curves.usage.get(lin)) {
            (true, Some(&u)) => scale.normalize(u, max_u),
            _ => 1.0,
        };
        let bin = frac_to_bin(frac, CURVE_BINS);
        for w in pts.windows(2) {
            let finite = |p: &(f32, f32)| p.0.is_finite() && p.1.is_finite();
            if finite(&w[0]) && finite(&w[1]) {
                bins[bin].push((bounds.to_pixel(w[0], ext), bounds.to_pixel(w[1], ext)));
            }
        }
    }
    if bins.iter().all(Vec::is_empty) {
        return Ok(Vec::new());
    }
    if curves.usage.is_empty() {
        warn!("no cell-lineage weights; drawing every principal curve at one width");
    } else {
        info!(
            "principal curves: {} lineage(s), stroke ∝ {scale:?}(cell usage), busiest carries {max_u:.0} cells",
            curves.n_lineages()
        );
    }

    let base = curve_base(radius_px);
    rasterize_binned_segments(&bins, ext, |k| curve_stroke(base, k), curve_alpha)
}

/// Rasterize the MST **once**, with stroke width encoding how many root→leaf paths
/// traverse each edge.
///
/// `curves_2d` holds one smooth principal curve per lineage, and every lineage
/// starts at the root — so the trunk is redrawn once per lineage. On a cord-blood
/// run that is 97 near-identical polylines stacked on the same pixels, which
/// saturates into an opaque mat and hides the branching it is supposed to show.
///
/// The tree is the union of those paths with no duplication: 199 edges, each drawn
/// once. Traversal count — how many of the 97 paths cross an edge — is exactly the
/// information the overplotting was trying (and failing) to convey, so it becomes
/// stroke width instead.
pub(super) fn build_tree_layer(
    nodes: &NodePositions,
    graph: &TrajectoryEdges,
    ext: Extent,
    radius_px: f32,
) -> Result<Vec<TopicLayer>> {
    let max_t = graph.traversals.values().copied().max().unwrap_or(1).max(1);

    // Width bins on the log-scaled traversal count: the trunk reads heavy, the twigs
    // stay legible. An edge with no recorded traversal draws at the base width.
    let mut bins: Vec<Vec<Seg>> = vec![Vec::new(); TREE_BINS];
    for e in &graph.edges {
        let (Some(&pa), Some(&pb)) = (nodes.by_index.get(&e.from), nodes.by_index.get(&e.to))
        else {
            continue;
        };
        let t = *graph
            .traversals
            .get(&(e.from.min(e.to), e.from.max(e.to)))
            .unwrap_or(&1);
        let frac = (f64::from(t).ln() / f64::from(max_t).max(2.0).ln()).clamp(0.0, 1.0);
        bins[frac_to_bin(frac as f32, TREE_BINS)].push((pa, pb));
    }
    if bins.iter().all(Vec::is_empty) {
        return Ok(Vec::new());
    }
    info!(
        "trajectory tree: {} edges drawn once (max {max_t} of the paths share one edge)",
        graph.edges.len()
    );

    let base = tree_base(radius_px);
    rasterize_binned_segments(&bins, ext, |k| tree_stroke(base, k), tree_alpha)
}

/// Velocity-grounded direction arrows, drawn independently of the backbone.
///
/// `velocity_flux` is the mean node velocity δ projected onto the edge, and δ is
/// gem's increment fit to the **unspliced** edges with the spliced θ held fixed —
/// so the arrow states what the spliced/unspliced contrast says, not what the graph
/// topology looks like. Its sign gives `directed_from → directed_to`; its magnitude
/// is the confidence. Below the cut the orientation is a coin flip (on a cord-blood
/// run, 54 of 199 edges sit under the absolute floor alone), and those edges are
/// left undirected rather than handed an arrowhead they have not earned.
pub(super) fn build_velocity_arrows(
    nodes: &NodePositions,
    graph: &TrajectoryEdges,
    ext: Extent,
    radius_px: f32,
) -> Result<Option<TopicLayer>> {
    // Arrowheads assert a direction, so only draw them where the velocity says one
    // exists: the top quartile of |flux|, and never below the absolute floor. Edge
    // traversal count is a topology statistic and must NOT gate the arrows.
    let mut mag: Vec<f32> = graph
        .edges
        .iter()
        .map(|e| e.velocity_flux.abs())
        .filter(|x| x.is_finite())
        .collect();
    mag.sort_by(f32::total_cmp);
    let cut = mag
        .get(mag.len().saturating_mul(3) / 4)
        .copied()
        .unwrap_or(0.0)
        .max(MIN_VELOCITY_FLUX);

    let arrows: Vec<Seg> = graph
        .edges
        .iter()
        .filter(|e| e.velocity_flux.abs() >= cut)
        .filter_map(|e| {
            let pf = nodes.by_index.get(&e.directed_from)?;
            let pt = nodes.by_index.get(&e.directed_to)?;
            Some((*pf, *pt))
        })
        .collect();
    info!(
        "velocity arrows: {} of {} edges directed (|flux| ≥ {cut:.3}); the rest left \
         undirected — the spliced/unspliced contrast is too weak to call",
        arrows.len(),
        graph.edges.len()
    );
    if arrows.is_empty() {
        return Ok(None);
    }
    // Scale the head to the CANVAS, not the cell radius: `--width/--height/--dpi`
    // move the figure's scale independently of `--point-size`.
    let head_len = arrow_head_len(ext);
    let png = rasterize_arrow_layer_png(
        &arrows,
        ext,
        arrow_stroke(head_len, radius_px),
        head_len,
        ARROW,
        ARROW_ALPHA,
    )?;
    Ok(Some(raster_layer(png, ARROW)))
}

/// scVelo-style cell-velocity field: gridded arrows (already projected to pixel
/// space) drawn as a muted-blue overlay, distinct from the backbone/trajectory
/// arrows. Reflects the local δ flow, independent of the trajectory topology.
pub(super) fn build_grid_velocity_arrows(
    arrows: &[Seg],
    ext: Extent,
    radius_px: f32,
) -> Result<Option<TopicLayer>> {
    if arrows.is_empty() {
        return Ok(None);
    }
    let head_len = arrow_head_len(ext);
    info!(
        "velocity-grid arrows: {} gridded cell-velocity vectors",
        arrows.len()
    );
    let png = rasterize_arrow_layer_png(
        arrows,
        ext,
        arrow_stroke(head_len, radius_px),
        head_len,
        VELOCITY_FIELD,
        ARROW_ALPHA,
    )?;
    Ok(Some(raster_layer(png, VELOCITY_FIELD)))
}

/// Build the trajectory-node point layer (dark dots) plus the vector overlay
/// carrying: a red star at the root node and a haloed `cell_type` label at each
/// non-`Cycling_Progenitor` node.
pub(super) fn build_nodes(
    prefix: &str,
    nodes: &NodePositions,
    ext: Extent,
    radius_px: f32,
    font_px: f32,
    mode: NodeLabels,
) -> Result<(TopicLayer, String)> {
    let pos_by_node = &nodes.by_name;

    let node_r = node_radius(radius_px);
    let png = rasterize_group_png(
        &nodes.pts_px,
        ext,
        RadiusSpec::Scalar(node_r),
        INK,
        1.0,
        PointShape::Circle,
    )?;
    let layer = raster_layer(png, INK);

    // Labels + root star from the trajectory annotation.
    let traj_path = format!("{prefix}.trajectory_annotation.parquet");
    let mut overlay = String::new();
    match read_str_columns(&traj_path, &["node", "role", "cell_type"]) {
        Ok(cols) => {
            let node_col = &cols[0];
            let role_col = &cols[1];
            let type_col = &cols[2];
            overlay.push_str("  <g id=\"trajectory-labels\">\n");
            let keep = select_labeled_nodes(node_col, role_col, type_col, pos_by_node, mode);
            let mut n_labeled = 0usize;
            for i in 0..node_col.len() {
                let Some(&(px, py)) = pos_by_node.get(&node_col[i]) else {
                    continue;
                };
                if role_col[i].as_ref() == "root" {
                    emit_star(&mut overlay, px, py, root_star_radius(node_r));
                }
                if keep.contains(&i) {
                    // Nudge the label just above the node marker.
                    emit_halo_text(
                        &mut overlay,
                        px,
                        py - node_label_dy(node_r, font_px),
                        font_px,
                        INK,
                        type_col[i].as_ref(),
                    );
                    n_labeled += 1;
                }
            }
            overlay.push_str("  </g>\n");
            info!(
                "labeled {n_labeled} of {} trajectory node(s) [--label-nodes {mode:?}]",
                node_col.len()
            );
        }
        Err(e) => warn!("no trajectory annotation ({e}); nodes drawn without labels"),
    }
    Ok((layer, overlay))
}

/////////////////////////////////////////////////////////////
// Vector SVG helpers (spliced on top of the raster stack) //
/////////////////////////////////////////////////////////////

/// Indices of the trajectory nodes that should carry a `cell_type` label.
///
/// The root is always included (it is the one node whose identity the reader must
/// know). Nodes with no call — an empty type, or `unassigned` — are never labeled:
/// printing "unassigned" on the figure states an absence as if it were a finding.
///
/// For [`NodeLabels::PerType`] each called type contributes exactly **one**
/// representative: the **medoid** of that type's nodes — the node minimising total
/// distance to its siblings — restricted to the terminal nodes when the type has
/// any, since a terminal is where that lineage actually ends. A medoid rather than
/// a centroid so the label lands *on* a real node, and rather than "first
/// encountered" so it sits in the middle of the type's territory instead of on a
/// stray outlier.
fn select_labeled_nodes(
    node_col: &[Box<str>],
    role_col: &[Box<str>],
    type_col: &[Box<str>],
    pos_by_node: &HashMap<Box<str>, (f32, f32)>,
    mode: NodeLabels,
) -> std::collections::HashSet<usize> {
    use std::collections::HashSet;
    let mut keep: HashSet<usize> = HashSet::new();
    if mode == NodeLabels::None {
        return keep;
    }
    let called = |i: usize| {
        let t = type_col[i].as_ref();
        !t.is_empty() && !t.eq_ignore_ascii_case(UNASSIGNED)
    };
    let is_root = |i: usize| role_col[i].as_ref() == "root";
    let is_terminal = |i: usize| role_col[i].as_ref() == "terminal";

    // The root always names itself, in every mode that draws labels at all.
    keep.extend((0..node_col.len()).filter(|&i| is_root(i) && called(i)));

    match mode {
        NodeLabels::None | NodeLabels::Root => {}
        NodeLabels::Terminal => {
            keep.extend((0..node_col.len()).filter(|&i| is_terminal(i) && called(i)))
        }
        NodeLabels::PerType => {
            // The root already names its own type; a second representative for it
            // would print the same string twice, right next to the star.
            let root_types: HashSet<&str> = keep.iter().map(|&i| type_col[i].as_ref()).collect();
            let mut by_type: HashMap<&str, Vec<usize>> = HashMap::new();
            for i in (0..node_col.len()).filter(|&i| called(i)) {
                let t = type_col[i].as_ref();
                if !root_types.contains(t) {
                    by_type.entry(t).or_default().push(i);
                }
            }
            for (_, members) in by_type {
                // A terminal is the most informative place to name a lineage's fate;
                // fall back to every node of the type when it has no terminal.
                let pool: Vec<usize> = match members
                    .iter()
                    .copied()
                    .filter(|&i| is_terminal(i))
                    .collect::<Vec<_>>()
                {
                    t if !t.is_empty() => t,
                    _ => members,
                };
                if let Some(rep) = medoid(&pool, node_col, pos_by_node) {
                    keep.insert(rep);
                }
            }
        }
    }
    keep
}

/// The member of `pool` minimising the summed Euclidean distance to the others —
/// a real node near the group's centre. `None` when no member has a position.
fn medoid(
    pool: &[usize],
    node_col: &[Box<str>],
    pos_by_node: &HashMap<Box<str>, (f32, f32)>,
) -> Option<usize> {
    let pts: Vec<(usize, (f32, f32))> = pool
        .iter()
        .filter_map(|&i| pos_by_node.get(&node_col[i]).map(|&p| (i, p)))
        .collect();
    let (first, _) = *pts.first()?;
    if pts.len() == 1 {
        return Some(first);
    }
    pts.iter()
        .map(|&(i, p)| {
            let cost: f32 = pts
                .iter()
                .map(|&(_, q)| ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt())
                .sum();
            (i, cost)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests;
