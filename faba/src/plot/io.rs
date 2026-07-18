//! Parquet reads for `faba plot` — every file this figure depends on is loaded
//! here, so the layer builders receive plain data and never touch a path.
//!
//! Each table is read exactly once per render and handed to whichever builders
//! need it: `nodes_2d` feeds the tree, the arrows *and* the node markers;
//! `curves_2d` feeds both the lineage count that `--trajectory auto` branches on
//! and the curve layer itself.

use anyhow::{Context, Result};
use log::warn;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use plot_utils::rasterize::{DataBounds, Extent};
use std::collections::HashMap;

/// The workspace-wide non-call label, matched (never produced) by this module.
pub(super) use enrichment::UNASSIGNED_LABEL as UNASSIGNED;

/// Position of `name` in a column-name list, with a file-qualified error.
pub(super) fn col_index(cols: &[Box<str>], name: &str, path: &str) -> Result<usize> {
    cols.iter()
        .position(|c| c.as_ref() == name)
        .ok_or_else(|| anyhow::anyhow!("column '{name}' not found in {path}"))
}

/// Read the named (string / `BYTE_ARRAY`) columns from a parquet file, one
/// `Vec<Box<str>>` per requested name, in request order. Non-string cells fall
/// back to the empty string.
pub(super) fn read_str_columns(path: &str, wanted: &[&str]) -> Result<Vec<Vec<Box<str>>>> {
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

//////////////////////
// Trajectory nodes //
//////////////////////

/// `nodes_2d.parquet`, projected into pixel space once and offered in the three
/// shapes its consumers want: by MST index (edges refer to nodes as `node_{i}`),
/// by node name (the annotation table joins on it), and as a bare ordered point
/// list (the raster layer). Nodes with a non-finite coordinate appear in none of
/// them.
pub(super) struct NodePositions {
    /// `node_{i}` → pixel.
    pub by_index: HashMap<usize, (f32, f32)>,
    /// Node name → pixel.
    pub by_name: HashMap<Box<str>, (f32, f32)>,
    /// Every finite node position, in row order.
    pub pts_px: Vec<(f32, f32)>,
}

pub(super) fn load_node_positions(
    prefix: &str,
    bounds: &DataBounds,
    ext: Extent,
) -> Result<NodePositions> {
    let path = format!("{prefix}.nodes_2d.parquet");
    let nodes =
        DMatrix::<f32>::from_parquet(&path).with_context(|| format!("reading nodes {path}"))?;
    let xi = col_index(&nodes.cols, "x", &path)?;
    let yi = col_index(&nodes.cols, "y", &path)?;

    let mut by_index = HashMap::new();
    let mut by_name = HashMap::new();
    let mut pts_px = Vec::with_capacity(nodes.rows.len());
    for (r, name) in nodes.rows.iter().enumerate() {
        let (x, y) = (nodes.mat[(r, xi)], nodes.mat[(r, yi)]);
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let p = bounds.to_pixel((x, y), ext);
        pts_px.push(p);
        by_name.insert(name.clone(), p);
        // `edges.parquet` and `lineages.parquet` refer to nodes by the `{i}` in
        // `node_{i}`; a name that does not parse simply has no edge endpoint.
        if let Some(i) = name.strip_prefix("node_").and_then(|s| s.parse().ok()) {
            by_index.insert(i, p);
        }
    }
    Ok(NodePositions {
        by_index,
        by_name,
        pts_px,
    })
}

/// Read `{prefix}.velocity_grid_2d.parquet` (scVelo-style gridded cell-velocity arrows
/// `[x, y, dx, dy]`, written by `faba lineage --layout umap`) and project each into
/// pixel space as a `Seg` (tail → head). Returns an empty vec when the file is absent
/// (the PHATE layout writes none), so callers can treat it as an optional overlay.
pub(super) fn load_velocity_grid(
    prefix: &str,
    bounds: &DataBounds,
    ext: Extent,
    scale: f32,
) -> Result<Vec<super::style::Seg>> {
    let path = format!("{prefix}.velocity_grid_2d.parquet");
    if !std::path::Path::new(&path).exists() {
        return Ok(Vec::new());
    }
    let g = DMatrix::<f32>::from_parquet(&path).with_context(|| format!("reading {path}"))?;
    let xi = col_index(&g.cols, "x", &path)?;
    let yi = col_index(&g.cols, "y", &path)?;
    let dxi = col_index(&g.cols, "dx", &path)?;
    let dyi = col_index(&g.cols, "dy", &path)?;
    let mut segs = Vec::with_capacity(g.mat.nrows());
    for r in 0..g.mat.nrows() {
        let (x, y) = (g.mat[(r, xi)], g.mat[(r, yi)]);
        let (dx, dy) = (g.mat[(r, dxi)] * scale, g.mat[(r, dyi)] * scale);
        if ![x, y, dx, dy].iter().all(|v| v.is_finite()) {
            continue;
        }
        segs.push((bounds.to_pixel((x, y), ext), bounds.to_pixel((x + dx, y + dy), ext)));
    }
    Ok(segs)
}

//////////////////////
// Trajectory edges //
//////////////////////

/// One MST edge, with the velocity call `faba lineage` made for it.
pub(super) struct Edge {
    /// Undirected endpoints, as MST node indices.
    pub from: usize,
    pub to: usize,
    /// Mean node velocity projected onto the edge; sign orients
    /// `directed_from → directed_to`, magnitude is the confidence.
    pub velocity_flux: f32,
    pub directed_from: usize,
    pub directed_to: usize,
}

/// The trajectory backbone: every MST edge, plus how many root→leaf paths cross
/// each one (keyed by the sorted endpoint pair).
pub(super) struct TrajectoryEdges {
    pub edges: Vec<Edge>,
    pub traversals: HashMap<(usize, usize), u32>,
}

pub(super) fn load_trajectory_edges(prefix: &str) -> Result<TrajectoryEdges> {
    let path = format!("{prefix}.edges.parquet");
    let e = DMatrix::<f32>::from_parquet(&path).with_context(|| format!("reading edges {path}"))?;
    let (fi, ti) = (
        col_index(&e.cols, "from", &path)?,
        col_index(&e.cols, "to", &path)?,
    );
    let fx = col_index(&e.cols, "velocity_flux", &path)?;
    let (dfi, dti) = (
        col_index(&e.cols, "directed_from", &path)?,
        col_index(&e.cols, "directed_to", &path)?,
    );
    let edges = (0..e.mat.nrows())
        .map(|i| Edge {
            from: e.mat[(i, fi)] as usize,
            to: e.mat[(i, ti)] as usize,
            velocity_flux: e.mat[(i, fx)],
            directed_from: e.mat[(i, dfi)] as usize,
            directed_to: e.mat[(i, dti)] as usize,
        })
        .collect();
    Ok(TrajectoryEdges {
        edges,
        traversals: load_traversals(prefix),
    })
}

/// How many root→leaf paths cross each undirected edge, from `lineages.parquet`.
/// Best-effort: an unreadable or malformed table yields an empty map, and every
/// edge then draws at the base width.
fn load_traversals(prefix: &str) -> HashMap<(usize, usize), u32> {
    let path = format!("{prefix}.lineages.parquet");
    let mut traversals = HashMap::new();
    let Ok(l) = DMatrix::<f32>::from_parquet(&path) else {
        return traversals;
    };
    let (Ok(li), Ok(si), Ok(ni)) = (
        col_index(&l.cols, "lineage", &path),
        col_index(&l.cols, "step", &path),
        col_index(&l.cols, "node", &path),
    ) else {
        return traversals;
    };
    let mut by_lin: HashMap<i64, Vec<(f32, usize)>> = HashMap::new();
    for i in 0..l.mat.nrows() {
        by_lin
            .entry(l.mat[(i, li)] as i64)
            .or_default()
            .push((l.mat[(i, si)], l.mat[(i, ni)] as usize));
    }
    for path in by_lin.values_mut() {
        path.sort_by(|a, b| a.0.total_cmp(&b.0));
        for w in path.windows(2) {
            let key = (w[0].1.min(w[1].1), w[0].1.max(w[1].1));
            *traversals.entry(key).or_insert(0) += 1;
        }
    }
    traversals
}

//////////////////////
// Principal curves //
//////////////////////

/// The Slingshot principal curves: one polyline per lineage, ordered along the
/// curve, in **data** space — plus how much cell mass each lineage carries.
pub(super) struct Curves {
    /// Lineage id → `(x, y)` ordered by the `grid` column.
    pub by_lineage: HashMap<i64, Vec<(f32, f32)>>,
    /// Lineage id → soft cell mass. Empty when the weights are unavailable.
    pub usage: HashMap<i64, f32>,
}

impl Curves {
    /// How many times the shared trunk would be redrawn if every curve were
    /// plotted: every lineage starts at the root.
    pub fn n_lineages(&self) -> usize {
        self.by_lineage.len()
    }
}

pub(super) fn load_curves(prefix: &str) -> Result<Curves> {
    let path = format!("{prefix}.curves_2d.parquet");
    let c =
        DMatrix::<f32>::from_parquet(&path).with_context(|| format!("reading curves {path}"))?;
    let li = col_index(&c.cols, "lineage", &path)?;
    let gi = col_index(&c.cols, "grid", &path)?;
    let xi = col_index(&c.cols, "x", &path)?;
    let yi = col_index(&c.cols, "y", &path)?;

    let mut staged: HashMap<i64, Vec<(f32, f32, f32)>> = HashMap::new();
    for i in 0..c.mat.nrows() {
        staged.entry(c.mat[(i, li)] as i64).or_default().push((
            c.mat[(i, gi)],
            c.mat[(i, xi)],
            c.mat[(i, yi)],
        ));
    }
    let by_lineage = staged
        .into_iter()
        .map(|(lin, mut pts)| {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            (lin, pts.into_iter().map(|(_, x, y)| (x, y)).collect())
        })
        .collect();

    Ok(Curves {
        by_lineage,
        usage: lineage_usage(prefix).unwrap_or_default(),
    })
}

/// Soft cell mass on each lineage: `Σ_cells w[cell, lineage]` from
/// `cell_lineage_weights.parquet`, the same membership weights the principal-curve
/// fit used. This is the lineage's *usage* — how much of the data it explains. See
/// [`super::style::CurveWidth::normalize`] for why the soft mass, and not hard
/// `argmax` ownership, is what stroke width may be scaled by.
fn lineage_usage(prefix: &str) -> Option<HashMap<i64, f32>> {
    let path = format!("{prefix}.cell_lineage_weights.parquet");
    let w = DMatrix::<f32>::from_parquet(&path)
        .inspect_err(|e| warn!("no cell-lineage weights ({e})"))
        .ok()?;
    let mut out = HashMap::new();
    for (c, name) in w.cols.iter().enumerate() {
        let Some(idx) = name
            .strip_prefix("lineage_")
            .and_then(|s| s.parse::<i64>().ok())
        else {
            continue;
        };
        let sum: f32 = (0..w.mat.nrows()).map(|r| w.mat[(r, c)]).sum();
        out.insert(idx, sum);
    }
    (!out.is_empty()).then_some(out)
}
