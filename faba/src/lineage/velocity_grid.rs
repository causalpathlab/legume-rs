//! Binning the per-cell velocity field into the grid of arrows a plot draws.

use anyhow::Result;
use log::info;

use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use std::collections::HashMap;

/// Running sums for one lattice bin of the velocity grid: the member cells' layout
/// positions and their projected 2D velocities, plus the member count that turns
/// those sums into means (and gates the bin on `MIN_PER_CELL`).
#[derive(Default, Clone, Copy)]
pub(super) struct GridBin {
    sum_x: f32,
    sum_y: f32,
    sum_dx: f32,
    sum_dy: f32,
    n: u32,
}

impl GridBin {
    /// Mean position and mean velocity of the bin's members. Callers check
    /// `n >= MIN_PER_CELL` first, so `n` is nonzero here.
    fn means(&self) -> (f32, f32, f32, f32) {
        let c = self.n as f32;
        (
            self.sum_x / c,
            self.sum_y / c,
            self.sum_dx / c,
            self.sum_dy / c,
        )
    }
}

/// scVelo-style velocity projection + gridding. For each cell, the 2D arrow is the
/// θ-neighbour transition-weighted mean displacement (weight = `max(0, cos(δ_i,
/// θ_j−θ_i))`); those are then averaged onto a `GRID×GRID` grid, keeping only cells
/// with ≥ `MIN_PER_CELL` members. Returns `(x, y, dx, dy)` per occupied grid cell
/// (a few hundred), unit-ish arrows scaled to the grid pitch.
pub(super) fn velocity_grid_arrows(
    cells_2d: &DMatrix<f32>,
    theta: &DMatrix<f32>,
    delta: &DMatrix<f32>,
    knn: usize,
) -> Vec<(f32, f32, f32, f32)> {
    use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
    const GRID: usize = 30;
    const MIN_PER_CELL: usize = 5;
    let (n, h) = (theta.nrows(), theta.ncols());
    if n == 0 {
        return Vec::new();
    }
    // θ-neighbour graph (identity space) for the transition projection.
    let Ok(graph) = KnnGraph::from_rows(
        theta,
        KnnGraphArgs {
            knn,
            block_size: 1000,
            reciprocal: false,
        },
    ) else {
        return Vec::new();
    };
    let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j) in &graph.edges {
        nbrs[i].push(j);
        nbrs[j].push(i);
    }
    // Per-cell 2D velocity via θ-transition weights on δ.
    let mut cell_vel = vec![(0f32, 0f32); n];
    for i in 0..n {
        let (mut vx, mut vy, mut wsum) = (0f32, 0f32, 0f32);
        let di = (0..h)
            .map(|c| delta[(i, c)] * delta[(i, c)])
            .sum::<f32>()
            .sqrt();
        if di < 1e-8 {
            continue;
        }
        for &j in &nbrs[i] {
            // cos(δ_i, θ_j − θ_i): the arrow points along the developmental-FORWARD flow.
            // gem's δ is the nascent (unspliced) increment on top of θ, so `θ+δ` is the
            // emerging/future state — the cell is heading toward `+δ`, i.e. from the
            // progenitor hub outward to the committed compartments.
            let (mut dot, mut dj2) = (0f32, 0f32);
            for c in 0..h {
                let dth = theta[(j, c)] - theta[(i, c)];
                dot += delta[(i, c)] * dth;
                dj2 += dth * dth;
            }
            let cos = if dj2 > 1e-12 {
                dot / (di * dj2.sqrt())
            } else {
                0.0
            };
            let wt = cos.max(0.0);
            if wt > 0.0 {
                let (dx, dy) = (
                    cells_2d[(j, 0)] - cells_2d[(i, 0)],
                    cells_2d[(j, 1)] - cells_2d[(i, 1)],
                );
                let dn = (dx * dx + dy * dy).sqrt().max(1e-8);
                vx += wt * dx / dn;
                vy += wt * dy / dn;
                wsum += wt;
            }
        }
        if wsum > 0.0 {
            cell_vel[i] = (vx / wsum, vy / wsum);
        }
    }
    // Grid-average onto a GRID×GRID lattice over the layout bounds.
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for i in 0..n {
        xmin = xmin.min(cells_2d[(i, 0)]);
        xmax = xmax.max(cells_2d[(i, 0)]);
        ymin = ymin.min(cells_2d[(i, 1)]);
        ymax = ymax.max(cells_2d[(i, 1)]);
    }
    let (wx, wy) = ((xmax - xmin).max(1e-6), (ymax - ymin).max(1e-6));
    let pitch = (wx / GRID as f32).min(wy / GRID as f32);
    let mut acc: HashMap<(usize, usize), GridBin> = HashMap::new();
    for i in 0..n {
        let gx = (((cells_2d[(i, 0)] - xmin) / wx * GRID as f32) as usize).min(GRID - 1);
        let gy = (((cells_2d[(i, 1)] - ymin) / wy * GRID as f32) as usize).min(GRID - 1);
        let e = acc.entry((gx, gy)).or_default();
        e.sum_x += cells_2d[(i, 0)];
        e.sum_y += cells_2d[(i, 1)];
        e.sum_dx += cell_vel[i].0;
        e.sum_dy += cell_vel[i].1;
        e.n += 1;
    }
    let mut out = Vec::new();
    for bin in acc.into_values() {
        if (bin.n as usize) < MIN_PER_CELL {
            continue;
        }
        let (mx, my, mdx, mdy) = bin.means();
        let mag = (mdx * mdx + mdy * mdy).sqrt();
        if mag < 1e-6 {
            continue;
        }
        // Scale each arrow to ~one grid pitch (unit direction × pitch).
        out.push((mx, my, mdx / mag * pitch, mdy / mag * pitch));
    }
    out
}

/// Write `{out}.velocity_grid_2d.parquet` — gridded arrows `[x, y, dx, dy]`.
pub(super) fn write_velocity_grid(arrows: &[(f32, f32, f32, f32)], path: &str) -> Result<()> {
    let mut m = DMatrix::<f32>::zeros(arrows.len(), 4);
    for (i, &(x, y, dx, dy)) in arrows.iter().enumerate() {
        m[(i, 0)] = x;
        m[(i, 1)] = y;
        m[(i, 2)] = dx;
        m[(i, 3)] = dy;
    }
    let cols: Vec<Box<str>> = ["x", "y", "dx", "dy"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    m.to_parquet_with_names(path, (None, None), Some(&cols))?;
    info!("Wrote {path} ({} gridded velocity arrows)", arrows.len());
    Ok(())
}

/////////////////////////////////////////
// Marker annotation of the trajectory //
/////////////////////////////////////////
