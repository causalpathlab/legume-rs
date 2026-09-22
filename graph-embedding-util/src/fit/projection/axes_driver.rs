//! Drive the per-axis projector over groups of cells: scatter each group's
//! result into global cell order, accumulate the gauge mean, centre at the end.

use super::block_sgd::{AxesProjector, CellGroup};
use crate::progress::new_progress_bar;
use log::info;

/// Every cell's latent in global order, centred, with the per-axis intercepts
/// and the mean that was removed (fold `⟨e^a_f, θ̄⟩` into every axis's bias to
/// keep the scores unchanged).
#[derive(Debug)]
pub struct AxesProjection {
    /// `[n_cells × H]` row-major.
    pub theta: Vec<f32>,
    /// `[A][n_cells]`; a cell no group carried is at the score-clamp floor.
    pub intercepts: Vec<Vec<f32>>,
    pub theta_mean: Vec<f32>,
}

/// Where every cell starts: row `cell_to_row[c]` of `rows` (`[R × H]`
/// row-major). Many cells may share a row, e.g. their pseudobulk's.
pub struct WarmStart<'a> {
    pub rows: &'a [f32],
    pub cell_to_row: &'a [usize],
}

impl WarmStart<'_> {
    fn check(&self, n_cells: usize, h: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.cell_to_row.len() == n_cells,
            "warm start maps {} cells, {n_cells} to solve",
            self.cell_to_row.len()
        );
        anyhow::ensure!(
            self.rows.len().is_multiple_of(h),
            "warm start rows have {} entries, not a multiple of {h}",
            self.rows.len()
        );
        let n_rows = self.rows.len() / h;
        if let Some(&r) = self.cell_to_row.iter().find(|&&r| r >= n_rows) {
            anyhow::bail!("warm start maps a cell to row {r} of {n_rows}");
        }
        Ok(())
    }

    /// `[n × H]` for the group's cells.
    fn gather(&self, cells: &[u32], h: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(cells.len() * h);
        for &c in cells {
            let r = self.cell_to_row[c as usize];
            out.extend_from_slice(&self.rows[r * h..(r + 1) * h]);
        }
        out
    }
}

/// Solve every group against `projector`, from zero or from `warm`, and
/// assemble the global tables. The gauge mean is taken over the cells the
/// groups carried.
pub fn project_cells_axes(
    projector: &AxesProjector,
    n_cells: usize,
    groups: impl Iterator<Item = anyhow::Result<CellGroup>>,
    warm: Option<&WarmStart<'_>>,
) -> anyhow::Result<AxesProjection> {
    let h = projector.h();
    if let Some(w) = warm {
        w.check(n_cells, h)?;
    }
    let mut theta = vec![0f32; n_cells * h];
    let floor = -(crate::cell_projection::SCORE_CLAMP as f32);
    let mut intercepts = vec![vec![floor; n_cells]; projector.n_axes()];
    let mut sum = vec![0f64; h];
    let mut n_seen = 0usize;
    let bar = new_progress_bar(n_cells as u64);
    bar.enable_steady_tick(std::time::Duration::from_millis(200));
    for group in groups {
        let group = group?;
        for &c in &group.cells {
            anyhow::ensure!(
                (c as usize) < n_cells,
                "cell id {c} outside the {n_cells}-cell axis"
            );
        }
        let init = warm.map(|w| w.gather(&group.cells, h));
        let out = projector.project_group(&group, init.as_deref(), &bar)?;
        for (i, &c) in group.cells.iter().enumerate() {
            let c = c as usize;
            let row = &out.theta[i * h..(i + 1) * h];
            theta[c * h..(c + 1) * h].copy_from_slice(row);
            for (k, x) in row.iter().enumerate() {
                sum[k] += f64::from(*x);
            }
            for (a, col) in out.intercepts.iter().enumerate() {
                intercepts[a][c] = col[i];
            }
        }
        n_seen += group.cells.len();
    }
    bar.finish_and_clear();
    let theta_mean: Vec<f32> = sum
        .iter()
        .map(|s| (s / n_seen.max(1) as f64) as f32)
        .collect();
    for row in theta.chunks_exact_mut(h) {
        for (x, m) in row.iter_mut().zip(&theta_mean) {
            *x -= m;
        }
    }
    info!(
        "Projection [axes]: {n_seen} of {n_cells} cells solved from {}, gauge mean removed",
        if warm.is_some() {
            "their warm start"
        } else {
            "zero"
        }
    );
    Ok(AxesProjection {
        theta,
        intercepts,
        theta_mean,
    })
}
