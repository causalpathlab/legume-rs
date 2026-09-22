//! Drive the per-axis projector over groups of cells: scatter each group's
//! result into global cell order, accumulate the gauge mean, centre at the end.

use super::block_sgd::{AxesProjector, CellGroup};
use crate::progress::new_progress_bar;
use log::info;

/// Every cell's latent in global order, centred, with the per-axis intercepts
/// and the mean that was removed (fold `⟨e^a_f, θ̄⟩` into every axis's bias to
/// keep the scores unchanged).
pub struct AxesProjection {
    /// `[n_cells × H]` row-major.
    pub theta: Vec<f32>,
    /// `[A][n_cells]`; a cell no group carried is at the score-clamp floor.
    pub intercepts: Vec<Vec<f32>>,
    pub theta_mean: Vec<f32>,
}

/// Solve every group against `projector` and assemble the global tables. The
/// gauge mean is taken over the cells the groups carried.
pub fn project_cells_axes(
    projector: &AxesProjector,
    n_cells: usize,
    groups: impl Iterator<Item = anyhow::Result<CellGroup>>,
) -> anyhow::Result<AxesProjection> {
    let h = projector.h();
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
        let out = projector.project_group(&group, &bar)?;
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
    info!("Projection [axes]: {n_seen} of {n_cells} cells solved, gauge mean removed");
    Ok(AxesProjection {
        theta,
        intercepts,
        theta_mean,
    })
}
