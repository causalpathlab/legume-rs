//! Rigid 2D rotation to put a chosen "root" anchor at the bottom of a
//! layout, with the "tip" anchor directly above it. Used by
//! `senna layout phate --orient-by-root` to give PHATE layouts a
//! pseudotime-aware canonical frame (PHATE coordinates are only defined
//! up to rotation/reflection, so this is principled, not a hack).

use crate::embed_common::*;

/// Rotate `pb_coords` (n_pb × 2) so that the centroid of the bottom-`q`
/// PBs by `pb_pt` sits below the centroid of the top-`q` PBs along the
/// y-axis (root at the bottom, tip at the top). PBs with NaN pseudotime
/// are excluded from anchor selection but still rotated.
///
/// Returns a fresh matrix; preserves all relative distances and the
/// PHATE manifold structure (rigid transform).
pub(crate) fn rotate_root_to_bottom(
    pb_coords: &Mat,
    pb_pt: &[f32],
    q: f32,
) -> anyhow::Result<Mat> {
    anyhow::ensure!(pb_coords.ncols() == 2, "expected 2D coords");
    anyhow::ensure!(
        pb_coords.nrows() == pb_pt.len(),
        "pb_coords rows ({}) != pb_pt len ({})",
        pb_coords.nrows(),
        pb_pt.len()
    );
    anyhow::ensure!(
        q > 0.0 && q <= 0.5,
        "--orient-tip-quantile must be in (0, 0.5]; got {q}"
    );

    let mut idx: Vec<usize> = (0..pb_pt.len())
        .filter(|&i| pb_pt[i].is_finite())
        .collect();
    anyhow::ensure!(
        idx.len() >= 2,
        "need ≥ 2 PBs with finite pseudotime for orientation; got {}",
        idx.len()
    );
    idx.sort_by(|&a, &b| {
        pb_pt[a]
            .partial_cmp(&pb_pt[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let bin = ((q * idx.len() as f32).ceil() as usize).max(1);
    let root_set = &idx[..bin];
    let tip_set = &idx[idx.len() - bin..];

    let mean_xy = |s: &[usize]| -> (f32, f32) {
        let n = s.len() as f32;
        let (mut x, mut y) = (0.0_f32, 0.0_f32);
        for &i in s {
            x += pb_coords[(i, 0)];
            y += pb_coords[(i, 1)];
        }
        (x / n, y / n)
    };
    let (rx, ry) = mean_xy(root_set);
    let (tx, ty) = mean_xy(tip_set);

    // Current root→tip angle vs the +y target (π/2). Rotation pivot is
    // root_xy so the root stays visually anchored after the transform.
    let theta = (ty - ry).atan2(tx - rx);
    let delta = std::f32::consts::FRAC_PI_2 - theta;
    let (s, c) = delta.sin_cos();

    info!(
        "Orienting layout by pseudotime: bottom-{bin} root anchor at ({rx:.3}, {ry:.3}), \
         top-{bin} tip anchor at ({tx:.3}, {ty:.3}), rotating by {:.3} rad",
        delta
    );

    let mut out = Mat::zeros(pb_coords.nrows(), 2);
    for i in 0..pb_coords.nrows() {
        let dx = pb_coords[(i, 0)] - rx;
        let dy = pb_coords[(i, 1)] - ry;
        out[(i, 0)] = c * dx - s * dy + rx;
        out[(i, 1)] = s * dx + c * dy + ry;
    }
    Ok(out)
}
