//! 2D convex hull (Andrew's monotone chain) and per-group summary stats
//! used for label placement in `senna plot`.
//!
//! ## References
//! - Andrew, A. M. (1979). "Another efficient algorithm for convex hulls
//!   in two dimensions." *Information Processing Letters* 9(5): 216–219.

/// A 2D point.
pub type Pt = (f32, f32);

/// Compute the convex hull of `pts` via Andrew's monotone chain.
///
/// Returns hull vertices in counter-clockwise order (first point *not*
/// repeated at the end). Degenerate cases:
/// - 0 points → empty vec.
/// - 1 point  → `[p]`.
/// - ≥2 collinear points → the two extreme points.
#[must_use]
pub fn convex_hull(pts: &[Pt]) -> Vec<Pt> {
    let n = pts.len();
    if n <= 1 {
        return pts.to_vec();
    }
    let mut sorted: Vec<Pt> = pts.to_vec();
    sorted.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Lower hull.
    let mut lower: Vec<Pt> = Vec::with_capacity(n);
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    // Upper hull.
    let mut upper: Vec<Pt> = Vec::with_capacity(n);
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    // Drop last of each — it's the first of the other.
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Signed cross product of `OA` × `OB`. Positive if `O→A→B` is a left
/// (CCW) turn, negative for right (CW), zero if collinear.
fn cross(o: Pt, a: Pt, b: Pt) -> f32 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Keep the `coverage` fraction of points closest to the coordinate-wise
/// median (Euclidean). Used to strip outliers before hull computation
/// so a single fringe cell doesn't drag the polygon across the plot.
///
/// `coverage` is clamped to `(0, 1]`. At 1.0 returns `pts` unchanged.
#[must_use]
pub fn trim_outliers_by_median(pts: &[Pt], coverage: f32) -> Vec<Pt> {
    let c = coverage.clamp(f32::MIN_POSITIVE, 1.0);
    if c >= 1.0 || pts.len() < 3 {
        return pts.to_vec();
    }
    let (mx, my) = median_xy(pts);
    let mut with_d: Vec<(f32, Pt)> = pts
        .iter()
        .map(|&p| {
            let dx = p.0 - mx;
            let dy = p.1 - my;
            (dx * dx + dy * dy, p)
        })
        .collect();
    with_d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let keep = ((pts.len() as f32) * c).ceil() as usize;
    let keep = keep.clamp(3, pts.len());
    with_d.into_iter().take(keep).map(|(_, p)| p).collect()
}

/// Coordinate-wise median of `pts`. Robust to outliers; returns `(NaN,
/// NaN)` if `pts` is empty (callers should skip groups with no points).
#[must_use]
pub fn median_xy(pts: &[Pt]) -> Pt {
    if pts.is_empty() {
        return (f32::NAN, f32::NAN);
    }
    let mut xs: Vec<f32> = pts.iter().map(|p| p.0).collect();
    let mut ys: Vec<f32> = pts.iter().map(|p| p.1).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = pts.len() / 2;
    (xs[mid], ys[mid])
}

/// Centroid of the convex hull polygon (area-weighted). For ≤2 hull
/// points, falls back to the arithmetic mean.
#[must_use]
pub fn hull_centroid(hull: &[Pt]) -> Pt {
    let n = hull.len();
    if n == 0 {
        return (f32::NAN, f32::NAN);
    }
    if n <= 2 {
        let mx = hull.iter().map(|p| p.0).sum::<f32>() / n as f32;
        let my = hull.iter().map(|p| p.1).sum::<f32>() / n as f32;
        return (mx, my);
    }
    let mut area2 = 0.0_f32;
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    for i in 0..n {
        let (x0, y0) = hull[i];
        let (x1, y1) = hull[(i + 1) % n];
        let c = x0 * y1 - x1 * y0;
        area2 += c;
        cx += (x0 + x1) * c;
        cy += (y0 + y1) * c;
    }
    if area2.abs() < f32::EPSILON {
        let mx = hull.iter().map(|p| p.0).sum::<f32>() / n as f32;
        let my = hull.iter().map(|p| p.1).sum::<f32>() / n as f32;
        return (mx, my);
    }
    let denom = 3.0 * area2;
    (cx / denom, cy / denom)
}
