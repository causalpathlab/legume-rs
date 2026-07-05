//! Velocity orientation of the (undirected) centroid MST.
//!
//! `faba gem` emits a per-cell velocity increment δ. Aggregated to node means,
//! δ gives each MST edge a signed **flux** — the projection of the mean velocity
//! onto the edge direction — which orients the tree and locates its root (the
//! node with the largest net outgoing flux, i.e. the velocity source).

use nalgebra::DMatrix;

/// Mean velocity per node: average δ over cells whose cluster (nearest centroid)
/// is `k`. Returns a `K × D` matrix; nodes with no cells stay zero.
pub fn aggregate_node_velocity(
    velocity: &DMatrix<f32>,
    cluster: &[usize],
    k: usize,
) -> DMatrix<f32> {
    let d = velocity.ncols();
    let mut v = DMatrix::<f32>::zeros(k, d);
    let mut counts = vec![0usize; k];
    for (i, &c) in cluster.iter().enumerate() {
        if c < k {
            for j in 0..d {
                v[(c, j)] += velocity[(i, j)];
            }
            counts[c] += 1;
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..d {
                v[(c, j)] /= counts[c] as f32;
            }
        }
    }
    v
}

/// Signed velocity flux along each undirected edge `(j,k)`:
/// `f = ½(v_j + v_k) · (centroid_k − centroid_j)`. Positive ⇒ flow `j → k`.
pub fn edge_velocity_flux(
    centroids: &DMatrix<f32>,
    node_velocity: &DMatrix<f32>,
    edges: &[(usize, usize)],
) -> Vec<f32> {
    let d = centroids.ncols();
    edges
        .iter()
        .map(|&(j, k)| {
            let mut f = 0f32;
            for c in 0..d {
                let dir = centroids[(k, c)] - centroids[(j, c)];
                let vmean = 0.5 * (node_velocity[(j, c)] + node_velocity[(k, c)]);
                f += vmean * dir;
            }
            f
        })
        .collect()
}

/// Orient each undirected edge by the sign of its flux: `(from, to)` points the
/// way velocity flows (`j → k` when `flux ≥ 0`, else `k → j`).
pub fn directed_edges(edges: &[(usize, usize)], flux: &[f32]) -> Vec<(usize, usize)> {
    edges
        .iter()
        .zip(flux)
        .map(|(&(j, k), &f)| if f >= 0.0 { (j, k) } else { (k, j) })
        .collect()
}

/// Root = node with maximum net outgoing flux (the velocity source). For an edge
/// `(j,k)` with flux `f` (positive ⇒ `j→k`), mass `f` leaves `j` and enters `k`,
/// so `outflow[j] += f`, `outflow[k] -= f`.
pub fn pick_velocity_root(edges: &[(usize, usize)], flux: &[f32], k: usize) -> usize {
    let mut outflow = vec![0f32; k];
    for (&(j, kk), &f) in edges.iter().zip(flux) {
        outflow[j] += f;
        outflow[kk] -= f;
    }
    (0..k)
        .max_by(|&a, &b| {
            outflow[a]
                .partial_cmp(&outflow[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
