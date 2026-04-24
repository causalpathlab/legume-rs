//! Per-topic spatial stratification and kNN matching for `cocoa spatial-diff`.
//!
//! Given cell-to-topic propensities and spatial coordinates, stratify cells
//! by θ_k into HIGH / LOW / DROP, build a spatial kNN graph, and expose
//! per-cell HIGH → LOW neighbor lookups used by the sufficient-stat
//! accumulator.

use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
use nalgebra::DMatrix;

pub const DEFAULT_SPATIAL_KNN: usize = 25;
pub const DEFAULT_HIGH_Q: f32 = 0.75;
pub const DEFAULT_LOW_Q: f32 = 0.25;

/// Per-cell stratum assignment for a single topic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stratum {
    High,
    Low,
    Drop,
}

/// Stratify cells within one topic k by quantile thresholds on θ_{·,k}.
///
/// Returns a vector of length n_cells.
pub fn stratify_topic(theta_k: &[f32], low_q: f32, high_q: f32) -> Vec<Stratum> {
    assert!((0.0..=1.0).contains(&low_q));
    assert!((0.0..=1.0).contains(&high_q));
    assert!(low_q < high_q);
    let mut sorted = theta_k.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return Vec::new();
    }
    let lo_idx = ((n as f32) * low_q) as usize;
    let hi_idx = ((n as f32) * high_q).min((n - 1) as f32) as usize;
    let lo_thr = sorted[lo_idx.min(n - 1)];
    let hi_thr = sorted[hi_idx];

    theta_k
        .iter()
        .map(|&v| {
            if v >= hi_thr {
                Stratum::High
            } else if v <= lo_thr {
                Stratum::Low
            } else {
                Stratum::Drop
            }
        })
        .collect()
}

/// Build a spatial kNN graph from cell coordinates.
///
/// `coords` is (n_cells × d); `knn` is the fan-out per cell.
pub fn build_spatial_graph(
    coords: &DMatrix<f32>,
    knn: usize,
    block_size: usize,
) -> anyhow::Result<KnnGraph> {
    KnnGraph::from_rows(
        coords,
        KnnGraphArgs {
            knn,
            block_size,
            reciprocal: false,
        },
    )
}

/// For one topic k: return `high_to_low[i]` = indices of spatial-kNN
/// neighbors of HIGH cell `i` that are in stratum LOW. Cells not in HIGH
/// get an empty list. Optional radius clipping is applied when `radius`
/// is `Some` — neighbors with edge distance > radius are dropped.
pub fn high_to_low_neighbors(
    graph: &KnnGraph,
    strata: &[Stratum],
    radius: Option<f32>,
) -> Vec<Vec<(usize, f32)>> {
    let n = graph.num_nodes();
    let mut out: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];

    // Build per-node neighbor+distance list from edge list.
    // edges are canonical (i < j). Symmetric expand.
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for (&(i, j), &d) in graph.edges.iter().zip(graph.distances.iter()) {
        if let Some(r) = radius {
            if d > r {
                continue;
            }
        }
        adj[i].push((j, d));
        adj[j].push((i, d));
    }

    for (i, nbrs) in adj.into_iter().enumerate() {
        if strata.get(i).copied() != Some(Stratum::High) {
            continue;
        }
        for (j, d) in nbrs {
            if strata.get(j).copied() == Some(Stratum::Low) {
                out[i].push((j, d));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratify_topic_splits_into_three_strata() {
        let theta = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let s = stratify_topic(&theta, 0.25, 0.75);
        // low_q=0.25 → idx 2 → 0.3; high_q=0.75 → idx 7 → 0.8
        assert_eq!(s[0], Stratum::Low); // 0.1 <= 0.3
        assert_eq!(s[2], Stratum::Low); // 0.3 <= 0.3
        assert_eq!(s[4], Stratum::Drop); // 0.5 in middle
        assert_eq!(s[7], Stratum::High); // 0.8 >= 0.8
        assert_eq!(s[9], Stratum::High); // 1.0 >= 0.8
    }

    #[test]
    fn empty_theta_yields_empty_strata() {
        let s = stratify_topic(&[], 0.25, 0.75);
        assert!(s.is_empty());
    }
}
