//! Gene-module resolution on the gene-gene network.
//!
//! Two stages:
//!   1. Iterative degree trim ("k-core"): drop genes with current-subgraph
//!      degree below `min_degree`, re-count, repeat until stable.
//!   2. Leiden on the surviving subgraph, via
//!      [`matrix_util::knn_graph::run_leiden`].
//!
//! Genes dropped by the trim get `None` as their module label; surviving
//! genes get `Some(m)` with `m` contiguous starting at 0.

use crate::gene_network::graph::GenePairGraph;
use log::info;

/// Per-gene degree within the subgraph induced by `keep` (count only edges
/// with both endpoints kept).
fn subgraph_degrees(graph: &GenePairGraph, keep: &[bool]) -> Vec<usize> {
    let mut deg = vec![0usize; graph.n_genes];
    for &(u, v) in &graph.gene_edges {
        if keep[u] && keep[v] {
            deg[u] += 1;
            deg[v] += 1;
        }
    }
    deg
}

/// Iteratively trim genes with in-subgraph degree below `min_degree`.
///
/// Returns a boolean mask of length `graph.n_genes`: `true` means the gene
/// survives, `false` means it was dropped in some round.
pub fn kcore_trim(graph: &GenePairGraph, min_degree: usize) -> Vec<bool> {
    let n = graph.n_genes;
    let mut alive = vec![true; n];
    if min_degree == 0 {
        return alive;
    }
    let mut round = 0usize;
    loop {
        let deg = subgraph_degrees(graph, &alive);
        let mut changed = false;
        for g in 0..n {
            if alive[g] && deg[g] < min_degree {
                alive[g] = false;
                changed = true;
            }
        }
        round += 1;
        if !changed {
            break;
        }
        // Safety stop for pathological cases (shouldn't trigger in practice).
        if round > n {
            break;
        }
    }
    let n_alive = alive.iter().filter(|&&b| b).count();
    info!(
        "k-core trim (min_degree={}): {}/{} genes retained after {} round(s)",
        min_degree, n_alive, n, round
    );
    alive
}

/// Run Leiden on the subgraph induced by `keep` and return per-gene
/// module labels (`None` for trimmed or isolated genes).
///
/// Module labels are contiguous `0..n_modules`.
pub fn leiden_gene_modules(
    graph: &GenePairGraph,
    keep: &[bool],
    resolution: f64,
    seed: u64,
) -> Vec<Option<usize>> {
    assert_eq!(keep.len(), graph.n_genes);

    // Restrict to non-isolated kept genes: a kept gene with all its
    // neighbors trimmed away would produce a singleton Leiden module with
    // no signal — drop those to None for cleaner output.
    let sub_degrees = subgraph_degrees(graph, keep);
    let sub_of: Vec<Option<usize>> = {
        let mut out = vec![None; graph.n_genes];
        let mut next = 0usize;
        for g in 0..graph.n_genes {
            if keep[g] && sub_degrees[g] > 0 {
                out[g] = Some(next);
                next += 1;
            }
        }
        out
    };
    let n_sub = sub_of.iter().filter(|o| o.is_some()).count();

    if n_sub == 0 {
        info!("leiden_gene_modules: empty subgraph");
        return vec![None; graph.n_genes];
    }

    // Build leiden::Network: node weights = subgraph degree, edge weights = 1.0.
    let mut total_edge_weight = 0.0f64;
    let mut lg = leiden::Graph::with_capacity(n_sub, graph.gene_edges.len());
    for g in 0..graph.n_genes {
        if sub_of[g].is_some() {
            lg.add_node(sub_degrees[g] as f32);
        }
    }
    for &(u, v) in &graph.gene_edges {
        if let (Some(su), Some(sv)) = (sub_of[u], sub_of[v]) {
            lg.add_edge((su as u32).into(), (sv as u32).into(), 1.0);
            total_edge_weight += 1.0;
        }
    }
    let network = leiden::Network::new_from_graph(lg);

    // The Leiden crate expects CPM-scale resolution. Convert from the
    // user-facing modularity γ via `γ / (2 * total_edge_weight)`, matching
    // matrix_util::knn_graph conventions.
    let cpm_resolution = if total_edge_weight > 0.0 {
        resolution / (2.0 * total_edge_weight)
    } else {
        resolution
    };

    let sub_labels =
        matrix_util::knn_graph::run_leiden(&network, n_sub, cpm_resolution, Some(seed as usize));

    // Compact labels to 0..K.
    let mut compact = sub_labels.clone();
    matrix_util::knn_graph::compact_labels(&mut compact);
    let n_modules = compact.iter().copied().max().map_or(0, |m| m + 1);

    let mut out = vec![None; graph.n_genes];
    for g in 0..graph.n_genes {
        if let Some(sub) = sub_of[g] {
            out[g] = Some(compact[sub]);
        }
    }
    info!(
        "leiden_gene_modules: {} modules over {} genes (resolution={:.3})",
        n_modules, n_sub, resolution
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gene_network::graph::test_graph_from_edges as graph_from_edges;

    #[test]
    fn test_kcore_min_degree_zero_keeps_all() {
        let g = graph_from_edges(&[(0, 1)], 3);
        let keep = kcore_trim(&g, 0);
        assert!(keep.iter().all(|&b| b));
    }

    #[test]
    fn test_kcore_drops_leaf() {
        // Triangle (0-1-2) plus a leaf 3 attached to 0.
        // With min_degree=2, leaf 3 (deg=1) drops; triangle survives at deg=2.
        let g = graph_from_edges(&[(0, 1), (0, 2), (1, 2), (0, 3)], 4);
        let keep = kcore_trim(&g, 2);
        assert_eq!(keep, vec![true, true, true, false]);
    }

    #[test]
    fn test_kcore_iterative_cascades() {
        // Chain: 0-1, 1-2, 2-3. All have degree ≤ 2, none ≥ 3 anywhere.
        // With min_degree=2: 0 (deg 1) drops; now 1 has deg 1 → drops; etc.
        let g = graph_from_edges(&[(0, 1), (1, 2), (2, 3)], 4);
        let keep = kcore_trim(&g, 2);
        assert!(keep.iter().all(|&b| !b));
    }

    #[test]
    fn test_kcore_triangle_at_k_equals_2() {
        let g = graph_from_edges(&[(0, 1), (0, 2), (1, 2)], 3);
        let keep = kcore_trim(&g, 2);
        assert!(keep.iter().all(|&b| b));
    }

    #[test]
    fn test_kcore_triangle_drops_at_k_equals_3() {
        let g = graph_from_edges(&[(0, 1), (0, 2), (1, 2)], 3);
        let keep = kcore_trim(&g, 3);
        assert!(keep.iter().all(|&b| !b));
    }

    #[test]
    fn test_leiden_two_disjoint_triangles_split() {
        // Two disjoint triangles: {0,1,2} and {3,4,5}.
        let g = graph_from_edges(&[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)], 6);
        let keep = vec![true; 6];
        let mods = leiden_gene_modules(&g, &keep, 1.0, 42);
        let m0 = mods[0].unwrap();
        let m3 = mods[3].unwrap();
        assert_ne!(
            m0, m3,
            "disjoint triangles should land in different modules"
        );
        assert_eq!(mods[1], Some(m0));
        assert_eq!(mods[2], Some(m0));
        assert_eq!(mods[4], Some(m3));
        assert_eq!(mods[5], Some(m3));
    }

    #[test]
    fn test_leiden_trimmed_genes_get_none() {
        // Triangle {0,1,2} + leaf 3 attached to 0.
        let g = graph_from_edges(&[(0, 1), (0, 2), (1, 2), (0, 3)], 4);
        let keep = vec![true, true, true, false];
        let mods = leiden_gene_modules(&g, &keep, 1.0, 1);
        assert!(mods[3].is_none());
        assert!(mods[0].is_some());
    }
}
