//! Hybrid cell community state: profile DC-SBM + hsblock tree structural model.
//!
//! Combines two sufficient-statistic stores under a single membership vector:
//! - `profile` (CellCommunityStats): per-community gene-expression aggregates,
//!   scored by the plug-in multinomial entropy J = Σ f(T) − Σ f(S).
//! - `structural` (hsblock::SufficientStats): per-block edge counts + cluster
//!   sizes/volumes, scored via Gamma-Poisson at each node of a fixed binary
//!   tree. Leaves own within-cluster edges, internal nodes own between-cluster
//!   (LCA) edges.
//!
//! Per-cell moves update both sides in lock-step. Combined objective:
//!   J(z) = J_profile(z) + β · J_tree(z),
//! with β = `hsbm_weight`.

use super::model::{compute_log_probs_for_cell as compute_profile_log_probs, CellCommunityStats};
use super::profiles::CellProfileStore;
use hsblock::btree::{BTree, GammaPoissonParam};
use hsblock::model::{poisson_score_cpu, tree_score_cpu};
use hsblock::sufficient_stats::{SufficientStats, WeightedEdge};

/// Per-cell weighted adjacency: `adj_hsbm[u]` = list of `(neighbor, weight)`.
pub type HsbmAdjList = Vec<Vec<(usize, f64)>>;

pub struct CellHybridStats {
    pub profile: CellCommunityStats,
    pub structural: SufficientStats,
    pub tree: BTree<GammaPoissonParam>,
    pub degree_corrected: bool,
    pub hsbm_weight: f64,
}

impl CellHybridStats {
    /// Build a combined state from a profile store, a weighted edge list, and
    /// initial leaf labels. `tree_depth` controls K = 2^(depth − 1).
    pub fn new(
        profiles: &CellProfileStore,
        edges: &[WeightedEdge],
        n_nodes: usize,
        tree_depth: usize,
        init_labels: &[usize],
        hsbm_weight: f64,
        degree_corrected: bool,
        a0: f64,
        b0: f64,
    ) -> Self {
        let tree = BTree::with_gamma_poisson(tree_depth, a0, b0);
        let k = tree.num_leaves();
        debug_assert!(init_labels.iter().all(|&c| c < k));
        let profile = CellCommunityStats::from_profiles(profiles, k, init_labels);
        let structural = SufficientStats::from_edges(edges, n_nodes, k, init_labels);
        Self {
            profile,
            structural,
            tree,
            degree_corrected,
            hsbm_weight,
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.profile.k
    }

    #[inline]
    pub fn n_cells(&self) -> usize {
        self.profile.n_cells
    }

    #[inline]
    pub fn membership(&self) -> &[usize] {
        &self.profile.membership
    }

    /// Move cell `u` from `old_c` to `new_c`, updating both stats and the
    /// shared membership.
    pub fn delta_move(
        &mut self,
        u: usize,
        old_c: usize,
        new_c: usize,
        profiles: &CellProfileStore,
        adj_hsbm: &[(usize, f64)],
    ) {
        if old_c == new_c {
            return;
        }
        self.profile.delta_move(u, old_c, new_c, profiles);
        self.structural.delta_move(u, old_c, new_c, adj_hsbm);
    }

    /// Combined score: J_profile + β · J_tree.
    pub fn total_score(&self) -> f64 {
        let prof = self.profile.total_score();
        let (node_edge, node_total) = self
            .structural
            .aggregate_to_tree(&self.tree, self.degree_corrected);
        let a0: Vec<f64> = (1..=self.tree.num_nodes())
            .map(|n| self.tree.node_params(n).0)
            .collect();
        let b0: Vec<f64> = (1..=self.tree.num_nodes())
            .map(|n| self.tree.node_params(n).1)
            .collect();
        let tree = tree_score_cpu(&a0, &b0, &node_edge[1..], &node_total[1..]);
        prof + self.hsbm_weight * tree
    }

    /// Per-cell combined Δ log-prob vector.
    ///
    /// `out[t] = Δprofile(u → t) + β · Δtree(u → t)`; `out[current_c]` is 0.
    pub fn compute_log_probs_for_cell(
        &self,
        u: usize,
        profiles: &CellProfileStore,
        adj_hsbm: &[(usize, f64)],
        log_weights: Option<&[f64]>,
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), self.k());
        // Profile deltas (writes into out).
        compute_profile_log_probs(u, &self.profile, profiles, log_weights, out);
        // Tree deltas accumulated on top.
        if self.hsbm_weight != 0.0 {
            let beta = self.hsbm_weight;
            let k = self.k();
            let current_c = self.profile.membership[u];
            let mut tree_deltas = vec![0.0f64; k];
            compute_tree_log_probs_for_cell(
                u,
                current_c,
                &self.structural,
                &self.tree,
                adj_hsbm,
                self.degree_corrected,
                &mut tree_deltas,
            );
            for (t, d) in tree_deltas.into_iter().enumerate() {
                if t != current_c {
                    out[t] += beta * d;
                }
            }
        }
    }
}

/// Per-leaf Δ log-prob of the tree (Gamma-Poisson) term for moving vertex `u`
/// from `current_c` to each candidate leaf `t`. Entry for `current_c` = 0.
///
/// Correct treatment of the tree score: per-pair (ci, cj) contributions are
/// aggregated into per-NODE (edge, total) deltas first, THEN the Poisson-Gamma
/// score is evaluated per node (since it is non-linear in its arguments).
pub fn compute_tree_log_probs_for_cell(
    u: usize,
    current_c: usize,
    stats: &SufficientStats,
    tree: &BTree<GammaPoissonParam>,
    neighbors: &[(usize, f64)],
    degree_corrected: bool,
    out: &mut [f64],
) {
    let k = stats.k;
    debug_assert_eq!(out.len(), k);
    let _ = u;

    // Current per-node aggregates.
    let (node_edge, node_total) = stats.aggregate_to_tree(tree, degree_corrected);
    let num_nodes = tree.num_nodes();

    // v's edge weight to each cluster (based on current memberships).
    let mut edge_to_cluster = vec![0.0f64; k];
    for &(nbr, w) in neighbors {
        edge_to_cluster[stats.membership[nbr]] += w;
    }
    let deg: f64 = edge_to_cluster.iter().sum();

    let mut delta_edge = vec![0.0f64; num_nodes + 1];
    let mut delta_total = vec![0.0f64; num_nodes + 1];
    let mut touched: Vec<usize> = Vec::new();

    for t in 0..k {
        if t == current_c {
            out[t] = 0.0;
            continue;
        }
        let new_size_s = stats.cluster_size[current_c] - 1.0;
        let new_size_t = stats.cluster_size[t] + 1.0;
        let new_vol_s = stats.cluster_volume[current_c] - deg;
        let new_vol_t = stats.cluster_volume[t] + deg;

        // Reset per-node delta accumulators and touched list.
        for &v in &touched {
            delta_edge[v] = 0.0;
            delta_total[v] = 0.0;
        }
        touched.clear();

        for ci in 0..k {
            for cj in ci..k {
                if ci != current_c && ci != t && cj != current_c && cj != t {
                    continue;
                }
                let lca_node = tree.lca(ci, cj);

                let old_edge = stats.edge_stat(ci, cj);
                let old_total = stats.total_stat(ci, cj, degree_corrected);

                let mut new_edge = old_edge;
                if ci == current_c && cj == current_c {
                    new_edge -= edge_to_cluster[current_c];
                } else if ci == t && cj == t {
                    new_edge += edge_to_cluster[t];
                } else if (ci == current_c && cj == t) || (ci == t && cj == current_c) {
                    new_edge = old_edge - edge_to_cluster[t] + edge_to_cluster[current_c];
                } else if ci == current_c {
                    new_edge -= edge_to_cluster[cj];
                } else if cj == current_c {
                    new_edge -= edge_to_cluster[ci];
                } else if ci == t {
                    new_edge += edge_to_cluster[cj];
                } else if cj == t {
                    new_edge += edge_to_cluster[ci];
                }

                let new_total = if degree_corrected {
                    let vol_ci = if ci == current_c {
                        new_vol_s
                    } else if ci == t {
                        new_vol_t
                    } else {
                        stats.cluster_volume[ci]
                    };
                    let vol_cj = if cj == current_c {
                        new_vol_s
                    } else if cj == t {
                        new_vol_t
                    } else {
                        stats.cluster_volume[cj]
                    };
                    if ci == cj {
                        vol_ci * vol_cj / 2.0
                    } else {
                        vol_ci * vol_cj
                    }
                } else {
                    let sz_ci = if ci == current_c {
                        new_size_s
                    } else if ci == t {
                        new_size_t
                    } else {
                        stats.cluster_size[ci]
                    };
                    let sz_cj = if cj == current_c {
                        new_size_s
                    } else if cj == t {
                        new_size_t
                    } else {
                        stats.cluster_size[cj]
                    };
                    if ci == cj {
                        sz_ci * (sz_ci - 1.0) / 2.0
                    } else {
                        sz_ci * sz_cj
                    }
                };

                if delta_edge[lca_node] == 0.0 && delta_total[lca_node] == 0.0 {
                    touched.push(lca_node);
                }
                delta_edge[lca_node] += new_edge - old_edge;
                delta_total[lca_node] += new_total - old_total;
            }
        }

        let mut delta = 0.0f64;
        for &v in &touched {
            let (a0, b0) = tree.node_params(v);
            let old_s = poisson_score_cpu(a0, b0, node_edge[v], node_total[v]);
            let new_s = poisson_score_cpu(
                a0,
                b0,
                node_edge[v] + delta_edge[v],
                node_total[v] + delta_total[v],
            );
            delta += new_s - old_s;
        }
        out[t] = delta;
    }
}

/// Build a weighted-adjacency list (cell → (neighbor, weight)) from an edge list.
pub fn build_hsbm_adj_list(n: usize, edges: &[WeightedEdge]) -> HsbmAdjList {
    let mut adj: HsbmAdjList = vec![Vec::new(); n];
    for &(i, j, w) in edges {
        adj[i].push((j, w as f64));
        adj[j].push((i, w as f64));
    }
    adj
}

/// Collapse fine KNN edges to super-edges by super-cell labels, summing weights.
pub fn coarsen_weighted_edges(
    fine_edges: &[(usize, usize)],
    super_labels: &[usize],
) -> Vec<WeightedEdge> {
    use rustc_hash::FxHashMap as HashMap;
    let mut map: HashMap<(usize, usize), f32> = HashMap::default();
    for &(i, j) in fine_edges {
        let a = super_labels[i];
        let b = super_labels[j];
        if a == b {
            // Within-super-cell edge: add to self-loop (within-block count).
            *map.entry((a, a)).or_insert(0.0) += 1.0;
        } else {
            let key = (a.min(b), a.max(b));
            *map.entry(key).or_insert(0.0) += 1.0;
        }
    }
    map.into_iter().map(|((i, j), w)| (i, j, w)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell_community::profiles::CellProfileStore;

    fn planted(n_per: usize, m: usize, k: usize) -> (CellProfileStore, Vec<(usize, usize)>, Vec<usize>) {
        let n = n_per * k;
        let mut profiles = vec![0.0f32; n * m];
        let mut labels = vec![0usize; n];
        for c in 0..k {
            for i in 0..n_per {
                let u = c * n_per + i;
                labels[u] = c;
                for g in 0..m {
                    profiles[u * m + g] = if g % k == c { 10.0 } else { 1.0 };
                }
            }
        }
        let mut edges = Vec::new();
        for c in 0..k {
            for i in 0..n_per {
                for d in 1..=3 {
                    let j = (i + d) % n_per;
                    edges.push((c * n_per + i, c * n_per + j));
                }
            }
        }
        (CellProfileStore::new(profiles, n, m), edges, labels)
    }

    #[test]
    fn test_delta_move_consistency() {
        let (store, fine_edges, truth) = planted(8, 6, 4);
        let weighted: Vec<WeightedEdge> = fine_edges
            .iter()
            .map(|&(i, j)| (i, j, 1.0f32))
            .collect();
        let n = store.n_cells;

        let init: Vec<usize> = (0..n).map(|u| (u * 3) % 4).collect();
        let adj = build_hsbm_adj_list(n, &weighted);
        let mut stats =
            CellHybridStats::new(&store, &weighted, n, 3, &init, 0.5, false, 1.0, 1.0);

        // Move a few cells.
        let mut current = init.clone();
        let moves = [(0usize, 1usize), (5, 2), (13, 0), (22, 3)];
        for &(u, new_c) in &moves {
            let old_c = current[u];
            stats.delta_move(u, old_c, new_c, &store, &adj[u]);
            current[u] = new_c;
        }

        // Rebuild from scratch and compare suff stats.
        let fresh =
            CellHybridStats::new(&store, &weighted, n, 3, &current, 0.5, false, 1.0, 1.0);
        for i in 0..stats.profile.gene_sum.len() {
            assert!((stats.profile.gene_sum[i] - fresh.profile.gene_sum[i]).abs() < 1e-9);
        }
        for i in 0..stats.structural.edge_counts.len() {
            assert!(
                (stats.structural.edge_counts[i] - fresh.structural.edge_counts[i]).abs() < 1e-9,
                "edge_count drift at {i}"
            );
        }
        for c in 0..stats.k() {
            assert!((stats.structural.cluster_size[c] - fresh.structural.cluster_size[c]).abs() < 1e-9);
            assert!((stats.structural.cluster_volume[c] - fresh.structural.cluster_volume[c]).abs() < 1e-9);
        }
        let _ = truth;
    }

    #[test]
    fn test_tree_delta_matches_brute_force() {
        let (store, fine_edges, _truth) = planted(6, 4, 4);
        let weighted: Vec<WeightedEdge> = fine_edges
            .iter()
            .map(|&(i, j)| (i, j, 1.0f32))
            .collect();
        let n = store.n_cells;

        let init: Vec<usize> = (0..n).map(|u| (u * 7) % 4).collect();
        let adj = build_hsbm_adj_list(n, &weighted);
        let stats =
            CellHybridStats::new(&store, &weighted, n, 3, &init, 1.0, false, 1.0, 1.0);

        let u = 5;
        let current_c = stats.membership()[u];
        let mut out = vec![0.0f64; stats.k()];
        compute_tree_log_probs_for_cell(
            u,
            current_c,
            &stats.structural,
            &stats.tree,
            &adj[u],
            false,
            &mut out,
        );

        // Brute force: compute full tree_score before and after each candidate move.
        let baseline = {
            let (ne, nt) = stats.structural.aggregate_to_tree(&stats.tree, false);
            let a0: Vec<f64> = (1..=stats.tree.num_nodes())
                .map(|n| stats.tree.node_params(n).0)
                .collect();
            let b0: Vec<f64> = (1..=stats.tree.num_nodes())
                .map(|n| stats.tree.node_params(n).1)
                .collect();
            tree_score_cpu(&a0, &b0, &ne[1..], &nt[1..])
        };
        for t in 0..stats.k() {
            if t == current_c {
                continue;
            }
            let mut trial = init.clone();
            trial[u] = t;
            let fresh = SufficientStats::from_edges(&weighted, n, stats.k(), &trial);
            let (ne, nt) = fresh.aggregate_to_tree(&stats.tree, false);
            let a0: Vec<f64> = (1..=stats.tree.num_nodes())
                .map(|n| stats.tree.node_params(n).0)
                .collect();
            let b0: Vec<f64> = (1..=stats.tree.num_nodes())
                .map(|n| stats.tree.node_params(n).1)
                .collect();
            let s_after = tree_score_cpu(&a0, &b0, &ne[1..], &nt[1..]);
            let expected = s_after - baseline;
            if (out[t] - expected).abs() >= 1e-8 {
                eprintln!("--- FAIL t={t} ---");
                eprintln!("got={:.9}, expected={:.9}", out[t], expected);

                // Recompute impl deltas for debugging.
                let (ne_before, nt_before) =
                    stats.structural.aggregate_to_tree(&stats.tree, false);
                let mut my_de = vec![0.0f64; stats.tree.num_nodes() + 1];
                let mut my_dt = vec![0.0f64; stats.tree.num_nodes() + 1];
                let kk = stats.k();
                let mut e2c = vec![0.0f64; kk];
                for &(nbr, w) in &adj[u] {
                    e2c[stats.structural.membership[nbr]] += w;
                }
                let deg_u: f64 = e2c.iter().sum();
                let nss = stats.structural.cluster_size[current_c] - 1.0;
                let nst = stats.structural.cluster_size[t] + 1.0;
                let nvs = stats.structural.cluster_volume[current_c] - deg_u;
                let nvt = stats.structural.cluster_volume[t] + deg_u;
                for ci in 0..kk {
                    for cj in ci..kk {
                        if ci != current_c && ci != t && cj != current_c && cj != t {
                            continue;
                        }
                        let lca_node = stats.tree.lca(ci, cj);
                        let old_edge = stats.structural.edge_stat(ci, cj);
                        let old_total = stats.structural.total_stat(ci, cj, false);
                        let mut new_edge = old_edge;
                        if ci == current_c && cj == current_c {
                            new_edge -= e2c[current_c];
                        } else if ci == t && cj == t {
                            new_edge += e2c[t];
                        } else if (ci == current_c && cj == t) || (ci == t && cj == current_c) {
                            new_edge = old_edge - e2c[t] + e2c[current_c];
                        } else if ci == current_c {
                            new_edge -= e2c[cj];
                        } else if cj == current_c {
                            new_edge -= e2c[ci];
                        } else if ci == t {
                            new_edge += e2c[cj];
                        } else if cj == t {
                            new_edge += e2c[ci];
                        }
                        let sz_ci = if ci == current_c { nss } else if ci == t { nst } else { stats.structural.cluster_size[ci] };
                        let sz_cj = if cj == current_c { nss } else if cj == t { nst } else { stats.structural.cluster_size[cj] };
                        let new_total = if ci == cj { sz_ci * (sz_ci - 1.0) / 2.0 } else { sz_ci * sz_cj };
                        my_de[lca_node] += new_edge - old_edge;
                        my_dt[lca_node] += new_total - old_total;
                        let _ = nvs; let _ = nvt;
                        eprintln!(
                            "  pair ({ci},{cj}) LCA={lca_node}: old_e={old_edge}, new_e={new_edge}, old_t={old_total}, new_t={new_total}"
                        );
                    }
                }
                for v in 1..=stats.tree.num_nodes() {
                    let my_ne = ne_before[v] + my_de[v];
                    let my_nt = nt_before[v] + my_dt[v];
                    if (my_ne - ne[v]).abs() > 1e-9 || (my_nt - nt[v]).abs() > 1e-9 {
                        eprintln!(
                            "  node {v} MISMATCH: my new (e={my_ne}, t={my_nt}) vs brute (e={}, t={})",
                            ne[v], nt[v]
                        );
                    }
                }
                panic!("mismatch");
            }
        }
    }
}
