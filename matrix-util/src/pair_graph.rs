//! `FeaturePairGraph` — undirected graph over named features (genes,
//! peaks, etc.) loaded from a two-column edge list (e.g. BioGRID, STRING,
//! peak-coaccessibility).
//!
//! Name resolution piggy-backs on [`crate::membership::GeneIndexResolver`]
//! (exact → delimiter → optional prefix). The result holds canonical
//! undirected edges (`u < v`), de-duplicated and sorted, ready for
//! downstream graph algorithms (Leiden, SGC propagation, link community).
//!
//! Every operation that needs row-neighbor access builds a directed CSR
//! once and runs the inner kernel via rayon. The shared
//! [`FeaturePairGraph::shared_neighbor_counts`] kernel drives SNN
//! augmentation, shared-neighbor QC pruning, and hub-degree capping.

use crate::common_io::read_lines_of_words_delim;
use crate::graph::AdjListGraph;
use crate::membership::{detect_delimiter, GeneIndexResolver};
use crate::parquet::{parquet_add_bytearray, parquet_add_string_column, ParquetWriter};
use log::info;
use parquet::basic::Type as ParquetType;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet as HashSet};

pub struct FeaturePairGraph {
    pub feature_names: Vec<Box<str>>,
    pub n_features: usize,
    pub feature_edges: Vec<(usize, usize)>,
}

//////////////////////////////////////////////////////////////////////
// Internal directed-CSR adjacency — built on demand, never stored. //
//////////////////////////////////////////////////////////////////////

struct AdjCsr {
    /// `[n_features + 1]` offsets into `col_idx`.
    row_ptr: Vec<usize>,
    /// `[2 · E]` directed neighbors, each row sorted ascending.
    col_idx: Vec<u32>,
}

impl AdjCsr {
    #[inline]
    fn row(&self, u: usize) -> &[u32] {
        &self.col_idx[self.row_ptr[u]..self.row_ptr[u + 1]]
    }
}

/// Sorted-merge intersection count for two ascending-sorted slices.
/// Used by every shared-neighbor / SNN kernel.
#[inline]
fn intersect_count(a: &[u32], b: &[u32]) -> usize {
    let (a, b) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut i = 0usize;
    let mut j = 0usize;
    let mut count = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

//////////////////
// Construction //
//////////////////

impl FeaturePairGraph {
    /// Build a feature-pair graph from an external two-column edge list.
    ///
    /// Names are matched against `feature_names` via
    /// `GeneIndexResolver` (exact → delimiter-stripped → optional prefix).
    /// Self-loops, duplicates, and edges referencing unknown names are
    /// dropped silently. The resulting `feature_edges` are canonical
    /// (`u < v`), unique, and sorted.
    pub fn from_edge_list(
        file_path: &str,
        feature_names: Vec<Box<str>>,
        allow_prefix: bool,
        delimiter: Option<char>,
    ) -> anyhow::Result<Self> {
        Self::from_edge_list_canon(file_path, feature_names, allow_prefix, delimiter, &|s| {
            s.into()
        })
    }

    /// Like [`Self::from_edge_list`] but canonicalizes both the feature
    /// axis names and each edge endpoint through `canon` before matching.
    /// Lets callers reuse a domain canonicalizer (e.g. `FeatureNameKind`
    /// that normalizes gene symbols *and* `chrX:start-end` loci) so an
    /// edge file with raw names resolves against a canonicalized axis.
    /// The stored `feature_names` remain the originals.
    pub fn from_edge_list_canon(
        file_path: &str,
        feature_names: Vec<Box<str>>,
        allow_prefix: bool,
        delimiter: Option<char>,
        canon: &dyn Fn(&str) -> Box<str>,
    ) -> anyhow::Result<Self> {
        let n_features = feature_names.len();
        // Resolver keyed on canonicalized names; index i still maps to
        // feature_names[i] because canonicalization preserves order.
        let canon_names: Vec<Box<str>> = feature_names.iter().map(|n| canon(n)).collect();
        let resolver = GeneIndexResolver::build(&canon_names, delimiter, allow_prefix);

        let file_delim = detect_delimiter(file_path);
        let read_out = read_lines_of_words_delim(file_path, file_delim, -1)?;

        let mut edge_set: HashSet<(usize, usize)> = Default::default();
        let mut n_matched = 0usize;
        let mut n_skipped = 0usize;
        for line in &read_out.lines {
            if line.len() < 2 {
                continue;
            }
            let idx1 = resolver.resolve(&canon(&line[0]));
            let idx2 = resolver.resolve(&canon(&line[1]));
            match (idx1, idx2) {
                (Some(i), Some(j)) if i != j => {
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    edge_set.insert((lo, hi));
                    n_matched += 1;
                }
                _ => {
                    n_skipped += 1;
                }
            }
        }

        let mut feature_edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
        feature_edges.par_sort_unstable();
        info!(
            "Feature-pair graph: {} edges loaded from {} ({} matched, {} skipped, {} unique)",
            read_out.lines.len(),
            file_path,
            n_matched,
            n_skipped,
            feature_edges.len(),
        );
        Ok(Self {
            feature_names,
            n_features,
            feature_edges,
        })
    }

    /// Keep only edges at the given indices.
    pub fn filter_edges(&mut self, keep_indices: &[usize]) {
        self.feature_edges = keep_indices
            .iter()
            .map(|&i| self.feature_edges[i])
            .collect();
    }

    pub fn num_edges(&self) -> usize {
        self.feature_edges.len()
    }

    pub fn num_features(&self) -> usize {
        self.n_features
    }

    /// Per-feature undirected degree from the canonical edge list.
    pub fn feature_degrees(&self) -> Vec<usize> {
        let mut d = vec![0usize; self.n_features];
        for &(u, v) in &self.feature_edges {
            d[u] += 1;
            d[v] += 1;
        }
        d
    }

    /// Build directed adjacency: `adj[g] = [(neighbor, edge_idx)]` where
    /// `neighbor > g` (so each undirected edge appears once).
    pub fn build_directed_adjacency(&self) -> Vec<Vec<(usize, usize)>> {
        let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.n_features];
        for (edge_idx, &(g1, g2)) in self.feature_edges.iter().enumerate() {
            adj[g1].push((g2, edge_idx));
        }
        adj
    }

    /////////////////////////////////////////////
    // CSR build (rayon-parallel per-row sort) //
    /////////////////////////////////////////////

    fn build_adj_csr(&self) -> AdjCsr {
        let n = self.n_features;
        let mut per_row: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        for &(u, v) in &self.feature_edges {
            per_row[u].push(v as u32);
            per_row[v].push(u as u32);
        }
        per_row.par_iter_mut().for_each(|row| row.sort_unstable());

        let total: usize = per_row.iter().map(Vec::len).sum();
        let mut row_ptr = Vec::with_capacity(n + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::with_capacity(total);
        for row in per_row.iter() {
            col_idx.extend_from_slice(row);
            row_ptr.push(col_idx.len());
        }
        AdjCsr { row_ptr, col_idx }
    }

    //////////////////////////////////
    // Shared kernel: |N(u) ∩ N(v)| //
    //////////////////////////////////

    /// Common-neighbor count for each `(u, v)` in `pairs`, computed in
    /// parallel via sorted-merge on the CSR rows. O(deg(u) + deg(v))
    /// per pair. Self-loops (`u == v`) yield `deg(u)` (correct but
    /// usually meaningless).
    pub fn shared_neighbor_counts(&self, pairs: &[(usize, usize)]) -> Vec<usize> {
        let csr = self.build_adj_csr();
        pairs
            .par_iter()
            .map(|&(u, v)| intersect_count(csr.row(u), csr.row(v)))
            .collect()
    }

    ///////////////////////////////////////////////////////////////////////
    // Mutations: SNN augment, shared-neighbor QC prune, hub-cap, k-core //
    ///////////////////////////////////////////////////////////////////////

    /// Augment with shared-neighbor edges: any unordered pair `(u, v)`
    /// with at least `min_shared` undirected neighbors in common gains a
    /// synthetic edge (unless one is already present). `min_shared = 0`
    /// is a no-op. Parallel over the outer node id; sorted-merge
    /// intersection avoids the HashSet rebuild that the old serial
    /// implementation paid per call.
    pub fn augment_with_snn(&mut self, min_shared: usize) {
        if min_shared == 0 {
            return;
        }
        let csr = self.build_adj_csr();
        let existing: HashSet<(u32, u32)> = self
            .feature_edges
            .iter()
            .map(|&(u, v)| (u as u32, v as u32))
            .collect();

        let new_edges: Vec<(usize, usize)> = (0..self.n_features)
            .into_par_iter()
            .filter(|&u| !csr.row(u).is_empty())
            .flat_map_iter(|u| {
                let mut seen: HashSet<u32> = HashSet::default();
                let mut local: Vec<(usize, usize)> = Vec::new();
                let ru = csr.row(u);
                for &m in ru {
                    for &v in csr.row(m as usize) {
                        if (v as usize) <= u {
                            continue;
                        }
                        if !seen.insert(v) {
                            continue;
                        }
                        if existing.contains(&(u as u32, v)) {
                            continue;
                        }
                        let rv = csr.row(v as usize);
                        if intersect_count(ru, rv) >= min_shared {
                            local.push((u, v as usize));
                        }
                    }
                }
                local
            })
            .collect();

        if !new_edges.is_empty() {
            let added = new_edges.len();
            self.feature_edges.extend(new_edges);
            self.feature_edges.par_sort_unstable();
            self.feature_edges.dedup();
            info!(
                "SNN augmentation (min_shared={}): +{} edges ({} total)",
                min_shared,
                added,
                self.feature_edges.len(),
            );
        }
    }

    /// QC prune: drop any edge `(u, v)` whose endpoints share fewer than
    /// `min_shared` neighbors in the current graph. Standard PPI
    /// denoising — an edge with no corroborating shared interactor is
    /// likely a noisy hit. `min_shared = 0` is a no-op.
    pub fn prune_by_shared_neighbors(&mut self, min_shared: usize) {
        if min_shared == 0 || self.feature_edges.is_empty() {
            return;
        }
        let initial = self.feature_edges.len();
        let snapshot = self.feature_edges.clone();
        let counts = self.shared_neighbor_counts(&snapshot);
        self.feature_edges = snapshot
            .into_par_iter()
            .zip(counts.into_par_iter())
            .filter_map(|(e, c)| (c >= min_shared).then_some(e))
            .collect();
        // par_iter zip may not preserve order; resort to canonical.
        self.feature_edges.par_sort_unstable();
        let kept = self.feature_edges.len();
        if kept != initial {
            info!(
                "shared-neighbor QC (min_shared={}): {} edges → {} edges",
                min_shared, initial, kept,
            );
        }
    }

    /// Per-node hard cap on degree, ranked by shared-neighbor count.
    /// For each node `u` with `deg(u) > max_degree`, sort its neighbors
    /// by `|N(u) ∩ N(v)|` descending (ties broken by neighbor id) and
    /// keep the top `max_degree`. Symmetric union — an edge survives iff
    /// *either* endpoint kept it. `max_degree = 0` is a no-op. Used to
    /// cap PPI hubs whose degree would otherwise blow up the per-cell
    /// sub-adjacency cache.
    pub fn cap_per_node_degree(&mut self, max_degree: usize) {
        if max_degree == 0 || self.feature_edges.is_empty() {
            return;
        }
        let initial = self.feature_edges.len();
        let snapshot = self.feature_edges.clone();
        let sn_scores = self.shared_neighbor_counts(&snapshot);

        let mut per_node: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); self.n_features];
        for (i, &(u, v)) in snapshot.iter().enumerate() {
            per_node[u].push((v as u32, sn_scores[i] as u32, i as u32));
            per_node[v].push((u as u32, sn_scores[i] as u32, i as u32));
        }

        let kept_per_node: Vec<Vec<u32>> = per_node
            .par_iter_mut()
            .map(|edges_of_u| {
                if edges_of_u.len() <= max_degree {
                    return edges_of_u.iter().map(|&(_, _, idx)| idx).collect();
                }
                edges_of_u.sort_unstable_by_key(|&(nbr, score, _)| (std::cmp::Reverse(score), nbr));
                edges_of_u
                    .iter()
                    .take(max_degree)
                    .map(|&(_, _, idx)| idx)
                    .collect()
            })
            .collect();

        let mut keep: Vec<bool> = vec![false; snapshot.len()];
        for kept_idxs in &kept_per_node {
            for &idx in kept_idxs {
                keep[idx as usize] = true;
            }
        }
        self.feature_edges = snapshot
            .into_iter()
            .zip(keep.iter())
            .filter_map(|(e, &k)| k.then_some(e))
            .collect();
        let kept = self.feature_edges.len();
        if kept != initial {
            info!(
                "per-node degree cap (max={}, SN-score, union): {} edges → {} edges",
                max_degree, initial, kept,
            );
        }
    }

    /// Iterative k-core: drop every feature whose current degree is
    /// `< min_degree`, then recompute degrees and repeat until the
    /// surviving subgraph is `(min_degree)`-degenerate. The feature axis
    /// itself is kept the same — only edges incident to pruned features
    /// are removed. `min_degree = 0` is a no-op.
    pub fn prune_by_min_degree(&mut self, min_degree: usize) {
        if min_degree == 0 || self.feature_edges.is_empty() {
            return;
        }
        let initial = self.feature_edges.len();
        loop {
            let degrees = self.feature_degrees();
            let drop: Vec<bool> = degrees.iter().map(|&d| d > 0 && d < min_degree).collect();
            if !drop.iter().any(|&x| x) {
                break;
            }
            self.feature_edges.retain(|&(u, v)| !drop[u] && !drop[v]);
            if self.feature_edges.is_empty() {
                break;
            }
        }
        let final_n = self.feature_edges.len();
        if final_n != initial {
            info!(
                "k-core pruning (min_degree={}): {} edges → {} edges",
                min_degree, initial, final_n,
            );
        }
    }

    ///////////////////////////////////////////////////////////////
    // Derived relations: second-order neighbours, diffusion top-k //
    ///////////////////////////////////////////////////////////////

    /// Second-order edges: unordered pairs `(u, v)` NOT directly linked
    /// that share at least `min_shared` neighbours, with the count. Where
    /// [`Self::augment_with_snn`] folds such pairs into the graph
    /// unweighted, this returns them on their own so a caller can treat
    /// "co-interactors" as a relation distinct from "interactors",
    /// weighted by how many partners they share. Each entry is
    /// `(u, v, shared, union)` so a caller can weight by the raw count or
    /// by the Jaccard overlap `shared / union`. `top_k > 0` keeps, for
    /// every node, only its `top_k` co-interactors by Jaccard (ties by
    /// count, then id) — the degree-normalised choice, since on a
    /// scale-free graph a raw count is dominated by the hubs every pair
    /// shares by chance — and a pair survives when either endpoint keeps
    /// it, which bounds the result at `n · top_k` where "every pair
    /// sharing one neighbour" would be quadratic. `min_shared = 0` yields
    /// nothing. Canonical `u < v`, sorted.
    pub fn shared_neighbor_edges(
        &self,
        min_shared: usize,
        top_k: usize,
    ) -> Vec<(usize, usize, usize, usize)> {
        if min_shared == 0 {
            return Vec::new();
        }
        let csr = self.build_adj_csr();
        let existing: HashSet<(u32, u32)> = self
            .feature_edges
            .iter()
            .map(|&(u, v)| (u as u32, v as u32))
            .collect();
        // Per node, every second-order partner with the count, then the
        // per-node top-k; the union of kept choices is folded to canonical
        // pairs.
        let per_node: Vec<Vec<(usize, usize, usize)>> = (0..self.n_features)
            .into_par_iter()
            .map(|u| {
                let ru = csr.row(u);
                if ru.is_empty() {
                    return Vec::new();
                }
                let mut seen: HashSet<u32> = HashSet::default();
                let mut local: Vec<(usize, usize, usize)> = Vec::new();
                for &m in ru {
                    for &v in csr.row(m as usize) {
                        let key = if (v as usize) < u {
                            (v, u as u32)
                        } else {
                            (u as u32, v)
                        };
                        if v as usize == u || !seen.insert(v) || existing.contains(&key) {
                            continue;
                        }
                        let rv = csr.row(v as usize);
                        let c = intersect_count(ru, rv);
                        if c >= min_shared {
                            local.push((v as usize, c, ru.len() + rv.len() - c));
                        }
                    }
                }
                if top_k > 0 && local.len() > top_k {
                    // Jaccard descending: c/un > c'/un' ⇔ c·un' > c'·un.
                    local.sort_unstable_by(|&(v, c, un), &(v2, c2, un2)| {
                        (c2 * un).cmp(&(c * un2)).then(c2.cmp(&c)).then(v.cmp(&v2))
                    });
                    local.truncate(top_k);
                }
                local
            })
            .collect();
        let mut out: Vec<(usize, usize, usize, usize)> = per_node
            .into_iter()
            .enumerate()
            .flat_map(|(u, vs)| {
                vs.into_iter()
                    .map(move |(v, c, un)| (u.min(v), u.max(v), c, un))
            })
            .collect();
        out.par_sort_unstable();
        out.dedup();
        out
    }

    /// Personalized PageRank from every node, truncated to its `k`
    /// strongest targets (self excluded), by the forward-push
    /// approximation (Andersen, Chung & Lang 2006): random walk with
    /// restart probability `alpha` on the unweighted graph, residual mass
    /// per node pushed until every residual is below `eps · degree`. Local
    /// and sparse, so the cost per source is `O(1/(eps · alpha))`
    /// regardless of graph size; sources run in parallel. Scores are the
    /// PPR mass in `(0, 1]`; the caller decides how to weight them.
    /// Isolated nodes get an empty list.
    pub fn personalized_pagerank_top_k(
        &self,
        alpha: f64,
        eps: f64,
        k: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let csr = self.build_adj_csr();
        let n = self.n_features;
        let alpha = alpha.clamp(1e-6, 1.0);
        let eps = eps.max(1e-12);
        (0..n)
            .into_par_iter()
            .map(|s| {
                if csr.row(s).is_empty() || k == 0 {
                    return Vec::new();
                }
                // Sparse push with a work queue; `p` and `r` live in hash maps
                // because a source touches a small neighbourhood.
                let mut p: FxHashMap<u32, f64> = FxHashMap::default();
                let mut r: FxHashMap<u32, f64> = FxHashMap::default();
                r.insert(s as u32, 1.0);
                let mut queue: Vec<u32> = vec![s as u32];
                let mut queued: HashSet<u32> = HashSet::default();
                queued.insert(s as u32);
                while let Some(u) = queue.pop() {
                    queued.remove(&u);
                    let du = csr.row(u as usize).len() as f64;
                    let ru = r.get(&u).copied().unwrap_or(0.0);
                    if du == 0.0 || ru < eps * du {
                        continue;
                    }
                    *p.entry(u).or_default() += alpha * ru;
                    let push = (1.0 - alpha) * ru / du;
                    r.insert(u, 0.0);
                    for &v in csr.row(u as usize) {
                        let rv = r.entry(v).or_default();
                        *rv += push;
                        let dv = csr.row(v as usize).len() as f64;
                        if *rv >= eps * dv && queued.insert(v) {
                            queue.push(v);
                        }
                    }
                }
                let mut top: Vec<(usize, f32)> = p
                    .into_iter()
                    .filter(|&(v, _)| v as usize != s)
                    .map(|(v, m)| (v as usize, m as f32))
                    .collect();
                top.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                top.truncate(k);
                top
            })
            .collect()
    }

    /// Symmetric adjacency-list view implementing
    /// `crate::graph::WeightedGraph` (for Leiden, SGC, etc.).
    pub fn to_adj_list(&self) -> AdjListGraph {
        AdjListGraph::from_unweighted_edges(self.n_features, &self.feature_edges)
    }

    /// Write the canonical edge list (two columns of feature names) to
    /// parquet. `col_names` lets callers pick `("gene1","gene2")`,
    /// `("peak1","peak2")`, etc.
    pub fn to_parquet(&self, file_path: &str, col_names: (&str, &str)) -> anyhow::Result<()> {
        let n_edges = self.feature_edges.len();
        let column_names: Vec<Box<str>> = vec![col_names.0.into(), col_names.1.into()];
        let column_types = vec![ParquetType::BYTE_ARRAY, ParquetType::BYTE_ARRAY];

        let shape = (n_edges, column_names.len());
        let writer = ParquetWriter::new(
            file_path,
            shape,
            (None, Some(&column_names)),
            Some(&column_types),
            None,
        )?;
        let row_names = writer.row_names_vec();
        let mut writer = writer.get_writer()?;
        let mut row_group_writer = writer.next_row_group()?;

        parquet_add_bytearray(&mut row_group_writer, row_names)?;

        let names1: Vec<Box<str>> = self
            .feature_edges
            .par_iter()
            .map(|&(g1, _)| self.feature_names[g1].clone())
            .collect();
        parquet_add_string_column(&mut row_group_writer, &names1)?;

        let names2: Vec<Box<str>> = self
            .feature_edges
            .par_iter()
            .map(|&(_, g2)| self.feature_names[g2].clone())
            .collect();
        parquet_add_string_column(&mut row_group_writer, &names2)?;

        row_group_writer.close()?;
        writer.close()?;
        Ok(())
    }
}

/// Synthetic test graph with feature names `g0..g{n-1}` from an unordered
/// edge list. Public so downstream test modules can reuse it.
pub fn test_graph_from_edges(edges: &[(usize, usize)], n_features: usize) -> FeaturePairGraph {
    let names: Vec<Box<str>> = (0..n_features).map(|i| format!("g{}", i).into()).collect();
    let mut canonical: Vec<(usize, usize)> = edges
        .iter()
        .map(|&(a, b)| if a < b { (a, b) } else { (b, a) })
        .collect();
    canonical.sort();
    canonical.dedup();
    FeaturePairGraph {
        feature_names: names,
        n_features,
        feature_edges: canonical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles {0,1,2} and {3,4,5} joined by the edge (2,3), plus a
    /// pendant 6 on node 0 and an isolated node 7.
    fn two_triangles() -> FeaturePairGraph {
        FeaturePairGraph {
            feature_names: (0..8).map(|i| format!("g{i}").into_boxed_str()).collect(),
            n_features: 8,
            feature_edges: vec![
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 5),
                (0, 6),
            ],
        }
    }

    #[test]
    fn shared_neighbor_edges_are_second_order_only_with_their_counts() {
        let g = two_triangles();
        // Pairs sharing ≥ 1 neighbour that are not directly linked:
        // (1,6) via 0; (2,6) via 0; (1,3) via 2; (0,3) via 2; (2,4) via 3;
        // (2,5) via 3. Directly linked pairs (0,1) etc. never appear.
        let snn = g.shared_neighbor_edges(1, 0);
        let pairs: Vec<(usize, usize, usize)> = snn.iter().map(|&(u, v, c, _)| (u, v, c)).collect();
        assert_eq!(
            pairs,
            vec![
                (0, 3, 1),
                (1, 3, 1),
                (1, 6, 1),
                (2, 4, 1),
                (2, 5, 1),
                (2, 6, 1)
            ]
        );
        assert!(snn
            .iter()
            .all(|&(u, v, _, _)| !g.feature_edges.contains(&(u, v))));
        // union = deg(u) + deg(v) − shared: (1,6) has degrees 2 and 1 → 2.
        assert!(snn.contains(&(1, 6, 1, 2)));
        assert!(snn.contains(&(0, 3, 1, 5)), "deg 3 + deg 3 − 1");
        assert!(
            g.shared_neighbor_edges(2, 0).is_empty(),
            "no pair shares two neighbours"
        );
        assert!(g.shared_neighbor_edges(0, 10).is_empty());
        // top-1 per node by Jaccard, and a pair survives when either side
        // keeps it. Here every pair survives: 2 keeps 6 (union 3 beats the
        // union-4 ties with 4 and 5), but 4 and 5 each have 2 as their only
        // candidate, and 3 keeps 1 (union 4) over 0 (union 5) while 0's only
        // candidate is 3.
        let top1 = g.shared_neighbor_edges(1, 1);
        assert_eq!(top1, snn);
        // Truncation bites on a star: the five leaves pairwise share the hub
        // (10 pairs, all Jaccard 1/2), and with k = 1 each leaf keeps the
        // lowest-id other leaf, so only leaf 1's four pairs survive.
        let star = FeaturePairGraph {
            feature_names: (0..6).map(|i| format!("g{i}").into_boxed_str()).collect(),
            n_features: 6,
            feature_edges: (1..6).map(|i| (0, i)).collect(),
        };
        assert_eq!(star.shared_neighbor_edges(1, 0).len(), 10);
        let s1: Vec<(usize, usize)> = star
            .shared_neighbor_edges(1, 1)
            .iter()
            .map(|&(u, v, _, _)| (u, v))
            .collect();
        assert_eq!(s1, vec![(1, 2), (1, 3), (1, 4), (1, 5)]);
        // A denser case: a 4-clique minus one edge — the missing pair shares 2.
        let h = FeaturePairGraph {
            feature_names: (0..4).map(|i| format!("g{i}").into_boxed_str()).collect(),
            n_features: 4,
            feature_edges: vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)],
        };
        assert_eq!(h.shared_neighbor_edges(2, 0), vec![(2, 3, 2, 2)]);
    }

    #[test]
    fn personalized_pagerank_ranks_the_own_triangle_above_the_far_one_and_skips_isolated_nodes() {
        let g = two_triangles();
        let ppr = g.personalized_pagerank_top_k(0.15, 1e-6, 3);
        assert_eq!(ppr.len(), 8);
        assert!(ppr[7].is_empty(), "isolated source has no targets");
        let top0: Vec<usize> = ppr[0].iter().map(|&(v, _)| v).collect();
        assert!(!top0.contains(&0), "self excluded");
        assert_eq!(top0.len(), 3);
        assert!(
            top0.contains(&1) && top0.contains(&2),
            "own triangle first: {top0:?}"
        );
        assert!(
            !top0.contains(&4) && !top0.contains(&5),
            "far triangle beyond the top 3: {top0:?}"
        );
        // Scores fall off with the rank and stay in (0, 1].
        let s0: Vec<f32> = ppr[0].iter().map(|&(_, m)| m).collect();
        assert!(s0.windows(2).all(|w| w[0] >= w[1]));
        assert!(s0.iter().all(|&m| m > 0.0 && m <= 1.0));
        // From node 4, node 2 (two hops via 3) outranks node 0 (three hops).
        let rank = |src: usize, v: usize| ppr[src].iter().position(|&(t, _)| t == v);
        let full = g.personalized_pagerank_top_k(0.15, 1e-7, 7);
        let rank_full = |src: usize, v: usize| full[src].iter().position(|&(t, _)| t == v).unwrap();
        assert!(rank_full(4, 2) < rank_full(4, 0));
        assert!(rank(4, 3).is_some() && rank(4, 5).is_some());
    }
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn names_of(names: &[&str]) -> Vec<Box<str>> {
        names.iter().map(|&s| s.into()).collect()
    }

    fn write_edge_file(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::with_suffix(".tsv").unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f.flush().unwrap();
        f
    }

    #[test]
    fn from_edge_list_exact_match() {
        let file = write_edge_file(&["TP53\tBRCA1", "BRCA1\tEGFR", "TP53\tEGFR"]);
        let names = names_of(&["TP53", "BRCA1", "EGFR", "MYC"]);
        let g = FeaturePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
            .unwrap();
        assert_eq!(g.num_features(), 4);
        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.feature_edges, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn from_edge_list_dedup_and_self_loop() {
        let file = write_edge_file(&["A\tB", "B\tA", "A\tB", "A\tA"]);
        let names = names_of(&["A", "B", "C"]);
        let g = FeaturePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
            .unwrap();
        assert_eq!(g.feature_edges, vec![(0, 1)]);
    }

    #[test]
    fn from_edge_list_unmatched_skipped() {
        let file = write_edge_file(&["TP53\tBRCA1", "UNK\tBRCA1", "TP53\tUNK2"]);
        let names = names_of(&["TP53", "BRCA1"]);
        let g = FeaturePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
            .unwrap();
        assert_eq!(g.feature_edges, vec![(0, 1)]);
    }

    #[test]
    fn from_edge_list_prefix_and_delim_match() {
        let file = write_edge_file(&["TP53\tBRCA1"]);
        let g_prefix = FeaturePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names_of(&["TP53.1", "BRCA1.2"]),
            true,
            None,
        )
        .unwrap();
        assert_eq!(g_prefix.num_edges(), 1);

        let g_delim = FeaturePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names_of(&["TP53.v1", "BRCA1.v2"]),
            false,
            Some('.'),
        )
        .unwrap();
        assert_eq!(g_delim.num_edges(), 1);
    }

    #[test]
    fn from_edge_list_csv() {
        let mut f = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(f, "A,B").unwrap();
        writeln!(f, "B,C").unwrap();
        f.flush().unwrap();
        let g = FeaturePairGraph::from_edge_list(
            f.path().to_str().unwrap(),
            names_of(&["A", "B", "C"]),
            false,
            None,
        )
        .unwrap();
        assert_eq!(g.feature_edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn feature_degrees_triangle() {
        let g = test_graph_from_edges(&[(0, 1), (0, 2), (1, 2)], 3);
        assert_eq!(g.feature_degrees(), vec![2, 2, 2]);
    }

    #[test]
    fn shared_neighbors_triangle() {
        // Triangle: every pair shares the third node as a shared neighbor.
        let g = test_graph_from_edges(&[(0, 1), (0, 2), (1, 2)], 3);
        let counts = g.shared_neighbor_counts(&[(0, 1), (0, 2), (1, 2)]);
        assert_eq!(counts, vec![1, 1, 1]);
    }

    #[test]
    fn shared_neighbors_path() {
        // Path 0-1-2-3: (0,2) shares {1}, (0,3) shares {}, (1,3) shares {2}.
        let g = test_graph_from_edges(&[(0, 1), (1, 2), (2, 3)], 4);
        let counts = g.shared_neighbor_counts(&[(0, 2), (0, 3), (1, 3)]);
        assert_eq!(counts, vec![1, 0, 1]);
    }

    #[test]
    fn snn_zero_is_noop() {
        let mut g = test_graph_from_edges(&[(0, 1), (1, 2)], 3);
        let before = g.feature_edges.clone();
        g.augment_with_snn(0);
        assert_eq!(g.feature_edges, before);
    }

    #[test]
    fn snn_two_hop() {
        let mut g = test_graph_from_edges(&[(0, 1), (1, 2)], 3);
        g.augment_with_snn(1);
        assert!(g.feature_edges.contains(&(0, 2)));
        assert_eq!(g.feature_edges.len(), 3);
    }

    #[test]
    fn snn_respects_min_shared() {
        let mut g = test_graph_from_edges(&[(0, 1), (1, 3)], 4);
        g.augment_with_snn(2);
        assert_eq!(g.feature_edges.len(), 2);
    }

    #[test]
    fn snn_no_duplicates() {
        let mut g = test_graph_from_edges(&[(0, 1), (0, 2), (1, 2)], 3);
        g.augment_with_snn(1);
        let mut dedup = g.feature_edges.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(dedup, g.feature_edges);
    }

    #[test]
    fn sn_prune_zero_is_noop() {
        let mut g = test_graph_from_edges(&[(0, 1), (1, 2)], 3);
        let before = g.feature_edges.clone();
        g.prune_by_shared_neighbors(0);
        assert_eq!(g.feature_edges, before);
    }

    #[test]
    fn sn_prune_drops_isolated_edge() {
        // 0-1 has no shared neighbor; 1-2-3-1 triangle is fully connected.
        let mut g = test_graph_from_edges(&[(0, 1), (1, 2), (1, 3), (2, 3)], 4);
        g.prune_by_shared_neighbors(1);
        assert!(!g.feature_edges.contains(&(0, 1)));
        assert!(g.feature_edges.contains(&(1, 2)));
        assert!(g.feature_edges.contains(&(1, 3)));
        assert!(g.feature_edges.contains(&(2, 3)));
        assert_eq!(g.feature_edges.len(), 3);
    }

    #[test]
    fn cap_zero_is_noop() {
        let mut g = test_graph_from_edges(&[(0, 1), (0, 2), (1, 2)], 3);
        let before = g.feature_edges.clone();
        g.cap_per_node_degree(0);
        assert_eq!(g.feature_edges, before);
    }

    #[test]
    fn cap_drops_zero_cn_edge_via_union() {
        // Hub 0 = {1,2,3,4} (deg 4). Node 4 = {0,5,6} (deg 3) with 5-6
        // also connected, so node 4's high-CN neighbors are {5,6}.
        // Triangles 0-1-2 and 0-1-3 give CN(0,1)=2, CN(0,2)=CN(0,3)=1,
        // CN(0,4)=0. Cap=2: hub 0 picks {1,2}; node 4 picks {5,6}.
        // Neither endpoint ranks (0,4) in its top-2; union drops it.
        let mut g = test_graph_from_edges(
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (4, 5),
                (4, 6),
                (5, 6),
            ],
            7,
        );
        g.cap_per_node_degree(2);
        assert!(!g.feature_edges.contains(&(0, 4)));
        // Highest-CN edge from hub 0 survives.
        assert!(g.feature_edges.contains(&(0, 1)));
    }

    #[test]
    fn cap_union_symmetric() {
        // Even if a hub's cap drops an edge, the *other* endpoint may
        // still keep it — union semantics. Star with hub 0 capped to 1,
        // but leaf 4 has only edge (0,4) so leaf 4 *must* keep it.
        let mut g = test_graph_from_edges(&[(0, 1), (0, 2), (0, 4), (1, 2)], 5);
        g.cap_per_node_degree(1);
        // Hub 0 picks its highest-CN neighbor; leaf 4's only neighbor is 0,
        // so (0,4) is in leaf 4's top-1 and survives via the union.
        assert!(g.feature_edges.contains(&(0, 4)));
    }

    #[test]
    fn directed_adjacency() {
        let g = test_graph_from_edges(&[(0, 1), (0, 2), (1, 2)], 3);
        let adj = g.build_directed_adjacency();
        assert_eq!(adj[0], vec![(1, 0), (2, 1)]);
        assert_eq!(adj[1], vec![(2, 2)]);
        assert!(adj[2].is_empty());
    }
}
