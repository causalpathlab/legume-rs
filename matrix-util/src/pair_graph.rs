//! `FeaturePairGraph` — undirected graph over named features (genes,
//! peaks, etc.) loaded from a two-column edge list (e.g. BioGRID, STRING,
//! peak-coaccessibility).
//!
//! Name resolution piggy-backs on [`crate::membership::GeneIndexResolver`]
//! (exact → delimiter → optional prefix). The result holds canonical
//! undirected edges (`u < v`), de-duplicated and sorted, ready for
//! downstream graph algorithms (Leiden, SGC propagation, link community).

use crate::common_io::read_lines_of_words_delim;
use crate::graph::AdjListGraph;
use crate::membership::{detect_delimiter, GeneIndexResolver};
use crate::parquet::{parquet_add_bytearray, parquet_add_string_column, ParquetWriter};
use log::info;
use parquet::basic::Type as ParquetType;
use rustc_hash::FxHashSet as HashSet;

pub struct FeaturePairGraph {
    pub feature_names: Vec<Box<str>>,
    pub n_features: usize,
    pub feature_edges: Vec<(usize, usize)>,
}

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
        let n_features = feature_names.len();
        let resolver = GeneIndexResolver::build(&feature_names, delimiter, allow_prefix);

        let file_delim = detect_delimiter(file_path);
        let read_out = read_lines_of_words_delim(file_path, file_delim, -1)?;

        let mut edge_set: HashSet<(usize, usize)> = Default::default();
        let mut n_matched = 0usize;
        let mut n_skipped = 0usize;

        for line in &read_out.lines {
            if line.len() < 2 {
                continue;
            }
            let idx1 = resolver.resolve(&line[0]);
            let idx2 = resolver.resolve(&line[1]);
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
        feature_edges.sort();

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

    /// Augment with shared-neighbor (SNN) edges: any unordered pair
    /// `(u, v)` with at least `min_shared` undirected neighbors in common
    /// gains a synthetic edge (unless one is already present).
    /// `min_shared = 0` is a no-op. Cost is roughly `O(Σ_u deg(u)²)` —
    /// fine for typical biological graphs.
    pub fn augment_with_snn(&mut self, min_shared: usize) {
        if min_shared == 0 {
            return;
        }

        let mut neighbor_set: Vec<HashSet<usize>> =
            (0..self.n_features).map(|_| HashSet::default()).collect();
        for &(u, v) in &self.feature_edges {
            neighbor_set[u].insert(v);
            neighbor_set[v].insert(u);
        }

        let mut existing: HashSet<(usize, usize)> = self.feature_edges.iter().copied().collect();

        let mut added = 0usize;
        for u in 0..self.n_features {
            if neighbor_set[u].is_empty() {
                continue;
            }
            let mut seen: HashSet<usize> = HashSet::default();
            for &m in &neighbor_set[u] {
                for &v in &neighbor_set[m] {
                    if v <= u {
                        continue;
                    }
                    if !seen.insert(v) {
                        continue;
                    }
                    if existing.contains(&(u, v)) {
                        continue;
                    }
                    let (a, b) = if neighbor_set[u].len() <= neighbor_set[v].len() {
                        (&neighbor_set[u], &neighbor_set[v])
                    } else {
                        (&neighbor_set[v], &neighbor_set[u])
                    };
                    let shared = a.iter().filter(|n| b.contains(n)).count();
                    if shared >= min_shared {
                        self.feature_edges.push((u, v));
                        existing.insert((u, v));
                        added += 1;
                    }
                }
            }
        }

        if added > 0 {
            self.feature_edges.sort();
            info!(
                "SNN augmentation (min_shared={}): +{} edges ({} total)",
                min_shared,
                added,
                self.feature_edges.len()
            );
        }
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
            .iter()
            .map(|&(g1, _)| self.feature_names[g1].clone())
            .collect();
        parquet_add_string_column(&mut row_group_writer, &names1)?;

        let names2: Vec<Box<str>> = self
            .feature_edges
            .iter()
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
    fn directed_adjacency() {
        let g = test_graph_from_edges(&[(0, 1), (0, 2), (1, 2)], 3);
        let adj = g.build_directed_adjacency();
        assert_eq!(adj[0], vec![(1, 0), (2, 1)]);
        assert_eq!(adj[1], vec![(2, 2)]);
        assert!(adj[2].is_empty());
    }
}
