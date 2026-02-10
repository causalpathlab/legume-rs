use crate::srt_common::*;
use crate::srt_knn_graph::{KnnGraph, KnnGraphArgs};
use matrix_util::common_io::read_lines_of_words_delim;
use matrix_util::membership::{detect_delimiter, Membership};
use matrix_util::parquet::*;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use parquet::basic::Type as ParquetType;

pub struct GenePairGraph {
    pub graph: KnnGraph,
    pub gene_names: Vec<Box<str>>,
    pub n_genes: usize,
    pub gene_edges: Vec<(usize, usize)>,
}

pub struct GenePairGraphArgs {
    pub knn: usize,
    pub block_size: usize,
    pub reciprocal: bool,
}

#[allow(dead_code)]
impl GenePairGraph {
    /// Build gene-gene KNN graph from gene × sample posterior means.
    ///
    /// Row-normalizes (center + unit variance per gene) so that
    /// Euclidean distance approximates Pearson correlation, then
    /// calls `KnnGraph::from_columns()` on the transposed matrix.
    pub fn from_posterior_means(
        posterior_means: &Mat,
        gene_names: Vec<Box<str>>,
        args: GenePairGraphArgs,
    ) -> anyhow::Result<Self> {
        let n_genes = posterior_means.nrows();

        // Row-normalize: center + unit variance per gene
        let mut normalized = posterior_means.clone();
        normalized.scale_rows_inplace();

        // Transpose to (n_samples × n_genes): columns = gene profiles
        // KnnGraph::from_columns expects d × n where each column is a point
        let transposed = normalized.transpose();

        info!(
            "Building gene KNN graph: {} genes, {} samples, k={}",
            n_genes,
            posterior_means.ncols(),
            args.knn,
        );

        let graph = KnnGraph::from_columns(
            &transposed,
            KnnGraphArgs {
                knn: args.knn,
                block_size: args.block_size,
                reciprocal: args.reciprocal,
            },
        )?;

        let gene_edges = graph.edges.clone();

        info!(
            "Gene graph: {} edges among {} genes",
            gene_edges.len(),
            n_genes,
        );

        Ok(Self {
            graph,
            gene_names,
            n_genes,
            gene_edges,
        })
    }

    /// Build gene-gene graph from an external edge list file (e.g., BioGRID).
    ///
    /// Reads a two-column TSV/CSV where each line is a gene-gene edge.
    /// Gene names are matched against the data's gene names using the
    /// Membership infrastructure (exact → delimiter-based → prefix matching).
    /// Unmatched edges are silently dropped.
    pub fn from_edge_list(
        file_path: &str,
        gene_names: Vec<Box<str>>,
        allow_prefix: bool,
        delimiter: Option<char>,
    ) -> anyhow::Result<Self> {
        let n_genes = gene_names.len();

        // Build membership: gene_name → gene_index (as string)
        // When a delimiter is set, also index by each component so that
        // compound names like "ENSG00000141510_TP53" can be matched by
        // either the ID ("ENSG00000141510") or the symbol ("TP53").
        let mut pairs_vec: Vec<(Box<str>, Box<str>)> = gene_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i.to_string().into_boxed_str()))
            .collect();

        if let Some(d) = delimiter {
            for (i, name) in gene_names.iter().enumerate() {
                let idx_str: Box<str> = i.to_string().into_boxed_str();
                for part in name.splitn(2, d) {
                    if part != name.as_ref() {
                        pairs_vec.push((part.into(), idx_str.clone()));
                    }
                }
            }
        }

        let membership = Membership::from_pairs(pairs_vec.into_iter(), allow_prefix);

        // Read the edge list file
        let file_delim = detect_delimiter(file_path);
        let read_out = read_lines_of_words_delim(file_path, file_delim, -1)?;

        // Match edges
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        let mut n_matched = 0usize;
        let mut n_skipped = 0usize;

        for line in &read_out.lines {
            if line.len() < 2 {
                continue;
            }
            let name1 = &line[0];
            let name2 = &line[1];

            let idx1 = membership.get(name1).and_then(|s| s.parse::<usize>().ok());
            let idx2 = membership.get(name2).and_then(|s| s.parse::<usize>().ok());

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

        let mut gene_edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
        gene_edges.sort();

        info!(
            "External gene network: {} edges loaded from {} ({} matched, {} skipped, {} unique)",
            read_out.lines.len(),
            file_path,
            n_matched,
            n_skipped,
            gene_edges.len(),
        );

        // Build a minimal KnnGraph with stub adjacency
        let n_edges = gene_edges.len();
        let mut coo = CooMatrix::new(n_genes, n_genes);
        for &(i, j) in &gene_edges {
            coo.push(i, j, 1.0);
            coo.push(j, i, 1.0);
        }
        let adjacency = CscMatrix::from(&coo);
        let distances = vec![f32::NAN; n_edges];

        let graph = KnnGraph {
            adjacency,
            edges: gene_edges.clone(),
            distances,
            n_nodes: n_genes,
        };

        Ok(Self {
            graph,
            gene_names,
            n_genes,
            gene_edges,
        })
    }

    /// Keep only edges at the given indices. Updates gene_edges, graph.edges, graph.distances.
    /// Note: graph.adjacency becomes stale (not used after graph construction).
    pub fn filter_edges(&mut self, keep_indices: &[usize]) {
        self.gene_edges = keep_indices.iter().map(|&i| self.gene_edges[i]).collect();
        self.graph.edges = self.gene_edges.clone();
        self.graph.distances = keep_indices.iter().map(|&i| self.graph.distances[i]).collect();
    }

    pub fn num_edges(&self) -> usize {
        self.gene_edges.len()
    }

    pub fn num_genes(&self) -> usize {
        self.n_genes
    }

    /// Build directed adjacency list: gene_adj[g] = [(neighbor, edge_idx)]
    /// where neighbor > g (to avoid double-counting edges)
    pub fn build_directed_adjacency(&self) -> Vec<Vec<(usize, usize)>> {
        let mut gene_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.n_genes];
        for (edge_idx, &(g1, g2)) in self.gene_edges.iter().enumerate() {
            // g1 < g2 guaranteed by KnnGraph
            gene_adj[g1].push((g2, edge_idx));
        }
        gene_adj
    }

    /// Gene pair names formatted as "GENE1:GENE2"
    pub fn edge_names(&self) -> Vec<Box<str>> {
        self.gene_edges
            .iter()
            .map(|&(g1, g2)| {
                format!("{}:{}", self.gene_names[g1], self.gene_names[g2]).into_boxed_str()
            })
            .collect()
    }

    /// Gene pair names with channel suffix: "GENE1:GENE2@+" and "GENE1:GENE2@-"
    pub fn edge_names_with_channels(&self) -> Vec<Box<str>> {
        let base = self.edge_names();
        let mut names = Vec::with_capacity(base.len() * 2);
        for name in base.iter() {
            names.push(format!("{}@+", name).into_boxed_str());
        }
        for name in base.iter() {
            names.push(format!("{}@-", name).into_boxed_str());
        }
        names
    }

    /// Build directed adjacency from external edge list and verify consistency
    /// with the `gene_edges` stored on self.
    pub fn verify_adjacency(&self) -> bool {
        for &(i, j) in &self.gene_edges {
            if i >= j {
                return false; // edges must be canonical
            }
        }
        // Check adjacency is symmetric and matches edges
        for &(i, j) in &self.gene_edges {
            let has_ij = self.graph.neighbors(j).contains(&i);
            let has_ji = self.graph.neighbors(i).contains(&j);
            if !has_ij || !has_ji {
                return false;
            }
        }
        true
    }

    /// Write gene graph as an edge list (gene1, gene2, distance) to parquet
    pub fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
        let n_edges = self.gene_edges.len();

        let column_names: Vec<Box<str>> = vec![
            "gene1".into(),
            "gene2".into(),
            "distance".into(),
        ];
        let column_types = vec![
            ParquetType::BYTE_ARRAY,
            ParquetType::BYTE_ARRAY,
            ParquetType::FLOAT,
        ];

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

        let gene1_names: Vec<Box<str>> = self
            .gene_edges
            .iter()
            .map(|&(g1, _)| self.gene_names[g1].clone())
            .collect();
        parquet_add_string_column(&mut row_group_writer, &gene1_names)?;

        let gene2_names: Vec<Box<str>> = self
            .gene_edges
            .iter()
            .map(|&(_, g2)| self.gene_names[g2].clone())
            .collect();
        parquet_add_string_column(&mut row_group_writer, &gene2_names)?;

        parquet_add_numeric_column(&mut row_group_writer, &self.graph.distances)?;

        row_group_writer.close()?;
        writer.close()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn gene_names(names: &[&str]) -> Vec<Box<str>> {
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
    fn test_from_edge_list_exact_match() {
        let file = write_edge_file(&["TP53\tBRCA1", "BRCA1\tEGFR", "TP53\tEGFR"]);
        let names = gene_names(&["TP53", "BRCA1", "EGFR", "MYC"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        assert_eq!(graph.num_genes(), 4);
        assert_eq!(graph.num_edges(), 3);
        // Edges should be canonical (i < j) and sorted
        assert_eq!(graph.gene_edges, vec![(0, 1), (0, 2), (1, 2)]);
        assert!(graph.verify_adjacency());
    }

    #[test]
    fn test_from_edge_list_dedup() {
        // Same edge in both directions + duplicate
        let file = write_edge_file(&["A\tB", "B\tA", "A\tB"]);
        let names = gene_names(&["A", "B", "C"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.gene_edges, vec![(0, 1)]);
    }

    #[test]
    fn test_from_edge_list_unmatched_genes_skipped() {
        let file = write_edge_file(&["TP53\tBRCA1", "UNKNOWN1\tBRCA1", "TP53\tUNKNOWN2"]);
        let names = gene_names(&["TP53", "BRCA1"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.gene_edges, vec![(0, 1)]);
    }

    #[test]
    fn test_from_edge_list_self_loops_skipped() {
        let file = write_edge_file(&["A\tA", "A\tB"]);
        let names = gene_names(&["A", "B"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.gene_edges, vec![(0, 1)]);
    }

    #[test]
    fn test_from_edge_list_prefix_matching() {
        // Data has "TP53.1", file has "TP53"
        let file = write_edge_file(&["TP53\tBRCA1"]);
        let names = gene_names(&["TP53.1", "BRCA1.2"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            true, // allow prefix
            None,
        )
        .unwrap();

        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn test_from_edge_list_delimiter_matching() {
        // Data has "TP53.v1", "BRCA1.v2"; file has "TP53", "BRCA1"
        // Delimiter '.' extracts base key for matching
        let file = write_edge_file(&["TP53\tBRCA1"]);
        let names = gene_names(&["TP53.v1", "BRCA1.v2"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            Some('.'),
        )
        .unwrap();

        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn test_from_edge_list_empty_file() {
        let file = write_edge_file(&[]);
        let names = gene_names(&["A", "B"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.num_genes(), 2);
    }

    #[test]
    fn test_from_edge_list_csv() {
        let mut f = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(f, "A,B").unwrap();
        writeln!(f, "B,C").unwrap();
        f.flush().unwrap();

        let names = gene_names(&["A", "B", "C"]);

        let graph = GenePairGraph::from_edge_list(
            f.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        assert_eq!(graph.num_edges(), 2);
        assert_eq!(graph.gene_edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn test_from_edge_list_adjacency_symmetric() {
        let file = write_edge_file(&["A\tB", "B\tC", "A\tC"]);
        let names = gene_names(&["A", "B", "C"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        assert!(graph.verify_adjacency());
        // Check symmetric neighbors
        assert!(graph.graph.neighbors(0).contains(&1));
        assert!(graph.graph.neighbors(1).contains(&0));
    }

    #[test]
    fn test_from_edge_list_distances_are_nan() {
        let file = write_edge_file(&["A\tB", "B\tC"]);
        let names = gene_names(&["A", "B", "C"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        assert_eq!(graph.graph.distances.len(), 2);
        assert!(graph.graph.distances.iter().all(|d| d.is_nan()));
    }

    #[test]
    fn test_edge_names() {
        let file = write_edge_file(&["TP53\tBRCA1", "BRCA1\tEGFR"]);
        let names = gene_names(&["TP53", "BRCA1", "EGFR"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        let edge_names = graph.edge_names();
        assert_eq!(edge_names.len(), 2);
        assert_eq!(&*edge_names[0], "TP53:BRCA1");
        assert_eq!(&*edge_names[1], "BRCA1:EGFR");
    }

    #[test]
    fn test_directed_adjacency() {
        let file = write_edge_file(&["A\tB", "A\tC", "B\tC"]);
        let names = gene_names(&["A", "B", "C"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        let adj = graph.build_directed_adjacency();
        // A(0) -> B(1) edge 0, A(0) -> C(2) edge 1
        assert_eq!(adj[0], vec![(1, 0), (2, 1)]);
        // B(1) -> C(2) edge 2
        assert_eq!(adj[1], vec![(2, 2)]);
        // C(2) -> nothing (no neighbor > 2)
        assert!(adj[2].is_empty());
    }

    #[test]
    fn test_edge_names_with_channels() {
        let file = write_edge_file(&["A\tB"]);
        let names = gene_names(&["A", "B"]);

        let graph = GenePairGraph::from_edge_list(
            file.path().to_str().unwrap(),
            names,
            false,
            None,
        )
        .unwrap();

        let ch_names = graph.edge_names_with_channels();
        assert_eq!(ch_names.len(), 2);
        assert_eq!(&*ch_names[0], "A:B@+");
        assert_eq!(&*ch_names[1], "A:B@-");
    }
}
