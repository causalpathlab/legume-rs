use crate::util::common::*;
use matrix_util::common_io::read_lines_of_words_delim;
use matrix_util::membership::{detect_delimiter, Membership};
use matrix_util::parquet::*;
use parquet::basic::Type as ParquetType;

pub struct GenePairGraph {
    pub gene_names: Vec<Box<str>>,
    pub n_genes: usize,
    pub gene_edges: Vec<(usize, usize)>,
}

impl GenePairGraph {
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
                for part in name.split(d) {
                    if part != name.as_ref() {
                        pairs_vec.push((part.into(), idx_str.clone()));
                    }
                }
            }
        }

        let membership = Membership::from_pairs(pairs_vec, allow_prefix);

        // Read the edge list file
        let file_delim = detect_delimiter(file_path);
        let read_out = read_lines_of_words_delim(file_path, file_delim, -1)?;

        // Match edges
        let mut edge_set: HashSet<(usize, usize)> = Default::default();
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

        Ok(Self {
            gene_names,
            n_genes,
            gene_edges,
        })
    }

    /// Keep only edges at the given indices.
    pub fn filter_edges(&mut self, keep_indices: &[usize]) {
        self.gene_edges = keep_indices.iter().map(|&i| self.gene_edges[i]).collect();
    }

    pub fn num_edges(&self) -> usize {
        self.gene_edges.len()
    }

    #[cfg(test)]
    pub fn num_genes(&self) -> usize {
        self.n_genes
    }

    /// Build directed adjacency list: gene_adj[g] = [(neighbor, edge_idx)]
    /// where neighbor > g (to avoid double-counting edges)
    pub fn build_directed_adjacency(&self) -> Vec<Vec<(usize, usize)>> {
        let mut gene_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.n_genes];
        for (edge_idx, &(g1, g2)) in self.gene_edges.iter().enumerate() {
            gene_adj[g1].push((g2, edge_idx));
        }
        gene_adj
    }

    /// Write gene graph as an edge list (gene1, gene2) to parquet
    pub fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
        let n_edges = self.gene_edges.len();

        let column_names: Vec<Box<str>> = vec!["gene1".into(), "gene2".into()];
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

        let graph =
            GenePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
                .unwrap();

        assert_eq!(graph.num_genes(), 4);
        assert_eq!(graph.num_edges(), 3);
        assert_eq!(graph.gene_edges, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn test_from_edge_list_dedup() {
        let file = write_edge_file(&["A\tB", "B\tA", "A\tB"]);
        let names = gene_names(&["A", "B", "C"]);

        let graph =
            GenePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
                .unwrap();

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.gene_edges, vec![(0, 1)]);
    }

    #[test]
    fn test_from_edge_list_unmatched_genes_skipped() {
        let file = write_edge_file(&["TP53\tBRCA1", "UNKNOWN1\tBRCA1", "TP53\tUNKNOWN2"]);
        let names = gene_names(&["TP53", "BRCA1"]);

        let graph =
            GenePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
                .unwrap();

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.gene_edges, vec![(0, 1)]);
    }

    #[test]
    fn test_from_edge_list_self_loops_skipped() {
        let file = write_edge_file(&["A\tA", "A\tB"]);
        let names = gene_names(&["A", "B"]);

        let graph =
            GenePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
                .unwrap();

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.gene_edges, vec![(0, 1)]);
    }

    #[test]
    fn test_from_edge_list_prefix_matching() {
        let file = write_edge_file(&["TP53\tBRCA1"]);
        let names = gene_names(&["TP53.1", "BRCA1.2"]);

        let graph = GenePairGraph::from_edge_list(file.path().to_str().unwrap(), names, true, None)
            .unwrap();

        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn test_from_edge_list_delimiter_matching() {
        let file = write_edge_file(&["TP53\tBRCA1"]);
        let names = gene_names(&["TP53.v1", "BRCA1.v2"]);

        let graph =
            GenePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, Some('.'))
                .unwrap();

        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn test_from_edge_list_empty_file() {
        let file = write_edge_file(&[]);
        let names = gene_names(&["A", "B"]);

        let graph =
            GenePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
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

        let graph =
            GenePairGraph::from_edge_list(f.path().to_str().unwrap(), names, false, None).unwrap();

        assert_eq!(graph.num_edges(), 2);
        assert_eq!(graph.gene_edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn test_edge_names() {
        let file = write_edge_file(&["TP53\tBRCA1", "BRCA1\tEGFR"]);
        let names = gene_names(&["TP53", "BRCA1", "EGFR"]);

        let graph =
            GenePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
                .unwrap();

        let edge_names: Vec<String> = graph
            .gene_edges
            .iter()
            .map(|&(g1, g2)| format!("{}:{}", graph.gene_names[g1], graph.gene_names[g2]))
            .collect();
        assert_eq!(edge_names.len(), 2);
        assert_eq!(edge_names[0], "TP53:BRCA1");
        assert_eq!(edge_names[1], "BRCA1:EGFR");
    }

    #[test]
    fn test_directed_adjacency() {
        let file = write_edge_file(&["A\tB", "A\tC", "B\tC"]);
        let names = gene_names(&["A", "B", "C"]);

        let graph =
            GenePairGraph::from_edge_list(file.path().to_str().unwrap(), names, false, None)
                .unwrap();

        let adj = graph.build_directed_adjacency();
        assert_eq!(adj[0], vec![(1, 0), (2, 1)]);
        assert_eq!(adj[1], vec![(2, 2)]);
        assert!(adj[2].is_empty());
    }
}
