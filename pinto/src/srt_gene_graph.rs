use crate::srt_common::*;
use crate::srt_knn_graph::{KnnGraph, KnnGraphArgs};
use matrix_util::parquet::*;
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
