//! Peak/gene/pb embeddings via `graph-embedding-util` FNE.
//!
//! One `fne::train` over the [`crate::p2g::context_graph`]: the link relation
//! and the pb levels' count relations share a table, so peaks and genes are
//! placed by both their cis links and the pseudobulks that express them.

use crate::common::Mat;
use crate::p2g::context_graph::build_context_graph;
use crate::p2g::link_map::PeakGeneEdge;
use crate::p2g::pb_levels::PbLevels;
use graph_embedding_util::fne::{train, FneConfig};
use graph_embedding_util::save_embedding;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::matrix::dense_mat_io::{axis_id_names, l2_normalize_rows_inplace};
use legume_numeric::matrix::traits::ConvertMatOps;
use log::info;

/// Row embeddings after FNE training (CPU `[n, dim]` f32).
#[derive(Clone, Debug)]
pub struct PeakGeneEmbeds {
    pub peak: Vec<Vec<f32>>,
    pub gene: Vec<Vec<f32>>,
    /// Pseudobulk rows per level, finest first.
    pub pb: Vec<Vec<Vec<f32>>>,
    pub dim: usize,
    pub peak_names: Vec<Box<str>>,
    pub gene_names: Vec<Box<str>>,
}

impl PeakGeneEmbeds {
    /// The finest pb rows, L2-normalized: the sample embedding that is
    /// clustered and written as `cell_embedding`.
    pub fn finest_unit_rows(&self) -> Mat {
        let rows = &self.pb[0];
        let mut m = Mat::from_fn(rows.len(), self.dim, |i, d| rows[i][d]);
        l2_normalize_rows_inplace(&mut m);
        m
    }
}

/// Train peak/gene/pb embeddings jointly: the scored link edges plus the
/// pb levels' count relations (`bins` SIMBA levels per modality).
pub fn train_peak_gene_embeds(
    edges: &[PeakGeneEdge],
    levels: &PbLevels,
    peak_names: &[Box<str>],
    gene_names: &[Box<str>],
    bins: usize,
    cfg: &FneConfig,
) -> anyhow::Result<PeakGeneEmbeds> {
    anyhow::ensure!(!edges.is_empty(), "no peak–gene edges to embed");
    let n_peaks = peak_names.len();
    let n_genes = gene_names.len();
    let graph = build_context_graph(edges, levels, n_peaks, n_genes, bins)?;
    info!(
        "Context graph: {} node types, {} relations, {} edges (repeats {:?})",
        graph.types.len(),
        graph.rels.len(),
        graph.edges.len(),
        graph.repeats
    );
    let cfg = FneConfig {
        relation_repeats: graph.repeats.clone(),
        ..cfg.clone()
    };

    let out = train(graph.edges, graph.types, graph.rels, &cfg)?;
    let types = &out.node_types;
    let table = out.embedding.to_vec2::<f32>()?;
    anyhow::ensure!(
        table.len() == types.n_total(),
        "embedding rows {} != nodes {}",
        table.len(),
        types.n_total()
    );
    let dim = cfg.dim;
    anyhow::ensure!(
        table.iter().all(|r| r.len() == dim),
        "embedding width mismatch"
    );

    // The table stacks the node types in order: peaks, genes, then each level.
    let mut rows = table.into_iter();
    let mut take = |n: usize| rows.by_ref().take(n).collect::<Vec<_>>();
    let peak = take(n_peaks);
    let gene = take(n_genes);
    let pb: Vec<Vec<Vec<f32>>> = (0..levels.n_levels())
        .map(|l| take(levels.n_pb(l)))
        .collect();
    Ok(PeakGeneEmbeds {
        peak,
        gene,
        pb,
        dim,
        peak_names: peak_names.to_vec(),
        gene_names: gene_names.to_vec(),
    })
}

/// Write `{prefix}.{peak,gene,cell}_embedding.parquet` via [`save_embedding`]:
/// `cell` rows are the normalized finest pb rows, named `pb_{i}`. With more
/// than one level, `{prefix}.pb_tree_embedding.parquet` holds every level's
/// raw rows named `L{level}:{i}`.
pub fn write_embedding_parquets(prefix: &str, embeds: &PeakGeneEmbeds) -> anyhow::Result<()> {
    anyhow::ensure!(embeds.dim > 0, "empty embedding dim");
    anyhow::ensure!(
        embeds.peak.len() == embeds.peak_names.len(),
        "peak rows vs names mismatch"
    );
    anyhow::ensure!(
        embeds.gene.len() == embeds.gene_names.len(),
        "gene rows vs names mismatch"
    );
    anyhow::ensure!(!embeds.pb.is_empty(), "no pb level to write");

    let peak_path = format!("{prefix}.peak_embedding.parquet");
    save_embedding(
        &peak_path,
        &rows_to_tensor(&embeds.peak, embeds.dim)?,
        &embeds.peak_names,
        "peak",
    )?;
    let gene_path = format!("{prefix}.gene_embedding.parquet");
    save_embedding(
        &gene_path,
        &rows_to_tensor(&embeds.gene, embeds.dim)?,
        &embeds.gene_names,
        "gene",
    )?;
    let cell = embeds.finest_unit_rows();
    let cell_path = format!("{prefix}.cell_embedding.parquet");
    save_embedding(
        &cell_path,
        &cell.to_tensor(&Device::Cpu)?,
        &axis_id_names("pb_", cell.nrows()),
        "cell",
    )?;
    info!(
        "Wrote embeddings: {peak_path} ({} × {}), {gene_path} ({} × {}), {cell_path} ({} × {})",
        embeds.peak.len(),
        embeds.dim,
        embeds.gene.len(),
        embeds.dim,
        cell.nrows(),
        embeds.dim
    );

    if embeds.pb.len() > 1 {
        let rows: Vec<Vec<f32>> = embeds.pb.iter().flatten().cloned().collect();
        let names: Vec<Box<str>> = embeds
            .pb
            .iter()
            .enumerate()
            .flat_map(|(l, level)| axis_id_names(&format!("L{l}:"), level.len()))
            .collect();
        let tree_path = format!("{prefix}.pb_tree_embedding.parquet");
        save_embedding(
            &tree_path,
            &rows_to_tensor(&rows, embeds.dim)?,
            &names,
            "pb",
        )?;
        info!(
            "Wrote pb tree embedding: {tree_path} ({} × {}, {} levels)",
            rows.len(),
            embeds.dim,
            embeds.pb.len()
        );
    }
    Ok(())
}

fn rows_to_tensor(rows: &[Vec<f32>], dim: usize) -> anyhow::Result<Tensor> {
    let n = rows.len();
    let mut flat = Vec::with_capacity(n * dim);
    for (i, row) in rows.iter().enumerate() {
        anyhow::ensure!(
            row.len() == dim,
            "ragged embedding at row {i}: len {} != {dim}",
            row.len()
        );
        flat.extend_from_slice(row);
    }
    Ok(Tensor::from_vec(flat, (n, dim), &Device::Cpu)?)
}
