//! Peak/gene embeddings via `graph-embedding-util` FNE.
//!
//! Thin wrapper: typed region+gene graph from [`crate::p2g::link_map`] edges →
//! `graph_embedding_util::fne::train`. No local NCE/PBG loop.

use crate::p2g::link_map::PeakGeneEdge;
use graph_embedding_util::fne::{
    train, FneConfig, NodeTypeTable, Relation, RelationPolarity, RelationTable, TypedEdgeList,
};
use graph_embedding_util::save_embedding;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::matrix::traits::ConvertMatOps;
use log::info;
use nalgebra::DMatrix;

/// Peak and gene row embeddings after FNE training (CPU `[n, dim]` f32).
#[derive(Clone, Debug)]
pub struct PeakGeneEmbeds {
    pub peak: Vec<Vec<f32>>,
    pub gene: Vec<Vec<f32>>,
    pub dim: usize,
    pub peak_names: Vec<Box<str>>,
    pub gene_names: Vec<Box<str>>,
}

/// Train peak/gene embeddings from the scored link edges via ge-util FNE.
pub fn train_peak_gene_embeds(
    edges: &[PeakGeneEdge],
    peak_names: &[Box<str>],
    gene_names: &[Box<str>],
    cfg: &FneConfig,
) -> anyhow::Result<PeakGeneEmbeds> {
    anyhow::ensure!(!edges.is_empty(), "no peak–gene edges to embed");
    let n_peaks = peak_names.len();
    let n_genes = gene_names.len();
    anyhow::ensure!(n_peaks > 0 && n_genes > 0, "need ≥1 peak and ≥1 gene");

    for e in edges {
        anyhow::ensure!(
            e.peak < n_peaks && e.gene < n_genes,
            "edge peak={} gene={} out of range (peaks={n_peaks}, genes={n_genes})",
            e.peak,
            e.gene
        );
        anyhow::ensure!(
            e.weight.is_finite() && e.weight > 0.0,
            "edge weight must be finite and positive, got {}",
            e.weight
        );
    }

    let types = NodeTypeTable::new(&[("region", n_peaks), ("gene", n_genes)])?;
    let region_t = types.index_of("region").expect("region type") as u16;
    let gene_t = types.index_of("gene").expect("gene type") as u16;
    let rels = RelationTable::new(
        vec![Relation {
            name: "region:gene/link".into(),
            lhs_type: region_t,
            rhs_type: gene_t,
            weight: 1.0,
            undirected: false,
            polarity: RelationPolarity::Friend,
        }],
        &types,
    )?;

    let gene_off = n_peaks as u32;
    let mut lhs = Vec::with_capacity(edges.len());
    let mut rhs = Vec::with_capacity(edges.len());
    let mut rel = Vec::with_capacity(edges.len());
    let mut weight = Vec::with_capacity(edges.len());
    for e in edges {
        lhs.push(e.peak as u32);
        rhs.push(gene_off + e.gene as u32);
        rel.push(0u16);
        weight.push(e.weight);
    }
    let edge_list = TypedEdgeList {
        lhs,
        rhs,
        rel,
        weight: Some(weight),
    };

    let out = train(edge_list, types, rels, cfg)?;
    let mut table = out.embedding.to_vec2::<f32>()?;
    anyhow::ensure!(
        table.len() == n_peaks + n_genes,
        "embedding rows {} != peaks+genes {}",
        table.len(),
        n_peaks + n_genes
    );
    let dim = cfg.dim;
    anyhow::ensure!(
        table.iter().all(|r| r.len() == dim),
        "embedding width mismatch"
    );

    let gene = table.split_off(n_peaks);
    let peak = table;
    Ok(PeakGeneEmbeds {
        peak,
        gene,
        dim,
        peak_names: peak_names.to_vec(),
        gene_names: gene_names.to_vec(),
    })
}

/// Write `{prefix}.{peak,gene,cell}_embedding.parquet` via [`save_embedding`].
pub fn write_embedding_parquets(
    prefix: &str,
    embeds: &PeakGeneEmbeds,
    cell_emb: &DMatrix<f32>,
    cell_names: &[Box<str>],
) -> anyhow::Result<()> {
    anyhow::ensure!(embeds.dim > 0, "empty embedding dim");
    anyhow::ensure!(
        embeds.peak.len() == embeds.peak_names.len(),
        "peak rows vs names mismatch"
    );
    anyhow::ensure!(
        embeds.gene.len() == embeds.gene_names.len(),
        "gene rows vs names mismatch"
    );
    anyhow::ensure!(
        cell_emb.ncols() == embeds.dim,
        "cell embedding width {} != feature dim {}",
        cell_emb.ncols(),
        embeds.dim
    );
    anyhow::ensure!(
        cell_emb.nrows() == cell_names.len(),
        "cell rows {} != cell_names {}",
        cell_emb.nrows(),
        cell_names.len()
    );

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
    let cell_path = format!("{prefix}.cell_embedding.parquet");
    save_embedding(
        &cell_path,
        &cell_emb.to_tensor(&Device::Cpu)?,
        cell_names,
        "cell",
    )?;

    info!(
        "Wrote embeddings: {peak_path} ({} × {}), {gene_path} ({} × {}), {cell_path} ({} × {})",
        embeds.peak.len(),
        embeds.dim,
        embeds.gene.len(),
        embeds.dim,
        cell_names.len(),
        embeds.dim
    );
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
