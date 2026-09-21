//! Peak/gene embeddings via `graph-embedding-util` FNE.
//!
//! Thin wrapper: typed region+gene graph from [`crate::p2g::abc_map`] edges →
//! `graph_embedding_util::fne::train`. No local NCE/PBG loop.

use crate::p2g::abc_map::PeakGeneEdge;
use graph_embedding_util::fne::{
    train, FneConfig, NodeTypeTable, Relation, RelationPolarity, RelationTable, TypedEdgeList,
};

/// Peak and gene row embeddings after FNE training (CPU `[n, dim]` f32).
#[derive(Clone, Debug)]
pub struct PeakGeneEmbeds {
    pub peak: Vec<Vec<f32>>,
    pub gene: Vec<Vec<f32>>,
    pub dim: usize,
    pub peak_names: Vec<Box<str>>,
    pub gene_names: Vec<Box<str>>,
}

/// Train peak/gene embeddings from rough ABC edges via ge-util FNE.
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
            name: "region:gene/abc".into(),
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
    let table = out.embedding.to_vec2::<f32>()?;
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

    let peak = table[..n_peaks].to_vec();
    let gene = table[n_peaks..].to_vec();
    Ok(PeakGeneEmbeds {
        peak,
        gene,
        dim,
        peak_names: peak_names.to_vec(),
        gene_names: gene_names.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2g::abc_map::PeakGeneEdge;
    use graph_embedding_util::fne::FneConfig;
    use legume_numeric::candle::candle_core::Device;

    #[test]
    fn fne_returns_finite_peak_and_gene_rows() {
        let peak_names: Vec<Box<str>> = vec!["chr1:100-200".into(), "chr1:300-400".into()];
        let gene_names: Vec<Box<str>> = vec!["GENE_A".into(), "GENE_B".into()];
        // Enough edges that FNE can run a couple of tiny epochs.
        let edges = vec![
            PeakGeneEdge {
                peak: 0,
                gene: 0,
                weight: 0.9,
            },
            PeakGeneEdge {
                peak: 0,
                gene: 1,
                weight: 0.2,
            },
            PeakGeneEdge {
                peak: 1,
                gene: 0,
                weight: 0.3,
            },
            PeakGeneEdge {
                peak: 1,
                gene: 1,
                weight: 0.8,
            },
            PeakGeneEdge {
                peak: 0,
                gene: 0,
                weight: 0.7,
            },
            PeakGeneEdge {
                peak: 1,
                gene: 1,
                weight: 0.6,
            },
        ];
        let cfg = FneConfig {
            dim: 8,
            epochs: 2,
            lr: 0.1,
            batch_size: 4,
            num_batch_negs: 2,
            num_uniform_negs: 2,
            wd: Some(0.0),
            wd_interval: 50,
            eval_fraction: 0.0,
            eval_min_per_relation: 1,
            relation_repeats: Vec::new(),
            preset: None,
            seed: 7,
            device: Device::Cpu,
        };

        let out = train_peak_gene_embeds(&edges, &peak_names, &gene_names, &cfg).unwrap();
        assert_eq!(out.dim, 8);
        assert_eq!(out.peak.len(), 2);
        assert_eq!(out.gene.len(), 2);
        assert_eq!(out.peak[0].len(), 8);
        assert_eq!(out.peak_names, peak_names);
        assert_eq!(out.gene_names, gene_names);
        assert!(
            out.peak
                .iter()
                .chain(out.gene.iter())
                .flatten()
                .all(|x| x.is_finite()),
            "non-finite embedding entries"
        );
    }
}
