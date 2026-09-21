//! FNE wrapper on a tiny edge list.

use chickpea::p2g::embed_ge::*;
use chickpea::p2g::link_map::PeakGeneEdge;
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
        batch_size: 4,
        num_batch_negs: 2,
        num_uniform_negs: 2,
        wd: Some(0.0),
        eval_fraction: 0.0,
        seed: 7,
        device: Device::Cpu,
        ..FneConfig::default()
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
