//! Joint FNE over the link relation and the pb-tree count relations.

mod common;

use chickpea::p2g::embed_ge::*;
use chickpea::p2g::link_map::PeakGeneEdge;
use chickpea::p2g::pb_levels::PbLevels;
use common::mat;
use graph_embedding_util::fne::FneConfig;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::dense_mat_io::axis_id_names as names;
use legume_numeric::matrix::utils::cosine;

fn cfg(seed: u64, epochs: usize) -> FneConfig {
    FneConfig {
        dim: 8,
        epochs,
        batch_size: 16,
        num_batch_negs: 4,
        num_uniform_negs: 4,
        wd: Some(0.0),
        eval_fraction: 0.0,
        seed,
        device: Device::Cpu,
        ..FneConfig::default()
    }
}

/// Two groups of finest pbs (8 + 8) under two coarse parents; group A
/// expresses genes 0-1 and peaks 0-2, group B genes 2-3 and peaks 3-5.
fn two_group_tree() -> PbLevels {
    let group = |s: usize| usize::from(s >= 8);
    let rna0 = mat(4, 16, |g, s| {
        if g / 2 == group(s) {
            5.0 + (s % 3) as f32
        } else {
            0.0
        }
    });
    let atac0 = mat(6, 16, |p, s| {
        if p / 3 == group(s) {
            4.0 + (s % 2) as f32
        } else {
            0.0
        }
    });
    let rna1 = mat(4, 2, |g, s| if g / 2 == s { 40.0 } else { 0.0 });
    let atac1 = mat(6, 2, |p, s| if p / 3 == s { 32.0 } else { 0.0 });
    PbLevels {
        rna: Some(vec![rna0, rna1]),
        atac: vec![atac0, atac1],
        parent: vec![(0..16).map(group).collect()],
    }
}

fn two_group_links() -> Vec<PeakGeneEdge> {
    [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 3)]
        .iter()
        .map(|&(peak, gene)| PeakGeneEdge {
            peak,
            gene,
            weight: 0.8,
        })
        .collect()
}

#[test]
fn fne_returns_finite_rows_for_peaks_genes_and_every_level() {
    let tree = two_group_tree();
    let out = train_peak_gene_embeds(
        &two_group_links(),
        &tree,
        &names("chr1:", 6),
        &names("G", 4),
        5,
        &cfg(7, 2),
    )
    .unwrap();
    assert_eq!(out.dim, 8);
    assert_eq!(out.peak.len(), 6);
    assert_eq!(out.gene.len(), 4);
    assert_eq!(out.pb.len(), 2);
    assert_eq!(out.pb[0].len(), 16);
    assert_eq!(out.pb[1].len(), 2);
    assert!(
        out.peak
            .iter()
            .chain(out.gene.iter())
            .chain(out.pb.iter().flatten())
            .flatten()
            .all(|x| x.is_finite()),
        "non-finite embedding entries"
    );
}

#[test]
fn pb_rows_separate_by_program_and_parents_sit_with_their_children() {
    let tree = two_group_tree();
    let out = train_peak_gene_embeds(
        &two_group_links(),
        &tree,
        &names("chr1:", 6),
        &names("G", 4),
        5,
        &cfg(11, 30),
    )
    .unwrap();
    let pb = &out.pb[0];
    let mean = |ids: std::ops::Range<usize>| -> Vec<f32> {
        let mut m = vec![0.0f32; out.dim];
        for i in ids.clone() {
            for d in 0..out.dim {
                m[d] += pb[i][d] / ids.len() as f32;
            }
        }
        m
    };
    let (ca, cb) = (mean(0..8), mean(8..16));
    // Every finest pb is nearer its own group's centroid.
    for (i, row) in pb.iter().enumerate() {
        let (own, other) = if i < 8 { (&ca, &cb) } else { (&cb, &ca) };
        assert!(
            cosine(row, own) > cosine(row, other),
            "pb {i} closer to the other group"
        );
    }
    // Each parent is nearer its own children than the other parent's.
    let (pa, pb1) = (&out.pb[1][0], &out.pb[1][1]);
    assert!(cosine(pa, &ca) > cosine(pa, &cb));
    assert!(cosine(pb1, &cb) > cosine(pb1, &ca));
    // Each gene is nearer the pbs that express it.
    for g in 0..4 {
        let (own, other) = if g < 2 { (&ca, &cb) } else { (&cb, &ca) };
        assert!(
            cosine(&out.gene[g], own) > cosine(&out.gene[g], other),
            "gene {g} closer to the other group"
        );
    }
}
