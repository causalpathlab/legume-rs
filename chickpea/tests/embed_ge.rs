//! Joint hierarchical embed over frozen pb units.

mod common;

use chickpea::p2g::embed_ge::*;
use chickpea::p2g::link_map::PeakGeneEdge;
use chickpea::p2g::pb_levels::PbLevels;
use common::mat;
use graph_embedding_util::fit::projection::CellGroup;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::dense_mat_io::axis_id_names as names;
use legume_numeric::matrix::utils::cosine;

fn cfg(seed: u64, epochs: usize) -> HierEmbedConfig {
    HierEmbedConfig {
        dim: 8,
        epochs,
        seed,
        device: Device::Cpu,
        n_gene_modules: 4,
        n_peak_modules: 4,
        units_per_step: 8,
        modules_per_unit: 2,
        lr: 0.1,
        merge_every: 0,
        merge_cosine: 0.95,
        cells_per_pb: 0,
    }
}

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

/// Phase 1 on the pseudobulks alone.
fn no_cells() -> CellGroup {
    CellGroup {
        cells: Vec::new(),
        axes: vec![Vec::new(), Vec::new()],
    }
}

#[test]
fn hier_returns_finite_rows_and_unit_scaled_steps() {
    let tree = two_group_tree();
    let rna = tree.rna.as_ref().unwrap()[0].clone();
    let c = cfg(7, 2);
    let out = train_peak_gene_embeds(
        &two_group_links(),
        &tree,
        &names("chr1:", 6),
        &names("G", 4),
        &rna,
        &no_cells(),
        &c,
    )
    .unwrap();
    assert_eq!(out.dim, 8);
    assert_eq!(out.peak.len(), 6);
    assert_eq!(out.gene.len(), 4);
    assert_eq!(out.pb.len(), 2);
    assert_eq!(out.pb[0].len(), 16);
    let n_u: usize = 16 + 2;
    assert_eq!(out.steps_per_epoch, n_u.div_ceil(c.units_per_step));
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
    let rna = tree.rna.as_ref().unwrap()[0].clone();
    let out = train_peak_gene_embeds(
        &two_group_links(),
        &tree,
        &names("chr1:", 6),
        &names("G", 4),
        &rna,
        &no_cells(),
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
    for (i, row) in pb.iter().enumerate() {
        let (own, other) = if i < 8 { (&ca, &cb) } else { (&cb, &ca) };
        assert!(
            cosine(row, own) > cosine(row, other),
            "pb {i} closer to the other group"
        );
    }
    let (pa, pb1) = (&out.pb[1][0], &out.pb[1][1]);
    assert!(cosine(pa, &ca) > cosine(pa, &cb));
    assert!(cosine(pb1, &cb) > cosine(pb1, &ca));
    // Each gene and each peak sits nearer the pbs that express it.
    for g in 0..4 {
        let (own, other) = if g < 2 { (&ca, &cb) } else { (&cb, &ca) };
        assert!(
            cosine(&out.gene[g], own) > cosine(&out.gene[g], other),
            "gene {g} closer to the other group"
        );
    }
    for p in 0..6 {
        let (own, other) = if p < 3 { (&ca, &cb) } else { (&cb, &ca) };
        assert!(
            cosine(&out.peak[p], own) > cosine(&out.peak[p], other),
            "peak {p} closer to the other group"
        );
    }
}
