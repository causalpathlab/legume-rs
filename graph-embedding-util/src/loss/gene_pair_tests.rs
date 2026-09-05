//! Tests for the gene-gene co-occurrence edge: the hop tree resolves a partner
//! for every cell, hop-sampled negatives never come from the positive's own
//! finest group, positives are two distinct genes of one cell drawn by count,
//! the hop histogram follows the requested weights, the batch has the edge
//! layout the loss reads, and the loss matches a host reference.

use super::{gene_pair_nce, GenePairSampler, HopTree, HopWeights};
use crate::loss::{CellFeatureSampler, NceObjective, PerBatchStratifiedCellSampler};
use crate::model::JointEmbedModel;
use candle_util::candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;

const N_GENES: usize = 6;

/// Three levels over eight cells, coarsest first.
///
/// finest:  f0 = {0,1}  f1 = {2,3}  f2 = {4,5}  f3 = {6}  f4 = {7}
/// middle:  m0 = {f0,f1}  m1 = {f2}  m2 = {f3}  m3 = {f4}
/// coarse:  c0 = {m0,m1}  c1 = {m2}  c2 = {m3}
///
/// So cell 0 has a sister at one hop (f1), cell 4 finds its first partner two
/// hops up (f0, f1 under c0), and cell 6 has none until the root.
fn tree_levels() -> Vec<Vec<usize>> {
    vec![
        vec![0, 0, 0, 0, 0, 0, 1, 2], // coarsest
        vec![0, 0, 0, 0, 1, 1, 2, 3], // middle
        vec![0, 0, 1, 1, 2, 2, 3, 4], // finest
    ]
}

fn cell(features: &[u32], counts: &[f32]) -> CellFeatureSampler {
    CellFeatureSampler {
        features: features.to_vec(),
        counts: counts.to_vec(),
        picker: WeightedIndex::new(counts.to_vec()).unwrap(),
    }
}

/// One batch sampler holding all eight cells. Every cell has at least two
/// genes; cell 7 is heavily skewed so the distinct-gene rejection is exercised.
fn samplers() -> Vec<PerBatchStratifiedCellSampler> {
    let cells: Vec<(u32, CellFeatureSampler)> = vec![
        (0, cell(&[0, 1, 2], &[5.0, 3.0, 1.0])),
        (1, cell(&[0, 1], &[2.0, 2.0])),
        (2, cell(&[2, 3], &[4.0, 1.0])),
        (3, cell(&[2, 3, 4], &[1.0, 1.0, 1.0])),
        (4, cell(&[3, 4], &[3.0, 3.0])),
        (5, cell(&[4, 5], &[1.0, 6.0])),
        (6, cell(&[0, 5], &[2.0, 2.0])),
        (7, cell(&[1, 5], &[100.0, 1.0])),
    ];
    let active_cells: Vec<u32> = cells.iter().map(|(c, _)| *c).collect();
    let per_cell: Vec<CellFeatureSampler> = cells.into_iter().map(|(_, s)| s).collect();
    vec![PerBatchStratifiedCellSampler {
        cell_picker: WeightedIndex::new(vec![1.0f32; active_cells.len()]).unwrap(),
        active_cells,
        per_cell,
        neg: WeightedIndex::new(vec![1.0f32; N_GENES]).unwrap(),
        feature_pool: (0..N_GENES as u32).collect(),
    }]
}

fn sampler_with(hops: HopWeights) -> GenePairSampler {
    GenePairSampler::new(&samplers(), &tree_levels(), &hops).unwrap()
}

#[test]
fn every_cell_has_a_partner_at_some_hop() {
    let tree = HopTree::new(&tree_levels(), &[true; 8]).unwrap();
    assert_eq!(tree.n_hops(), 3);
    let mut rng = StdRng::seed_from_u64(1);
    let expected_first_hop = [1usize, 1, 1, 1, 2, 2, 3, 3];
    for (c, &expected) in expected_first_hop.iter().enumerate() {
        let (partner, hop) = tree
            .draw_partner(c, 1, &mut rng)
            .unwrap_or_else(|| panic!("cell {c} found no partner"));
        assert_ne!(partner, tree.fine_of(c), "cell {c} partnered with itself");
        assert_eq!(
            hop, expected,
            "cell {c} resolved at hop {hop}, expected {expected}"
        );
    }
    // Asking from a higher hop must never come back lower.
    for c in 0..8 {
        let (_, hop) = tree.draw_partner(c, 2, &mut rng).unwrap();
        assert!(hop >= 2, "cell {c}: asked for hop 2, got {hop}");
    }
}

#[test]
fn hop_negatives_never_come_from_the_positive_group() {
    let s = sampler_with(HopWeights::Uniform);
    let tree = HopTree::new(&tree_levels(), &[true; 8]).unwrap();
    let mut rng = StdRng::seed_from_u64(2);
    let sams = samplers();
    let batch = s.sample_batch(&sams, 2000, 5, &mut rng);
    assert_eq!(batch.neg_cells.len(), batch.pos_cells.len() * 5);
    for (b, &c) in batch.pos_cells.iter().enumerate() {
        for k in 0..5 {
            let cn = batch.neg_cells[b * 5 + k];
            assert_ne!(
                tree.fine_of(cn as usize),
                tree.fine_of(c as usize),
                "negative cell {cn} shares the finest group of positive cell {c}"
            );
        }
    }
}

#[test]
fn positive_pairs_are_distinct_genes_from_one_cell() {
    let s = sampler_with(HopWeights::Uniform);
    let mut rng = StdRng::seed_from_u64(3);
    let sams = samplers();
    let batch = s.sample_batch(&sams, 4000, 1, &mut rng);
    assert_eq!(batch.pos_g.len(), 4000);
    let mut top_gene_hits = 0usize;
    let mut cell0_draws = 0usize;
    for ((&c, &g), &h) in batch.pos_cells.iter().zip(&batch.pos_g).zip(&batch.pos_h) {
        assert_ne!(g, h, "cell {c} drew the same gene twice");
        let pf = &sams[0].per_cell[c as usize];
        assert!(pf.features.contains(&g), "gene {g} not in cell {c}");
        assert!(pf.features.contains(&h), "gene {h} not in cell {c}");
        if c == 0 {
            cell0_draws += 1;
            if g == 0 {
                top_gene_hits += 1;
            }
        }
    }
    // Cell 0's first gene carries 5/9 of its mass; the first draw must reflect it.
    let frac = top_gene_hits as f64 / cell0_draws.max(1) as f64;
    assert!(
        (frac - 5.0 / 9.0).abs() < 0.08,
        "gene 0 drawn first in {frac:.2} of cell-0 pairs, expected ~0.56"
    );
}

#[test]
fn hop_counts_follow_the_requested_weights() {
    let mut rng = StdRng::seed_from_u64(4);
    let sams = samplers();
    // Root: every negative resolves at the top.
    let root = sampler_with(HopWeights::Root).sample_batch(&sams, 500, 4, &mut rng);
    assert!(
        root.hops.iter().all(|&h| h == 3),
        "root weights must land every draw at hop 3"
    );
    // Sisters: cells with a sister resolve at 1; the others escalate, never below 1.
    let sis = sampler_with(HopWeights::Sisters).sample_batch(&sams, 2000, 1, &mut rng);
    for (&c, &h) in sis.pos_cells.iter().zip(&sis.hops) {
        let expected = [1u8, 1, 1, 1, 2, 2, 3, 3][c as usize];
        assert_eq!(
            h, expected,
            "cell {c} under sisters-only resolved at hop {h}"
        );
    }
    // Uniform over three hops, measured on cell 0 which can resolve at any hop.
    let uni = sampler_with(HopWeights::Uniform).sample_batch(&sams, 6000, 1, &mut rng);
    let mut hist = [0usize; 4];
    let mut n = 0usize;
    for (&c, &h) in uni.pos_cells.iter().zip(&uni.hops) {
        if c == 0 {
            hist[h as usize] += 1;
            n += 1;
        }
    }
    for (h, &count) in hist.iter().enumerate().skip(1) {
        let f = count as f64 / n as f64;
        assert!(
            (f - 1.0 / 3.0).abs() < 0.06,
            "hop {h}: {f:.2} of cell-0 draws, expected ~0.33"
        );
    }
}

#[test]
fn explicit_hop_weights_are_validated() {
    assert!(HopWeights::Explicit(vec![1.0, 2.0]).weights(3).is_err());
    assert!(HopWeights::Explicit(vec![0.0, 0.0, 0.0])
        .weights(3)
        .is_err());
    assert!(HopWeights::Explicit(vec![1.0, -1.0, 1.0])
        .weights(3)
        .is_err());
    assert_eq!(HopWeights::Near.weights(3).unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(HopWeights::Far.weights(3).unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn a_single_finest_group_is_refused() {
    let levels = vec![vec![0usize; 8], vec![0usize; 8], vec![0usize; 8]];
    assert!(GenePairSampler::new(&samplers(), &levels, &HopWeights::Uniform).is_err());
}

#[test]
fn the_batch_has_the_edge_layout_the_loss_reads() {
    let s = sampler_with(HopWeights::Uniform);
    let mut rng = StdRng::seed_from_u64(5);
    let batch = s.sample_batch(&samplers(), 37, 3, &mut rng);
    assert_eq!(batch.pos_cells.len(), 37);
    assert_eq!(batch.pos_g.len(), 37);
    assert_eq!(batch.pos_h.len(), 37);
    assert_eq!(batch.neg_h.len(), 37 * 3);
    assert_eq!(batch.neg_cells.len(), 37 * 3);
    assert_eq!(batch.hops.len(), 37 * 3);
    assert_eq!(batch.n_negatives, 3);
}

/// The loss is the same softmax-NCE over `ρ_g·ρ_h + b_g + b_h` that a host
/// implementation computes, on a free (uncomposed) model.
#[test]
fn gene_pair_loss_matches_a_host_reference() {
    const H: usize = 3;
    let dev = Device::Cpu;
    let rho: Vec<f32> = (0..N_GENES * H)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.2)
        .collect();
    let b: Vec<f32> = (0..N_GENES).map(|g| 0.1 * g as f32 - 0.2).collect();
    let model = JointEmbedModel {
        e_feat: Tensor::from_vec(rho.clone(), (N_GENES, H), &dev).unwrap(),
        e_cell: Tensor::zeros((1, H), candle_util::candle_core::DType::F32, &dev).unwrap(),
        b_feat: Tensor::from_vec(b.clone(), N_GENES, &dev).unwrap(),
        b_cell: Tensor::zeros(1, candle_util::candle_core::DType::F32, &dev).unwrap(),
        factor: None,
        adapter: None,
        modules: None,
        embedding_dim: H,
    };
    let batch = super::GenePairBatch {
        pos_cells: vec![0, 1],
        pos_g: vec![0, 3],
        pos_h: vec![1, 4],
        neg_cells: vec![9, 9, 9, 9],
        neg_h: vec![2, 5, 0, 1],
        hops: vec![1, 1, 1, 1],
        n_negatives: 2,
    };
    let got = gene_pair_nce(&model, batch, NceObjective::Softmax, &dev)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let score = |g: usize, h: usize| -> f64 {
        let dot: f32 = (0..H).map(|d| rho[g * H + d] * rho[h * H + d]).sum();
        f64::from(dot + b[g] + b[h])
    };
    let lse = |xs: &[f64]| {
        let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
    };
    let l0 = lse(&[score(0, 1), score(0, 2), score(0, 5)]) - score(0, 1);
    let l1 = lse(&[score(3, 4), score(3, 0), score(3, 1)]) - score(3, 4);
    let want = ((l0 + l1) / 2.0) as f32;
    assert!(
        (got - want).abs() < 1e-4,
        "gene-pair loss {got} vs host reference {want}"
    );
}
