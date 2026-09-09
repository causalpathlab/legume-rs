//! Tests for the cell-cell sampling primitives and chain NCE losses.
//!
//! Re-imports the public API via `crate::loss::*` so the tests don't
//! care which submodule each item lives in.

use crate::loss::cell::LevelSiblingPool;
use crate::loss::{
    build_per_batch_unit_samplers, sample_unit_chain_batch, ChainGroupFilter, UnitChainBatchArgs,
};

#[test]
fn cell_cell_sampler_skips_cross_batch_edges() {
    // 4 cells, 2 batches. Edges: (0,1) within batch 0, (2,3) within
    // batch 1, (1,2) cross-batch.
    let edges = vec![(0u32, 1), (2, 3), (1, 2)];
    let batch_membership = vec![0u32, 0, 1, 1];
    let (samplers, stats) =
        build_per_batch_unit_samplers(&edges, &batch_membership, 2, 4, 0.75, None);

    assert_eq!(
        stats.cross_batch_dropped, 1,
        "expected one cross-batch edge dropped"
    );
    assert_eq!(stats.group_mismatch_dropped, 0);
    let s0 = samplers[0]
        .as_ref()
        .expect("batch 0 has within-batch edges");
    let s1 = samplers[1]
        .as_ref()
        .expect("batch 1 has within-batch edges");
    assert_eq!(s0.edge_indices, vec![0]);
    assert_eq!(s1.edge_indices, vec![1]);
    assert_eq!(s0.unit_pool, vec![0, 1]);
    assert_eq!(s1.unit_pool, vec![2, 3]);
}

#[test]
fn cell_cell_sampler_empty_batch_returns_none() {
    let edges = vec![(0u32, 1)];
    let batch_membership = vec![0u32, 0, 1, 1];
    let (samplers, stats) =
        build_per_batch_unit_samplers(&edges, &batch_membership, 2, 4, 0.75, None);
    assert_eq!(stats.cross_batch_dropped, 0);
    assert_eq!(stats.group_mismatch_dropped, 0);
    assert!(samplers[0].is_some());
    assert!(samplers[1].is_none(), "batch 1 has no edges → None");
}

#[test]
fn chain_pools_prune_parents_without_siblings() {
    // 4 cells, one batch. Parent pb_0 = {0,0,1,1}; finer pb_1 = {0,0,1,1}
    // — every parent has exactly ONE child pb at level 1, so no anchor
    // has a true sibling. The map should drop both parents.
    let edges = vec![(0u32, 1), (2, 3)];
    let batch_membership = vec![0u32; 4];
    let unit_to_group_per_level: Vec<Vec<usize>> = vec![
        vec![0, 0, 1, 1], // L=0 parent
        vec![0, 0, 1, 1], // L=1 self — same partition as parent
    ];
    let filter = ChainGroupFilter {
        unit_to_group_per_level: &unit_to_group_per_level,
        levels: &[0, 1],
    };
    let (samplers, _) =
        build_per_batch_unit_samplers(&edges, &batch_membership, 1, 4, 0.75, Some(filter));
    let s = samplers[0].as_ref().unwrap();
    let LevelSiblingPool::ByParent(by_parent) = &s.chain_pools[1] else {
        panic!("expected ByParent at chain position 1");
    };
    assert!(
        by_parent.is_empty(),
        "parents whose children are all the same pb at this level should be dropped"
    );
}

#[test]
fn chain_pools_group_by_parent_pb() {
    // 8 cells, one batch. Two-level chain over level 0 (coarse,
    // 2 pbs) and level 1 (fine, 4 pbs, where cells [0,1] and [2,3]
    // share parent pb_0=0; [4,5] and [6,7] share parent pb_0=1).
    let edges = vec![(0u32, 1), (2, 3), (4, 5), (6, 7)];
    let batch_membership = vec![0u32; 8];
    let unit_to_group_per_level: Vec<Vec<usize>> = vec![
        vec![0, 0, 0, 0, 1, 1, 1, 1], // L=0 coarse: {0..3} ↦ 0; {4..7} ↦ 1
        vec![0, 0, 1, 1, 2, 2, 3, 3], // L=1 fine
    ];
    let filter = ChainGroupFilter {
        unit_to_group_per_level: &unit_to_group_per_level,
        levels: &[0, 1],
    };
    let (samplers, _stats) =
        build_per_batch_unit_samplers(&edges, &batch_membership, 1, 8, 0.75, Some(filter));
    let s = samplers[0]
        .as_ref()
        .expect("batch 0 has within-pb edges at every chain level");

    assert_eq!(s.chain_pools.len(), 2);
    // Chain position 0 (coarsest) is the Root — no by_parent pool.
    assert!(matches!(s.chain_pools[0], LevelSiblingPool::Root));
    // Chain position 1: by_parent groups by L=0 pb id.
    let LevelSiblingPool::ByParent(by_parent) = &s.chain_pools[1] else {
        panic!("expected ByParent at chain position 1");
    };
    let mut parent0 = by_parent.get(&0).cloned().expect("parent pb 0 present");
    parent0.sort();
    assert_eq!(parent0, vec![0, 1, 2, 3]);
    let mut parent1 = by_parent.get(&1).cloned().expect("parent pb 1 present");
    parent1.sort();
    assert_eq!(parent1, vec![4, 5, 6, 7]);
}

#[test]
fn sibling_negative_draws_share_parent_differ_at_self() {
    // Same 8-cell setup; verify that sibling-pool draws at the fine
    // chain level always produce cells with same L=0 pb as the anchor
    // but different L=1 pb (i.e. real siblings in the pb tree).
    use rand::SeedableRng;
    let edges = vec![(0u32, 1), (4, 5)];
    let batch_membership = vec![0u32; 8];
    let unit_to_group_per_level: Vec<Vec<usize>> =
        vec![vec![0, 0, 0, 0, 1, 1, 1, 1], vec![0, 0, 1, 1, 2, 2, 3, 3]];
    let filter = ChainGroupFilter {
        unit_to_group_per_level: &unit_to_group_per_level,
        levels: &[0, 1],
    };
    let (samplers, _) =
        build_per_batch_unit_samplers(&edges, &batch_membership, 1, 8, 0.75, Some(filter));
    let s = samplers[0].as_ref().unwrap();

    let pb_l0: &[usize] = &unit_to_group_per_level[0];
    let pb_l1: &[usize] = &unit_to_group_per_level[1];
    let pb_maps: Vec<&[usize]> = vec![pb_l0, pb_l1];

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let (batch, stats) = sample_unit_chain_batch(
        UnitChainBatchArgs {
            edges: &edges,
            batch_sampler: s,
            n_positives: 128, // exercise both positives many times
            n_negatives: 4,
            unit_to_group_per_level: &pb_maps,
        },
        &mut rng,
    );

    // No fallbacks expected: every anchor's parent pb has 2 children at L=1.
    assert_eq!(stats.per_level_fallback[1], 0);
    // Fine-level negatives must agree with anchor at L=0 (sibling) and
    // disagree at L=1.
    let k = 4;
    for b in 0..batch.left_units.len() {
        let u = batch.left_units[b];
        let pu_l0 = pb_l0[u as usize];
        let pu_l1 = pb_l1[u as usize];
        for kk in 0..k {
            let w = batch.per_level_neg[1][b * k + kk];
            assert_eq!(
                pb_l0[w as usize], pu_l0,
                "fine-level neg should share parent pb with anchor"
            );
            assert_ne!(
                pb_l1[w as usize], pu_l1,
                "fine-level neg should differ from anchor at this level"
            );
        }
    }
}

#[test]
fn cell_cell_sampler_filters_pb_mismatched_edges() {
    // 4 cells in one batch. Edges: (0,1) same pb at L0, (2,3) same
    // pb at L0, (0,2) different pb at L0 — should drop the last.
    let edges = vec![(0u32, 1), (2, 3), (0, 2)];
    let batch_membership = vec![0u32; 4];
    let unit_to_group_per_level: Vec<Vec<usize>> = vec![vec![0, 0, 1, 1]];
    let filter = ChainGroupFilter {
        unit_to_group_per_level: &unit_to_group_per_level,
        levels: &[0],
    };
    let (samplers, stats) =
        build_per_batch_unit_samplers(&edges, &batch_membership, 1, 4, 0.75, Some(filter));
    assert_eq!(stats.cross_batch_dropped, 0);
    assert_eq!(stats.group_mismatch_dropped, 1);
    let s0 = samplers[0]
        .as_ref()
        .expect("batch 0 has within-batch within-pb edges");
    assert_eq!(s0.edge_indices, vec![0, 1]);
}

/// The Gram-trace `embedding_ridge` must equal the elementwise
/// `λ·mean_n‖x_n‖²` it replaced, in value AND in the gradient it sends
/// to the table (the reformulation is a memory fix, not a model change).
#[test]
fn embedding_ridge_matches_elementwise_form() {
    use candle_util::candle_core::{DType, Device, Var};
    let dev = Device::Cpu;
    let var = Var::rand(-1.0f32, 1.0, (7, 5), &dev).unwrap();
    let x = var.as_tensor();
    let lambda = 0.25;

    let ridge = crate::loss::embedding_ridge(x, lambda).unwrap();
    let reference = x
        .sqr()
        .unwrap()
        .sum(1)
        .unwrap()
        .mean_all()
        .unwrap()
        .affine(lambda, 0.0)
        .unwrap();
    let a = ridge.to_scalar::<f32>().unwrap();
    let b = reference.to_scalar::<f32>().unwrap();
    assert!((a - b).abs() < 1e-5, "value: {a} vs {b}");

    let g_new = ridge
        .backward()
        .unwrap()
        .get(x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let g_ref = reference
        .backward()
        .unwrap()
        .get(x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(g_new.len(), g_ref.len());
    for (i, (gn, gr)) in g_new.iter().zip(g_ref.iter()).enumerate() {
        assert!((gn - gr).abs() < 1e-5, "grad[{i}]: {gn} vs {gr}");
    }
    let _ = DType::F32;
}

/////////////////////////////////////////////////////////////
// Single-modality equivalence of the per-modality negative //
// pools. Runs cannot be diffed on disk (bge does not       //
// reproduce byte-for-byte across processes), so the guard  //
// is in-process: with one panel the sampler must consume   //
// the RNG exactly as the global draw did, and emit the     //
// same negatives.                                          //
/////////////////////////////////////////////////////////////

use crate::coarsen::identity_axis;
use crate::loss::feat::{
    sample_per_batch_stratified_edge_batch, sample_stratified_edge_batch, CellFeatureSampler,
    PerBatchStratifiedCellSampler, PerBatchStratifiedEdgeBatchArgs, StratifiedEdgeBatchArgs,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

fn cell_sampler(
    modality: Option<std::sync::Arc<crate::loss::ModalityPools>>,
) -> PerBatchStratifiedCellSampler {
    let per_cell: Vec<CellFeatureSampler> = (0..3)
        .map(|c| {
            let features: Vec<u32> = vec![c, (c + 1) % 5, (c + 2) % 5];
            let counts = vec![3.0f32, 2.0, 1.0];
            let picker = WeightedIndex::new(counts.clone()).unwrap();
            CellFeatureSampler {
                features,
                counts,
                picker,
            }
        })
        .collect();
    PerBatchStratifiedCellSampler {
        cell_picker: WeightedIndex::new(vec![1.0f32; 3]).unwrap(),
        active_cells: vec![0, 1, 2],
        per_cell,
        neg: WeightedIndex::new(vec![1.0f32; 5]).unwrap(),
        feature_pool: vec![0, 1, 2, 3, 4],
        modality,
    }
}

/// One panel is no panel: the batch a single-modality sampler emits must be
/// bit-for-bit what it emitted before the split existed. The reference here is
/// the pre-change loop — `batch_size * n_negatives` uniform draws off `neg`.
#[test]
fn one_modality_cell_axis_draws_exactly_the_old_negatives() {
    let coarsening = identity_axis(3);
    let s = cell_sampler(None);
    let (bs, k) = (16usize, 4usize);

    let mut rng = StdRng::seed_from_u64(4242);
    let got = sample_per_batch_stratified_edge_batch(
        PerBatchStratifiedEdgeBatchArgs {
            sampler: &s,
            cell_coarsening: &coarsening.coarsenings[0],
            batch_size: bs,
            n_negatives: k,
            module_pools: None,
        },
        &mut rng,
    );

    // Reference: the positives are drawn the same way, then the negatives are
    // one flat run of uniform draws over the whole pool.
    let mut want_rng = StdRng::seed_from_u64(4242);
    let mut want_feats = Vec::new();
    for _ in 0..bs {
        let lc = s.cell_picker.sample(&mut want_rng);
        let pf = &s.per_cell[lc];
        want_feats.push(pf.features[pf.picker.sample(&mut want_rng)]);
    }
    let want_neg: Vec<u32> = (0..bs * k)
        .map(|_| s.feature_pool[s.neg.sample(&mut want_rng)])
        .collect();

    assert_eq!(got.fine_feats, want_feats);
    assert_eq!(got.neg_feats, want_neg);
    assert_eq!(got.neg_feats.len(), bs * k);
}

/// The same guard on the pb axis, whose global draw alternates uniform and
/// degree-weighted. The alternation must stay keyed on the negative's index.
#[test]
fn one_modality_pb_axis_draws_exactly_the_old_negatives() {
    let t = |cell, feature, count| crate::data::Triplet {
        cell,
        feature,
        count,
    };
    let triplets = [
        t(0, 0, 5.0),
        t(0, 1, 3.0),
        t(0, 2, 1.0),
        t(1, 1, 4.0),
        t(1, 3, 2.0),
        t(1, 4, 6.0),
    ];
    let s =
        crate::loss::build_stratified_sampler(&triplets, 2, 5, 1.0, None, None).expect("sampler");
    assert!(s.modality.is_none(), "one panel must leave the pools off");
    let (bs, k) = (12usize, 3usize);

    let mut rng = StdRng::seed_from_u64(77);
    let got = sample_stratified_edge_batch(
        StratifiedEdgeBatchArgs {
            sampler: &s,
            batch_size: bs,
            n_negatives: k,
            module_pools: None,
        },
        &mut rng,
    );

    let mut want_rng = StdRng::seed_from_u64(77);
    let mut want_feats = Vec::new();
    for _ in 0..bs {
        let lp = s.pb_picker.sample(&mut want_rng);
        let pf = &s.per_pb[lp];
        want_feats.push(pf.features[pf.picker.sample(&mut want_rng)]);
    }
    let want_neg: Vec<u32> = (0..bs * k)
        .map(|i| {
            let local = if i % 2 == 0 {
                s.neg.sample(&mut want_rng)
            } else {
                s.neg_by_degree.sample(&mut want_rng)
            };
            s.feature_pool[local]
        })
        .collect();

    assert_eq!(got.fine_feats, want_feats);
    assert_eq!(got.neg_feats, want_neg);
}
