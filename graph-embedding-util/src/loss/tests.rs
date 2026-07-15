//! Tests for the cell-cell sampling primitives and chain NCE losses.
//!
//! Re-imports the public API via `crate::loss::*` so the tests don't
//! care which submodule each item lives in.

use crate::loss::cell::LevelSiblingPool;
use crate::loss::{
    build_per_batch_cell_samplers, cell_cell_nce_loss_per_level_batched_gated,
    cell_cell_nce_loss_per_level_gated, sample_cell_chain_batch, CellChainBatch,
    CellChainBatchArgs, PbChainFilter,
};

#[test]
fn cell_cell_sampler_skips_cross_batch_edges() {
    // 4 cells, 2 batches. Edges: (0,1) within batch 0, (2,3) within
    // batch 1, (1,2) cross-batch.
    let edges = vec![(0u32, 1), (2, 3), (1, 2)];
    let batch_membership = vec![0u32, 0, 1, 1];
    let (samplers, stats) =
        build_per_batch_cell_samplers(&edges, &batch_membership, 2, 4, 0.75, None);

    assert_eq!(
        stats.cross_batch_dropped, 1,
        "expected one cross-batch edge dropped"
    );
    assert_eq!(stats.pb_mismatch_dropped, 0);
    let s0 = samplers[0]
        .as_ref()
        .expect("batch 0 has within-batch edges");
    let s1 = samplers[1]
        .as_ref()
        .expect("batch 1 has within-batch edges");
    assert_eq!(s0.edge_indices, vec![0]);
    assert_eq!(s1.edge_indices, vec![1]);
    assert_eq!(s0.cell_pool, vec![0, 1]);
    assert_eq!(s1.cell_pool, vec![2, 3]);
}

#[test]
fn cell_cell_sampler_empty_batch_returns_none() {
    let edges = vec![(0u32, 1)];
    let batch_membership = vec![0u32, 0, 1, 1];
    let (samplers, stats) =
        build_per_batch_cell_samplers(&edges, &batch_membership, 2, 4, 0.75, None);
    assert_eq!(stats.cross_batch_dropped, 0);
    assert_eq!(stats.pb_mismatch_dropped, 0);
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
    let cell_to_pb_per_level: Vec<Vec<usize>> = vec![
        vec![0, 0, 1, 1], // L=0 parent
        vec![0, 0, 1, 1], // L=1 self — same partition as parent
    ];
    let filter = PbChainFilter {
        cell_to_pb_per_level: &cell_to_pb_per_level,
        levels: &[0, 1],
    };
    let (samplers, _) =
        build_per_batch_cell_samplers(&edges, &batch_membership, 1, 4, 0.75, Some(filter));
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
    let cell_to_pb_per_level: Vec<Vec<usize>> = vec![
        vec![0, 0, 0, 0, 1, 1, 1, 1], // L=0 coarse: {0..3} ↦ 0; {4..7} ↦ 1
        vec![0, 0, 1, 1, 2, 2, 3, 3], // L=1 fine
    ];
    let filter = PbChainFilter {
        cell_to_pb_per_level: &cell_to_pb_per_level,
        levels: &[0, 1],
    };
    let (samplers, _stats) =
        build_per_batch_cell_samplers(&edges, &batch_membership, 1, 8, 0.75, Some(filter));
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
    let cell_to_pb_per_level: Vec<Vec<usize>> =
        vec![vec![0, 0, 0, 0, 1, 1, 1, 1], vec![0, 0, 1, 1, 2, 2, 3, 3]];
    let filter = PbChainFilter {
        cell_to_pb_per_level: &cell_to_pb_per_level,
        levels: &[0, 1],
    };
    let (samplers, _) =
        build_per_batch_cell_samplers(&edges, &batch_membership, 1, 8, 0.75, Some(filter));
    let s = samplers[0].as_ref().unwrap();

    let pb_l0: &[usize] = &cell_to_pb_per_level[0];
    let pb_l1: &[usize] = &cell_to_pb_per_level[1];
    let pb_maps: Vec<&[usize]> = vec![pb_l0, pb_l1];

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let (batch, stats) = sample_cell_chain_batch(
        CellChainBatchArgs {
            edges: &edges,
            batch_sampler: s,
            batch_size: 128, // exercise both positives many times
            n_negatives: 4,
            pb_maps: &pb_maps,
        },
        &mut rng,
    );

    // No fallbacks expected: every anchor's parent pb has 2 children at L=1.
    assert_eq!(stats.per_level_fallback[1], 0);
    // Fine-level negatives must agree with anchor at L=0 (sibling) and
    // disagree at L=1.
    let k = 4;
    for b in 0..batch.left_cells.len() {
        let u = batch.left_cells[b];
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
fn batched_gated_matches_per_gene_gated() {
    // The batched gene-modulated per-level loss must match stacking G
    // calls to the single-gene gated version, gene-by-gene.
    use crate::model::{JointEmbedModel, ModelArgs, ModelInit};
    use candle_util::candle_core::Device;
    use candle_util::candle_nn::VarMap;
    use rand::SeedableRng;

    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let n_cells = 8;
    let n_genes = 4;
    let embedding_dim = 4;
    let e_cell_init = nalgebra::DMatrix::<f32>::from_row_slice(
        n_cells,
        embedding_dim,
        &[
            0.10, -0.20, 0.30, 0.40, //
            0.40, 0.50, -0.60, 0.10, //
            -0.70, 0.80, 0.90, -0.30, //
            0.05, -0.15, 0.25, 0.35, //
            0.20, 0.10, -0.10, 0.00, //
            -0.05, 0.45, -0.25, 0.15, //
            0.55, -0.35, 0.15, -0.05, //
            -0.45, 0.25, 0.35, -0.15, //
        ],
    );
    let e_gene_init = nalgebra::DMatrix::<f32>::from_row_slice(
        n_genes,
        embedding_dim,
        &[
            0.30, -0.10, 0.20, 0.10, //
            -0.20, 0.40, 0.00, 0.30, //
            0.10, 0.20, -0.30, 0.40, //
            0.50, -0.20, 0.10, -0.10, //
        ],
    );
    let b_cell_init: Vec<f32> = (0..n_cells).map(|i| 0.01 * i as f32).collect();
    let b_gene_init: Vec<f32> = (0..n_genes).map(|i| -0.02 * i as f32).collect();
    let model = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features: n_genes,
            n_cells,
            embedding_dim,
            seed: 0,
        },
        &ModelInit {
            e_feat: Some(&e_gene_init),
            e_cell: Some(&e_cell_init),
            b_feat: &b_gene_init,
            b_cell: &b_cell_init,
        },
        &varmap,
        &dev,
    )
    .expect("model");

    let edges = vec![(0u32, 1), (2, 3), (4, 5), (6, 7)];
    let batch_membership = vec![0u32; n_cells];
    let cell_to_pb_per_level: Vec<Vec<usize>> =
        vec![vec![0, 0, 0, 0, 1, 1, 1, 1], vec![0, 0, 1, 1, 2, 2, 3, 3]];
    let filter = PbChainFilter {
        cell_to_pb_per_level: &cell_to_pb_per_level,
        levels: &[0, 1],
    };
    let (samplers, _) =
        build_per_batch_cell_samplers(&edges, &batch_membership, 1, n_cells, 0.75, Some(filter));
    let s = samplers[0].as_ref().unwrap();
    let pb_maps: Vec<&[usize]> = vec![&cell_to_pb_per_level[0], &cell_to_pb_per_level[1]];

    // 3 genes drawn with deterministic seeds.
    let gene_ids: Vec<u32> = vec![0, 2, 3];
    let g = gene_ids.len();
    let mut single_results: Vec<Vec<f32>> = Vec::with_capacity(g);
    let mut batches_for_batched: Vec<CellChainBatch> = Vec::with_capacity(g);
    for (gi, &gid) in gene_ids.iter().enumerate() {
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(200 + gi as u64);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(200 + gi as u64);
        let (batch_a, _) = sample_cell_chain_batch(
            CellChainBatchArgs {
                edges: &edges,
                batch_sampler: s,
                batch_size: 6,
                n_negatives: 2,
                pb_maps: &pb_maps,
            },
            &mut rng_a,
        );
        let (batch_b, _) = sample_cell_chain_batch(
            CellChainBatchArgs {
                edges: &edges,
                batch_sampler: s,
                batch_size: 6,
                n_negatives: 2,
                pb_maps: &pb_maps,
            },
            &mut rng_b,
        );
        let single = cell_cell_nce_loss_per_level_gated(&model, batch_a, gid, None, &dev)
            .expect("single gated")
            .to_vec1::<f32>()
            .unwrap();
        single_results.push(single);
        batches_for_batched.push(batch_b);
    }
    let batched = cell_cell_nce_loss_per_level_batched_gated(
        &model,
        batches_for_batched,
        &gene_ids,
        None,
        None,
        &dev,
    )
    .expect("batched gated")
    .to_vec2::<f32>()
    .unwrap();

    for (gi, expected) in single_results.iter().enumerate() {
        for (lvl, &exp) in expected.iter().enumerate() {
            let got = batched[gi][lvl];
            let diff = (exp - got).abs();
            assert!(
                diff < 1e-5,
                "gene_id={} (chunk pos {gi}) level {lvl}: single={exp:.6} batched={got:.6} diff={diff:.2e}",
                gene_ids[gi]
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
    let cell_to_pb_per_level: Vec<Vec<usize>> = vec![vec![0, 0, 1, 1]];
    let filter = PbChainFilter {
        cell_to_pb_per_level: &cell_to_pb_per_level,
        levels: &[0],
    };
    let (samplers, stats) =
        build_per_batch_cell_samplers(&edges, &batch_membership, 1, 4, 0.75, Some(filter));
    assert_eq!(stats.cross_batch_dropped, 0);
    assert_eq!(stats.pb_mismatch_dropped, 1);
    let s0 = samplers[0]
        .as_ref()
        .expect("batch 0 has within-batch within-pb edges");
    assert_eq!(s0.edge_indices, vec![0, 1]);
}
