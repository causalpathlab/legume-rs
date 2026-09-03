use super::*;
use crate::simba::graph::{EdgeList, RelationTable};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;

/// 20 edges over three levels (7 / 12 / 1), unique (cell, gene) pairs.
fn three_relation_graph() -> EdgeList {
    let mut cell = Vec::new();
    let mut gene = Vec::new();
    let mut level = Vec::new();
    for (l, n) in [(1u8, 7usize), (2, 12), (3, 1)] {
        for i in 0..n {
            cell.push((cell.len() % 9) as u32);
            gene.push((gene.len() / 9 * 3 + i % 3) as u32);
            level.push(l);
        }
    }
    EdgeList {
        n_cells: 9,
        n_genes: 12,
        cell,
        gene,
        level,
    }
}

#[test]
fn every_batch_holds_a_single_relation_padded_to_a_multiple_of_the_chunk_size() {
    let edges = three_relation_graph();
    let rel = RelationTable::from_levels(&edges.levels_present());
    let level_of: HashMap<(u32, u32), u8> = (0..edges.len())
        .map(|i| ((edges.cell[i], edges.gene[i]), edges.level[i]))
        .collect();
    assert_eq!(level_of.len(), edges.len(), "fixture pairs are unique");
    let (batch_size, c, u) = (5usize, 4usize, 2usize);
    let mut batcher = EpochBatcher::new(&edges, 0..edges.len(), &rel, batch_size);
    let mut rng = StdRng::seed_from_u64(3);
    let mut seen: Vec<(u32, u32)> = Vec::new();
    let mut n_batches = 0;
    while let Some(b) = batcher.next_batch(&edges, &rel, c, u, &mut rng) {
        n_batches += 1;
        assert!(b.n_real >= 1 && b.n_real <= batch_size);
        assert_eq!(b.c, c);
        assert_eq!(b.u, u);
        assert_eq!(b.k, b.n_real.div_ceil(c));
        assert_eq!(b.lhs.len(), b.k * c);
        assert_eq!(b.rhs.len(), b.k * c);
        assert_eq!(b.row_w.len(), b.k * c);
        assert_eq!(b.col_valid.len(), b.k * c);
        let levels: Vec<u8> = (0..b.n_real)
            .map(|i| level_of[&(b.lhs[i], b.rhs[i])])
            .collect();
        assert!(
            levels.iter().all(|&l| l == levels[0]),
            "one relation per batch"
        );
        assert!((0..b.n_real).all(|i| b.row_w[i] == rel.weight(levels[0])));
        seen.extend((0..b.n_real).map(|i| (b.lhs[i], b.rhs[i])));
    }
    assert!(n_batches >= 4, "7/5 + 12/5 + 1/5 batches at least");
    assert_eq!(batcher.remaining(), 0);
    seen.sort_unstable();
    let mut all: Vec<(u32, u32)> = level_of.keys().copied().collect();
    all.sort_unstable();
    assert_eq!(seen, all, "every edge is drawn exactly once per epoch");
}

#[test]
fn pad_rows_carry_zero_weight_and_uniform_negatives_are_drawn_per_chunk() {
    let edges = three_relation_graph();
    let rel = RelationTable::from_levels(&edges.levels_present());
    // Only the level-2 edges (12 of them) → one batch of 12 at batch_size 20,
    // c = 5 → k = 3 chunks, 3 pad rows at the end.
    let start = 7;
    let mut batcher = EpochBatcher::new(&edges, start..start + 12, &rel, 20);
    let mut rng = StdRng::seed_from_u64(11);
    let b = batcher
        .next_batch(&edges, &rel, 5, 4, &mut rng)
        .expect("one batch");
    assert_eq!(b.n_real, 12);
    assert_eq!(b.k, 3);
    for i in 0..12 {
        assert_eq!(b.row_w[i], rel.weight(2));
        assert_eq!(b.col_valid[i], 1.0);
    }
    for i in 12..15 {
        assert_eq!(b.row_w[i], 0.0, "pad row {i} has no loss weight");
        assert_eq!(b.col_valid[i], 0.0, "pad row {i} is not a negative");
        assert_eq!(b.lhs[i], 0);
        assert_eq!(b.rhs[i], 0);
    }
    assert_eq!(b.uni_lhs.len(), 3 * 4, "u uniform cells per chunk");
    assert_eq!(b.uni_rhs.len(), 3 * 4, "u uniform genes per chunk");
    assert!(b.uni_lhs.iter().all(|&i| (i as usize) < edges.n_cells));
    assert!(b.uni_rhs.iter().all(|&i| (i as usize) < edges.n_genes));
    assert!(batcher.next_batch(&edges, &rel, 5, 4, &mut rng).is_none());
}
