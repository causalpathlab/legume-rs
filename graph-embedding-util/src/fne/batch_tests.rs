use super::*;
use crate::fne::graph::{NodeTypeTable, Relation, RelationTable, TypedEdgeList};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;

fn types() -> NodeTypeTable {
    NodeTypeTable::new(&[("gene", 9), ("term", 12), ("cell_type", 4)]).unwrap()
}

fn relations(t: &NodeTypeTable) -> RelationTable {
    RelationTable::new(
        vec![
            Relation {
                name: "ppi".into(),
                lhs_type: 0,
                rhs_type: 0,
                weight: 1.0,
                undirected: true,
            },
            Relation {
                name: "go".into(),
                lhs_type: 0,
                rhs_type: 1,
                weight: 2.0,
                undirected: false,
            },
            Relation {
                name: "marker".into(),
                lhs_type: 0,
                rhs_type: 2,
                weight: 3.0,
                undirected: false,
            },
        ],
        t,
    )
    .unwrap()
}

/// 20 edges over three relations (7 / 12 / 1), unique (lhs, rhs) pairs,
/// already grouped by relation.
fn three_relation_graph(t: &NodeTypeTable) -> (TypedEdgeList, Vec<std::ops::Range<usize>>) {
    let mut e = TypedEdgeList::default();
    let mut w = Vec::new();
    // ppi: gene i – gene (i + 1) for i in 0..7
    for i in 0..7u32 {
        e.lhs.push(i);
        e.rhs.push(i + 1);
        e.rel.push(0);
        w.push(0.5);
    }
    // go: gene (i % 9) – term (5th offset) i
    for i in 0..12u32 {
        e.lhs.push(i % 9);
        e.rhs.push(t.global(1, i));
        e.rel.push(1);
        w.push(1.0 + i as f32);
    }
    // marker: gene 3 – cell_type 1
    e.lhs.push(3);
    e.rhs.push(t.global(2, 1));
    e.rel.push(2);
    w.push(0.25);
    e.weight = Some(w);
    (e, vec![0..7, 7..19, 19..20])
}

#[test]
fn every_batch_holds_a_single_relation_padded_to_a_multiple_of_the_chunk_size() {
    let t = types();
    let rels = relations(&t);
    let (edges, blocks) = three_relation_graph(&t);
    edges.validate(&t, &rels).unwrap();
    let key_of: HashMap<(u32, u32), (u16, f32)> = (0..edges.len())
        .map(|i| {
            (
                (edges.lhs[i], edges.rhs[i]),
                (edges.rel[i], edges.edge_weight(i)),
            )
        })
        .collect();
    assert_eq!(key_of.len(), edges.len(), "fixture pairs are unique");
    let (batch_size, c, u) = (5usize, 4usize, 2usize);
    let mut batcher = EpochBatcher::new(&blocks, batch_size);
    let mut rng = StdRng::seed_from_u64(3);
    let mut seen: Vec<(u32, u32)> = Vec::new();
    let mut n_batches = 0;
    while let Some(b) = batcher.next_batch(&edges, &t, &rels, c, u, &mut rng) {
        n_batches += 1;
        assert!(b.n_real >= 1 && b.n_real <= batch_size);
        assert_eq!(b.c, c);
        assert_eq!(b.u, u);
        assert_eq!(b.k, b.n_real.div_ceil(c));
        assert_eq!(b.lhs.len(), b.k * c);
        assert_eq!(b.rhs.len(), b.k * c);
        assert_eq!(b.row_w.len(), b.k * c);
        assert_eq!(b.col_valid.len(), b.k * c);
        for i in 0..b.n_real {
            let (r, w) = key_of[&(b.lhs[i], b.rhs[i])];
            assert_eq!(r as usize, b.rel, "one relation per batch");
            assert_eq!(
                b.row_w[i],
                rels.get(b.rel).weight * w,
                "row weight is relation × edge weight"
            );
        }
        seen.extend((0..b.n_real).map(|i| (b.lhs[i], b.rhs[i])));
    }
    assert!(n_batches >= 4, "7/5 + 12/5 + 1/5 batches at least");
    assert_eq!(batcher.remaining(), 0);
    seen.sort_unstable();
    let mut all: Vec<(u32, u32)> = key_of.keys().copied().collect();
    all.sort_unstable();
    assert_eq!(seen, all, "every edge is drawn exactly once per epoch");
}

#[test]
fn uniform_negatives_stay_inside_the_relations_own_node_types() {
    let t = types();
    let rels = relations(&t);
    let (edges, blocks) = three_relation_graph(&t);
    let mut rng = StdRng::seed_from_u64(11);
    let mut seen_rel = [false; 3];
    for _ in 0..20 {
        let mut batcher = EpochBatcher::new(&blocks, 20);
        while let Some(b) = batcher.next_batch(&edges, &t, &rels, 5, 16, &mut rng) {
            seen_rel[b.rel] = true;
            let r = rels.get(b.rel);
            let lr = t.range(r.lhs_type as usize);
            let rr = t.range(r.rhs_type as usize);
            assert_eq!(b.uni_lhs.len(), b.k * 16);
            assert_eq!(b.uni_rhs.len(), b.k * 16);
            assert!(
                b.uni_lhs.iter().all(|i| lr.contains(i)),
                "lhs negatives of `{}` inside `{}`",
                r.name,
                t.name(r.lhs_type as usize)
            );
            assert!(
                b.uni_rhs.iter().all(|i| rr.contains(i)),
                "rhs negatives of `{}` inside `{}`",
                r.name,
                t.name(r.rhs_type as usize)
            );
            if b.rel == 1 {
                // gene → term: the two sides are different ranges, so no
                // rhs negative can be a gene id.
                assert!(b.uni_rhs.iter().all(|&i| i >= 9));
            }
        }
    }
    assert!(seen_rel.iter().all(|&s| s), "every relation was batched");
}

#[test]
fn a_same_type_relation_draws_lhs_and_rhs_negatives_independently() {
    let t = types();
    let rels = relations(&t);
    let (edges, _) = three_relation_graph(&t);
    let mut rng = StdRng::seed_from_u64(5);
    // Only the ppi block; the other relations hand out nothing.
    let mut batcher = EpochBatcher::new(&[0..7, 7..7, 19..19], 7);
    let b = batcher
        .next_batch(&edges, &t, &rels, 7, 32, &mut rng)
        .expect("one batch");
    assert_eq!(b.rel, 0);
    assert!(b.uni_lhs.iter().all(|&i| i < 9));
    assert!(b.uni_rhs.iter().all(|&i| i < 9));
    assert_ne!(
        b.uni_lhs, b.uni_rhs,
        "independent draws over the same range"
    );
}

#[test]
fn pad_rows_carry_zero_weight_and_are_not_negatives() {
    let t = types();
    let rels = relations(&t);
    let (edges, _) = three_relation_graph(&t);
    // Only the go block (12 edges) → one batch of 12 at batch_size 20,
    // c = 5 → k = 3 chunks, 3 pad rows at the end.
    // Blocks are in relation order; the other two relations hand out nothing.
    let mut batcher = EpochBatcher::new(&[0..0, 7..19, 19..19], 20);
    let mut rng = StdRng::seed_from_u64(11);
    let b = batcher
        .next_batch(&edges, &t, &rels, 5, 4, &mut rng)
        .expect("one batch");
    assert_eq!(b.n_real, 12);
    assert_eq!(b.k, 3);
    assert_eq!(b.rel, 1);
    for i in 0..12 {
        assert!(b.row_w[i] > 0.0);
        assert_eq!(b.col_valid[i], 1.0);
    }
    for i in 12..15 {
        assert_eq!(b.row_w[i], 0.0, "pad row {i} has no loss weight");
        assert_eq!(b.col_valid[i], 0.0, "pad row {i} is not a negative");
        assert_eq!(b.lhs[i], 0);
        assert_eq!(b.rhs[i], 0);
    }
    assert_eq!(b.uni_lhs.len(), 3 * 4, "u uniform lhs per chunk");
    assert_eq!(b.uni_rhs.len(), 3 * 4, "u uniform rhs per chunk");
    assert!(batcher
        .next_batch(&edges, &t, &rels, 5, 4, &mut rng)
        .is_none());
}

#[test]
fn a_repeated_relation_is_drawn_that_many_times_per_epoch() {
    let t = types();
    let rels = relations(&t);
    let (edges, blocks) = three_relation_graph(&t);
    // go (relation 1) three times, marker (relation 2) twice, ppi once.
    let mut entries = Vec::new();
    for (r, k) in [(0usize, 1usize), (1, 3), (2, 2)] {
        for _ in 0..k {
            entries.push((r, blocks[r].clone()));
        }
    }
    let mut batcher = EpochBatcher::from_entries(entries, 5);
    assert_eq!(batcher.remaining(), 7 + 3 * 12 + 2);
    let mut rng = StdRng::seed_from_u64(3);
    let mut seen: HashMap<(u32, u32), usize> = HashMap::new();
    while let Some(b) = batcher.next_batch(&edges, &t, &rels, 4, 2, &mut rng) {
        for i in 0..b.n_real {
            *seen.entry((b.lhs[i], b.rhs[i])).or_default() += 1;
            let r = rels.get(b.rel);
            assert!(t.range(r.lhs_type as usize).contains(&b.lhs[i]));
        }
    }
    for i in 0..edges.len() {
        let want = [1, 3, 2][edges.rel[i] as usize];
        assert_eq!(
            seen[&(edges.lhs[i], edges.rhs[i])],
            want,
            "edge {i} visited {want} times"
        );
    }
}

#[test]
fn an_empty_block_is_a_relation_with_nothing_to_hand_out() {
    let t = types();
    let rels = relations(&t);
    let (edges, _) = three_relation_graph(&t);
    let mut rng = StdRng::seed_from_u64(1);
    let mut batcher = EpochBatcher::new(&[0..7, 7..7, 19..20], 100);
    let mut rels_seen = Vec::new();
    while let Some(b) = batcher.next_batch(&edges, &t, &rels, 4, 1, &mut rng) {
        rels_seen.push(b.rel);
    }
    rels_seen.sort_unstable();
    assert_eq!(rels_seen, vec![0, 2]);
}
