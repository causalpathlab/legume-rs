use super::*;

fn types() -> NodeTypeTable {
    NodeTypeTable::new(&[("gene", 5), ("term", 3), ("cell_type", 2)]).unwrap()
}

fn rel(name: &str, l: u16, r: u16, w: f32, undirected: bool) -> Relation {
    Relation {
        name: name.into(),
        lhs_type: l,
        rhs_type: r,
        weight: w,
        undirected,
        polarity: RelationPolarity::Friend,
    }
}

#[test]
fn node_type_ranges_are_contiguous_cover_every_node_and_invert() {
    let t = types();
    assert_eq!(t.len(), 3);
    assert_eq!(t.range(0), 0..5);
    assert_eq!(t.range(1), 5..8);
    assert_eq!(t.range(2), 8..10);
    assert_eq!(t.n_total(), 10);
    assert_eq!(t.n_nodes(1), 3);
    assert_eq!(t.index_of("term"), Some(1));
    assert_eq!(t.index_of("word"), None);
    for g in 0..10u32 {
        let (ty, local) = t.local(g);
        assert!(t.range(ty).contains(&g));
        assert_eq!(t.global(ty, local), g);
    }
    assert_eq!(t.type_of(0), 0);
    assert_eq!(t.type_of(4), 0);
    assert_eq!(t.type_of(5), 1);
    assert_eq!(t.type_of(9), 2);
}

#[test]
fn node_type_table_refuses_empty_duplicate_and_zero_sized_types() {
    assert!(NodeTypeTable::new(&[]).is_err());
    assert!(NodeTypeTable::new(&[("gene", 0)]).is_err());
    assert!(NodeTypeTable::new(&[("gene", 1), ("gene", 2)]).is_err());
}

#[test]
fn relation_table_checks_type_indices_weights_and_the_undirected_flag() {
    let t = types();
    assert!(RelationTable::new(vec![], &t).is_err());
    assert!(RelationTable::new(vec![rel("x", 0, 7, 1.0, false)], &t).is_err());
    assert!(RelationTable::new(vec![rel("x", 0, 1, -1.0, false)], &t).is_err());
    assert!(RelationTable::new(vec![rel("x", 0, 1, f32::NAN, false)], &t).is_err());
    assert!(
        RelationTable::new(vec![rel("x", 0, 1, 1.0, true)], &t).is_err(),
        "undirected across two types"
    );
    let ok = RelationTable::new(
        vec![rel("ppi", 0, 0, 1.0, true), rel("go", 0, 1, 2.0, false)],
        &t,
    )
    .unwrap();
    assert_eq!(ok.len(), 2);
    assert_eq!(ok.get(1).weight, 2.0);
}

#[test]
fn edge_list_validation_catches_ids_outside_the_relations_types() {
    let t = types();
    let rels = RelationTable::new(vec![rel("go", 0, 1, 1.0, false)], &t).unwrap();
    let good = TypedEdgeList {
        lhs: vec![0, 4],
        rhs: vec![5, 7],
        rel: vec![0, 0],
        weight: None,
    };
    good.validate(&t, &rels).unwrap();
    let bad_rhs = TypedEdgeList {
        lhs: vec![0],
        rhs: vec![8], // a cell_type, not a term
        rel: vec![0],
        weight: None,
    };
    assert!(bad_rhs.validate(&t, &rels).is_err());
    let bad_rel = TypedEdgeList {
        lhs: vec![0],
        rhs: vec![5],
        rel: vec![3],
        weight: None,
    };
    assert!(bad_rel.validate(&t, &rels).is_err());
    let bad_w = TypedEdgeList {
        lhs: vec![0],
        rhs: vec![5],
        rel: vec![0],
        weight: Some(vec![-0.5]),
    };
    assert!(bad_w.validate(&t, &rels).is_err());
}

#[test]
fn grouping_by_relation_keeps_every_edge_with_its_weight_and_reports_the_blocks() {
    let mut e = TypedEdgeList {
        lhs: vec![0, 1, 2, 3, 4],
        rhs: vec![5, 6, 7, 5, 6],
        rel: vec![1, 0, 1, 2, 0],
        weight: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5]),
    };
    let blocks = e.group_by_relation(3);
    assert_eq!(blocks, vec![0..2, 2..4, 4..5]);
    for (r, b) in blocks.iter().enumerate() {
        assert!(b.clone().all(|i| e.rel[i] as usize == r));
    }
    // Stable: relation 0 keeps (1, 0.2) before (4, 0.5).
    assert_eq!(&e.lhs[0..2], &[1, 4]);
    assert_eq!(&e.weight.as_ref().unwrap()[0..2], &[0.2, 0.5]);
    assert_eq!(e.counts_per_relation(3), vec![2, 2, 1]);
    // Every (lhs, weight) pair survives.
    let mut pairs: Vec<(u32, f32)> = e
        .lhs
        .iter()
        .zip(e.weight.as_ref().unwrap())
        .map(|(&l, &w)| (l, w))
        .collect();
    pairs.sort_by_key(|a| a.0);
    assert_eq!(
        pairs,
        vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)]
    );
}
