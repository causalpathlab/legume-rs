use super::*;

/// Build a level with known counts and a hand-written super-edge list. Column
/// `p` of `counts` is filled with `p + 1`, so any mis-summing shows up as a
/// wrong integer rather than a plausible-looking float.
fn level(super_edges: Vec<(usize, usize)>, n_pb: usize, n_genes: usize) -> LevelPseudobulk {
    let counts = Mat::from_fn(n_genes, n_pb, |_, p| (p + 1) as f32);
    LevelPseudobulk {
        cell_labels: Vec::new(),
        counts,
        // One fine edge per super-edge keeps `fine_to_super` well-formed; the
        // collapse does not read it, but the invariant should hold anyway.
        fine_to_super: (0..super_edges.len()).map(Some).collect(),
        super_edges,
        e_pb: Mat::zeros(n_pb, 1),
        pb_offset: 0,
    }
}

/// Intra-group fine edges are reported as `None`, not as a super-edge.
#[test]
fn internal_fine_edges_have_no_super_edge() {
    let mut lvl = level(vec![(0, 1)], 2, 3);
    lvl.fine_to_super = vec![Some(0), None, None];
    assert_eq!(lvl.n_internal_fine_edges(), 2);
    assert_eq!(lvl.n_super_edges(), 1);
}
