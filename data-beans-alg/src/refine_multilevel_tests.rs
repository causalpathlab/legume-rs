use super::*;

#[test]
fn test_build_candidate_sets_fallback() {
    // pbsamp 0 has siblings [0,1] but BBKNN lands only on a non-sibling group.
    let siblings = vec![vec![0usize, 1], vec![0, 1]];
    let bbknn = vec![vec![1usize], vec![0]];
    let pbsamp_to_group = vec![0usize, 1];
    // Neighbor of sc0 is sc1 (group 1) → intersection {1}.
    // sc0 is in group 0; intersection lacks 0 so 0 is appended.
    let cand = build_candidate_sets(&siblings, &bbknn, &pbsamp_to_group);
    assert_eq!(cand[0], vec![0, 1]);
    assert_eq!(cand[1], vec![0, 1]);
}

#[test]
fn child_offset_within_parent_is_bounded_per_parent() {
    // Two parents; child codes nest in them. Offsets are local, first-seen
    // indices within each parent (so they're bounded by children-per-parent,
    // not by the global child-code count).
    let parent = vec![0usize, 0, 0, 1, 1, 1];
    let child = vec![10usize, 10, 11, 22, 23, 22];
    let off = child_offset_within_parent(&child, &parent);
    // parent 0: 10→0, 11→1 ; parent 1: 22→0, 23→1
    assert_eq!(off, vec![0, 0, 1, 0, 1, 0]);

    // Subdividing a *scrambled* refined parent by the offset stays bounded
    // by #parent_groups × max_children_per_parent — no cross-product blowup.
    let refined_parent = vec![3usize, 3, 3, 8, 8, 8];
    let (_lab, k) = project_to_refinement(&off, &refined_parent);
    assert!(k <= 4, "expected ≤ 2 parents × 2 offsets, got {k}");
}

#[test]
fn test_project_to_refinement_splits_straddling_groups() {
    // child group 0 straddles parents A=0 and B=1; group 1 sits wholly
    // under parent B. Re-projection must split group 0 into two and
    // produce a strict refinement of `parent`.
    let child = vec![0usize, 0, 0, 1, 1];
    let parent = vec![0usize, 0, 1, 1, 1];
    let (reproj, k) = project_to_refinement(&child, &parent);

    // 3 distinct (child, parent) pairs: (0,0), (0,1), (1,1).
    assert_eq!(k, 3);
    // Entities 0,1 → (0,0); entity 2 → (0,1); entities 3,4 → (1,1).
    assert_eq!(reproj[0], reproj[1]);
    assert_ne!(reproj[0], reproj[2]);
    assert_eq!(reproj[3], reproj[4]);
    assert_ne!(reproj[2], reproj[3]);

    // Strict refinement: every reprojected group maps to one parent.
    let mut group_parent = std::collections::HashMap::new();
    for (&g, &p) in reproj.iter().zip(parent.iter()) {
        assert_eq!(*group_parent.entry(g).or_insert(p), p);
    }
}
