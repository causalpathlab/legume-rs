//! Unit tests for the TreeBH primitives (`super` = [`crate::treebh`]).

use super::*;

/// Augmented tree mirroring an internal "T" label with CD8/CD4 children:
///   0 root → 1 A("T") → {2 A_self, 3 B_node("CD8")→4 B_self, 5 C_node("CD4")→6 C_self}
/// leaves (carrying data p) are the *_self nodes 2/4/6.
fn tree() -> (Vec<Vec<usize>>, Vec<usize>) {
    let children = vec![
        vec![1],       // 0 root
        vec![2, 3, 5], // 1 A (T)
        vec![],        // 2 A_self
        vec![4],       // 3 B_node (CD8)
        vec![],        // 4 B_self
        vec![6],       // 5 C_node (CD4)
        vec![],        // 6 C_self
    ];
    let postorder = vec![2, 4, 3, 6, 5, 1, 0];
    (children, postorder)
}

fn run(p_tself: f64, p_cd8: f64, p_cd4: f64) -> Vec<bool> {
    let (children, postorder) = tree();
    let mut leaf_p = vec![None; 7];
    leaf_p[2] = Some(p_tself);
    leaf_p[4] = Some(p_cd8);
    leaf_p[6] = Some(p_cd4);
    let cp = combine_bottom_up(&children, &postorder, &leaf_p);
    descend(&children, 0, &cp, 0.1, false)
}

#[test]
fn peaked_subtype_descends() {
    // Generic-T present AND CD8 strong, CD4 absent → resolve to CD8.
    let r = run(0.001, 0.001, 0.9);
    assert!(r[4], "CD8 self should be rejected");
    assert!(!r[6], "CD4 self should NOT be rejected");
    assert!(r[2], "T self rejected");
}

#[test]
fn flat_siblings_stop_at_parent() {
    // Generic-T strong but CD8≈CD4 weak → resolve to T, abstain on subtype.
    let r = run(0.001, 0.2, 0.2);
    assert!(r[2], "T self rejected (resolved at T)");
    assert!(!r[4] && !r[6], "neither subtype rejected (abstain)");
}

#[test]
fn all_weak_cannot_explain() {
    // Nothing significant anywhere → only the virtual root is 'rejected'.
    let r = run(0.9, 0.9, 0.9);
    assert!(r[0]);
    assert!(r.iter().skip(1).all(|&x| !x), "no real hypothesis rejected");
}

#[test]
fn simes_and_bh_basics() {
    assert!((simes(&[0.01, 0.5, 0.9]) - 0.03).abs() < 1e-9); // min(3*.01/1, 3*.5/2, 3*.9/3)
    assert_eq!(bh_reject(&[0.001, 0.2, 0.2], 0.1, false), vec![0]);
    assert!(bh_reject(&[0.9, 0.9], 0.1, false).is_empty());
}
