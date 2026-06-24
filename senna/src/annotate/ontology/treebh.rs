//! TreeBH — hierarchical, multi-resolution FDR testing on a tree.
//!
//! Bogomolov, Peterson, Benjamini & Sabatti, "Hypotheses on a tree: new error
//! rates and testing strategies", Biometrika 108(3):575–590, 2021
//! (DOI 10.1093/biomet/asaa086).
//!
//! Pure routines over a flat tree given as a children-adjacency list. The caller
//! designates a `root` that is treated as always-rejected; its children form
//! level 1. Hypotheses are tested top-down: Benjamini–Hochberg is applied within
//! each family (a node's children) at a working target shrunk by the proportion
//! of rejections along the ancestor path, and a child-family is tested only if
//! its parent was rejected. This controls the level-specific selective FDR at
//! every resolution simultaneously.

/// Simes' combined p-value for a family of p-values:
/// `min_i ( m · p_(i) / i )` over the sorted `p_(1) ≤ … ≤ p_(m)`.
#[must_use]
pub fn simes(ps: &[f64]) -> f64 {
    let m = ps.len();
    if m == 0 {
        return 1.0;
    }
    let mut s: Vec<f64> = ps.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut best = f64::INFINITY;
    for (rank, &p) in s.iter().enumerate() {
        let i = (rank + 1) as f64;
        best = best.min(m as f64 * p / i);
    }
    best.min(1.0)
}

/// Benjamini–Hochberg rejection set within a family at level `alpha`. With
/// `by = true`, applies the Benjamini–Yekutieli correction (`alpha /= H_m`,
/// `H_m = Σ 1/i`), valid under arbitrary dependence. Returns the indices into
/// `ps` that are rejected.
#[must_use]
pub fn bh_reject(ps: &[f64], alpha: f64, by: bool) -> Vec<usize> {
    let m = ps.len();
    if m == 0 {
        return Vec::new();
    }
    let alpha = if by {
        let hm: f64 = (1..=m).map(|i| 1.0 / i as f64).sum();
        alpha / hm
    } else {
        alpha
    };
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        ps[a]
            .partial_cmp(&ps[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Largest rank k with p_(k) ≤ (k/m)·alpha.
    let mut k_max: Option<usize> = None;
    for (rank, &idx) in order.iter().enumerate() {
        let i = (rank + 1) as f64;
        if ps[idx] <= (i / m as f64) * alpha {
            k_max = Some(rank);
        }
    }
    match k_max {
        Some(r) => order[..=r].to_vec(),
        None => Vec::new(),
    }
}

/// Bottom-up Simes combination. `leaf_p[i] = Some(p)` marks a leaf carrying a
/// data p-value; `None` marks an internal node whose p is combined from its
/// children. `postorder` must list nodes children-before-parents. Returns the
/// combined p-value per node.
#[must_use]
pub fn combine_bottom_up(
    children: &[Vec<usize>],
    postorder: &[usize],
    leaf_p: &[Option<f64>],
) -> Vec<f64> {
    let mut cp = vec![1.0f64; children.len()];
    for &node in postorder {
        if let Some(p) = leaf_p[node] {
            cp[node] = p.clamp(0.0, 1.0);
        } else {
            let kids = &children[node];
            cp[node] = if kids.is_empty() {
                1.0
            } else {
                simes(&kids.iter().map(|&c| cp[c]).collect::<Vec<_>>())
            };
        }
    }
    cp
}

/// Top-down TreeBH. `root` is treated as always-rejected; its children are the
/// level-1 family. Returns a per-node rejection mask (with `root` set true).
/// `q` is the per-level selective-FDR target; `by` toggles the BY correction
/// within families.
#[must_use]
pub fn descend(children: &[Vec<usize>], root: usize, cp: &[f64], q: f64, by: bool) -> Vec<bool> {
    let mut rejected = vec![false; children.len()];
    rejected[root] = true;
    // (node whose children-family to test, gamma = ∏ r_A/n_A over ancestors).
    let mut stack: Vec<(usize, f64)> = vec![(root, 1.0)];
    while let Some((v, gamma)) = stack.pop() {
        let fam = &children[v];
        if fam.is_empty() {
            continue;
        }
        let ps: Vec<f64> = fam.iter().map(|&c| cp[c]).collect();
        let rej = bh_reject(&ps, q * gamma, by);
        if rej.is_empty() {
            continue; // STOP — this node abstains over its children.
        }
        let child_gamma = gamma * (rej.len() as f64 / fam.len() as f64);
        for &i in &rej {
            let c = fam[i];
            rejected[c] = true;
            stack.push((c, child_gamma));
        }
    }
    rejected
}

#[cfg(test)]
mod tests {
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
}
