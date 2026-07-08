//! Local, root-free branch structure of the centroid MST — the substrate for the
//! root-invariant sibling-branch association test (`faba dyn-assoc`).
//!
//! The instability of pseudotime-based matching comes from committing to a single
//! global **root**. A **junction** (a node where the tree diverges) is instead a
//! *local* feature: on the undirected MST it is any node of degree ≥ 3, and the
//! subtrees meeting there are **sibling branches** — the alternative fates — no matter
//! where the root is. Removing a junction `J` splits the tree into `deg(J)` components;
//! each is one branch off `J`. Every node is then assigned to its **nearest** junction,
//! the branch it sits on relative to that junction, and its graph **distance** from it —
//! `(junction, sibling_branch, dist_from_junction)`. `dist_from_junction` is the local,
//! root-invariant axis cells are matched on (replacing global pseudotime).
//!
//! The velocity orientation ([`super::orient::directed_edges`]) is kept only as an
//! annotation: a branch is `downstream` (a diverging fate) when its incident edge points
//! away from the junction, else it is the `upstream` trunk / shared context. Nothing here
//! depends on a root existing.

use std::collections::VecDeque;

/// One node's placement relative to its nearest junction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeBranch {
    /// MST node id of the nearest junction (degree-≥3 node).
    pub junction: usize,
    /// Which incident branch of that junction this node lies on (`Some(0..deg(junction))`),
    /// or `None` when the node *is* the junction (the hub).
    pub branch: Option<usize>,
    /// Graph distance (edge count) from the node to its junction.
    pub dist: u32,
    /// `true` when this branch's incident edge points *away* from the junction under the
    /// velocity orientation (a diverging fate); `false` for the upstream trunk. Always
    /// `false` for an unoriented tree and for the junction hub itself.
    pub downstream: bool,
}

/// Local branch structure over the `k` MST nodes.
pub struct BranchTopology {
    /// Per-node placement; `None` only when the tree has no junction at all (a bare path
    /// or a single edge — no divergence to contrast).
    pub node_branch: Vec<Option<NodeBranch>>,
    /// Junction node ids (degree ≥ 3 in the undirected MST), ascending.
    pub junctions: Vec<usize>,
}

/// Undirected adjacency lists from the MST edge set.
fn undirected_adjacency(edges: &[(usize, usize)], k: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); k];
    for &(a, b) in edges {
        if a < k && b < k && a != b {
            adj[a].push(b);
            adj[b].push(a);
        }
    }
    adj
}

/// Detect junctions and assign every node to its nearest junction + sibling branch.
///
/// `edges` is the undirected MST edge set (`k` nodes); `directed` is the same edges after
/// velocity orientation ([`super::orient::directed_edges`]) — pass an empty slice for an
/// unoriented tree (every branch is then `downstream = false`).
pub fn detect_branches(
    edges: &[(usize, usize)],
    directed: &[(usize, usize)],
    k: usize,
) -> BranchTopology {
    let adj = undirected_adjacency(edges, k);
    let junctions: Vec<usize> = (0..k).filter(|&n| adj[n].len() >= 3).collect();

    // Fast lookup: is edge (from → to) oriented this way under velocity?
    let directed_set: std::collections::HashSet<(usize, usize)> =
        directed.iter().copied().collect();

    let mut node_branch: Vec<Option<NodeBranch>> = vec![None; k];
    let mut best_dist: Vec<u32> = vec![u32::MAX; k];

    for &j in &junctions {
        // Each incident neighbour of `j` roots one branch (one component of `tree \ {j}`).
        // Pre-mark `j` as visited (best_dist 0) so BFS never crosses back through it — that
        // stand-in for parent-tracking makes the queue entry `(node, dist)`, no parent field.
        best_dist[j] = 0;
        for (branch_idx, &start) in adj[j].iter().enumerate() {
            let downstream = directed_set.contains(&(j, start));
            let mut queue: VecDeque<(usize, u32)> = VecDeque::new();
            queue.push_back((start, 1));
            while let Some((node, dist)) = queue.pop_front() {
                if dist >= best_dist[node] {
                    continue; // already assigned to a nearer junction (or is the junction itself)
                }
                best_dist[node] = dist;
                node_branch[node] = Some(NodeBranch {
                    junction: j,
                    branch: Some(branch_idx),
                    dist,
                    downstream,
                });
                for &nb in &adj[node] {
                    queue.push_back((nb, dist + 1));
                }
            }
        }
        // Claim `j` as the hub (its `best_dist` was already forced to 0 above).
        node_branch[j] = Some(NodeBranch {
            junction: j,
            branch: None,
            dist: 0,
            downstream: false,
        });
    }

    BranchTopology {
        node_branch,
        junctions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Y: trunk 0–1–2, junction at 2, two fates 2–3 and 2–4.
    fn y_tree() -> Vec<(usize, usize)> {
        vec![(0, 1), (1, 2), (2, 3), (2, 4)]
    }

    #[test]
    fn detects_the_single_junction() {
        let t = detect_branches(&y_tree(), &[], 5);
        assert_eq!(t.junctions, vec![2], "node 2 is the only degree-≥3 node");
    }

    #[test]
    fn assigns_nodes_to_branches_and_distances() {
        let t = detect_branches(&y_tree(), &[], 5);
        let nb = |n: usize| t.node_branch[n].expect("assigned");
        // Junction hub.
        assert_eq!(nb(2).branch, None);
        assert_eq!(nb(2).dist, 0);
        // Trunk side: 1 is one step out, 0 is two steps — same branch.
        assert_eq!(nb(1).junction, 2);
        assert_eq!(nb(1).dist, 1);
        assert_eq!(nb(0).dist, 2);
        assert_eq!(nb(0).branch, nb(1).branch, "0 and 1 share the trunk branch");
        // Fates 3 and 4 are distinct single-node branches, each one step out.
        assert_eq!(nb(3).dist, 1);
        assert_eq!(nb(4).dist, 1);
        assert_ne!(nb(3).branch, nb(4).branch, "3 and 4 are different fates");
        assert_ne!(nb(3).branch, nb(1).branch, "fate 3 differs from the trunk");
    }

    #[test]
    fn orientation_marks_downstream_fates() {
        // Orient trunk toward the junction (1→2) and fates away (2→3, 2→4).
        let directed = vec![(0, 1), (1, 2), (2, 3), (2, 4)];
        let t = detect_branches(&y_tree(), &directed, 5);
        assert!(t.node_branch[3].unwrap().downstream, "fate 3 is downstream");
        assert!(t.node_branch[4].unwrap().downstream, "fate 4 is downstream");
        assert!(
            !t.node_branch[1].unwrap().downstream,
            "trunk (2→? no) is upstream"
        );
    }

    #[test]
    fn nearest_junction_wins_with_two_junctions() {
        // 0–1–2(J)–3–4(J)–5, plus fates 2–6 and 4–7 so 2 and 4 are both degree-3.
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (4, 7)];
        let t = detect_branches(&edges, &[], 8);
        assert_eq!(t.junctions, vec![2, 4]);
        // Node 3 is equidistant (1) from both junctions; the first-scanned (2) claims it by
        // strict `<`, which is deterministic — assert it is assigned to one of them at dist 1.
        let nb3 = t.node_branch[3].unwrap();
        assert_eq!(nb3.dist, 1);
        assert!(nb3.junction == 2 || nb3.junction == 4);
        // Node 0 is nearest junction 2 (dist 2), not 4 (dist 4).
        assert_eq!(t.node_branch[0].unwrap().junction, 2);
        assert_eq!(t.node_branch[0].unwrap().dist, 2);
        // Node 5 is nearest junction 4.
        assert_eq!(t.node_branch[5].unwrap().junction, 4);
    }

    #[test]
    fn bare_path_has_no_junction() {
        let t = detect_branches(&[(0, 1), (1, 2), (2, 3)], &[], 4);
        assert!(t.junctions.is_empty());
        assert!(t.node_branch.iter().all(|x| x.is_none()));
    }
}
