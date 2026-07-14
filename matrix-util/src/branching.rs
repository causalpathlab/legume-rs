//! Maximum-weight spanning **branching** (rooted forest) over a directed graph,
//! via Chu–Liu/Edmonds.
//!
//! A branching is a set of arcs with **in-degree ≤ 1** at every node and **no
//! cycles** — equivalently, a forest of arborescences, each edge pointing away
//! from its tree's root. [`max_branching`] returns the branching of maximum total
//! arc weight, subject to a per-node **`root_affinity`**: the value a node gets for
//! staying a root (taking no parent). A node adopts a real parent only when its best
//! incoming arc beats its `root_affinity`, so `root_affinity` is the knob that trades
//! one big tree (low affinity) against many small ones (high affinity).
//!
//! Implemented by adding a virtual super-root with an arc of weight `root_affinity[v]`
//! into every node, then computing a maximum spanning **arborescence** from it (every
//! node reachable, in-degree 1) — the super-root's children are the forest roots.
//! At `K ≤ few hundred` nodes the cubic-ish contraction cost is negligible.

use std::collections::HashSet;

/// A maximum-weight rooted forest over `n` nodes.
pub struct Branching {
    /// `parent[v] = Some(u)` for the chosen arc `u → v`; `None` when `v` is a root.
    pub parent: Vec<Option<usize>>,
    /// Component (tree) id per node, dense in `0..roots.len()`, ordered by ascending root.
    pub tree: Vec<usize>,
    /// Root node ids (in-degree 0), ascending.
    pub roots: Vec<usize>,
}

/// One directed arc in the contracted working graph. `orig` indexes the original
/// (super-root-augmented) arc list and is stable across contraction levels; `landed`
/// is the arc's target in the *current* level's numbering (which differs from `v` only
/// for an arc collapsed onto a cycle super-node — there `landed` is the pre-collapse
/// cycle member, needed to know where to break the cycle on expansion).
struct Arc {
    u: usize,
    v: usize,
    w: f64,
    orig: usize,
    landed: usize,
}

/// Maximum-weight spanning branching. `arcs` are `(u, v, weight)` directed `u → v`
/// over nodes `0..n`; `root_affinity[v]` is the score for leaving `v` a root. Arcs into
/// a node compete with that node's `root_affinity`. Ties resolve to the earliest arc.
pub fn max_branching(n: usize, arcs: &[(usize, usize, f32)], root_affinity: &[f32]) -> Branching {
    assert_eq!(root_affinity.len(), n, "root_affinity must have length n");
    if n == 0 {
        return Branching {
            parent: Vec::new(),
            tree: Vec::new(),
            roots: Vec::new(),
        };
    }

    // Super-root = node `n`, with an arc n → v of weight root_affinity[v] into every v.
    let sroot = n;
    let mut all: Vec<Arc> = Vec::with_capacity(arcs.len() + n);
    for (i, &(u, v, w)) in arcs.iter().enumerate() {
        if u == v {
            continue; // self-loops never help
        }
        all.push(Arc {
            u,
            v,
            w: w as f64,
            orig: i,
            landed: v,
        });
    }
    let sroot_orig_base = arcs.len();
    for (v, &aff) in root_affinity.iter().enumerate() {
        all.push(Arc {
            u: sroot,
            v,
            w: aff as f64,
            orig: sroot_orig_base + v,
            landed: v,
        });
    }

    let used = solve(n + 1, sroot, &all);

    // Reconstruct parent pointers from the chosen original arc ids.
    let mut parent: Vec<Option<usize>> = vec![None; n];
    for &orig in &used {
        let (u, v) = if orig < sroot_orig_base {
            (arcs[orig].0, arcs[orig].1)
        } else {
            (sroot, orig - sroot_orig_base)
        };
        if v < n {
            parent[v] = if u == sroot { None } else { Some(u) };
        }
    }

    // Roots = in-degree-0 nodes; component ids by ascending root.
    let roots: Vec<usize> = (0..n).filter(|&v| parent[v].is_none()).collect();
    let mut root_of_comp = vec![usize::MAX; n];
    for (c, &r) in roots.iter().enumerate() {
        root_of_comp[r] = c;
    }
    let mut tree = vec![usize::MAX; n];
    for v in 0..n {
        // walk to the root, then paint the path with its component id
        let mut path = Vec::new();
        let mut x = v;
        while tree[x] == usize::MAX && root_of_comp[x] == usize::MAX {
            path.push(x);
            match parent[x] {
                Some(p) => x = p,
                None => break,
            }
        }
        let comp = if tree[x] != usize::MAX {
            tree[x]
        } else {
            root_of_comp[x]
        };
        for &p in &path {
            tree[p] = comp;
        }
        tree[v] = comp;
    }

    Branching {
        parent,
        tree,
        roots,
    }
}

/// Maximum spanning arborescence rooted at `root` over `n` nodes; returns the chosen
/// original arc ids. Every non-root node is assumed reachable (the super-root guarantees
/// this at the top level, and every contracted cycle keeps its super-root arc).
fn solve(n: usize, root: usize, arcs: &[Arc]) -> Vec<usize> {
    // Best (max-weight) incoming arc per node.
    let mut best: Vec<Option<usize>> = vec![None; n];
    for (i, a) in arcs.iter().enumerate() {
        if a.v == root {
            continue;
        }
        match best[a.v] {
            None => best[a.v] = Some(i),
            Some(j) if a.w > arcs[j].w => best[a.v] = Some(i),
            _ => {}
        }
    }

    // Find a cycle in the functional parent graph (v → arcs[best[v]].u).
    let par = |v: usize| best[v].map(|i| arcs[i].u);
    let mut color = vec![0u8; n]; // 0 white, 1 gray, 2 black
    let mut cycle: Option<Vec<usize>> = None;
    for s in 0..n {
        if color[s] != 0 || s == root {
            continue;
        }
        let mut stack: Vec<usize> = Vec::new();
        let mut v = s;
        loop {
            if v == root || color[v] == 2 {
                break;
            }
            if color[v] == 1 {
                let start = stack.iter().position(|&x| x == v).unwrap();
                cycle = Some(stack[start..].to_vec());
                break;
            }
            color[v] = 1;
            stack.push(v);
            match par(v) {
                Some(p) => v = p,
                None => break,
            }
        }
        for &x in &stack {
            color[x] = 2;
        }
        if cycle.is_some() {
            break;
        }
    }

    // No cycle: the best arcs already form the arborescence.
    let Some(cyc) = cycle else {
        let mut used = Vec::new();
        for (v, slot) in best.iter().enumerate() {
            if v == root {
                continue;
            }
            if let Some(i) = slot {
                used.push(arcs[*i].orig);
            }
        }
        return used;
    };

    // Contract the cycle into a single super-node `cnode`.
    let in_cycle: HashSet<usize> = cyc.iter().copied().collect();
    let mut map = vec![usize::MAX; n];
    let mut next = 0;
    for (v, m) in map.iter_mut().enumerate() {
        if !in_cycle.contains(&v) {
            *m = next;
            next += 1;
        }
    }
    let cnode = next;
    next += 1;
    for &v in &cyc {
        map[v] = cnode;
    }
    let new_n = next;

    let mut new_arcs: Vec<Arc> = Vec::with_capacity(arcs.len());
    for a in arcs {
        let nu = map[a.u];
        let nv = map[a.v];
        if nu == nv {
            continue; // internal to the cycle (or self) — drop
        }
        let w = if in_cycle.contains(&a.v) {
            // reduced weight for maximum: gain of this arc minus the cycle arc it replaces
            a.w - arcs[best[a.v].unwrap()].w
        } else {
            a.w
        };
        new_arcs.push(Arc {
            u: nu,
            v: nv,
            w,
            orig: a.orig,
            landed: a.v,
        });
    }

    let sub_used = solve(new_n, map[root], &new_arcs);
    let sub_set: HashSet<usize> = sub_used.iter().copied().collect();

    // The one used arc entering `cnode` breaks the cycle at `v_enter`.
    let mut v_enter = None;
    for a in &new_arcs {
        if a.v == cnode && sub_set.contains(&a.orig) {
            v_enter = Some(a.landed);
            break;
        }
    }
    let v_enter = v_enter.expect("contracted cycle must have an entering arc");

    let mut used = sub_used;
    for &x in &cyc {
        if x != v_enter {
            used.push(arcs[best[x].unwrap()].orig);
        }
    }
    used
}

#[cfg(test)]
mod tests;
