use super::*;

/// Total weight of a branching under the given arcs + root_affinity.
fn score(b: &Branching, arcs: &[(usize, usize, f32)], root_affinity: &[f32]) -> f32 {
    let mut s = 0.0;
    for (v, p) in b.parent.iter().enumerate() {
        match p {
            None => s += root_affinity[v],
            Some(u) => {
                s += arcs
                    .iter()
                    .filter(|&&(a, c, _)| a == *u && c == v)
                    .map(|&(_, _, w)| w)
                    .fold(f32::MIN, f32::max);
            }
        }
    }
    s
}

/// Brute-force optimum over all in-degree-≤1 acyclic selections, for validation.
fn brute(n: usize, arcs: &[(usize, usize, f32)], root_affinity: &[f32]) -> f32 {
    // Each node picks a parent among incoming arcs or "root". Enumerate choices,
    // reject cyclic ones, keep max score. n is tiny in tests.
    let mut incoming: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for &(u, v, w) in arcs {
        if u != v {
            incoming[v].push((u, w));
        }
    }
    let mut best = f32::MIN;
    let mut choice = vec![0usize; n]; // index into option list per node
    let options: Vec<Vec<Option<(usize, f32)>>> = (0..n)
        .map(|v| {
            let mut o: Vec<Option<(usize, f32)>> = vec![None];
            for &(u, w) in &incoming[v] {
                o.push(Some((u, w)));
            }
            o
        })
        .collect();
    loop {
        // build parent + score
        let mut parent = vec![None; n];
        let mut sc = 0.0;
        for v in 0..n {
            match options[v][choice[v]] {
                None => sc += root_affinity[v],
                Some((u, w)) => {
                    parent[v] = Some(u);
                    sc += w;
                }
            }
        }
        // acyclic check
        let mut ok = true;
        for start in 0..n {
            let mut seen = vec![false; n];
            let mut x = start;
            let mut steps = 0;
            while let Some(p) = parent[x] {
                if seen[x] || steps > n {
                    ok = false;
                    break;
                }
                seen[x] = true;
                x = p;
                steps += 1;
            }
            if !ok {
                break;
            }
        }
        if ok && sc > best {
            best = sc;
        }
        // odometer increment
        let mut i = 0;
        loop {
            if i == n {
                return best;
            }
            choice[i] += 1;
            if choice[i] < options[i].len() {
                break;
            }
            choice[i] = 0;
            i += 1;
        }
    }
}

#[test]
fn simple_chain_orients_away_from_root() {
    // 0 -> 1 -> 2 strongly; root_affinity low so one tree forms.
    let arcs = vec![(0, 1, 5.0), (1, 2, 5.0), (1, 0, 0.1), (2, 1, 0.1)];
    let ra = vec![1.0, 0.0, 0.0];
    let b = max_branching(3, &arcs, &ra);
    assert_eq!(b.roots, vec![0]);
    assert_eq!(b.parent, vec![None, Some(0), Some(1)]);
    assert_eq!(b.tree, vec![0, 0, 0]);
}

#[test]
fn convergence_keeps_max_weight_parent() {
    // Both 0 and 2 point into 1; the heavier parent (0→1, w=9) must win.
    let arcs = vec![(0, 1, 9.0), (2, 1, 3.0), (0, 2, 4.0)];
    let ra = vec![1.0, 0.0, 0.0];
    let b = max_branching(3, &arcs, &ra);
    assert_eq!(b.parent[1], Some(0), "node 1 keeps its max-weight parent");
    assert_eq!(b.parent[2], Some(0), "node 2 chains off 0");
}

#[test]
fn virtual_root_cuts_weak_link_into_a_forest() {
    // 0->1 strong, 1->2 very weak; a high affinity on node 2 makes it its own root.
    let arcs = vec![(0, 1, 8.0), (1, 2, 0.05)];
    let ra = vec![1.0, 0.0, 5.0];
    let b = max_branching(3, &arcs, &ra);
    assert_eq!(b.parent[2], None, "weak link cut, node 2 becomes a root");
    assert!(b.roots.contains(&0) && b.roots.contains(&2));
    assert_ne!(b.tree[0], b.tree[2], "two separate trees");
}

#[test]
fn resolves_a_two_cycle() {
    // Mutual max arcs 0<->1 form a 2-cycle; Edmonds must break it.
    let arcs = vec![(0, 1, 5.0), (1, 0, 4.0)];
    let ra = vec![1.0, 0.5];
    let b = max_branching(2, &arcs, &ra);
    // one of them is a root, the other its child; acyclic.
    let n_roots = b.roots.len();
    assert!(n_roots >= 1);
    // exactly one real arc chosen
    let chosen: usize = b.parent.iter().filter(|p| p.is_some()).count();
    assert_eq!(chosen, 1);
}

#[test]
fn matches_brute_force_on_random_small_graphs() {
    // Deterministic pseudo-random graphs (no rng dep): vary structure by index.
    for seed in 0u32..40 {
        let n = 4 + (seed as usize % 3); // 4..6
        let mut arcs = Vec::new();
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
        let mut nextw = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            ((x % 100) as f32) / 10.0
        };
        for u in 0..n {
            for v in 0..n {
                if u != v && (nextw() > 4.0) {
                    arcs.push((u, v, nextw()));
                }
            }
        }
        let ra: Vec<f32> = (0..n).map(|_| nextw() * 0.5).collect();
        let b = max_branching(n, &arcs, &ra);
        // validity: acyclic, in-degree ≤ 1 (guaranteed by parent[] shape)
        let got = score(&b, &arcs, &ra);
        let opt = brute(n, &arcs, &ra);
        assert!(
            (got - opt).abs() < 1e-3,
            "seed {seed}: branching score {got} != optimum {opt}"
        );
    }
}
