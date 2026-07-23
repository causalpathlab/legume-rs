//! Fixed pseudobulk lineage structure for the velocity-drift SEM residual (velocity-KNN).
//!
//! Given the analytic pb-level velocity readout ([`PbLevelVelocity`]: identity
//! `θ_pb` + velocity `δ_pb` per pb node per collapse level), build a **directed**
//! neighbour graph over the pb nodes of each level, oriented by velocity: an edge
//! `i → j` is kept when `j` is velocity-forward of `i` (`⟨θ_j − θ_i, δ_i⟩ > 0`).
//! This fixed structure feeds the velocity-drift SEM residual
//! `Σ_{i→j} w_ij ‖ê_j − ê_i − s·v̂_i‖²` added to phase-1 training (see
//! `crate::training::PbSemTerm`).

use super::projection::PbLevelVelocity;

/// Nearest pb neighbours considered per node when building the directed graph.
pub const DEFAULT_LINEAGE_KNN: usize = 10;
/// Velocity-drift step `s` in the SEM residual `ê_j − ê_i − s·v̂_i`.
pub const DEFAULT_SEM_STEP: f32 = 1.0;
/// Weight `λ_sem` multiplying the SEM residual in the composite loss.
pub const DEFAULT_SEM_WEIGHT: f32 = 0.1;
/// θ-space neighbours averaged when smoothing the pb velocity field.
pub const DEFAULT_SMOOTH_KNN: usize = 10;

/// Fixed directed lineage structure over the pb nodes of one collapse level.
/// Edges are `(parent i, child j, weight)` with `j` velocity-forward of `i`;
/// `velocity` is the per-node **unit** drift `v̂` flattened `[n_pb × h]` (zero
/// rows for nodes whose `‖δ‖ ≈ 0`, which emit no outgoing edges).
pub struct PbLineageLevel {
    pub n_pb: usize,
    pub edges: Vec<(u32, u32, f32)>,
    pub velocity: Vec<f32>,
}

/// Build one [`PbLineageLevel`] per collapse level (same order as `pb_vel`). With
/// `mst`, each level is a **minimum spanning tree** over the θ geometry, velocity-oriented
/// into a DAG (sparse, `n−1` edges) — [`build_one_level_mst`] — instead of the dense
/// velocity-KNN ([`build_one_level`]).
pub fn build_pb_lineage(
    pb_vel: &[PbLevelVelocity],
    h: usize,
    knn: usize,
    mst: bool,
) -> Vec<PbLineageLevel> {
    pb_vel
        .iter()
        .map(|lvl| {
            if mst {
                build_one_level_mst(lvl, h)
            } else {
                build_one_level(lvl, h, knn)
            }
        })
        .collect()
}

/// Smooth + confidence-gate the pb velocity field of every level (see
/// [`smooth_pb_velocity`]). θ is unchanged; only δ is denoised. Applied to the pb
/// readout before it orients the lineage graph / SEM drift / cell-lift, so all
/// consumers see the same stabilized velocity.
pub fn smooth_pb_velocity_levels(
    pb_vel: &[PbLevelVelocity],
    h: usize,
    knn: usize,
) -> Vec<PbLevelVelocity> {
    pb_vel
        .iter()
        .map(|lvl| smooth_pb_velocity(lvl, h, knn))
        .collect()
}

/// Velocity-graph smoothing of one level's pb velocity field: replace each node's raw δ
/// with a Gaussian-kernel average of its θ-space KNN neighbours' δ,
/// `v̄ᵢ = Σ_{j∈KNN(i)∪{i}} exp(−d²ᵢⱼ/σ²ᵢ)·δⱼ`. Summing **raw** δ (not unit v̂) lets
/// higher-magnitude — more confident — neighbours dominate. θ is untouched.
///
/// Non-circular: uses only δ + θ geometry, never pseudotime — so it denoises the
/// exogenous velocity anchor without dissolving the independence that breaks the
/// structure-from-embedding circularity. Opt-in (`--lineage-smooth`); on the clean sim
/// it is a wash (no noise to remove, and it can blur branch-point velocity), so the
/// payoff is on noisy real spliced/unspliced data.
pub fn smooth_pb_velocity(vel: &PbLevelVelocity, h: usize, knn: usize) -> PbLevelVelocity {
    let n = vel.n_pb;
    if n < 2 {
        return PbLevelVelocity {
            n_pb: n,
            theta: vel.theta.clone(),
            delta: vel.delta.clone(),
        };
    }
    let theta = &vel.theta;
    let k = knn.min(n - 1);

    // Gaussian-kernel average over each node's k nearest θ-neighbours (self included).
    let mut delta = vec![0f32; n * h];
    for i in 0..n {
        let mut nbrs = knn_sorted(theta, i, n, h);
        nbrs.truncate(k);
        // Adaptive bandwidth σ² = mean squared distance to the k neighbours (>0).
        let sig2 = (nbrs.iter().map(|&(d, _)| d).sum::<f32>() / k.max(1) as f32).max(1e-8);
        // Self contributes weight 1 (d = 0); neighbours a Gaussian of their θ distance.
        let out = &mut delta[i * h..i * h + h];
        out.copy_from_slice(row(&vel.delta, i, h));
        for &(d, j) in &nbrs {
            let w = (-d / sig2).exp();
            let dj = row(&vel.delta, j, h);
            for c in 0..h {
                out[c] += w * dj[c];
            }
        }
    }
    PbLevelVelocity {
        n_pb: n,
        theta: theta.clone(),
        delta,
    }
}

/// Per-node unit velocity `v̂` (‖·‖-normalized `δ`; zero rows with `has_vel = false`
/// where `‖δ‖ ≈ 0`, orientation undefined). Shared by the lineage graph, the pb-DAG
/// term, and the cell lift so all orient off the identical velocity field.
pub(crate) fn unit_velocity(lvl: &PbLevelVelocity, h: usize) -> (Vec<f32>, Vec<bool>) {
    let n = lvl.n_pb;
    let mut velocity = vec![0f32; n * h];
    let mut has_vel = vec![false; n];
    for i in 0..n {
        let d = row(&lvl.delta, i, h);
        let nrm = d.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 1e-8 {
            for k in 0..h {
                velocity[i * h + k] = d[k] / nrm;
            }
            has_vel[i] = true;
        }
    }
    (velocity, has_vel)
}

/// The other `n-1` nodes ranked by θ distance to node `i` (nearest first), as
/// `(dist², j)`. Callers `take(k)`/`truncate(k)` the neighbour count they need.
fn knn_sorted(theta: &[f32], i: usize, n: usize, h: usize) -> Vec<(f32, usize)> {
    let ti = row(theta, i, h);
    let mut nbrs: Vec<(f32, usize)> = (0..n)
        .filter(|&j| j != i)
        .map(|j| (dist2(ti, row(theta, j, h)), j))
        .collect();
    nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    nbrs
}

/// Row `i` (length `h`) of a flattened `[n × h]` row-major buffer.
pub(crate) fn row(m: &[f32], i: usize, h: usize) -> &[f32] {
    &m[i * h..(i + 1) * h]
}

/// Squared Euclidean distance between two equal-length rows.
pub(crate) fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

// The node loops index several parallel `[n × h]` buffers at stride `h` (θ, δ,
// v̂) by node id — a plain range loop is clearer than juggling zipped chunks.
#[allow(clippy::needless_range_loop)]
fn build_one_level(lvl: &PbLevelVelocity, h: usize, knn: usize) -> PbLineageLevel {
    let n = lvl.n_pb;
    let theta = &lvl.theta;
    let (velocity, has_vel) = unit_velocity(lvl, h);

    let k = knn.min(n.saturating_sub(1));
    let mut edges = Vec::new();
    for i in 0..n {
        if !has_vel[i] {
            continue; // no orientation → no outgoing edges
        }
        let ti = row(theta, i, h);
        let nbrs = knn_sorted(theta, i, n, h); // nearest by identity distance
        let vi = row(&velocity, i, h);
        for &(_, j) in nbrs.iter().take(k) {
            // Keep i → j only when j is velocity-forward of i.
            let tj = row(theta, j, h);
            let fwd: f32 = (0..h).map(|c| (tj[c] - ti[c]) * vi[c]).sum();
            if fwd > 0.0 {
                edges.push((i as u32, j as u32, 1.0f32));
            }
        }
    }

    PbLineageLevel {
        n_pb: n,
        edges,
        velocity,
    }
}

/// One collapse level as a **minimum spanning tree** over the θ geometry (Prim, O(n²) —
/// `n_pb` is small), each tree edge oriented by velocity into a DAG. A tree is acyclic
/// under any per-edge orientation, and the downstream root-anchoring
/// ([`crate::fit::lift::pb_trajectory`]) re-orients globally — so this enforces a sparse,
/// single-tree lineage (`n−1` edges) in place of the dense velocity-KNN.
#[allow(clippy::needless_range_loop)]
fn build_one_level_mst(lvl: &PbLevelVelocity, h: usize) -> PbLineageLevel {
    let n = lvl.n_pb;
    let theta = &lvl.theta;
    let (velocity, has_vel) = unit_velocity(lvl, h);
    if n < 2 {
        return PbLineageLevel {
            n_pb: n,
            edges: Vec::new(),
            velocity,
        };
    }

    // Prim's MST rooted at node 0, θ-distance weighted. `best_d[j]`/`best_p[j]` track the
    // cheapest edge connecting a not-yet-in-tree node `j` to the current tree.
    let mut in_tree = vec![false; n];
    let mut best_d = vec![f32::INFINITY; n];
    let mut best_p = vec![0usize; n];
    in_tree[0] = true;
    for j in 1..n {
        best_d[j] = dist2(row(theta, 0, h), row(theta, j, h));
    }
    let mut tree: Vec<(usize, usize)> = Vec::with_capacity(n - 1);
    for _ in 1..n {
        let (mut u, mut ud) = (usize::MAX, f32::INFINITY);
        for j in 0..n {
            if !in_tree[j] && best_d[j] < ud {
                ud = best_d[j];
                u = j;
            }
        }
        if u == usize::MAX {
            break; // no finite edge left (degenerate); leave the tree partial
        }
        in_tree[u] = true;
        tree.push((best_p[u], u));
        for j in 0..n {
            if !in_tree[j] {
                let d = dist2(row(theta, u, h), row(theta, j, h));
                if d < best_d[j] {
                    best_d[j] = d;
                    best_p[j] = u;
                }
            }
        }
    }

    // Orient each tree edge by velocity: from the velocity-bearing endpoint toward the
    // other. Neither endpoint has velocity ⇒ keep (a, b) — orientation is moot for
    // acyclicity, and root-anchoring fixes global direction downstream.
    let edges = tree
        .into_iter()
        .map(|(a, b)| {
            let (i, j) = orient_by_velocity(a, b, theta, &velocity, &has_vel, h);
            (i as u32, j as u32, 1.0f32)
        })
        .collect();

    PbLineageLevel {
        n_pb: n,
        edges,
        velocity,
    }
}

/// Orient an undirected edge `{a, b}` from the velocity-bearing endpoint toward the other
/// (`i → j` iff `⟨θ_j − θ_i, v̂_i⟩ ≥ 0`). Falls back to `(a, b)` when neither has velocity.
fn orient_by_velocity(
    a: usize,
    b: usize,
    theta: &[f32],
    velocity: &[f32],
    has_vel: &[bool],
    h: usize,
) -> (usize, usize) {
    let fwd = |i: usize, j: usize| -> f32 {
        let (ti, tj, vi) = (row(theta, i, h), row(theta, j, h), row(velocity, i, h));
        (0..h).map(|c| (tj[c] - ti[c]) * vi[c]).sum()
    };
    if has_vel[a] {
        if fwd(a, b) >= 0.0 {
            (a, b)
        } else {
            (b, a)
        }
    } else if has_vel[b] {
        if fwd(b, a) >= 0.0 {
            (b, a)
        } else {
            (a, b)
        }
    } else {
        (a, b)
    }
}

/// Weight `λ` on the θ-pseudotime DAG's SEM residual (the second lineage term). The
/// unspliced counts δ orients on are sparse, so a δ-only DAG drops every pb with no
/// usable velocity; this term orients the SAME θ-KNN by a root-anchored θ-pseudotime
/// (dense identity manifold) instead, keeping those pbs in the lineage.
pub const DEFAULT_THETA_SEM_WEIGHT: f32 = 0.1;

/// Build one θ-pseudotime DAG [`PbLineageLevel`] per collapse level (same order as
/// `pb_vel`). The velocity DAG (`vel_levels`) supplies only each level's ROOT (its
/// net velocity source, a single robust choice); everything else — topology,
/// orientation, drift — comes from θ. See [`build_theta_dag_level`].
pub fn build_theta_dag(
    pb_vel: &[PbLevelVelocity],
    vel_levels: &[PbLineageLevel],
    h: usize,
    knn: usize,
) -> Vec<PbLineageLevel> {
    pb_vel
        .iter()
        .zip(vel_levels)
        .map(|(lvl, vel)| build_theta_dag_level(&lvl.theta, lvl.n_pb, h, knn, vel))
        .collect()
}

/// θ-pseudotime DAG for one level: the **same θ-KNN topology** oriented by a
/// root-anchored θ-pseudotime `τ` (Dijkstra distance from the velocity source over
/// the symmetric θ-KNN) instead of by the sparse velocity. An edge `i → j` is kept
/// when `τ_j > τ_i` — defined wherever θ is, so no pb is dropped for lacking δ. The
/// `velocity` field is repurposed to the per-pb pseudotime-gradient direction `ĝ_i`
/// (unit mean of `θ_j − θ_i` over τ-forward neighbours), so a [`crate::training::PbSemTerm`]
/// drifts the embedding `ê_j → ê_i + s·ĝ_i` along the identity manifold's flow.
fn build_theta_dag_level(
    theta: &[f32],
    n: usize,
    h: usize,
    knn: usize,
    vel: &PbLineageLevel,
) -> PbLineageLevel {
    if n == 0 {
        return PbLineageLevel {
            n_pb: 0,
            edges: vec![],
            velocity: vec![],
        };
    }
    let k = knn.min(n.saturating_sub(1));
    // Symmetric θ-KNN adjacency, θ-distance edge weights (the manifold metric).
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for i in 0..n {
        for &(d2, j) in knn_sorted(theta, i, n, h).iter().take(k) {
            let w = d2.sqrt();
            adj[i].push((j, w));
            adj[j].push((i, w)); // symmetrize (a KNN edge is bidirectional here)
        }
    }
    // τ = θ-geodesic distance from the lineage root (velocity net-source).
    let tau = theta_pseudotime(&adj, velocity_source(vel, n), n);
    // Orient the θ-KNN by τ, and set ĝ_i = unit mean of (θ_j − θ_i) over τ-forward
    // neighbours (the local direction of increasing pseudotime).
    let mut velocity = vec![0f32; n * h];
    let mut edges = Vec::new();
    for i in 0..n {
        let ti = row(theta, i, h);
        let mut g = vec![0f32; h];
        let mut forward = 0u32;
        for &(j, _) in &adj[i] {
            if tau[j] > tau[i] {
                edges.push((i as u32, j as u32, 1.0f32));
                let tj = row(theta, j, h);
                for c in 0..h {
                    g[c] += tj[c] - ti[c];
                }
                forward += 1;
            }
        }
        if forward > 0 {
            let nrm = g.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for c in 0..h {
                velocity[i * h + c] = g[c] / nrm;
            }
        }
    }
    PbLineageLevel {
        n_pb: n,
        edges,
        velocity,
    }
}

/// Net velocity source of a directed level: the node with max `(out − in)` degree
/// (the lineage origin, `τ ≈ 0`); ties break to the lowest id, and an edge-less
/// graph falls back to node 0.
fn velocity_source(vel: &PbLineageLevel, n: usize) -> usize {
    let mut deg = vec![0i32; n];
    for &(i, j, _) in &vel.edges {
        deg[i as usize] += 1;
        deg[j as usize] -= 1;
    }
    (0..n).max_by_key(|&i| deg[i]).unwrap_or(0)
}

/// Dijkstra shortest-path distance from `root` over a symmetric weighted adjacency.
/// Unreachable nodes get `max finite distance + 1` so they stay finite (and sort
/// after reachable ones) for the `τ_j > τ_i` orientation test.
fn theta_pseudotime(adj: &[Vec<(usize, f32)>], root: usize, n: usize) -> Vec<f32> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;
    // Min-heap on distance (reverse the f32 order; ties by node id are irrelevant).
    struct St(f32, usize);
    impl PartialEq for St {
        fn eq(&self, o: &Self) -> bool {
            self.0 == o.0
        }
    }
    impl Eq for St {}
    impl PartialOrd for St {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for St {
        fn cmp(&self, o: &Self) -> Ordering {
            o.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
        }
    }
    let mut dist = vec![f32::INFINITY; n];
    dist[root] = 0.0;
    let mut pq = BinaryHeap::new();
    pq.push(St(0.0, root));
    while let Some(St(d, u)) = pq.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, w) in &adj[u] {
            let nd = d + w;
            if nd < dist[v] {
                dist[v] = nd;
                pq.push(St(nd, v));
            }
        }
    }
    let max_finite = dist
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(0.0f32, f32::max);
    for d in dist.iter_mut() {
        if !d.is_finite() {
            *d = max_finite + 1.0;
        }
    }
    dist
}

#[cfg(test)]
mod tests;
