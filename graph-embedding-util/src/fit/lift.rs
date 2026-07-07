//! Phase-2 cell-lineage lift (M3): evaluation-only pseudotime + fate.
//!
//! Phase 1 shapes the shared dictionary through a directed pb-DAG (fixed M1 or
//! learnable M2); phase 2 analytically projects every cell onto that dictionary
//! (identity `θ_c` + velocity `δ_c`). This module lifts the pb-level structure to
//! cells **by evaluation only** — no extra training:
//!
//! 1. **pb pseudotime `τ_pb`** — velocity-integration least squares on the directed
//!    adjacency: every forward edge `i→j` asks `τ_j − τ_i ≈ step` (child one
//!    velocity-step ahead), solved as a ridge-regularized graph-Laplacian system.
//! 2. **pb fate `π_pb`** — absorption probabilities of an absorbing Markov chain on
//!    the forward-oriented adjacency; the sinks (no outgoing edge) are the fates.
//! 3. **cell lift** — landmark-assign each cell to the pb nodes by a softmax over
//!    `‖θ_c − θ_p‖²`, then blend the pb pseudotime/fate, adding a local
//!    along-velocity correction `⟨θ_c − θ_p, v̂_p⟩`.
//! 4. **QC** — `ambiguity_c` = normalized entropy of the landmark weights (a cell
//!    wedged between landmarks is flagged, never silently placed).
//!
//! Lives here (not in `faba`) so the pb integration + lift is unit-testable on
//! synthetic chains; the `faba` wrapper only turns the result into parquet.

use super::lineage::{dist2, row, unit_velocity};
use super::projection::PbLevelVelocity;
use nalgebra::{DMatrix, DVector};

/// Edge weights below this are treated as absent when building the directed graph
/// (the learnable `W` never reaches exactly zero; the fixed graph uses weight 1).
const EDGE_EPS: f32 = 1e-6;
/// Ridge added to the graph Laplacian's diagonal to fix its constant null space
/// (the pseudotime is identifiable only up to an additive constant otherwise).
const LAPLACIAN_RIDGE: f64 = 1e-3;
/// A source is kept as a root only if it reaches ≥ this fraction of the best source's
/// reach. A true root drains most of the graph; a single flipped branch manufactures a
/// spurious source reaching only that branch — this ratio drops it while keeping
/// genuine co-equal roots (distinct lineage origins each reaching a large subtree).
const ROOT_REACH_FRAC: f64 = 0.5;
/// Strong diagonal weight pinning each root to `τ = 0` in the Laplacian solve
/// (a Dirichlet anchor that removes the global additive-constant freedom).
const ROOT_ANCHOR: f64 = 1e3;

/// Directed pb trajectory derived from the oriented pb-DAG of one collapse level.
pub struct PbTrajectory {
    pub n_pb: usize,
    /// pb pseudotime `τ_pb`, min-max normalized to `[0, 1]`, length `n_pb`.
    pub tau: Vec<f32>,
    /// Root (source) pb node ids — where the velocity flow emanates (`τ ≈ 0`). A
    /// velocity-directed DAG can have several sources, so multiple roots are allowed
    /// (distinct lineage origins); spurious flip-sources are dropped by reach mass.
    pub roots: Vec<usize>,
    /// Top source's reachable fraction of the graph `[0, 1]` — the QC decisiveness score.
    pub decisiveness: f32,
    /// Terminal (sink) pb node ids — the fate basis (nodes with no forward edge out).
    pub terminals: Vec<usize>,
    /// Absorption distribution `[n_pb × n_terminals]` row-major: each pb node's fate
    /// over the terminals (a terminal is one-hot on itself).
    pub fate: Vec<f32>,
    /// τ-per-embedding-distance scale (mean `|Δτ| / ‖Δθ‖` over forward edges), used
    /// to put the per-cell along-velocity correction in the same units as `τ`.
    pub tau_scale: f32,
}

/// Unsupervised per-run **diagnostics + hygiene floor** for lineage runs (no ground
/// truth). An agent exploring random seeds reads this to reject broken runs
/// (`flag == "underfit"`) and inspect a run's structure. NOTE: these are *diagnostics*,
/// not a validated quality ranker — on the branching sim `root_decisiveness` correlated
/// with quality on one seed batch but not another (coarse values + pipeline
/// non-determinism), so it must not be trusted for fine seed selection. Reliable fine
/// selection of the genuine tail needs a marker-prior root, not these scores.
pub struct LineageQc {
    /// Functional-backbone top-source reach `[0, 1]` — a structural summary (how much of
    /// the backbone one source drains). A diagnostic, NOT a reliable ranker (see above).
    pub root_decisiveness: f32,
    /// Mean local velocity coherence (scVelo-style): how well each pb node's `v̂` agrees
    /// with its θ-neighbours'. Meaningful for M1; ≈0 for M2 (which collapses the δ readout).
    pub velocity_coherence: f32,
    /// Number of inferred roots (sources). `1`–few expected; many ⇒ fragmented field.
    pub n_roots: usize,
    /// Number of terminal fates. Should roughly match the number of lineages.
    pub n_terminals: usize,
    /// Mean per-cell landmark ambiguity, `[0, 1]` (lower = more confident placement).
    pub mean_ambiguity: f32,
    /// Final-epoch refine loss (NCE ≈ neg-log-lik) — a fit-hygiene signal ONLY. It is
    /// orientation-invariant, so it says nothing about trajectory quality.
    pub likelihood: f32,
    /// `"ok"` | `"underfit"` (no terminal structure / collapsed backbone — reject). This
    /// binary FLOOR is the reliable part; the other fields are diagnostics, not a ranker.
    pub flag: Box<str>,
}

/// Per-cell lineage lifted from a [`PbTrajectory`] (evaluation only).
pub struct CellLineage {
    /// Per-cell pseudotime `τ_c` in `[0, 1]`, length `n_cells`.
    pub tau: Vec<f32>,
    /// Per-cell landmark ambiguity = normalized entropy of the softmax weights in
    /// `[0, 1]` (0 = pinned to one pb landmark, 1 = uniform over all). QC signal.
    pub ambiguity: Vec<f32>,
    /// Per-cell fate `[n_cells × n_terminals]` row-major (blend of pb fates).
    pub fate: Vec<f32>,
    /// pb node ids of the terminals (fate columns), mirrored from [`PbTrajectory`].
    pub terminals: Vec<usize>,
    /// Collapse level the lift ran on (finest, by default).
    pub level: usize,
}

/// Directed edges `(i, j, w)` with `w > EDGE_EPS` from a dense `[n × n]` row-major
/// adjacency (M2 learnable `W`, forward-masked; negatives are dropped). The lift
/// only follows forward mass, so sub-`eps` and negative entries carry no edge.
pub fn dense_to_edges(w: &[f32], n: usize) -> Vec<(usize, usize, f32)> {
    let mut edges = Vec::new();
    for (i, rowi) in w.chunks_exact(n).enumerate() {
        for (j, &wij) in rowi.iter().enumerate() {
            if wij > EDGE_EPS {
                edges.push((i, j, wij));
            }
        }
    }
    edges
}

/// Boolean reachability: `reach[s*n + t]` is true when `t` is reachable from `s` over
/// the directed edges (`s` reaches itself). `n_pb` is small, so an O(n·(n+e)) BFS from
/// every node is fine.
fn reachability(n: usize, edges: &[(usize, usize, f32)]) -> Vec<bool> {
    let mut adj = vec![Vec::new(); n];
    for &(i, j, _) in edges {
        adj[i].push(j);
    }
    let mut reach = vec![false; n * n];
    for s in 0..n {
        reach[s * n + s] = true;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if !reach[s * n + v] {
                    reach[s * n + v] = true;
                    stack.push(v);
                }
            }
        }
    }
    reach
}

/// Infer the lineage roots: the **sources** of the velocity-directed graph (nodes not
/// reachable from outside their own strongly-connected component — where the flow
/// emanates). Multiple sources are kept (a DAG/forest can have several lineage
/// origins), but only those whose reachable mass is ≥ [`ROOT_REACH_FRAC`] of the best
/// source's — dropping the spurious sources a single flipped branch would create. One
/// representative per source SCC. Empty when there are no edges.
fn find_roots(n: usize, edges: &[(usize, usize, f32)]) -> Vec<usize> {
    if edges.is_empty() {
        return Vec::new();
    }
    let reach = reachability(n, edges);
    // `r` is a source iff every node that reaches `r` is also reachable from `r`
    // (i.e. nothing outside `r`'s SCC drains into it).
    let is_source = |r: usize| (0..n).all(|j| j == r || !reach[j * n + r] || reach[r * n + j]);
    let mass = |r: usize| (0..n).filter(|&k| reach[r * n + k]).count();

    // One representative per source SCC (skip SCC members already covered), ranked so
    // the reach threshold is taken against the strongest source.
    let mut sources: Vec<(usize, usize)> = Vec::new(); // (reach mass, rep node)
    let mut covered = vec![false; n];
    for r in 0..n {
        if covered[r] || !is_source(r) {
            continue;
        }
        for k in 0..n {
            if reach[r * n + k] && reach[k * n + r] {
                covered[k] = true; // same SCC
            }
        }
        sources.push((mass(r), r));
    }
    let best = sources.iter().map(|&(m, _)| m).max().unwrap_or(0);
    let thresh = (ROOT_REACH_FRAC * best as f64).ceil() as usize;
    sources
        .into_iter()
        .filter(|&(m, _)| m >= thresh.max(1))
        .map(|(_, r)| r)
        .collect()
}

/// QC **decisiveness** diagnostic: reduce the graph to its *functional* backbone (each
/// node keeps only its single strongest forward out-edge), then return the top source's
/// reachable fraction `[0, 1]`. The reduction makes this comparable across M1's sparse
/// velocity-KNN and M2's dense learned `W` (a dense graph trivially reaches everything,
/// so raw reach ≡ 1 is useless). A structural summary only — its correlation with true
/// quality is unstable across runs, so it is a diagnostic, not a reliable ranker.
fn functional_decisiveness(n: usize, edges: &[(usize, usize, f32)]) -> f32 {
    if edges.is_empty() {
        return 0.0;
    }
    // Strongest forward out-edge per source node → a functional (≤1-out) graph.
    let mut best_out: Vec<Option<(usize, f32)>> = vec![None; n];
    for &(i, j, w) in edges {
        if best_out[i].is_none_or(|(_, bw)| w > bw) {
            best_out[i] = Some((j, w));
        }
    }
    let backbone: Vec<(usize, usize, f32)> = best_out
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.map(|(j, w)| (i, j, w)))
        .collect();
    let reach = reachability(n, &backbone);
    let best = (0..n)
        .map(|s| (0..n).filter(|&k| reach[s * n + k]).count())
        .max()
        .unwrap_or(0);
    best as f32 / n.max(1) as f32
}

/// Mean local velocity coherence (scVelo-style "velocity confidence"): the average, over
/// pb nodes, of the mean cosine between a node's unit velocity `v̂` and its θ-KNN
/// neighbours' — an unsupervised measure of how smooth/coherent the velocity field is.
fn velocity_coherence(vel: &PbLevelVelocity, h: usize, knn: usize) -> f32 {
    let n = vel.n_pb;
    if n < 2 {
        return 0.0;
    }
    let (vhat, has_vel) = super::lineage::unit_velocity(vel, h);
    let k = knn.min(n - 1);
    let mut sum = 0f32;
    let mut cnt = 0usize;
    for i in 0..n {
        if !has_vel[i] {
            continue;
        }
        let ti = row(&vel.theta, i, h);
        let mut nbrs: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (dist2(ti, row(&vel.theta, j, h)), j))
            .collect();
        nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let vi = row(&vhat, i, h);
        let (mut acc, mut m) = (0f32, 0usize);
        for &(_, j) in nbrs.iter().take(k) {
            if has_vel[j] {
                let vj = row(&vhat, j, h);
                acc += (0..h).map(|c| vi[c] * vj[c]).sum::<f32>();
                m += 1;
            }
        }
        if m > 0 {
            sum += acc / m as f32;
            cnt += 1;
        }
    }
    if cnt > 0 {
        sum / cnt as f32
    } else {
        0.0
    }
}

/// Multi-source hop distance from the root set over the **undirected** edge set
/// (`u32::MAX` for nodes disconnected from every root). Each node's distance is to its
/// nearest root, so in a multi-root forest every subtree is measured from its own origin.
fn root_distance(n: usize, edges: &[(usize, usize, f32)], roots: &[usize]) -> Vec<u32> {
    let mut adj = vec![Vec::new(); n];
    for &(i, j, _) in edges {
        adj[i].push(j);
        adj[j].push(i);
    }
    let mut dist = vec![u32::MAX; n];
    let mut q = std::collections::VecDeque::new();
    for &r in roots {
        dist[r] = 0;
        q.push_back(r);
    }
    while let Some(u) = q.pop_front() {
        for &v in &adj[u] {
            if dist[v] == u32::MAX {
                dist[v] = dist[u] + 1;
                q.push_back(v);
            }
        }
    }
    dist
}

/// Re-orient every edge to point *away* from the roots — from the endpoint nearer a
/// root to the farther one. This overrides a locally-flipped velocity edge with the
/// globally-consistent root geometry (the actual fix for whole-subtree inversions);
/// equal-distance edges keep their original direction.
fn reorient_by_root(edges: &[(usize, usize, f32)], dist: &[u32]) -> Vec<(usize, usize, f32)> {
    edges
        .iter()
        .map(|&(i, j, w)| {
            if dist[j] >= dist[i] {
                (i, j, w)
            } else {
                (j, i, w)
            }
        })
        .collect()
}

/// Velocity-integration least squares: every forward edge `i→j` contributes
/// `w_ij (τ_j − τ_i − step)²`; minimizing gives the ridge-regularized graph-Laplacian
/// system `(L + εI) τ = g`. `L` is the weighted Laplacian, `g` the net forward push.
/// Each root is pinned to `τ ≈ 0` by a strong diagonal anchor ([`ROOT_ANCHOR`]),
/// removing the additive-constant freedom. Returns raw `τ`; `[0.0; n]` with no edges.
fn integrate_pseudotime(
    n: usize,
    edges: &[(usize, usize, f32)],
    step: f32,
    roots: &[usize],
) -> Vec<f64> {
    if edges.is_empty() {
        return vec![0.0; n];
    }
    let mut m = DMatrix::<f64>::zeros(n, n);
    let mut g = DVector::<f64>::zeros(n);
    for &(i, j, w) in edges {
        let w = f64::from(w);
        let s = f64::from(step);
        m[(i, i)] += w;
        m[(j, j)] += w;
        m[(i, j)] -= w;
        m[(j, i)] -= w;
        g[j] += w * s;
        g[i] -= w * s;
    }
    for k in 0..n {
        m[(k, k)] += LAPLACIAN_RIDGE;
    }
    for &r in roots {
        m[(r, r)] += ROOT_ANCHOR; // Dirichlet anchor τ_r → 0 (g[r] stays 0)
    }
    m.lu()
        .solve(&g)
        .map(|v| v.iter().copied().collect())
        .unwrap_or_else(|| vec![0.0; n])
}

/// Absorbing-Markov-chain fate. Sinks (no forward out-edge) are the absorbing
/// terminals; every transient node's fate is its absorption distribution
/// `B = (I − Q)^{-1} R`. Returns `(terminals, fate[n × k])` row-major, terminals
/// one-hot on themselves. Empty terminals ⇒ `(vec![], vec![])`.
fn absorbing_fate(n: usize, edges: &[(usize, usize, f32)]) -> (Vec<usize>, Vec<f32>) {
    // Forward out-weight per node; a node with none is a sink (terminal).
    let mut out_w = vec![0f64; n];
    let mut adj = DMatrix::<f64>::zeros(n, n);
    for &(i, j, w) in edges {
        let w = f64::from(w);
        adj[(i, j)] += w;
        out_w[i] += w;
    }
    let terminals: Vec<usize> = (0..n).filter(|&i| out_w[i] < f64::from(EDGE_EPS)).collect();
    let k = terminals.len();
    if k == 0 || k == n {
        // No structure to absorb into (or everything is a sink): no fate.
        return (Vec::new(), Vec::new());
    }
    let term_col: std::collections::HashMap<usize, usize> =
        terminals.iter().enumerate().map(|(c, &t)| (t, c)).collect();
    let transient: Vec<usize> = (0..n).filter(|i| !term_col.contains_key(i)).collect();
    let t_row: std::collections::HashMap<usize, usize> =
        transient.iter().enumerate().map(|(r, &i)| (i, r)).collect();
    let nt = transient.len();

    // Row-normalized transition blocks among transient (Q) and into terminals (R).
    let mut q = DMatrix::<f64>::zeros(nt, nt);
    let mut r = DMatrix::<f64>::zeros(nt, k);
    for (ri, &i) in transient.iter().enumerate() {
        let denom = out_w[i];
        if denom <= 0.0 {
            continue;
        }
        for j in 0..n {
            let p = adj[(i, j)] / denom;
            if p == 0.0 {
                continue;
            }
            if let Some(&col) = term_col.get(&j) {
                r[(ri, col)] += p;
            } else if let Some(&rj) = t_row.get(&j) {
                q[(ri, rj)] += p;
            }
        }
    }

    // B = (I − Q)^{-1} R. A tiny ridge guards residual cycles (M2 acyclicity is
    // soft, so Q may not be strictly substochastic on every row).
    let mut im_q = DMatrix::<f64>::identity(nt, nt) - &q;
    for d in 0..nt {
        im_q[(d, d)] += LAPLACIAN_RIDGE;
    }
    let b = match im_q.lu().solve(&r) {
        Some(b) => b,
        None => return (terminals, Vec::new()),
    };

    let mut fate = vec![0f32; n * k];
    for (&t, &col) in &term_col {
        fate[t * k + col] = 1.0; // terminal → one-hot on itself
    }
    for (ri, &i) in transient.iter().enumerate() {
        for col in 0..k {
            fate[i * k + col] = b[(ri, col)] as f32;
        }
    }
    (terminals, fate)
}

/// Build the [`PbTrajectory`] for one level from its velocity readout and directed
/// edges (dense `W` → [`dense_to_edges`] for M2, fixed-lineage edges for M1).
pub fn pb_trajectory(
    vel: &PbLevelVelocity,
    edges: &[(usize, usize, f32)],
    h: usize,
    step: f32,
) -> PbTrajectory {
    let n = vel.n_pb;

    // Root-anchoring stage: infer the source(s) of the velocity-directed graph, then
    // re-orient every edge to point away from the nearest root. This overrides
    // locally-flipped δ with the globally-consistent root geometry, so a whole subtree
    // can no longer integrate backward. All downstream steps use the re-oriented edges.
    let roots = find_roots(n, edges);
    let dist = root_distance(n, edges, &roots);
    let oriented = reorient_by_root(edges, &dist);
    let decisiveness = functional_decisiveness(n, &oriented);

    let raw = integrate_pseudotime(n, &oriented, step, &roots);

    // Min-max normalize τ to [0, 1] (roots sit at ≈ the minimum by construction).
    let (lo, hi) = raw
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| {
            (l.min(x), h.max(x))
        });
    let span = (hi - lo).max(1e-12);
    let tau: Vec<f32> = raw.iter().map(|&x| ((x - lo) / span) as f32).collect();

    // τ-per-distance scale: mean |Δτ| / ‖Δθ‖ over the re-oriented edges (1 when none).
    let mut num = 0f64;
    let mut den = 0f64;
    for &(i, j, _) in &oriented {
        let dt = (f64::from(tau[j]) - f64::from(tau[i])).abs();
        let ti = row(&vel.theta, i, h);
        let tj = row(&vel.theta, j, h);
        let dl = ti
            .iter()
            .zip(tj)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum::<f64>()
            .sqrt();
        if dl > 1e-8 {
            num += dt;
            den += dl;
        }
    }
    let tau_scale = if den > 0.0 { (num / den) as f32 } else { 1.0 };

    let (terminals, fate) = absorbing_fate(n, &oriented);
    PbTrajectory {
        n_pb: n,
        tau,
        roots,
        decisiveness,
        terminals,
        fate,
        tau_scale,
    }
}

/// Assemble the unsupervised [`LineageQc`] diagnostics for one run from its trajectory,
/// velocity readout, cell lift, and final refine loss. Bakes the binary `underfit` floor
/// in; the remaining fields are diagnostics (not a validated ranker). `knn` = lineage KNN.
pub fn compute_lineage_qc(
    traj: &PbTrajectory,
    vel: &PbLevelVelocity,
    lin: &CellLineage,
    likelihood: f32,
    h: usize,
    knn: usize,
) -> LineageQc {
    let coherence = velocity_coherence(vel, h, knn);
    let mean_ambiguity = if lin.ambiguity.is_empty() {
        0.0
    } else {
        lin.ambiguity.iter().sum::<f32>() / lin.ambiguity.len() as f32
    };
    // Binary FLOOR only — reject a run with no terminal structure or a collapsed
    // backbone (decisiveness ≈ 0). NOT gated on coherence (M2 collapses δ while its
    // `W`-trajectory is great). Everything else is `ok`. The other fields are diagnostics
    // — decisiveness is NOT a reliable fine ranker (coarse + pipeline non-determinism).
    let flag = if traj.terminals.is_empty() || traj.decisiveness < 0.12 {
        "underfit"
    } else {
        "ok"
    };
    LineageQc {
        root_decisiveness: traj.decisiveness,
        velocity_coherence: coherence,
        n_roots: traj.roots.len(),
        n_terminals: traj.terminals.len(),
        mean_ambiguity,
        likelihood,
        flag: flag.into(),
    }
}

/// Lift the pb trajectory to cells: landmark-assign each cell `θ_c` to the pb nodes
/// `θ_p` by a softmax over `‖θ_c − θ_p‖²`, then blend `τ_pb` (plus a local
/// along-velocity correction) and `π_pb`. Evaluation only — no training. Cells are
/// independent, so the per-cell work is rayon-parallel (like the phase-2 projection
/// it follows).
///
/// `theta_c` is the phase-2 per-cell identity `[n_cells × h]` row-major (raw `θ`).
// Loops index several parallel `[n_pb × h]` / `[n_pb × k]` / `[· × k]` buffers at
// stride `h`/`k` (θ_pb, v̂_p, τ_pb, fate) by node/terminal id — a plain range loop
// reads clearer than juggling zipped chunks in lockstep.
#[allow(clippy::needless_range_loop)]
pub fn lift_cells(
    theta_c: &[f32],
    n_cells: usize,
    vel: &PbLevelVelocity,
    traj: &PbTrajectory,
    h: usize,
    level: usize,
) -> CellLineage {
    use rayon::prelude::*;

    let n_pb = vel.n_pb;
    let k = traj.terminals.len();
    let (vhat, _) = unit_velocity(vel, h); // per-node unit v̂ (zero rows where ‖δ‖≈0)

    // Softmax bandwidth: mean nearest-landmark squared distance across cells (a
    // scale that adapts to how tightly the pb landmarks tile the latent space).
    let near_sum: f64 = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let tc = row(theta_c, c, h);
            let best = (0..n_pb)
                .map(|p| dist2(tc, row(&vel.theta, p, h)))
                .fold(f32::INFINITY, f32::min);
            if best.is_finite() {
                f64::from(best)
            } else {
                0.0
            }
        })
        .sum();
    let bandwidth = ((near_sum / n_cells.max(1) as f64) as f32).max(1e-6);
    let ln_pb = (n_pb as f32).max(2.0).ln();

    // One parallel pass per cell → (τ_c, ambiguity_c, fate_row). For each landmark the
    // difference `θ_c − θ_p` is walked ONCE, yielding both the softmax distance `d²`
    // and the along-velocity projection `⟨θ_c − θ_p, v̂_p⟩`.
    let per_cell: Vec<(f32, f32, Vec<f32>)> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let tc = row(theta_c, c, h);
            let mut d2 = vec![0f32; n_pb];
            let mut proj = vec![0f32; n_pb];
            for p in 0..n_pb {
                let tp = row(&vel.theta, p, h);
                let vp = row(&vhat, p, h);
                let (mut dd, mut pj) = (0f32, 0f32);
                for d in 0..h {
                    let diff = tc[d] - tp[d];
                    dd += diff * diff;
                    pj += diff * vp[d];
                }
                d2[p] = dd;
                proj[p] = pj;
            }
            // Softmax over −d²/bandwidth (shift by the min for numerical stability).
            let dmin = d2.iter().copied().fold(f32::INFINITY, f32::min);
            let mut w = vec![0f32; n_pb];
            let mut wsum = 0f32;
            for p in 0..n_pb {
                let e = (-(d2[p] - dmin) / bandwidth).exp();
                w[p] = e;
                wsum += e;
            }
            let mut fate_row = vec![0f32; k];
            if wsum <= 0.0 {
                return (0.0, 0.0, fate_row);
            }
            let (mut tau_c, mut ent) = (0f32, 0f32);
            for p in 0..n_pb {
                let m = w[p] / wsum;
                if m > 1e-12 {
                    ent -= m * m.ln();
                }
                tau_c += m * (traj.tau[p] + proj[p] * traj.tau_scale);
                for col in 0..k {
                    fate_row[col] += m * traj.fate[p * k + col];
                }
            }
            (
                tau_c.clamp(0.0, 1.0),
                (ent / ln_pb).clamp(0.0, 1.0),
                fate_row,
            )
        })
        .collect();

    let mut tau = vec![0f32; n_cells];
    let mut ambiguity = vec![0f32; n_cells];
    let mut fate = vec![0f32; n_cells * k];
    for (c, (t, a, fr)) in per_cell.into_iter().enumerate() {
        tau[c] = t;
        ambiguity[c] = a;
        if k > 0 {
            fate[c * k..c * k + k].copy_from_slice(&fr);
        }
    }

    CellLineage {
        tau,
        ambiguity,
        fate,
        terminals: traj.terminals.clone(),
        level,
    }
}

#[cfg(test)]
mod tests;
