//! Fixed pseudobulk lineage structure for the velocity-drift SEM residual (M1).
//!
//! Given the analytic pb-level velocity readout ([`PbLevelVelocity`]: identity
//! `θ_pb` + velocity `δ_pb` per pb node per collapse level), build a **directed**
//! neighbour graph over the pb nodes of each level, oriented by velocity: an edge
//! `i → j` is kept when `j` is velocity-forward of `i` (`⟨θ_j − θ_i, δ_i⟩ > 0`).
//! This fixed structure feeds the velocity-drift SEM residual
//! `Σ_{i→j} w_ij ‖ê_j − ê_i − s·v̂_i‖²` added to phase-1 training (see
//! `crate::training::PbSemTerm`). M2 replaces this fixed graph with a learnable
//! DAGMA `W`; the residual form is unchanged.

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

impl PbLineageLevel {
    /// Dense `[n_pb × n_pb]` row-major adjacency: `[i·n + j] = w` for edge `(i→j)`,
    /// `0` elsewhere. Warm-starts the learnable pb-DAG `W` (M2) from this fixed
    /// velocity-oriented structure so SGD refines from a correctly-oriented start
    /// instead of zeros. The `(i, j)` layout matches `PbDagTerm`'s forward mask.
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        let n = self.n_pb;
        let mut w = vec![0f32; n * n];
        for &(i, j, wt) in &self.edges {
            w[i as usize * n + j as usize] = wt;
        }
        w
    }
}

/// Build one [`PbLineageLevel`] per collapse level (same order as `pb_vel`).
pub fn build_pb_lineage(pb_vel: &[PbLevelVelocity], h: usize, knn: usize) -> Vec<PbLineageLevel> {
    pb_vel
        .iter()
        .map(|lvl| build_one_level(lvl, h, knn))
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

#[cfg(test)]
mod tests;
