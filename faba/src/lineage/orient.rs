//! Per-edge velocity direction over the centroid candidate graph.
//!
//! `faba gem` emits a per-cell velocity increment δ. For each candidate edge this module
//! projects the adjacent cells' δ onto the edge axis and tests whether the mean flow is
//! significantly directed ([`edge_directionality`]), yielding a `Forward`/`Reverse`/`Abstain`
//! call per edge — the weights the max-weight branching turns into a rooted forest.
//! [`aggregate_node_velocity`] is kept for the per-node mean-velocity output.

use faba::hypothesis_tests::{benjamini_hochberg, bootstrap_mean_ci, mean, sign_flip_pvalue};
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashSet;

/// Mean velocity per node: average δ over cells whose cluster (nearest centroid)
/// is `k`. Returns a `K × D` matrix; nodes with no cells stay zero.
pub fn aggregate_node_velocity(
    velocity: &DMatrix<f32>,
    cluster: &[usize],
    k: usize,
) -> DMatrix<f32> {
    let d = velocity.ncols();
    let mut v = DMatrix::<f32>::zeros(k, d);
    let mut counts = vec![0usize; k];
    for (i, &c) in cluster.iter().enumerate() {
        if c < k {
            for j in 0..d {
                v[(c, j)] += velocity[(i, j)];
            }
            counts[c] += 1;
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..d {
                v[(c, j)] /= counts[c] as f32;
            }
        }
    }
    v
}

////////////////////////////////////////////////////////////////////////////////
// Per-edge directionality: a statistically-tested velocity direction per edge  //
// of a geometric candidate graph, so directions δ can't support are abstained. //
////////////////////////////////////////////////////////////////////////////////

/// The velocity call for one candidate edge `(a, b)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeCall {
    /// Velocity confidently flows `a → b`.
    Forward,
    /// Velocity confidently flows `b → a`.
    Reverse,
    /// Direction not statistically supported (weak, ambiguous, or too few cells).
    Abstain,
}

/// A candidate edge with its tested velocity direction. `edge = (a, b)` is stored with
/// `a < b`; `flux > 0` means `a → b`. Statistics are `NaN` when the edge auto-abstained
/// (fewer than `min_cells` cells or a degenerate axis).
#[derive(Clone, Debug)]
pub struct EdgeDirection {
    pub edge: (usize, usize),
    /// Euclidean distance between the two centroids.
    pub geom_dist: f32,
    /// Mean per-cell projected velocity onto the `a → b` axis (signed).
    pub flux: f32,
    /// Bootstrap standard error of `flux`.
    pub se: f32,
    pub ci_lo: f32,
    pub ci_hi: f32,
    /// Sign-flip permutation p-value (H0: mean projection = 0).
    pub p: f32,
    /// BH-adjusted `p` across all candidate edges.
    pub q: f32,
    pub n_cells: usize,
    pub call: EdgeCall,
    /// `1 − q` for a call, `0` when abstained.
    pub confidence: f32,
    /// Whether this candidate edge is in the geometric MST.
    pub in_mst: bool,
}

/// Tuning for [`edge_directionality`].
#[derive(Clone, Debug)]
pub struct EdgeDirectionConfig {
    /// Cell bootstrap resamples (SE/CI).
    pub n_boot: usize,
    /// Sign-flip permutation draws.
    pub n_perm: usize,
    /// q cutoff and CI level (the abstain bar).
    pub alpha: f64,
    /// Minimum cells on an edge before it can be called.
    pub min_cells: usize,
    pub seed: u64,
}

/// Squared Euclidean distance between rows `a` and `b` of `m`.
pub(crate) fn row_sqdist(m: &DMatrix<f32>, a: usize, b: usize) -> f32 {
    let mut s = 0f32;
    for c in 0..m.ncols() {
        let d = m[(a, c)] - m[(b, c)];
        s += d * d;
    }
    s
}

/// Canonical undirected-edge key `(min, max)` — the sort order every `dirs` lookup uses.
pub(crate) fn undirected(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

impl EdgeDirection {
    /// An abstained edge (direction not asserted): all stats `NaN`, confidence 0.
    pub(crate) fn abstain(
        edge: (usize, usize),
        geom_dist: f32,
        flux: f32,
        n_cells: usize,
        in_mst: bool,
    ) -> Self {
        EdgeDirection {
            edge,
            geom_dist,
            flux,
            se: f32::NAN,
            ci_lo: f32::NAN,
            ci_hi: f32::NAN,
            p: f32::NAN,
            q: f32::NAN,
            n_cells,
            call: EdgeCall::Abstain,
            confidence: 0.0,
            in_mst,
        }
    }
}

/// Geometry-only directions for every MST edge (all abstained). Used when velocity is absent
/// or `--no-edge-direction`, so the max-weight branching reduces to the geometric MST.
pub fn mst_only_directions(
    centroids: &DMatrix<f32>,
    mst_edges: &[(usize, usize)],
) -> Vec<EdgeDirection> {
    mst_edges
        .iter()
        .map(|&(a, b)| {
            let edge = undirected(a, b);
            let geom_dist = row_sqdist(centroids, edge.0, edge.1).max(0.0).sqrt();
            EdgeDirection::abstain(edge, geom_dist, f32::NAN, 0, true)
        })
        .collect()
}

/// Candidate edge set = the MST ∪ each node's `k_cand` nearest centroids, deduped and
/// stored as `(min, max)`. The non-MST candidates are the alternative parents that let
/// the branching *rewire* rather than only cut.
pub fn candidate_edges(
    centroids: &DMatrix<f32>,
    mst_edges: &[(usize, usize)],
    k_cand: usize,
) -> Vec<(usize, usize)> {
    let k = centroids.nrows();
    let mut set: HashSet<(usize, usize)> = HashSet::new();
    for &(a, b) in mst_edges {
        set.insert(undirected(a, b));
    }
    for a in 0..k {
        let mut d: Vec<(f32, usize)> = (0..k)
            .filter(|&b| b != a)
            .map(|b| (row_sqdist(centroids, a, b), b))
            .collect();
        d.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(_, b) in d.iter().take(k_cand) {
            set.insert(undirected(a, b));
        }
    }
    let mut edges: Vec<(usize, usize)> = set.into_iter().collect();
    edges.sort_unstable();
    edges
}

/// Test the velocity direction of every candidate edge. For edge `(a, b)` the sample is
/// the per-cell projected velocity `g_i = δ_i · û` over cells assigned (by `labels`) to
/// node `a` or `b`; a cell bootstrap gives the SE/CI and a sign-flip null gives the
/// p-value. Calls are BH-adjusted across edges and thresholded at `alpha` (with a CI that
/// must clear 0). Runs one edge per rayon task.
pub fn edge_directionality(
    centroids: &DMatrix<f32>,
    velocity: &DMatrix<f32>,
    labels: &[usize],
    cand_edges: &[(usize, usize)],
    mst_edges: &[(usize, usize)],
    cfg: &EdgeDirectionConfig,
) -> Vec<EdgeDirection> {
    let k = centroids.nrows();
    let d = centroids.ncols();
    let mst_set: HashSet<(usize, usize)> =
        mst_edges.iter().map(|&(a, b)| undirected(a, b)).collect();

    // Cells per node, once.
    let mut node_cells: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in labels.iter().enumerate() {
        if c < k {
            node_cells[c].push(i);
        }
    }

    // Per-edge statistic (p computed here; q filled in the serial BH pass below).
    let mut dirs: Vec<EdgeDirection> = cand_edges
        .par_iter()
        .enumerate()
        .map(|(ei, &(a, b))| {
            let (a, b) = undirected(a, b);
            let geom_dist = row_sqdist(centroids, a, b).max(0.0).sqrt();
            let in_mst = mst_set.contains(&(a, b));

            // Unit axis a → b.
            let mut axis = vec![0f32; d];
            let mut nrm = 0f32;
            for c in 0..d {
                let u = centroids[(b, c)] - centroids[(a, c)];
                axis[c] = u;
                nrm += u * u;
            }
            nrm = nrm.sqrt();

            let abstain =
                |flux: f32, n: usize| EdgeDirection::abstain((a, b), geom_dist, flux, n, in_mst);

            // Projected velocities of the edge's cells.
            let mut g: Vec<f32> = Vec::new();
            if nrm > 0.0 {
                for &cell in node_cells[a].iter().chain(node_cells[b].iter()) {
                    let mut proj = 0f32;
                    for c in 0..d {
                        proj += velocity[(cell, c)] * axis[c];
                    }
                    g.push(proj / nrm);
                }
            }
            let n = g.len();
            if nrm == 0.0 || n < cfg.min_cells {
                return abstain(if n > 0 { mean(&g) } else { f32::NAN }, n);
            }

            // Direction of the mean projection: bootstrap CI/SE, then a sign-flip null.
            let gbar = mean(&g);
            let mut rng = SmallRng::seed_from_u64(cfg.seed ^ (ei as u64).wrapping_mul(0x9E3779B9));
            let (se, ci_lo, ci_hi) = bootstrap_mean_ci(&g, cfg.n_boot, cfg.alpha, &mut rng);
            let p = sign_flip_pvalue(&g, cfg.n_perm, &mut rng);

            EdgeDirection {
                edge: (a, b),
                geom_dist,
                flux: gbar,
                se,
                ci_lo,
                ci_hi,
                p,
                q: f32::NAN, // filled below
                n_cells: n,
                call: EdgeCall::Abstain, // decided below
                confidence: 0.0,
                in_mst,
            }
        })
        .collect();

    // BH across the testable edges (finite p); auto-abstained edges keep q = NaN.
    let testable: Vec<usize> = (0..dirs.len()).filter(|&i| dirs[i].p.is_finite()).collect();
    let ps: Vec<f32> = testable.iter().map(|&i| dirs[i].p).collect();
    let qs = benjamini_hochberg(&ps);
    for (slot, &i) in testable.iter().enumerate() {
        let q = qs[slot];
        let ed = &mut dirs[i];
        ed.q = q;
        let ci_clears_zero = ed.ci_lo > 0.0 || ed.ci_hi < 0.0;
        if (q as f64) <= cfg.alpha && ci_clears_zero {
            ed.call = if ed.flux > 0.0 {
                EdgeCall::Forward
            } else {
                EdgeCall::Reverse
            };
            ed.confidence = 1.0 - q;
        }
    }
    dirs
}

#[cfg(test)]
mod tests;
