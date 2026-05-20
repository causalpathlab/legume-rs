//! Archetypal analysis (Cutler & Breiman, 1994) over the rows of a
//! dense matrix, plus a k-sweep with elbow selection.
//!
//! Given `Z [N, H]` (N points as rows in an H-dim space) we seek K
//! archetypes `α [K, H]` and weights `θ [N, K]` with each `θ_n` on the
//! simplex `Δ^{K-1}`, minimizing `‖Z − θ α‖²_F`. Archetypes are kept
//! **inside the convex hull** of the data: each `α_k` is maintained as a
//! running convex combination of data rows via Frank–Wolfe, so no
//! `[K, N]` coefficient matrix is ever materialized.
//!
//! Both phases are embarrassingly parallel over points:
//! - **E-step** — each `θ_n` is an independent simplex-constrained
//!   least-squares solved by Frank–Wolfe (parallel `map` over rows).
//! - **M-step** — the archetype gradient is `G = θᵀZ − (θᵀθ) α`
//!   (two Gram terms, a parallel reduction); each archetype's
//!   Frank–Wolfe vertex is `argmax_n (G_k · z_n)` (a parallel reduction).
//!
//! Used by `senna bge --resolve-etm` to recover the ETM topic-side
//! embedding α (= archetypes) and per-cell proportions θ from a frozen
//! bipartite-graph cell embedding, with no encoder/decoder training.

use crate::clustering::{Kmeans, KmeansArgs};
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::borrow::Cow;

/// Hyperparameters for a single archetypal fit.
#[derive(Clone, Debug)]
pub struct AaArgs {
    /// Number of archetypes K.
    pub k: usize,
    /// Outer alternating (E/M) iterations.
    pub max_iter: usize,
    /// Frank–Wolfe inner steps per per-point / per-archetype subproblem.
    pub fw_iters: usize,
    /// Relative-RSS change below which the outer loop stops early.
    pub tol: f32,
    /// RNG seed (empty-cluster fallback during init, subsampling).
    pub seed: u64,
    /// Optional cap on rows used to *fit* archetypes; θ is still assigned
    /// for every row at the end. `None` fits on all rows.
    pub subsample: Option<usize>,
}

impl Default for AaArgs {
    fn default() -> Self {
        Self {
            k: 10,
            max_iter: 50,
            fw_iters: 30,
            tol: 1e-4,
            seed: 42,
            subsample: None,
        }
    }
}

/// Result of an archetypal fit.
pub struct AaResult {
    /// Archetypes `α [K, H]` (= ETM topic embeddings).
    pub alpha: DMatrix<f32>,
    /// Per-row simplex weights `θ [N, K]` (= per-cell topic proportions).
    pub theta: DMatrix<f32>,
    /// Reconstruction `‖Z − θ α‖²_F` over all N rows.
    pub rss: f32,
}

/// Run archetypal analysis on the rows of `z [N, H]`.
///
/// Archetypes are fit on `z` (optionally subsampled per [`AaArgs::subsample`]);
/// the returned θ is then assigned for every row of the full `z`.
pub fn archetypal_analysis(z: &DMatrix<f32>, args: &AaArgs) -> AaResult {
    let fit = subsample_rows(z, args.subsample, args.seed);
    let alpha = fit_archetypes(&fit, args).0;
    let theta = assign_theta(z, &alpha, args.fw_iters);
    let rss = reconstruction_rss(z, &theta, &alpha);
    AaResult { alpha, theta, rss }
}

/// Sweep `k_range`, fit archetypes for each K on a single shared (optionally
/// subsampled) fit set, pick the RSS elbow, then return a full fit at that K.
///
/// Mirrors the BIC/elbow K-selection pattern in `cnv::kmeans_init::select_kmeans_k`.
/// Returns `(chosen_k, full_result)`.
pub fn select_archetype_k(z: &DMatrix<f32>, k_range: &[usize], args: &AaArgs) -> (usize, AaResult) {
    assert!(!k_range.is_empty(), "select_archetype_k: empty k_range");
    let fit = subsample_rows(z, args.subsample, args.seed);

    // Fit each K once on the shared fit set, keeping its archetypes (α is
    // only [K, H] — cheap to retain) so the chosen K isn't refit afterward.
    let mut fits: Vec<(f32, DMatrix<f32>)> = k_range
        .iter()
        .map(|&k| {
            let (alpha, rss) = fit_archetypes(&fit, &AaArgs { k, ..args.clone() });
            log::info!("archetypal K-sweep: k={k} fit-RSS={rss:.4}");
            (rss, alpha)
        })
        .collect();

    let rss_fit: Vec<f32> = fits.iter().map(|(r, _)| *r).collect();
    let bi = elbow_index(k_range, &rss_fit);
    log::info!("archetypal K-sweep selected k={}", k_range[bi]);

    // Reuse the archetypes already fit for the chosen K; only θ needs full z.
    let (_, alpha) = fits.swap_remove(bi);
    let theta = assign_theta(z, &alpha, args.fw_iters);
    let rss = reconstruction_rss(z, &theta, &alpha);
    (k_range[bi], AaResult { alpha, theta, rss })
}

/////////////////////////////////////////////////////////////////////
// Core fit                                                          //
/////////////////////////////////////////////////////////////////////

/// Alternating E/M fit on `fit [M, H]`. Returns `(alpha [K, H], fit_rss)`,
/// where `fit_rss` is the reconstruction error over the fit rows only.
fn fit_archetypes(fit: &DMatrix<f32>, args: &AaArgs) -> (DMatrix<f32>, f32) {
    let m = fit.nrows();
    let k = args.k.min(m.max(1));

    let mut alpha = init_archetypes(fit, k, args.seed);
    let mut prev_rss = f32::INFINITY;
    let mut rss = f32::INFINITY;

    for it in 0..args.max_iter {
        // E-step: per-row simplex weights against current archetypes.
        let theta = assign_theta(fit, &alpha, args.fw_iters);

        // M-step gradient terms: G = θᵀZ − (θᵀθ) α  (two Gram products).
        let gtx = theta.tr_mul(fit); // [K, H]
        let gtg = theta.tr_mul(&theta); // [K, K]
        let g = &gtx - &gtg * &alpha; // [K, H]

        // Frank–Wolfe vertex per archetype: the data row most aligned with
        // its residual-gradient direction (keeps α_k inside the hull).
        let vertices = best_vertices(fit, &g, k);
        for kk in 0..k {
            let z_star = fit.row(vertices[kk]).transpose(); // [H]
            let d = &z_star - alpha.row(kk).transpose(); // [H]
            let denom = d.norm_squared() * gtg[(kk, kk)];
            if denom <= f32::EPSILON {
                continue;
            }
            // Closed-form line search for the quadratic along d:
            // f(γ) = const − 2γ d·G_k + γ² ‖d‖² (θᵀθ)_kk.
            let gamma = (d.dot(&g.row(kk).transpose()) / denom).clamp(0.0, 1.0);
            let new_row = alpha.row(kk).transpose() + gamma * d;
            alpha.row_mut(kk).copy_from(&new_row.transpose());
        }

        rss = reconstruction_rss(fit, &theta, &alpha);
        let rel = (prev_rss - rss).abs() / prev_rss.max(f32::EPSILON);
        log::debug!("archetypal fit k={k} iter={it} rss={rss:.4} rel={rel:.2e}");
        if rel < args.tol {
            break;
        }
        prev_rss = rss;
    }
    (alpha, rss)
}

/// k-means centroids as the initial archetypes. Empty clusters fall back
/// to a random data row.
fn init_archetypes(z: &DMatrix<f32>, k: usize, seed: u64) -> DMatrix<f32> {
    let (m, h) = (z.nrows(), z.ncols());
    let membership = z.kmeans_rows(KmeansArgs {
        num_clusters: k,
        max_iter: 100,
    });
    let mut alpha = DMatrix::<f32>::zeros(k, h);
    let mut counts = vec![0usize; k];
    for (i, &c) in membership.iter().enumerate() {
        let c = c.min(k - 1);
        let sum = alpha.row(c).transpose() + z.row(i).transpose();
        alpha.row_mut(c).copy_from(&sum.transpose());
        counts[c] += 1;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    use rand::RngExt;
    for (c, &count) in counts.iter().enumerate() {
        if count > 0 {
            let avg = alpha.row(c) / count as f32;
            alpha.row_mut(c).copy_from(&avg);
        } else {
            let r = rng.random_range(0..m);
            alpha.row_mut(c).copy_from(&z.row(r));
        }
    }
    alpha
}

/////////////////////////////////////////////////////////////////////
// Sub-solvers                                                       //
/////////////////////////////////////////////////////////////////////

/// E-step: assign simplex weights `θ [N, K]` for every row of `z` against
/// fixed archetypes `α [K, H]`. Parallel over rows; each row is written
/// once, so no shared-mutable state is needed.
fn assign_theta(z: &DMatrix<f32>, alpha: &DMatrix<f32>, fw_iters: usize) -> DMatrix<f32> {
    let n = z.nrows();
    let k = alpha.nrows();
    let rows: Vec<DVector<f32>> = (0..n)
        .into_par_iter()
        .map(|i| simplex_lsq(alpha, &z.row(i).transpose(), fw_iters))
        .collect();
    let mut theta = DMatrix::<f32>::zeros(n, k);
    for (i, row) in rows.into_iter().enumerate() {
        theta.row_mut(i).copy_from(&row.transpose());
    }
    theta
}

/// Solve `min_{a ∈ Δ^{K-1}} ‖x − αᵀ a‖²` by Frank–Wolfe.
fn simplex_lsq(alpha: &DMatrix<f32>, x: &DVector<f32>, fw_iters: usize) -> DVector<f32> {
    let k = alpha.nrows();
    let mut a = DVector::<f32>::from_element(k, 1.0 / k as f32);
    for _ in 0..fw_iters {
        let pred = alpha.tr_mul(&a); // αᵀ a   [H]
        let r = x - &pred; // residual   [H]
        let grad = alpha * &r * (-2.0); // ∂/∂a   [K]
                                        // Linear-minimization oracle over the simplex: the vertex e_j with
                                        // the smallest gradient component.
        let j = argmin(&grad);
        let mut d = -&a;
        d[j] += 1.0; // d = e_j − a
        let ad = alpha.tr_mul(&d); // αᵀ d   [H]
        let denom = ad.norm_squared();
        if denom <= f32::EPSILON {
            break;
        }
        let gamma = (r.dot(&ad) / denom).clamp(0.0, 1.0);
        a += gamma * d;
    }
    a
}

/// For each archetype k, the fit-row index maximizing `G_k · z_n`
/// (Frank–Wolfe vertex). Parallel reduction over rows; per-archetype
/// running max, no `[K, N]` matrix.
fn best_vertices(z: &DMatrix<f32>, g: &DMatrix<f32>, k: usize) -> Vec<usize> {
    let init = || (vec![f32::NEG_INFINITY; k], vec![0usize; k]);
    let (_, idx) = (0..z.nrows())
        .into_par_iter()
        .fold(init, |(mut mv, mut mi), n| {
            let gv = g * z.row(n).transpose(); // [K]
            for kk in 0..k {
                if gv[kk] > mv[kk] {
                    mv[kk] = gv[kk];
                    mi[kk] = n;
                }
            }
            (mv, mi)
        })
        .reduce(init, |(mut amv, mut ami), (bmv, bmi)| {
            for kk in 0..k {
                if bmv[kk] > amv[kk] {
                    amv[kk] = bmv[kk];
                    ami[kk] = bmi[kk];
                }
            }
            (amv, ami)
        });
    idx
}

/////////////////////////////////////////////////////////////////////
// Helpers                                                           //
/////////////////////////////////////////////////////////////////////

/// `‖Z − θ α‖²_F`.
fn reconstruction_rss(z: &DMatrix<f32>, theta: &DMatrix<f32>, alpha: &DMatrix<f32>) -> f32 {
    let pred = theta * alpha;
    (z - pred).norm_squared()
}

fn argmin(v: &DVector<f32>) -> usize {
    let mut best = 0;
    let mut best_v = f32::INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x < best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Take a deterministic random subset of `cap` rows when `cap < N`;
/// otherwise borrow all rows (no copy).
fn subsample_rows(z: &DMatrix<f32>, cap: Option<usize>, seed: u64) -> Cow<'_, DMatrix<f32>> {
    let n = z.nrows();
    match cap {
        Some(m) if m < n => {
            let mut rng = StdRng::seed_from_u64(seed);
            let idx = rand::seq::index::sample(&mut rng, n, m).into_vec();
            let mut out = DMatrix::<f32>::zeros(m, z.ncols());
            for (r, &i) in idx.iter().enumerate() {
                out.row_mut(r).copy_from(&z.row(i));
            }
            Cow::Owned(out)
        }
        _ => Cow::Borrowed(z),
    }
}

/// Index of the K with the largest perpendicular distance from the chord
/// joining the first and last point of the (axis-normalized) RSS curve —
/// the classic "elbow". Falls back to the first K for <3 points.
fn elbow_index(ks: &[usize], rss: &[f32]) -> usize {
    let n = ks.len();
    if n < 3 {
        return 0;
    }
    let kf: Vec<f32> = ks.iter().map(|&k| k as f32).collect();
    let (kmin, kmax) = (kf[0], kf[n - 1]);
    let (rmin, rmax) = rss
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &r| {
            (lo.min(r), hi.max(r))
        });
    let kspan = (kmax - kmin).max(f32::EPSILON);
    let rspan = (rmax - rmin).max(f32::EPSILON);
    let xs: Vec<f32> = kf.iter().map(|&k| (k - kmin) / kspan).collect();
    let ys: Vec<f32> = rss.iter().map(|&r| (r - rmin) / rspan).collect();

    let (x0, y0) = (xs[0], ys[0]);
    let (x1, y1) = (xs[n - 1], ys[n - 1]);
    let den = ((y1 - y0).powi(2) + (x1 - x0).powi(2))
        .sqrt()
        .max(f32::EPSILON);

    let mut best = 0;
    let mut best_d = f32::NEG_INFINITY;
    for i in 0..n {
        let num = ((y1 - y0) * xs[i] - (x1 - x0) * ys[i] + x1 * y0 - y1 * x0).abs();
        let d = num / den;
        if d > best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    /// Build `Z = θ A` with θ drawn on the simplex and A a set of planted
    /// archetypes well separated in H-space.
    fn planted(n: usize, k: usize, h: usize, seed: u64) -> (DMatrix<f32>, DMatrix<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        // Archetypes far apart: scaled standard basis-ish points.
        let mut a = DMatrix::<f32>::zeros(k, h);
        for kk in 0..k {
            for hh in 0..h {
                a[(kk, hh)] = if hh % k == kk {
                    5.0
                } else {
                    rng.random_range(-0.2..0.2)
                };
            }
        }
        let mut z = DMatrix::<f32>::zeros(n, h);
        for i in 0..n {
            // First K points are pure (θ = e_i) so the planted archetypes are
            // genuine vertices of the data hull (otherwise hull-constrained AA
            // cannot — and should not — reach them). The rest are Dirichlet-ish
            // interior mixtures.
            let theta = if i < k {
                let mut e = DVector::<f32>::zeros(k);
                e[i] = 1.0;
                e
            } else {
                let mut w: Vec<f32> = (0..k)
                    .map(|_| -rng.random_range(0.0f32..1.0).ln())
                    .collect();
                let s: f32 = w.iter().sum();
                w.iter_mut().for_each(|x| *x /= s);
                DVector::from_vec(w)
            };
            let row = a.tr_mul(&theta); // [H]
            z.row_mut(i).copy_from(&row.transpose());
        }
        (z, a)
    }

    #[test]
    fn recovers_planted_archetypes() {
        let (k, h, n) = (4, 8, 600);
        let (z, a_true) = planted(n, k, h, 1);
        let res = archetypal_analysis(
            &z,
            &AaArgs {
                k,
                max_iter: 100,
                fw_iters: 50,
                tol: 1e-6,
                seed: 7,
                subsample: None,
            },
        );
        // Each true archetype must be matched by some recovered archetype.
        for kt in 0..k {
            let mut best = f32::INFINITY;
            for kr in 0..k {
                let d = (a_true.row(kt) - res.alpha.row(kr)).norm();
                best = best.min(d);
            }
            assert!(
                best < 0.75,
                "true archetype {kt} unmatched (min dist {best})"
            );
        }
    }

    #[test]
    fn theta_rows_are_simplex() {
        let (k, h, n) = (3, 6, 300);
        let (z, _) = planted(n, k, h, 2);
        let res = archetypal_analysis(
            &z,
            &AaArgs {
                k,
                ..Default::default()
            },
        );
        for i in 0..n {
            let s: f32 = res.theta.row(i).sum();
            assert!((s - 1.0).abs() < 1e-3, "row {i} sums to {s}");
            assert!(
                res.theta.row(i).iter().all(|&x| x >= -1e-5),
                "row {i} negative"
            );
        }
    }

    #[test]
    fn sweep_picks_planted_k() {
        let (k, h, n) = (4, 8, 500);
        let (z, _) = planted(n, k, h, 3);
        let krange: Vec<usize> = (2..=8).collect();
        let (best, _) = select_archetype_k(
            &z,
            &krange,
            &AaArgs {
                fw_iters: 40,
                ..Default::default()
            },
        );
        assert!(
            (3..=5).contains(&best),
            "elbow picked k={best}, expected ~4"
        );
    }
}
