//! Principal-component scores from the randomized SVD, plus the 2D
//! coordinates a UMAP-style layout starts its SGD from.
//!
//! One representation, two consumers. A kNN graph built on the PC scores
//! rather than on the raw latent, and an SGD init that is the leading two
//! of those scores rather than a random draw — which is what the reference
//! implementations do: scanpy's `pp.neighbors` runs on `X_pca`, and
//! uwot/UMAP default to a spectral init with PCA as the documented cheap
//! stand-in, falling back to a uniform draw only when neither is available.
//!
//! Pair with [`crate::knn_graph::KnnGraph::from_rows`] and
//! [`crate::umap::Umap::fit`].

use crate::traits::RandomizedAlgs;
use nalgebra::DMatrix;
use rand::{rngs::SmallRng, RngExt, SeedableRng};

type Mat = DMatrix<f32>;

/// UMAP reference implementations init in `[-10, 10]`, and the SGD clamps
/// every per-edge gradient step at ±4 — so an init spread far from this
/// scale either takes hundreds of epochs to expand out of (too tight) or
/// gets bulldozed a clamp at a time (too wide).
pub const INIT_SCALE: f32 = 10.0;

/// Relative jitter added to the PC init, as a fraction of [`INIT_SCALE`]
/// (uwot's `scale_and_jitter` uses the same 1e-4).
const JITTER_FRAC: f32 = 1e-4;

/// Principal-component scores of `data` (rows = points, columns = features)
/// from a randomized SVD, with the `skip` leading components dropped.
///
/// Returns `[n × k]`, `k ≤ rank`: column `c` is `u[·, skip+c] · s[skip+c]`,
/// the coordinate of each row on component `skip + c`. `k` falls short of
/// `rank` only when the matrix has fewer than `rank + skip` directions to
/// give.
///
/// **No centering pass runs first, and `skip = 1` is why.** On the
/// nonnegative latents this workspace lays out — topic proportions, the
/// unit-normalized simplex rows of a cosine geometry — every row loads
/// positively on the first right singular vector, so that vector *is* the
/// mean profile and the first score measures how average a point is rather
/// than how it differs from the others. Dropping it removes the mean the
/// way centering would, without materializing a centered copy. Pass
/// `skip = 0` for data that is already centered: there the first component
/// is real structure, usually the most of it.
pub fn pc_scores(data: &Mat, rank: usize, skip: usize) -> anyhow::Result<Mat> {
    let n = data.nrows();
    let want = (rank + skip).min(n).min(data.ncols());
    if rank == 0 || want <= skip {
        return Err(anyhow::anyhow!(
            "no components left: {n} × {} data, rank={rank}, skip={skip}",
            data.ncols()
        ));
    }
    let (u, s, _v) = data.rsvd(want)?;
    let keep = u.ncols().saturating_sub(skip).min(rank);
    if keep == 0 {
        return Err(anyhow::anyhow!(
            "randomized SVD returned {} component(s), skip={skip}",
            u.ncols()
        ));
    }
    let mut out = Mat::zeros(n, keep);
    for c in 0..keep {
        out.set_column(c, &(u.column(skip + c) * s[skip + c]));
    }
    Ok(out)
}

/// Row-major `n × 2` SGD init from PC scores: columns 0 and 1 of `scores`,
/// rescaled so the largest `|coordinate|` is [`INIT_SCALE`], plus a jitter
/// of `JITTER_FRAC · INIT_SCALE`.
///
/// The jitter is not cosmetic. Two points with identical scores sit at
/// exactly zero distance, where UMAP's attractive gradient is zero and the
/// repulsive one is a fixed constant along a zero vector — the pair can
/// never separate. Falls back to [`random_init_2d`] when `scores` has fewer
/// than two columns or no spread at all.
#[must_use]
pub fn init_2d_from_scores(scores: &Mat, seed: u64) -> Vec<f32> {
    let n = scores.nrows();
    if scores.ncols() < 2 {
        return random_init_2d(n, seed);
    }
    let max_abs = scores
        .columns(0, 2)
        .iter()
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    if max_abs <= 0.0 || !max_abs.is_finite() {
        return random_init_2d(n, seed);
    }
    let scale = INIT_SCALE / max_abs;

    let mut rng = SmallRng::seed_from_u64(seed ^ 0x5151_5151_5151_5151);
    let jitter = INIT_SCALE * JITTER_FRAC;
    let mut init = Vec::with_capacity(n * 2);
    for i in 0..n {
        for c in 0..2 {
            init.push(scores[(i, c)] * scale + rng.random_range(-jitter..jitter));
        }
    }
    init
}

/// Seeded uniform draw over `[-INIT_SCALE, INIT_SCALE]²`, row-major `n × 2`
/// — the init to fall back on when there is no usable PC structure.
#[must_use]
pub fn random_init_2d(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n * 2)
        .map(|_| rng.random_range(-INIT_SCALE..INIT_SCALE))
        .collect()
}

/// [`pc_scores`] and [`init_2d_from_scores`] in one call, with the fallback
/// policy in one place: when the SVD fails or leaves fewer than two usable
/// components, the returned scores are `None` and the init is a
/// [`random_init_2d`] draw.
///
/// A caller that also wants its kNN graph on the PCs reads the `None` as
/// "build it on the raw features instead", so the graph and the init never
/// disagree about which space they are in.
#[must_use]
pub fn pc_layout_init(data: &Mat, rank: usize, skip: usize, seed: u64) -> (Option<Mat>, Vec<f32>) {
    match pc_scores(data, rank, skip) {
        Ok(scores) if scores.ncols() >= 2 => {
            let init = init_2d_from_scores(&scores, seed);
            (Some(scores), init)
        }
        Ok(_) | Err(_) => (None, random_init_2d(data.nrows(), seed)),
    }
}

#[cfg(test)]
mod tests;
