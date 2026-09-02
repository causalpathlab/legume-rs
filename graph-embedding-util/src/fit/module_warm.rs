//! Warm start for the learned gene modules: a seeded k-means over the feature
//! profiles at the finest pseudobulk level.
//!
//! A membership learned from a cold start is rich-get-richer — the exact module
//! term rewards putting every feature in the module that already scores well — so
//! the partition starts from the data's own co-expression structure and is held
//! there for the first epochs (`FeatModules::set_frozen`). The profile is the
//! batch-corrected pseudobulk count matrix the fit already built, so this costs
//! nothing extra.

use log::info;
use matrix_util::principal_graph::kmeans_centroids_seeded;
use matrix_util::rand_util::name_seed;
use matrix_util::traits::SampleOps;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Widest profile k-means runs on directly; a pseudobulk axis longer than this is
/// sketched down to [`WARM_SKETCH_DIM`] with a seeded Gaussian first.
pub const WARM_PROFILE_MAX_DIM: usize = 1024;
pub const WARM_SKETCH_DIM: usize = 64;
/// Lloyd iterations for the warm-start clustering.
const WARM_KMEANS_ITER: usize = 30;

/// Cluster the rows of a `[features × pseudobulks]` count profile into `n_modules`
/// groups. Columns are depth-normalized, `log1p`-transformed, each row centred and
/// scaled to unit norm (so the clustering reads the SHAPE of a feature's profile,
/// not its abundance), then k-means with a seeded kmeans++ init. Returns one
/// module id per feature.
#[must_use]
pub fn warm_start_module_labels(profile: &DMatrix<f32>, n_modules: usize, seed: u64) -> Vec<u32> {
    let (d, s) = (profile.nrows(), profile.ncols());
    if d == 0 || n_modules < 2 {
        return vec![0u32; d];
    }
    // Column depth normalization to a common scale, then log1p.
    let col_tot: Vec<f32> = (0..s)
        .map(|j| profile.column(j).iter().sum::<f32>().max(1e-8))
        .collect();
    let mean_tot = col_tot.iter().sum::<f32>() / s.max(1) as f32;
    let mut z = DMatrix::<f32>::zeros(d, s);
    // nalgebra is column-major: fill by column.
    for j in 0..s {
        let scale = mean_tot / col_tot[j];
        for i in 0..d {
            z[(i, j)] = (profile[(i, j)] * scale).ln_1p();
        }
    }
    // Row centre + unit norm, in parallel over rows via a row-major scratch.
    let mut rows: Vec<Vec<f32>> = (0..d).map(|i| z.row(i).iter().copied().collect()).collect();
    rows.par_iter_mut().for_each(|r| {
        let mean = r.iter().sum::<f32>() / s.max(1) as f32;
        for x in r.iter_mut() {
            *x -= mean;
        }
        let norm = r.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in r.iter_mut() {
            *x /= norm;
        }
    });
    let mut z = DMatrix::<f32>::from_fn(d, s, |i, j| rows[i][j]);
    if s > WARM_PROFILE_MAX_DIM {
        let basis =
            DMatrix::<f32>::rnorm_seeded(s, WARM_SKETCH_DIM, name_seed(seed, "module_warm"));
        z = (&z * basis) / (WARM_SKETCH_DIM as f32).sqrt();
    }
    let (_, labels) = kmeans_centroids_seeded(&z, n_modules, WARM_KMEANS_ITER, seed);
    let mut sizes = vec![0usize; n_modules];
    for &l in &labels {
        sizes[l.min(n_modules - 1)] += 1;
    }
    info!(
        "module warm start: k-means over {d} feature profiles × {s} pseudobulk(s) → {n_modules} \
         modules, sizes min {} / median {} / max {}",
        sizes.iter().min().copied().unwrap_or(0),
        {
            let mut v = sizes.clone();
            v.sort_unstable();
            v[v.len() / 2]
        },
        sizes.iter().max().copied().unwrap_or(0),
    );
    labels.into_iter().map(|l| l as u32).collect()
}

#[cfg(test)]
mod tests;
