//! Warm start for the learned gene modules: a seeded k-means over the feature
//! profiles at the finest pseudobulk level.
//!
//! A membership learned from a cold start is rich-get-richer — the exact module
//! term rewards putting every feature in the module that already scores well — so
//! the partition starts from the data's own co-expression structure and is held
//! there for the first epochs (`FeatModules::set_frozen`). The profile is the
//! batch-corrected pseudobulk count matrix the fit already built, so this costs
//! nothing extra.

use super::config::ParentModulesOwned;
use log::info;
use matrix_util::principal_graph::kmeans_centroids_seeded;
use matrix_util::rand_util::name_seed;
use matrix_util::traits::SampleOps;
use nalgebra::DMatrix;

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
    let unit = crate::transfer::unit_log_profile_rows(profile);
    let mut z = DMatrix::<f32>::from_fn(d, s, |i, j| unit[i][j]);
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

/// Membership logits `[D × M]` for a fit warm-started from a parent: a matched
/// feature takes the parent's membership row verbatim (a simplex point, which
/// sparsemax reproduces exactly), an unmatched one is initialized through the
/// parent's modules from its nearest matched neighbours by profile, or the
/// parent's module-average membership below the similarity floor.
#[must_use]
pub fn parent_module_logits(parent: &ParentModulesOwned, profiles: &DMatrix<f32>) -> DMatrix<f32> {
    use crate::transfer::{align_gene_axis, AlignInputs, ModuleTables};
    let d = parent.row_to_parent.len();
    let m = parent.pi.ncols();
    let al = align_gene_axis(&AlignInputs {
        rho: &parent.rho,
        b_feat: None,
        modules: Some(ModuleTables {
            pi: &parent.pi,
            mu: &parent.mu,
        }),
        new_to_train: &parent.row_to_parent,
        profiles_new: Some(profiles),
        knobs: parent.knobs,
    });
    let membership = al
        .membership
        .as_ref()
        .expect("module tables given ⇒ membership present");
    let mut logits = DMatrix::<f32>::zeros(d, m);
    let (mut n_init, mut n_diffuse) = (0usize, 0usize);
    for g in 0..d {
        let union = al.new_to_union[g].expect("every feature is matched or initialized");
        logits.set_row(g, &membership.row(union));
        if !al.is_scored(union) {
            n_init += 1;
            if al.provenance[union].as_ref().is_some_and(|p| p.diffuse) {
                n_diffuse += 1;
            }
        }
    }
    info!(
        "module warm start from a parent: {} features carry the parent's membership, \
         {n_init} initialized through its modules ({n_diffuse} on the diffuse prior)",
        d - n_init
    );
    logits
}

#[cfg(test)]
mod tests;
