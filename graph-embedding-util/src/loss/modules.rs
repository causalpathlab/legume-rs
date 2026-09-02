//! The module-level terms of the composite objective, for a model whose feature
//! side is [`crate::model::FeatModules`].
//!
//! Two levels. The EXACT term ([`module_softmax_loss`]) scores every module
//! against a unit's module-pooled counts with a full softmax over `M` — no
//! sampling, so every module receives gradient on every step, which is what
//! lets a feature that is never drawn inherit a trained row. The within-module
//! NCE reuses the ordinary sampled-softmax with negatives drawn from the
//! positive's own module ([`ModulePools`]); with one module per feature `μ`
//! cancels there and it trains only the residual, the feature bias and the cell
//! side. Together: `log p(g|c) = log p(m|c) + log p(g|m,c)`.
//!
//! Gene dropout ([`masked_membership`]) hides a random subset of features when
//! the counts are pooled — the module keeps its expected count through the
//! survivors — so `μ` is fit under exactly the perturbation a later dataset with a
//! different gene panel applies.

use candle_util::candle_core::{Device, Result, Tensor};
use candle_util::candle_nn::ops::log_softmax;
use rand::{Rng, RngExt};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use rayon::prelude::*;

/// Floor on a pooled or marginal mass before it divides or is logged.
const EPS: f64 = 1e-6;

///////////////////////////////////
// Dropout and the pooled counts //
///////////////////////////////////

/// A `[D, 1]` survivor mask with each feature kept with probability `1 − p_drop`,
/// drawn from the caller's RNG so it only touches the stream when modules are on.
pub fn draw_gene_keep_mask(
    n_features: usize,
    p_drop: f32,
    rng: &mut impl Rng,
    dev: &Device,
) -> Result<Tensor> {
    let keep: Vec<f32> = (0..n_features)
        .map(|_| {
            if rng.random::<f32>() < p_drop {
                0.0
            } else {
                1.0
            }
        })
        .collect();
    Tensor::from_vec(keep, (n_features, 1), dev)
}

/// `π̃ = keep ⊙ π`, rescaled per MODULE so each column keeps the mass it had
/// before the mask: a dropped feature contributes exactly zero, and the survivors
/// of its modules are scaled up to stand in for it. The scale is detached — it is
/// a normalizer, not a parameter — so the gradient into `π` is the plain masked
/// one. A module with no survivor stays at zero.
pub fn masked_membership(pi: &Tensor, keep: &Tensor) -> Result<Tensor> {
    let kept = pi.broadcast_mul(keep)?; // [D, M]
    let full_mass = pi.sum_keepdim(0)?.detach(); // [1, M]
    let kept_mass = kept.sum_keepdim(0)?.detach().clamp(EPS, f64::INFINITY)?;
    let scale = full_mass.div(&kept_mass)?;
    kept.broadcast_mul(&scale)
}

/// Module-pooled counts `x̃ = X · π̃`, `[U, M]`. One matmul, with autograd into
/// the membership.
pub fn pool_module_counts(x_dense: &Tensor, pi_masked: &Tensor) -> Result<Tensor> {
    x_dense.matmul(pi_masked)
}

/// Scatter sparse count rows into one dense `[U, D]` block on `dev`. Rows are
/// `(features, counts)` slices aligned per unit; filled with rayon, one unit per
/// task.
pub fn dense_count_block(
    rows: &[(&[u32], &[f32])],
    n_features: usize,
    dev: &Device,
) -> Result<Tensor> {
    let u = rows.len();
    let mut flat = vec![0f32; u * n_features];
    flat.par_chunks_mut(n_features)
        .zip(rows.par_iter())
        .for_each(|(dst, (feats, counts))| {
            for (&f, &c) in feats.iter().zip(counts.iter()) {
                dst[f as usize] += c;
            }
        });
    Tensor::from_vec(flat, (u, n_features), dev)
}

////////////////////
// The exact term //
////////////////////

/// The exact cell–module term for one block of units:
///
/// ```text
///   s_um   = e_u · μ_m + b_m
///   ℓ_u    = − Σ_m ( x̃_um / Σ_m' x̃_um' ) · log_softmax_m(s_u)_m
///   L      = mean_u ℓ_u
/// ```
///
/// Per-unit normalization makes each unit one unit of weight, the same as the NCE
/// gives it (a mean over positives drawn per unit), so the two levels are
/// commensurate at `λ = 1`. The cell bias cancels in the softmax and is not
/// scored. A unit with no counts on the surviving features contributes zero.
pub fn module_softmax_loss(
    e_units: &Tensor,
    mu: &Tensor,
    b_module: &Tensor,
    x_cm: &Tensor,
) -> Result<Tensor> {
    let s = e_units.matmul(&mu.t()?)?.broadcast_add(b_module)?; // [U, M]
    let log_q = log_softmax(&s, 1)?;
    let total = x_cm.sum_keepdim(1)?; // [U, 1]
    let p = x_cm.broadcast_div(&total.clamp(EPS, f64::INFINITY)?)?;
    p.mul(&log_q)?.sum(1)?.neg()?.mean(0)
}

////////////
// Priors //
////////////

/// Load-balance prior `KL(π̄ ‖ Uniform_M) = Σ_m π̄_m log(M π̄_m)` on the feature-
/// marginal occupancy `π̄_m = mean_g π_gm`. Zero when every module carries the same
/// share of the features; largest when one module holds them all — which is the
/// direction the exact term rewards on its own. A uniform target, not a decaying
/// ladder: for modules a decaying prior is precisely the collapse direction.
pub fn module_balance_prior(pi: &Tensor) -> Result<Tensor> {
    let m = pi.dim(1)? as f64;
    let occ = pi.mean(0)?; // [M]
    let occ_safe = occ.clamp(EPS, f64::INFINITY)?;
    occ.mul(&occ_safe.affine(m, 0.0)?.log()?)?.sum_all()
}

/// Mean row entropy `mean_g −Σ_m π_gm log π_gm`. Exact zeros contribute nothing.
pub fn module_row_entropy(pi: &Tensor) -> Result<Tensor> {
    let safe = pi.clamp(EPS, f64::INFINITY)?;
    pi.mul(&safe.log()?)?.sum(1)?.neg()?.mean(0)
}

/////////////////////////////////
// Within-module negative pools //
/////////////////////////////////

/// Host-side view of the membership for the within-module negative draw, rebuilt
/// once per epoch from the current `π` (the pools lag the membership by at most an
/// epoch, the same granularity as the gate mask). One per sampler, because each
/// sampler has its own expressed-feature pool.
pub struct ModulePools {
    /// Per feature: the nonzero `(module, weight)` entries of its membership row.
    /// Shared by every sampler's pools; only `members` differs.
    rows: std::sync::Arc<Vec<Vec<(u32, f32)>>>,
    /// Per module: the features with nonzero membership in it, restricted to this
    /// sampler's feature pool.
    members: Vec<Vec<u32>>,
}

/// The per-feature membership rows, built once per refresh and shared across the
/// samplers' pools.
pub fn membership_rows_host(
    pi_host: &[f32],
    n_features: usize,
    n_modules: usize,
) -> std::sync::Arc<Vec<Vec<(u32, f32)>>> {
    assert_eq!(pi_host.len(), n_features * n_modules, "membership shape");
    let rows: Vec<Vec<(u32, f32)>> = pi_host
        .par_chunks(n_modules)
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(_, &w)| w > 0.0)
                .map(|(m, &w)| (m as u32, w))
                .collect()
        })
        .collect();
    std::sync::Arc::new(rows)
}

impl ModulePools {
    /// Build one sampler's pools from the shared rows and its feature pool.
    #[must_use]
    pub fn build(
        rows: std::sync::Arc<Vec<Vec<(u32, f32)>>>,
        n_modules: usize,
        feature_pool: &[u32],
    ) -> Self {
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_modules];
        for &f in feature_pool {
            for &(m, _) in &rows[f as usize] {
                members[m as usize].push(f);
            }
        }
        Self { rows, members }
    }

    /// Per-module member counts, for the diagnostics.
    #[must_use]
    pub fn member_counts(&self) -> Vec<usize> {
        self.members.iter().map(Vec::len).collect()
    }

    /// Push `k` negatives for the positive `feat`: pick one of its modules with
    /// probability ∝ its membership weight, then draw uniformly from that module's
    /// members. Returns `false` — pushing nothing — when the chosen module has
    /// fewer than two members in this pool (a lone feature has nothing to contrast
    /// with), so the caller falls back to the global draw.
    pub fn draw_negatives(
        &self,
        feat: u32,
        k: usize,
        out: &mut Vec<u32>,
        rng: &mut impl Rng,
    ) -> bool {
        let row = &self.rows[feat as usize];
        if row.is_empty() {
            return false;
        }
        let m = if row.len() == 1 {
            row[0].0
        } else {
            let w: Vec<f32> = row.iter().map(|&(_, w)| w).collect();
            let Ok(pick) = WeightedIndex::new(w) else {
                return false;
            };
            row[pick.sample(rng)].0
        };
        let pool = &self.members[m as usize];
        if pool.len() < 2 {
            return false;
        }
        for _ in 0..k {
            out.push(pool[rng.random_range(0..pool.len())]);
        }
        true
    }
}

/////////////////
// Diagnostics //
/////////////////

/// Per-epoch membership summary, from the host copy of `π`.
pub struct MembershipDiagnostics {
    /// `max_m π̄_m / (1/M)`: 1 is balanced, `M` is total collapse.
    pub max_occupancy_ratio: f32,
    /// Modules whose argmax count is below `small_floor`.
    pub n_small_modules: usize,
    /// `mean_g H(π_g)` in nats.
    pub mean_row_entropy: f32,
    /// Mean number of modules a feature sits on with nonzero weight.
    pub mean_row_support: f32,
}

#[must_use]
pub fn membership_diagnostics(
    pi_host: &[f32],
    n_features: usize,
    n_modules: usize,
    small_floor: usize,
) -> MembershipDiagnostics {
    let mut occ = vec![0f64; n_modules];
    let mut argmax_count = vec![0usize; n_modules];
    let mut ent = 0f64;
    let mut support = 0usize;
    for row in pi_host.chunks(n_modules) {
        let mut best = 0usize;
        for (m, &w) in row.iter().enumerate() {
            occ[m] += f64::from(w);
            if w > 0.0 {
                support += 1;
                ent -= f64::from(w) * f64::from(w).ln();
            }
            if w > row[best] {
                best = m;
            }
        }
        argmax_count[best] += 1;
    }
    let d = n_features.max(1) as f64;
    let max_occ = occ.iter().copied().fold(0f64, f64::max) / d;
    MembershipDiagnostics {
        max_occupancy_ratio: (max_occ * n_modules as f64) as f32,
        n_small_modules: argmax_count.iter().filter(|&&c| c < small_floor).count(),
        mean_row_entropy: (ent / d) as f32,
        mean_row_support: (support as f64 / d) as f32,
    }
}

#[cfg(test)]
mod tests;
