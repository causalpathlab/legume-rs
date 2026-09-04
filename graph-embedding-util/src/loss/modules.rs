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
//!
//! [`module_step_loss`], [`module_priors`] and [`log_membership_diagnostics`] are
//! the pieces every trainer with a module model shares (geu's composite trainer,
//! `pinto cage`), so the objective is written once.

use crate::model::FeatModules;
use candle_util::candle_core::{DType, Device, Result, Tensor};
use candle_util::candle_nn::ops::log_softmax;
use candle_util::sgvb::l2_normalize_dim;
use log::info;
use rand::{Rng, RngExt};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Floor on a pooled or marginal mass before it divides or is logged.
const EPS: f64 = 1e-6;

/// A module with fewer argmax features than this is reported as small.
pub const MODULE_SMALL_FLOOR: usize = 5;

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
/// of its modules are scaled up to stand in for it. Detached throughout: the only
/// consumer is the exact term's target, which never trains the membership. A
/// module with no survivor stays at zero.
pub fn masked_membership(pi: &Tensor, keep: &Tensor) -> Result<Tensor> {
    let pi = pi.detach();
    let kept = pi.broadcast_mul(keep)?; // [D, M]
    let full_mass = pi.sum_keepdim(0)?; // [1, M]
    let kept_mass = kept.sum_keepdim(0)?.clamp(EPS, f64::INFINITY)?;
    let scale = full_mass.div(&kept_mass)?;
    kept.broadcast_mul(&scale)
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
///
/// The target `x̃` is DETACHED: this term trains `μ`, `b` and the cell side
/// given the partition, and never the partition itself. Letting the gradient
/// reach `π` through the target turns the term into a cross-entropy against a
/// target it may reshape, and the cheapest reshaping is to put every expressed
/// feature into one module — the target is then the same one-hot for every unit
/// and the module bias fits it at zero loss. Measured on real marrow data: every
/// marker gene at weight 1.0 in one module, with the identity pushed into the
/// residual. The membership learns from the within-module NCE, which is
/// discriminative and indifferent to a merge, and from the priors.
pub fn module_softmax_loss(
    e_units: &Tensor,
    mu: &Tensor,
    b_module: &Tensor,
    x_cm: &Tensor,
) -> Result<Tensor> {
    let s = e_units.matmul(&mu.t()?)?.broadcast_add(b_module)?; // [U, M]
    let log_q = log_softmax(&s, 1)?;
    let x_cm = x_cm.detach();
    let total = x_cm.sum_keepdim(1)?; // [U, 1]
    let p = x_cm.broadcast_div(&total.clamp(EPS, f64::INFINITY)?)?;
    p.mul(&log_q)?.sum(1)?.neg()?.mean(0)
}

/// One trainer step of the exact term: pool the units' dense count block
/// `[U, D]` through the (dropout-masked, detached) membership `[D, M]`, score
/// every module against the units' embeddings `[U, H]`, and weight by
/// `lambda_module`. The one definition both geu's composite trainer and
/// `pinto cage` call.
pub fn module_step_loss(
    modules: &FeatModules,
    pi_masked: &Tensor,
    x_dense: &Tensor,
    e_units: &Tensor,
    lambda_module: f32,
) -> Result<Tensor> {
    let x_cm = x_dense.matmul(pi_masked)?; // [U, M]
    module_softmax_loss(e_units, &modules.mu, &modules.b_module, &x_cm)?
        .affine(f64::from(lambda_module), 0.0)
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

/// The membership priors for one optimizer step, or `None` while the membership
/// is frozen (it is detached then, and they would be constants).
pub fn module_priors(
    modules: &FeatModules,
    pi: &Tensor,
    lambda_balance: f32,
) -> Result<Option<Tensor>> {
    if modules.is_frozen() || lambda_balance <= 0.0 {
        return Ok(None);
    }
    Ok(Some(
        module_balance_prior(pi)?.affine(f64::from(lambda_balance), 0.0)?,
    ))
}

/////////////////////////////////
// Within-module negative pools //
/////////////////////////////////

/// Host-side view of the membership for the within-module negative draw, rebuilt
/// once per epoch from the current `π` (the pools lag the membership by at most an
/// epoch, the same granularity as the gate mask). One per sampler, because each
/// sampler has its own expressed-feature pool.
pub struct ModulePools {
    /// Per feature: the `(module, cumulative weight)` entries of its membership
    /// row ABOVE the uniform level `1/M`, so a draw is one uniform number and a
    /// scan. Shared by every sampler's pools; only `members` differs.
    rows: std::sync::Arc<Vec<Vec<(u32, f32)>>>,
    /// Per module: the features with above-uniform membership in it, restricted
    /// to this sampler's feature pool.
    members: Vec<Vec<u32>>,
    /// Positives whose module was too small in this pool to contrast within, so
    /// the caller drew globally. Read and reset by the diagnostics.
    fallbacks: AtomicUsize,
}

/// The per-feature membership rows for the pools, built once per refresh and
/// shared across the samplers' pools.
///
/// Only entries ABOVE the uniform level `1/M` count as membership here. The
/// warm start deliberately keeps every module in every row's sparsemax support
/// at a sliver of mass, so taking every nonzero would put every feature in every
/// module: a "within-module" draw that is the global draw at `M ×` the memory.
pub fn membership_rows_host(
    pi_host: &[f32],
    n_features: usize,
    n_modules: usize,
) -> std::sync::Arc<Vec<Vec<(u32, f32)>>> {
    assert_eq!(pi_host.len(), n_features * n_modules, "membership shape");
    let floor = 1.0 / n_modules as f32;
    let rows: Vec<Vec<(u32, f32)>> = pi_host
        .par_chunks(n_modules)
        .map(|row| {
            let mut cum = 0f32;
            row.iter()
                .enumerate()
                .filter(|(_, &w)| w > floor)
                .map(|(m, &w)| {
                    cum += w;
                    (m as u32, cum)
                })
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
        Self {
            rows,
            members,
            fallbacks: AtomicUsize::new(0),
        }
    }

    /// Push `k` negatives for the positive `feat`: pick one of its modules with
    /// probability ∝ its membership weight, then draw uniformly from that module's
    /// members. Returns `false` — pushing nothing, and counting the fallback —
    /// when the feature has no above-uniform module or the chosen module has
    /// fewer than two members in this pool (a lone feature has nothing to
    /// contrast with), so the caller falls back to the global draw.
    pub fn draw_negatives(
        &self,
        feat: u32,
        k: usize,
        out: &mut Vec<u32>,
        rng: &mut impl Rng,
    ) -> bool {
        let row = &self.rows[feat as usize];
        let Some(&(_, total)) = row.last() else {
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let m = if row.len() == 1 {
            row[0].0
        } else {
            let r = rng.random::<f32>() * total;
            row.iter()
                .find(|&&(_, cum)| r < cum)
                .map_or(row[row.len() - 1].0, |&(m, _)| m)
        };
        let pool = &self.members[m as usize];
        if pool.len() < 2 {
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        for _ in 0..k {
            out.push(pool[rng.random_range(0..pool.len())]);
        }
        true
    }

    /// Fallbacks since the last call, and reset.
    pub fn take_fallbacks(&self) -> usize {
        self.fallbacks.swap(0, Ordering::Relaxed)
    }
}

/////////////////////////
// Dictionary geometry //
/////////////////////////

/// Wang & Isola's uniformity term (arXiv:2005.10242) on a row-L2-normalized
/// table:
///
/// ```text
///   L = ln( (1 / (K(K−1))) · Σ_{a≠b} exp( −t · ‖x̂_a − x̂_b‖² ) )
/// ```
///
/// `‖x̂_a − x̂_b‖² = 2 − 2cos`, so this is a soft-min over cosine: the CLOSEST
/// pair dominates. That is the instrument we want, because the pathology is
/// duplicate directions, not mild global correlation — a mean squared cosine
/// weights every pair alike and is blind to one duplicated pair among many
/// well-spread ones. It keeps repelling past orthogonality too, where a squared
/// cosine is flat.
///
/// Normalization is INTERNAL to the term and nowhere else: without a norm
/// constraint the objective is gamed by rescaling rows, and everything outside
/// this function stays on the raw dot product the model is scored with. For
/// finite `K` the minimizer is a spherical code, not the uniform measure; the
/// clean-optimum story is asymptotic.
///
/// Do NOT read this value back as an evaluation metric — it is insensitive to
/// dimensional collapse. Evaluation is the participation ratio
/// ([`dictionary_participation_ratio`]) plus module purity.
pub fn dictionary_uniformity(table: &Tensor, t: f32) -> Result<Tensor> {
    let k = table.dim(0)?;
    if k < 2 {
        // No pairs: nothing to repel. A zero of the table's dtype and device.
        return table.zeros_like()?.sum_all();
    }
    let x = l2_normalize_dim(table, 1)?; // [K, H], unit rows
    let cos = x.matmul(&x.t()?)?; // [K, K]
                                  // `‖x̂_a − x̂_b‖² = 2 − 2cos`, then the kernel; the diagonal (distance 0,
                                  // kernel 1) is masked out exactly rather than subtracted, so the row
                                  // normalization's `+ε` cannot leak into the mean.
    let kernel = cos.affine(-2.0, 2.0)?.affine(-f64::from(t), 0.0)?.exp()?;
    let off_diagonal = (kernel.ones_like()? - Tensor::eye(k, DType::F32, table.device())?)?;
    let pairs = (k * (k - 1)) as f64;
    kernel
        .mul(&off_diagonal)?
        .sum_all()?
        .affine(1.0 / pairs, 0.0)?
        .log()
}

/// The dictionary-geometry penalty for one step: [`dictionary_uniformity`] on
/// the module dictionary `μ`, weighted by `lambda`; `None` when `lambda <= 0`,
/// so a zero weight never enters the graph.
///
/// Deliberately NOT gated on `is_frozen()`. The membership freeze detaches `π`;
/// it does not stop `μ` training — the exact cell–module term trains it every
/// step of the warm-up. A μ-penalty suppressed there would be absent for the
/// whole quarter of the run in which the dictionary is laid down from its randn
/// init.
///
/// Why a repulsion term exists at all: nothing on the dictionary side forbids
/// two identical modules. Duplication is a SYMMETRY of the mixture likelihood —
/// `π μ` is unchanged if two rows of `μ` are merged and the membership mass is
/// redistributed between them, and `π` absorbs exactly that. That is why
/// [`module_balance_prior`] cannot fix direction: it constrains how much mass
/// each module holds, and the duplicate configuration satisfies it perfectly.
pub fn module_dictionary_prior(
    modules: &FeatModules,
    lambda: f32,
    t: f32,
) -> Result<Option<Tensor>> {
    if lambda <= 0.0 {
        return Ok(None);
    }
    Ok(Some(
        dictionary_uniformity(&modules.mu, t)?.affine(f64::from(lambda), 0.0)?,
    ))
}

/// Participation ratio of the dictionary's directions,
/// `‖μ‖_F⁴ / ‖μ μᵀ‖_F²` — the `(Σλ)²/Σλ²` of the Gram's spectrum, without an
/// eigendecomposition. `M` for orthonormal rows, `1` when every row points the
/// same way. The EVALUATION readout the stages of the dictionary work are gated
/// on; never a training term.
pub fn dictionary_participation_ratio(mu: &Tensor) -> Result<f32> {
    let mu = mu.detach();
    let fro2: f32 = mu.sqr()?.sum_all()?.to_scalar()?;
    let gram2: f32 = mu.matmul(&mu.t()?)?.sqr()?.sum_all()?.to_scalar()?;
    Ok(if gram2 > 0.0 {
        fro2 * fro2 / gram2
    } else {
        0.0
    })
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

/// One `info` line per epoch on the membership: occupancy, small modules, row
/// entropy and support, the residual's share of the row norm, and how many
/// positives fell back to global negatives. Collapse shows up here long before it
/// shows up in the loss. `pi_host` is the current `[D × M]` membership.
pub fn log_membership_diagnostics(
    modules: &FeatModules,
    pi_host: &[f32],
    epoch: usize,
    epochs: usize,
    fallbacks: usize,
) -> Result<()> {
    let m = modules.n_modules;
    let n_features = modules.logits.dim(0)?;
    let dg = membership_diagnostics(pi_host, n_features, m, MODULE_SMALL_FLOOR);
    let r2: f32 = modules
        .residual
        .detach()
        .sqr()?
        .sum(1)?
        .mean_all()?
        .to_scalar()?;
    let rho2: f32 = modules
        .compose()?
        .detach()
        .sqr()?
        .sum(1)?
        .mean_all()?
        .to_scalar()?;
    // The dictionary's effective directions: the readout every geometry term on
    // μ is trying to move, and the gate for adding another.
    let pr = dictionary_participation_ratio(&modules.mu)?;
    info!(
        "epoch {}/{} modules{}: max occupancy {:.2}x uniform, {} of {} modules below {} \
         features, row entropy {:.3} nats, {:.2} modules/feature, mean ‖r‖² {:.3} vs \
         ‖ρ‖² {:.3}, dictionary PR {:.2} of {}, {} positives fell back to global negatives",
        epoch + 1,
        epochs,
        if modules.is_frozen() { " (frozen)" } else { "" },
        dg.max_occupancy_ratio,
        dg.n_small_modules,
        m,
        MODULE_SMALL_FLOOR,
        dg.mean_row_entropy,
        dg.mean_row_support,
        r2,
        rho2,
        pr,
        modules.mu.dim(1)?,
        fallbacks,
    );
    Ok(())
}

#[cfg(test)]
mod tests;
