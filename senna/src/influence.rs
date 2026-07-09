//! Influence core — first-order counterfactual estimates of "what would
//! training on this batch actually do?".
//!
//! **Estimand.** We differentiate the same **multinomial fit objective** that
//! `probe` reports as the outcome `Y`:
//! ```text
//!   ℓ_j(α) = Σ_g p_gj · log recon_gj,     p_j = x_j / Σ_g x_gj
//!   recon_j = Σ_k θ_jk β_k,               β_k = softmax_g(α_k · ρᵀ)
//! ```
//! with respect to the decoder's **topic embeddings** `α ∈ ℝ^{K×H}` — the
//! topic-side generative parameters. The `β = softmax_g(α·ρᵀ)` parameterization is
//! the embedded topic model of Dieng, Ruiz & Blei (2020). `ρ` (gene embedding) and
//! the encoder are held fixed, matching the design rule "scope influence to
//! β/anchors, not the encoder".
//!
//! **Closed-form gradient** (no autograd, no decoder rebuild):
//! ```text
//!   r_gj = p_gj / recon_gj                (nonzero only on observed genes)
//!   c_jk = Σ_g β_kg r_gj                  m_k  = Σ_g β_kg ρ_g
//!   u_jk = Σ_g β_kg r_gj ρ_g
//!   ∂ℓ_j/∂α_k = θ_jk · ( u_jk − c_jk · m_k )
//! ```
//!
//! **Counterfactual.** From per-cell mean gradients `ḡ` and empirical Fisher
//! diagonals `F` (mean squared per-cell gradient), the EWC-regularized one-step
//! update and the two axes of the decision plane are
//! ```text
//!   ḡ_Δ   = ḡ_new − ḡ_cal                 (query-specific pull, see below)
//!   Δ     = ḡ_Δ / (κ·F_cal + F_new)
//!   τ_new = ḡ_Δ·Δ   − ½ Δᵀ F_new Δ        (differential benefit of the query)
//!   τ_old = ḡ_cal·Δ − ½ Δᵀ F_cal Δ        (change on old data; < 0 ⇒ forgetting)
//! ```
//! This is the one-step / influence-function estimate of a refit (Hampel 1974;
//! Koh & Liang 2017; Giordano et al. 2019) under an EWC quadratic anchor
//! (Kirkpatrick et al. 2017), with `κ` playing the role of the generalized-Bayes
//! prior temperature (Bissiri, Holmes & Walker 2016). `ḡᵀF⁻¹ḡ` is Rao's score
//! statistic (Rao 1948).
//!
//! Regularizing by `(κ·F_cal + F_new)` rather than `F_cal` alone is what keeps
//! the axes independent: an unregularized Newton step collapses them to
//! `τ_old = −τ_new`. Novelty along low-`F_cal` (unexplored) directions is cheap
//! and safe; conflict along high-`F_cal` (well-known) directions is toxic.
//!
//! **Why the contrast `ḡ_new − ḡ_cal` (the control arm).** The model is *trained*
//! on a masked-NB loss over pseudobulks, so its gradient of this per-cell objective
//! is **not** zero even on in-distribution data. That non-zero direction is
//! *common-mode*: every batch shares it, and updating along it improves the fit of
//! new and old data alike. Left in, it swamps `τ_new`. Treating the calibration
//! batch as a **control arm** and taking the difference removes it — a
//! negative-control / difference-in-differences adjustment (Lipsitch, Tchetgen
//! Tchetgen & Cohen 2010). The estimand becomes "the effect of adding *this* batch
//! rather than an equally-sized in-distribution batch", which is precisely what
//! novelty-relative-to-reference means.
//!
//! Consequence: with query ≡ calibration, `ḡ_Δ = 0 ⇒ Δ = 0 ⇒ τ_new = τ_old = 0` —
//! an exact null, which is what makes the axes usable as calibrated statistics.
//! The first-order term `ḡ_cal·Δ` is retained in `τ_old` (no optimality assumed).
//!
//! **Reading `τ` as an average treatment effect.** Take unit = cell, treatment =
//! "apply `Δ` to the model", outcome `Y_j = ℓ_j`. A second-order expansion of
//! `(1/n) Σ_j [Y_j(1) − Y_j(0)]` about `α`, with `−Hessian ≈ F`, is exactly the
//! `ḡ·Δ − ½ΔᵀFΔ` form above. So `τ_old` is a **sample average treatment effect** on
//! the calibration cells of the intervention `Δ`, in log-likelihood-per-count units
//! (Neyman 1923; Rubin 1974). Two honest caveats:
//!  1. `Δ` is chosen *using* the query batch and `τ_new` is then evaluated *on* that
//!     same batch, so `τ_new` is optimistic — the honest version needs cross-fitting
//!     (Chernozhukov et al. 2018), as in one-step / TMLE estimators
//!     (van der Laan & Rubin 2006).
//!  2. As written, `τ_new` uses `ḡ_Δ` rather than `ḡ_new`, so it is a
//!     difference-in-differences-flavoured contrast, **not** the ATE on the query.
//!     Using the contrast only to *define* `Δ` (and scoring with `ḡ_new`) would make
//!     the pair `(τ_new, τ_old)` two ATEs of the same intervention.
//!
//! **Caveat: this is the *empirical* Fisher.** `F` is the mean of squared per-cell
//! gradients. It coincides with the Fisher — and hence with `−Hessian` — only at a
//! well-specified optimum, which is exactly the assumption we *cannot* make here
//! (`ḡ_cal ≠ 0`). Treat `F` as a positive-definite preconditioner and `τ` as a
//! **screening statistic**, not a calibrated second-order prediction (Kunstner,
//! Hennig & Balles 2019; Martens 2020; Amari 1998). A generalized Gauss-Newton
//! curvature (Schraudolph 2002) is the principled upgrade.
//!
//! # References
//! - Amari (1998) *Natural gradient works efficiently in learning.* Neural Comput. 10:251.
//! - Bissiri, Holmes & Walker (2016) *A general framework for updating belief
//!   distributions.* J. R. Stat. Soc. B 78:1103.
//! - Chernozhukov et al. (2018) *Double/debiased machine learning for treatment and
//!   structural parameters.* Econometrics J. 21:C1.
//! - Dieng, Ruiz & Blei (2020) *Topic modeling in embedding spaces.* TACL 8:439.
//! - Giordano, Stephenson, Liu, Jordan & Broderick (2019) *A Swiss army infinitesimal
//!   jackknife.* AISTATS, PMLR 89:1139.
//! - Hampel (1974) *The influence curve and its role in robust estimation.* JASA 69:383.
//! - Kirkpatrick et al. (2017) *Overcoming catastrophic forgetting in neural networks.*
//!   PNAS 114:3521.
//! - Koh & Liang (2017) *Understanding black-box predictions via influence functions.* ICML.
//! - Kunstner, Hennig & Balles (2019) *Limitations of the empirical Fisher approximation
//!   for natural gradient descent.* NeurIPS.
//! - Lipsitch, Tchetgen Tchetgen & Cohen (2010) *Negative controls: a tool for detecting
//!   confounding and bias in observational studies.* Epidemiology 21:383.
//! - Martens (2020) *New insights and perspectives on the natural gradient method.* JMLR 21:1.
//! - Nguyen, Li, Bui & Turner (2018) *Variational continual learning.* ICLR.
//! - Rao (1948) *Large sample tests of statistical hypotheses concerning several parameters.*
//!   Math. Proc. Camb. Phil. Soc. 44:50.
//! - Rubin (1974) *Estimating causal effects of treatments in randomized and nonrandomized
//!   studies.* J. Educ. Psychol. 66:688.
//! - Schraudolph (2002) *Fast curvature matrix-vector products for second-order gradient
//!   descent.* Neural Comput. 14:1723.
//! - van der Laan & Rubin (2006) *Targeted maximum likelihood learning.* Int. J. Biostat. 2(1).

use crate::embed_common::*;
use crate::topic::eval::GeneRemap;
use data_beans::sparse_io_vector::SparseIoVec;
use rayon::prelude::*;

#[cfg(test)]
mod tests;

/// Mean per-cell gradient and empirical Fisher diagonal over `α` (flattened
/// row-major `[K × H]`).
pub(crate) struct GradStats {
    pub n_cells: usize,
    pub g_mean: Vec<f64>,
    pub fisher: Vec<f64>,
}

/// Read-only per-block context for [`block_gradient_sums`].
pub(crate) struct BlockCtx<'a> {
    pub theta_kn: &'a Mat,
    pub exp_beta: &'a Mat,
    pub rho_dh: &'a Mat,
    pub m_kh: &'a Mat,
    pub recon_dn: &'a Mat,
}

/// Accumulate `Σ_j ∂ℓ_j/∂α` and `Σ_j (∂ℓ_j/∂α)²` over a block of cells.
///
/// `cells[jloc]` holds the cell's `(training-gene index, count)` nonzeros.
/// Returns `(g_sum, f_sum, n_used)`; cells with zero total count are skipped.
pub(crate) fn block_gradient_sums(
    cells: &[Vec<(usize, f32)>],
    ctx: &BlockCtx<'_>,
) -> (Vec<f64>, Vec<f64>, usize) {
    let kk = ctx.exp_beta.ncols();
    let hh = ctx.rho_dh.ncols();
    let mut g_sum = vec![0f64; kk * hh];
    let mut f_sum = vec![0f64; kk * hh];
    let mut c = vec![0f64; kk];
    let mut u = vec![0f64; kk * hh];
    let mut n_used = 0usize;

    for (jloc, nz) in cells.iter().enumerate() {
        let total: f64 = nz.iter().map(|&(_, x)| f64::from(x)).sum();
        if total <= 0.0 {
            continue;
        }
        n_used += 1;
        c.iter_mut().for_each(|v| *v = 0.0);
        u.iter_mut().for_each(|v| *v = 0.0);

        for &(g, x) in nz {
            let recon = f64::from(ctx.recon_dn[(g, jloc)]).max(1e-12);
            let r = (f64::from(x) / total) / recon;
            for (k, ck) in c.iter_mut().enumerate() {
                let w = f64::from(ctx.exp_beta[(g, k)]) * r;
                if w == 0.0 {
                    continue;
                }
                *ck += w;
                let base = k * hh;
                for (h, uh) in u[base..base + hh].iter_mut().enumerate() {
                    *uh += w * f64::from(ctx.rho_dh[(g, h)]);
                }
            }
        }

        for (k, &ck) in c.iter().enumerate() {
            let th = f64::from(ctx.theta_kn[(k, jloc)]);
            let base = k * hh;
            for h in 0..hh {
                let grad = th * (u[base + h] - ck * f64::from(ctx.m_kh[(k, h)]));
                g_sum[base + h] += grad;
                f_sum[base + h] += grad * grad;
            }
        }
    }
    (g_sum, f_sum, n_used)
}

/// Column-normalize `exp(β)` so each topic is a gene-simplex.
pub(crate) fn gene_simplex_beta(beta_dk: &Mat) -> Mat {
    let mut exp_beta = beta_dk.map(f32::exp);
    for k in 0..exp_beta.ncols() {
        let s: f32 = exp_beta.column(k).sum();
        if s > 0.0 {
            for g in 0..exp_beta.nrows() {
                exp_beta[(g, k)] /= s;
            }
        }
    }
    exp_beta
}

/// Mean per-cell gradient + empirical Fisher diagonal of the multinomial fit
/// objective w.r.t. `α`, over every cell of `data_vec`.
pub(crate) fn topic_gradient_stats(
    data_vec: &SparseIoVec,
    z_nk: &Mat,
    beta_dk: &Mat,
    rho_dh: &Mat,
    gene_remap: Option<&GeneRemap>,
    minibatch_size: usize,
) -> anyhow::Result<GradStats> {
    let exp_beta = gene_simplex_beta(beta_dk);
    anyhow::ensure!(
        exp_beta.nrows() == rho_dh.nrows(),
        "influence: dictionary genes ({}) != feature-embedding genes ({})",
        exp_beta.nrows(),
        rho_dh.nrows()
    );
    let m_kh = exp_beta.transpose() * rho_dh; // [K, H]
    let p = exp_beta.ncols() * rho_dh.ncols();
    let ntot = data_vec.num_columns();

    let parts: Vec<(Vec<f64>, Vec<f64>, usize)> = create_jobs(ntot, 0, Some(minibatch_size))
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<(Vec<f64>, Vec<f64>, usize)> {
            let csc = data_vec.read_columns_csc(lb..ub)?;
            let n_block = csc.ncols();
            let theta_kn = z_nk.rows(lb, n_block).map(f32::exp).transpose();
            let recon_dn = &exp_beta * &theta_kn;

            let cells: Vec<Vec<(usize, f32)>> = (0..n_block)
                .map(|jloc| {
                    let col = csc.col(jloc);
                    col.row_indices()
                        .iter()
                        .zip(col.values().iter())
                        .filter_map(|(&row_new, &val)| {
                            let row_train = match gene_remap {
                                Some(rm) => rm.new_to_train[row_new],
                                None => Some(row_new),
                            };
                            row_train.map(|t| (t, val))
                        })
                        .collect()
                })
                .collect();

            let ctx = BlockCtx {
                theta_kn: &theta_kn,
                exp_beta: &exp_beta,
                rho_dh,
                m_kh: &m_kh,
                recon_dn: &recon_dn,
            };
            Ok(block_gradient_sums(&cells, &ctx))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut g = vec![0f64; p];
    let mut f = vec![0f64; p];
    let mut n_used = 0usize;
    for (gs, fs, n) in parts {
        for i in 0..p {
            g[i] += gs[i];
            f[i] += fs[i];
        }
        n_used += n;
    }
    anyhow::ensure!(n_used > 0, "influence: no cells with positive counts");

    let n = n_used as f64;
    Ok(GradStats {
        n_cells: n_used,
        g_mean: g.iter().map(|v| v / n).collect(),
        fisher: f.iter().map(|v| v / n).collect(),
    })
}

/// The two counterfactual axes and the one-step update that produces them.
pub(crate) struct Counterfactual {
    /// Predicted per-cell fit gain on the new batch attributable to its
    /// **query-specific** direction (control-arm contrast). `0` when query ≡ calibration.
    pub tau_new: f64,
    /// Predicted per-cell fit change on the old (calibration) data; `< 0` ⇒ forgetting.
    pub tau_old: f64,
    /// ‖ḡ_new − ḡ_cal‖ — how far the query pulls beyond an in-distribution batch.
    pub g_delta_norm: f64,
    /// Per-topic ‖Δ_k‖ — which topics a joint retrain would move.
    pub delta_norm_per_topic: Vec<f64>,
}

/// EWC-regularized one-step counterfactual on the control-arm contrast:
/// `Δ = (ḡ_new − ḡ_cal) / (κ·F_cal + F_new)`.
pub(crate) fn counterfactual(
    new: &GradStats,
    old: &GradStats,
    kappa: f64,
    n_topics: usize,
) -> Counterfactual {
    let p = new.g_mean.len();
    let hh = p / n_topics.max(1);
    let mut delta = vec![0f64; p];
    let (mut tau_new, mut tau_old, mut g_delta_sq) = (0f64, 0f64, 0f64);

    for (i, d_slot) in delta.iter_mut().enumerate() {
        // Control-arm contrast: strip the common-mode pull that any
        // in-distribution batch would also produce.
        let g_delta = new.g_mean[i] - old.g_mean[i];
        g_delta_sq += g_delta * g_delta;

        let d = g_delta / (kappa * old.fisher[i] + new.fisher[i] + 1e-12);
        *d_slot = d;
        tau_new += g_delta * d - 0.5 * d * d * new.fisher[i];
        tau_old += old.g_mean[i] * d - 0.5 * d * d * old.fisher[i];
    }

    let delta_norm_per_topic = (0..n_topics)
        .map(|k| {
            (0..hh)
                .map(|h| delta[k * hh + h].powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    Counterfactual {
        tau_new,
        tau_old,
        g_delta_norm: g_delta_sq.sqrt(),
        delta_norm_per_topic,
    }
}
